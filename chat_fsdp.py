# -*- coding: utf-8 -*-
"""Interactive chat script for a finetuned gpt-oss-20b / Qwen3 model (FSDP checkpoints)."""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

import torch
from loguru import logger
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
)

# silence HF warnings
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# ------------------------------------------------------------
# SDPA 互換ラッパ（PyTorch 2.4+ / それ未満 両対応）
# ------------------------------------------------------------
def _build_sdpa_ctx():
    try:
        from torch.nn.attention import sdpa_kernel as _new_sdpa_kernel, SDPBackend
        def _ctx():
            return _new_sdpa_kernel(
                backends=[
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ]
            )
        return _ctx
    except Exception:
        def _ctx():
            return torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_mem_efficient=True, enable_math=True
            )
        return _ctx

sdpa_ctx = _build_sdpa_ctx()

# ------------------------------------------------------------
# Paths (defaults)
# ------------------------------------------------------------
DATA_PATH = Path(os.getcwd())
CKPT_DIR = DATA_PATH / "checkpoints" / "last"
CKPT_PATH = CKPT_DIR / "final_checkpoint.pth"          # FSDP 最終
ALT_PT    = DATA_PATH / "checkpoints" / "pytorch_model.pt"  # 中間
CONFIG_PATH = CKPT_DIR / "config.json"                 # 学習側 save_config のもの

SIZE_LOOKUP = {1024: "0.6B", 2048: "1.7B", 2560: "4B", 2880: "20B", 4096: "8B"}

def infer_size_from_state_dict(sd: dict) -> str:
    for k, w in sd.items():
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            hidden = w.shape[-1]
            if hidden in SIZE_LOOKUP:
                return SIZE_LOOKUP[hidden]
    raise RuntimeError("Could not infer model size from checkpoint tensors.")

def _load_ckpt(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    if "model_state_dict" not in ckpt:
        # 互換：まれにそのまま state_dict が入っている場合
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            ckpt = {"model_state_dict": ckpt}
        else:
            raise RuntimeError(f"Invalid checkpoint format: keys={list(ckpt.keys())[:5]}")
    return ckpt

def _read_training_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    try:
        obj = json.loads(config_path.read_text())
        # 学習側 save_config は {"model_name": "...", "tokenizer_config": {...}}
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _build_model_from_model_name(model_name: str):
    """
    モデルの全重みはDLせず、AutoConfig→from_config でアーキテクチャだけ構築。
    """
    logger.info(f"Building model from config of [{model_name}] (no base weights download)")
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    return model, tok

def _fallback_build_qwen3(size_label: str):
    # 最終手段（ネット不可等で AutoConfig 取れない時）
    logger.warning(f"Falling back to Qwen3 skeleton: size={size_label}")
    qcfg = Qwen3Config()  # 最低限。必要なら独自の BaseModel 実装に差し替えてください。
    model = Qwen3ForCausalLM(qcfg)
    tok = AutoTokenizer.from_pretrained(f"Qwen/Qwen3-{size_label}", trust_remote_code=True)
    return model, tok

def load_model(checkpoint_path: Path, config_path: Path, dtype: torch.dtype = torch.bfloat16):
    ckpt = _load_ckpt(checkpoint_path)
    train_cfg = _read_training_config(config_path)

    model_name = train_cfg.get("model_name", "") or train_cfg.get("_name_or_path", "")
    model = None
    tok = None

    if model_name:
        try:
            model, tok = _build_model_from_model_name(model_name)
        except Exception as e:
            logger.warning(f"Failed to build from model_name={model_name}: {e}")

    if model is None:
        # config.json が無い／壊れている時はサイズ推定 → Qwen3 骨組み
        size = infer_size_from_state_dict(ckpt["model_state_dict"])
        model, tok = _fallback_build_qwen3(size)

    # 重みロード（FSDP FULL_STATE を想定）
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        logger.warning(f"Missing keys: {len(missing)} (showing up to 10): {missing[:10]}")
    if unexpected:
        logger.warning(f"Unexpected keys: {len(unexpected)} (showing up to 10): {unexpected[:10]}")

    # dtype & device 未設定（後で .to(device, dtype) する）
    if getattr(model.config, "pad_token_id", None) is None and getattr(model.config, "eos_token_id", None) is not None:
        model.config.pad_token_id = model.config.eos_token_id

    # サイズ表記（推定）
    try:
        any_w = next(t for t in model.state_dict().values() if t.ndim == 2)
        size_label = SIZE_LOOKUP.get(any_w.shape[-1], "unknown")
    except Exception:
        size_label = "unknown"

    return model, tok, size_label

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(CKPT_PATH),
                   help="FSDPで保存した .pth / .pt のパス（final でも中間でもOK）")
    p.add_argument("--config", default=str(CONFIG_PATH),
                   help="学習側 save_config() が出力した config.json のパス")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = p.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    model, tokenizer, sz = load_model(Path(args.checkpoint), Path(args.config), dtype=dtype)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=dtype)
    model.eval()

    # sdpa 有効化
    print(f"\n🚀 Chat ready — model size **{sz}**, dtype={dtype}, device={device}. Type [exit] to quit.\n")
    if device.type == "cuda":
        print("Using SDPA attention backends.\n")

    # stdin: ensure UTF-8
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=True,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
    )

    with sdpa_ctx():
        while True:
            try:
                usr = input("👤 You: ")
                if usr.strip().lower() == "[exit]":
                    print("👋 Bye!"); break

                inputs = tokenizer(usr, return_tensors="pt").to(device)
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype if dtype != torch.float32 else None):
                    out = model.generate(**inputs, **gen_kwargs)

                # 先頭のプロンプトを安全に取り除く
                gen = out[0][inputs["input_ids"].shape[1]:]
                ans = tokenizer.decode(gen, skip_special_tokens=True)
                print(f"🤖 Akabeko: {ans}\n")

            except KeyboardInterrupt:
                print("\nInterrupted. Type [exit] to quit.")
            except UnicodeDecodeError as e:
                print(f"Unicode error: {e}")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
