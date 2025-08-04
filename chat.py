# -*- coding: utf-8 -*-
"""Interactive chat script for a finetuned Qwen‑2 model.

* Automatically infers model size (0.5B / 1B / 3B / 7B) from:
  1. `config.json` saved alongside the checkpoint; or
  2. hidden_size extracted from the checkpoint weights.
* Loads the correct `Qwen2Config` via `BaseModel.get_qwen2_config`.
* Starts a simple REPL; type `[exit]` to quit.
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

import torch
from loguru import logger
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer

from lib.base_model import BaseModel

# silence HF warnings
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = Path(os.getcwd())
CKPT_DIR = DATA_PATH / "checkpoints/last"
CKPT_PATH = CKPT_DIR / "final_checkpoint.pth"
CONFIG_PATH = CKPT_DIR / "config.json"  # optional

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
SIZE_LOOKUP = {
    1024: "0.6B",
    2048: "1.7B",
    2560: "4B",
    4096: "8B",
}

def infer_size_from_state_dict(sd: dict) -> str:
    """Detect hidden_size from any linear weight and map -> size label."""
    # look for embed_tokens or first transformer block
    for key, w in sd.items():
        if w.ndim == 2:  # [vocab, hidden] or [hidden, hidden]
            hidden = w.shape[-1]
            if hidden in SIZE_LOOKUP:
                return SIZE_LOOKUP[hidden]
    raise RuntimeError("Could not infer model size from checkpoint.")


@staticmethod
def get_qwen3_config(size: str) -> Qwen3Config:
    model_id = f"Qwen/Qwen3-{size}"
    return Qwen3Config.from_pretrained(model_id, trust_remote_code=True)

def load_model(checkpoint_path: Path, config_path: Path) -> tuple[Qwen3ForCausalLM, AutoTokenizer, str]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if config_path.exists():
        logger.info(f"Loading config from {config_path}")
        model_config_dict = json.loads(config_path.read_text())
        model_config = Qwen3Config.from_dict(model_config_dict)
        hidden_size = model_config.hidden_size
    else:
        logger.warning("config.json not found — inferring size from checkpoint weights …")
        hidden_size = None
        for w in ckpt["model_state_dict"].values():
            if w.ndim == 2:
                hidden_size = w.shape[-1]
                break
        if hidden_size not in SIZE_LOOKUP:
            raise RuntimeError("Could not infer model size from checkpoint.")
        size_label = SIZE_LOOKUP[hidden_size]
        model_config = BaseModel.get_qwen3_config(size_label)
        logger.info(f"Detected size = {size_label}")

    # tokenizer の読み込み（config.jsonがある場合、そこから model_id 推測）
    if config_path.exists() and "model_type" in model_config.to_dict():
        model_type = model_config.to_dict()["model_type"]
        # 仮に model_id を推定するなら名前で生成
        size_label = SIZE_LOOKUP.get(model_config.hidden_size, "unknown")
        model_id = f"Qwen/Qwen3-{size_label}"
    else:
        model_id = f"Qwen/Qwen3-{SIZE_LOOKUP.get(model_config.hidden_size, 'unknown')}"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = Qwen3ForCausalLM(model_config)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()

    size_label = SIZE_LOOKUP.get(model_config.hidden_size, "unknown")
    return model, tokenizer, size_label

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(CKPT_PATH))
    p.add_argument("--config", default=str(CONFIG_PATH))
    args = p.parse_args()

    model, tokenizer, sz = load_model(Path(args.checkpoint), Path(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # stdin: ensure UTF‑8
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    print(f"\n🚀 Chat ready — model size **{sz}**. Type [exit] to quit.\n")

    while True:
        try:
            usr = input("👤 You: ")
            if usr.strip().lower() == "[exit]":
                print("👋 Bye!"); break
            inp = tokenizer(usr, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inp, max_length=100, temperature=0.7, top_k=50, top_p=0.9, do_sample=True)
            ans = tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"🤖 Akabeko: {ans}\n")
        except KeyboardInterrupt:
            print("\nInterrupted. Type [exit] to quit.")
        except UnicodeDecodeError as e:
            print(f"Unicode error: {e}")

if __name__ == "__main__":
    main()
