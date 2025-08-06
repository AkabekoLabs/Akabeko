# -*- coding: utf-8 -*-
"""Interactive chat script for a finetuned Qwen3 / gpt-oss-20b model."""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

import torch
from loguru import logger
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

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
    2880: "6.7B",  # 追加: OSS-20B hidden_size
    4096: "8B",
}

def infer_size_from_state_dict(sd: dict) -> str:
    """Detect hidden_size from any linear weight and map -> size label."""
    for key, w in sd.items():
        if w.ndim == 2:  # [vocab, hidden] or [hidden, hidden]
            hidden = w.shape[-1]
            if hidden in SIZE_LOOKUP:
                return SIZE_LOOKUP[hidden]
    raise RuntimeError("Could not infer model size from checkpoint.")

def load_model(checkpoint_path: Path, config_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if config_path.exists():
        logger.info(f"Loading config from {config_path}")
        model_config_dict = json.loads(config_path.read_text())
        model_type = model_config_dict.get("model_type", "")

        if model_type == "gpt_oss":
            # AutoConfig.from_dict は存在しないため、直接 AutoModelForCausalLM.from_pretrained を使う
            model_id = model_config_dict.get("_name_or_path", "openai/gpt-oss-20b")
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif model_type == "qwen3":
            model_config = Qwen3Config.from_dict(model_config_dict)
            model = Qwen3ForCausalLM(model_config)
            tokenizer = AutoTokenizer.from_pretrained(model_config_dict.get("_name_or_path", "Qwen/Qwen3"), trust_remote_code=True)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            raise RuntimeError(f"Unknown model_type: {model_type}")

        hidden_size = model.config.hidden_size
    else:
        logger.warning("config.json not found — inferring size from checkpoint weights …")
        hidden_size = infer_size_from_state_dict(ckpt["model_state_dict"])
        size_label = SIZE_LOOKUP.get(hidden_size, "unknown")
        model_config = BaseModel.get_qwen3_config(size_label)
        model = Qwen3ForCausalLM(model_config)
        tokenizer = AutoTokenizer.from_pretrained(f"Qwen/Qwen3-{size_label}", trust_remote_code=True)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info(f"Detected size = {size_label}")

    model.config.pad_token_id = model.config.eos_token_id
    model.eval()

    size_label = SIZE_LOOKUP.get(hidden_size, "unknown")
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
