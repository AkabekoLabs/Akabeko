# utils.py
import os
import json
import datetime
import torch
from torch.utils.data import ConcatDataset
from torch.nn.parallel import DistributedDataParallel as DDP

from .akabeko_dataset import AkabekoDataset
from .muon_optimizer import Muon

# ---------- 汎用 ---------- #
def format_hms(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def human_readable(num: int, suffix: str = "B") -> str:
    for unit in ["", "K", "M", "G", "T"]:
        if abs(num) < 1000.0:
            return f"{num:.1f}{unit}{suffix}"
        num /= 1000.0
    return f"{num:.1f}P{suffix}"

# ---------- チェックポイント ---------- #
# ---------- チェックポイント ---------- #
def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    step: int,
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
) -> None:                                 # ← 閉じカッコと戻り値
    is_final = filename == "final_checkpoint.pth"

    if is_final:
        # 👇 final は stepディレクトリなしでそのまま保存
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        metadata_path  = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
    else:
        # 👇 通常は step_000000 ディレクトリ作成
        step_dir       = os.path.join(checkpoint_dir, f"step_{step:06d}")
        os.makedirs(step_dir, exist_ok=True)
        checkpoint_path = os.path.join(step_dir, filename)
        metadata_path   = os.path.join(step_dir, "checkpoint_metadata.json")

    model_to_save = model.module if isinstance(model, DDP) else model

    # ---- 重みとオプティマイザ ----
    torch.save(
        {
            "epoch": epoch,
            "step":  step,
            "model_state_dict":     model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    # ---- メタ情報 ----
    with open(metadata_path, "w") as f:
        json.dump({"last_epoch": epoch, "last_step": step}, f)

    # ---- config.json を同期保存 ----
    cfg_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(model_to_save.config.to_dict(), f, indent=2, ensure_ascii=False)

    # ---- latest_checkpoint.txt を更新 ----
    with open(os.path.join(checkpoint_dir, "latest_checkpoint.txt"), "w") as f:
        f.write(checkpoint_path + "\n")

# ---------- データセット ---------- #
def get_traindata(pt_dir):
    pt_files = [f for f in os.listdir(pt_dir) if f.endswith(".pt")]
    pt_files.sort()  # 順番を安定させる

    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in: {pt_dir}")

    datasets = []
    for pt in pt_files:
        path = os.path.join(pt_dir, pt)
        print(f"📦 Loading: {path}")
        ds = torch.load(path, weights_only=False, mmap=True)
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]
    else:
        return ConcatDataset(datasets)

from bitsandbytes.optim import AdamW8bit

def get_optimizer(optimizer_name, model, lr=1e-3, wd=0.1):
    if optimizer_name == "adamw8bit":
        return AdamW8bit(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == "muon":
        from lib.muon import Muon
        return Muon(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")