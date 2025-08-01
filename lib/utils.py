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
def save_checkpoint(model, optimizer, epoch: int, step: int,
                    checkpoint_dir: str, filename: str = "checkpoint.pth") -> None:
    is_final = filename == "final_checkpoint.pth"

    if is_final:
        # 👇 final は stepディレクトリなしでそのまま保存
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        metadata_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
    else:
        # 👇 通常は step_000000 ディレクトリ作成
        step_dir = os.path.join(checkpoint_dir, f"step_{step:06d}")
        os.makedirs(step_dir, exist_ok=True)
        checkpoint_path = os.path.join(step_dir, filename)
        metadata_path = os.path.join(step_dir, "checkpoint_metadata.json")

    model_to_save = model.module if isinstance(model, DDP) else model

    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    metadata = {"last_epoch": epoch, "last_step": step}
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    # latest_checkpoint.txt を常に更新（オプション）
    latest_link = os.path.join(checkpoint_dir, "latest_checkpoint.txt")
    with open(latest_link, "w") as f:
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


# ---------- オプティマイザ ---------- #
def get_optimizer(name: str, model, lr: float = 1e-3, wd: float = 0.1):
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))

    if name == "muon":
        muon_params = [p for n, p in model.named_parameters()
                       if p.ndim >= 2 and "embed_tokens" not in n and "lm_head" not in n]
        
        # 🔥 ここで Tensor の ID で比較
        muon_param_ids = {id(p) for p in muon_params}
        adamw_params = [p for n, p in model.named_parameters()
                        if id(p) not in muon_param_ids]

        return Muon(lr=lr, wd=wd, muon_params=muon_params, adamw_params=adamw_params)

    raise ValueError(f"Unsupported optimizer: {name}")

