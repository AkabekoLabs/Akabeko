import os
import json
import time
import argparse
import datetime
from glob import glob
from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from lib.akabeko_dataset_gpt import AkabekoDataset
from lib.utils import format_hms, get_optimizer
import bitsandbytes as bnb

# ===== FSDP =====
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, BackwardPrefetch, MixedPrecision
)
from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# =========================================================
# sdpa_kernel 互換ラッパ（新旧APIどちらでも動く）
# =========================================================
def _build_sdpa_ctx():
    """
    PyTorch >= 2.4: torch.nn.attention.sdpa_kernel(backends=[SDPBackend....])
    それ未満:        torch.backends.cuda.sdp_kernel(enable_flash=..., enable_mem_efficient=..., enable_math=...)
    """
    try:
        from torch.nn.attention import sdpa_kernel as _new_sdpa_kernel, SDPBackend
        def _ctx():
            return _new_sdpa_kernel(
                backends=[
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH
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

# =========================================================
# パス
# =========================================================
DATA_PATH = os.getcwd()
PT_DIR = os.path.join(DATA_PATH, "tokenized_dataset")
CHECKPOINT_DIR = os.path.join(DATA_PATH, "checkpoints")
LOG_DIR = os.path.join(DATA_PATH, "logs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =========================================================
# collate（最後の保険で hard cap も可能）
# =========================================================
def custom_collate(batch, hard_cap=None):
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None
    input_ids = [sample["input_ids"] for sample in batch]
    attention_mask = [sample["attention_mask"] for sample in batch]
    labels = [sample["labels"] for sample in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    if hard_cap is not None and input_ids_padded.size(1) > hard_cap:
        L = hard_cap
        input_ids_padded      = input_ids_padded[:, :L]
        attention_mask_padded = attention_mask_padded[:, :L]
        labels_padded         = labels_padded[:, :L]

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded
    }

# =========================================================
# tokenizer設定保存
# =========================================================
def save_config(tokenizer, output_dir):
    config_path = os.path.join(output_dir, "config.json")
    if getattr(tokenizer, "is_fast", False):
        tokenizer_json_str = tokenizer.backend_tokenizer.to_str()
    else:
        tokenizer_json_str = tokenizer.to_json_string()
    tokenizer_config = json.loads(tokenizer_json_str)
    config = {"model_name": tokenizer.name_or_path, "tokenizer_config": tokenizer_config}
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

# =========================================================
# FSDP対応の安全な保存（FULL_STATEをrank0に集約→CPU保存）
# =========================================================
def fsdp_safe_save(model: FSDP, optimizer, epoch, step, checkpoint_dir, filename="pytorch_model.pt"):
    """
    全rankで呼び出し必須。内部で FULL_STATE を rank0 に集約し rank0 のみ書き込み。
    """
    is_dist = dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    is_rank0 = (rank == 0)

    if is_rank0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # すべてのrankで足並みを揃える
    if is_dist:
        dist.barrier()

    state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_cfg):
        full_state = model.state_dict()

    if is_rank0:
        save_path = os.path.join(checkpoint_dir, filename)
        torch.save(
            {
                "model_state_dict": full_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "step": step,
            },
            save_path,
        )
        print(f"[Rank 0] Saved FSDP FULL_STATE to: {save_path}")

    if is_dist:
        dist.barrier()

# 中間チェックポイント：元のフォルダ構成に近い形で ep_step サブフォルダへ保存
def save_mid_checkpoint(model, optimizer, epoch, step, base_dir, tokenizer):
    subdir = os.path.join(base_dir, f"ep{epoch+1}_step{step+1:06d}")
    fsdp_safe_save(model, optimizer, epoch, step, checkpoint_dir=subdir, filename="pytorch_model.pt")
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        save_config(tokenizer, subdir)

# 最終チェックポイント：/last に保存
def save_final_checkpoint(model, optimizer, epoch, step, base_dir, tokenizer):
    last_dir = os.path.join(base_dir, "last")
    fsdp_safe_save(model, optimizer, epoch, step, checkpoint_dir=last_dir, filename="final_checkpoint.pth")
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        save_config(tokenizer, last_dir)

# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="adamw8bit",
                        choices=["muon", "adamw", "adamw8bit", "pagedadamw8bit"])
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save", type=int, default=2000)
    parser.add_argument("--dataset_dir", type=str, default=PT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hf_model", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=896)
    parser.add_argument("--accumulation_steps", type=int, default=1)  # 安全側で1
    parser.add_argument("--log_interval", type=int, default=1)
    args = parser.parse_args()

    # ---------------- DDP init ----------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(hours=2))
    device = torch.device(f"cuda:{local_rank}")

    try:
        # --------------- backend hints ---------------
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # --------------- tokenizer ---------------
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=False)

        # --------------- load dataset (.pt) ---------------
        dataset_pt_files = sorted(glob(os.path.join(args.dataset_dir, "*.pt")))
        raw_data = []
        if local_rank == 0:
            print(f"{args.dataset_dir} 内に {len(dataset_pt_files)} 個の .pt を検出")
        for pt_file in dataset_pt_files:
            if local_rank == 0:
                print(f"読み込み中: {pt_file}")
            data = torch.load(pt_file, weights_only=False, mmap=True)
            if isinstance(data, dict):
                raw_data.append([data])
            elif isinstance(data, list):
                for sample in data:
                    if isinstance(sample, dict):
                        raw_data.append([sample])
                    elif isinstance(sample, list):
                        raw_data.append(sample)
            else:
                if local_rank == 0:
                    print(f"想定外のデータ形式: {pt_file}")
        if local_rank == 0:
            print(f"読み込んだ生サンプル数: {len(raw_data)}")

        train_dataset = AkabekoDataset(raw_data, tokenizer)
        MAX_LEN = args.max_len
        train_dataset = [s for s in train_dataset if len(s["input_ids"]) <= MAX_LEN]
        if local_rank == 0:
            print(f"max_len={MAX_LEN} 適用後サイズ: {len(train_dataset)}")

        # --------------- DataLoader / Sampler ---------------
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, seed=0, drop_last=True
        )
        collate = (lambda b: custom_collate(b, hard_cap=args.max_len))
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=2,  # 安全め
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate,
        )

        # --------------- Model (bf16 + SDPA) ---------------
        attn_impl = "sdpa"
        if local_rank == 0:
            print("FlashAttention 無効、SDPA 使用（互換 sdpa_ctx で制御）")

        # HFモデル読込（CPUのまま）→ FSDPでshardに移す（.to(device)はしない）
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # ---- FSDP wrap（FULL_SHARD + auto wrap）----
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        # 層ごとに自動wrap（全モデル一括wrapはNG）
        auto_wrap = partial(size_based_auto_wrap_policy, min_num_params=50_000_000)

        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap,
            mixed_precision=mp,
            use_orig_params=True,
            device_id=device,
            limit_all_gathers=True,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            sync_module_states=True,
        )

        # --------------- Optimizer / Scheduler ---------------
        # OptimizerはFSDPラップ後の model.parameters() を使うこと！
        if args.optimizer == "pagedadamw8bit":
            if not hasattr(bnb.optim, "PagedAdamW8bit"):
                raise RuntimeError("bitsandbytes に PagedAdamW8bit がありません。")
            optimizer = bnb.optim.PagedAdamW8bit(
                model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95)
            )
            if local_rank == 0:
                print("Using bitsandbytes PagedAdamW8bit")
        elif args.optimizer == "adamw8bit":
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95)
            )
            if local_rank == 0:
                print("Using bitsandbytes AdamW8bit")
        elif args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95)
            )
            if local_rank == 0:
                print("Using torch.optim.AdamW")
        elif args.optimizer == "muon":
            optimizer = get_optimizer("muon", model.parameters(), args.lr, args.wd)
            if local_rank == 0:
                print("Using Muon optimizer via get_optimizer")
        else:
            raise ValueError(f"Unknown optimizer: {args.optimizer}")

        accumulation_steps = max(1, args.accumulation_steps)
        steps_per_epoch_optimizer = max(1, (len(train_loader) + accumulation_steps - 1) // accumulation_steps)
        num_optimizer_steps = steps_per_epoch_optimizer * args.epoch
        warmup_steps = max(1, min(1000, int(0.03 * num_optimizer_steps)))
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_optimizer_steps,
            num_cycles=0.5,
        )

        # --------------- logging ---------------
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        logfile_name = f"{date_str}_{len(train_dataset)}samples_{args.optimizer}.csv"
        logfile_path = os.path.join(LOG_DIR, logfile_name)

        total_start_time = time.time()
        total_steps = len(train_loader) * args.epoch
        step = -1

        # 協調スキップ用フラグ
        oom_window_flag = torch.zeros(1, device=device)
        skip_window = False

        # --------------- Train loop ---------------
        with sdpa_ctx():
            for epoch in range(args.epoch):
                sampler.set_epoch(epoch)

                for step, batch in enumerate(train_loader):
                    if batch is None:
                        continue

                    # ウィンドウ先頭でフラグをリセット
                    if (step % accumulation_steps) == 0:
                        oom_window_flag.zero_()
                        skip_window = False

                    # H2D
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    if batch["attention_mask"].sum(dim=1).eq(0).all():
                        continue

                    # この step が蓄積境界か
                    boundary = ((step + 1) % accumulation_steps == 0)

                    # 既にウィンドウ中に OOM/NaN が出ていれば、この step も同期しない
                    sync_ctx = nullcontext() if (boundary and not skip_window) else model.no_sync()

                    local_bad = False  # OOM or NaN の検出
                    with sync_ctx:
                        try:
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                outputs = model(**batch)
                                loss = outputs.loss / accumulation_steps

                            if not torch.isfinite(loss):
                                local_bad = True
                                optimizer.zero_grad(set_to_none=True)
                                if local_rank == 0:
                                    print(f"[Rank {local_rank}] NaN/Inf 検出: step {step+1}（この蓄積ウィンドウは破棄予定）")
                            else:
                                loss.backward()

                        except torch.cuda.OutOfMemoryError:
                            local_bad = True
                            torch.cuda.empty_cache()
                            optimizer.zero_grad(set_to_none=True)
                            if local_rank == 0:
                                print(f"[Rank {local_rank}] OOM 検出: step {step+1}（この蓄積ウィンドウは破棄予定）")

                    # rank 間でフラグを集約
                    if local_bad:
                        oom_window_flag.fill_(1.0)
                    dist.all_reduce(oom_window_flag, op=dist.ReduceOp.MAX)
                    if oom_window_flag.item() > 0:
                        skip_window = True

                    # 蓄積境界で更新 or 破棄
                    if boundary:
                        if skip_window:
                            window_start = step - (accumulation_steps - 1)
                            window_end = step
                            if local_rank == 0:
                                print(f"[Rank 0] 協調スキップ: step {window_start+1}〜{window_end+1} を破棄（OOM/NaN）")
                            optimizer.zero_grad(set_to_none=True)
                            torch.cuda.empty_cache()
                        else:
                            FSDP.clip_grad_norm_(model, max_norm=1.0)
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            torch.cuda.empty_cache()

                    # ログ（skip_window 中は loss 表示を抑制）
                    if local_rank == 0 and not skip_window and 'loss' in locals():
                        current_step = epoch * len(train_loader) + step + 1
                        elapsed = time.time() - total_start_time
                        eta = (elapsed / current_step) * (total_steps - current_step) if current_step > 0 else 0.0
                        unscaled_loss = loss.item() * accumulation_steps
                        if (step + 1) % args.log_interval == 0:
                            print(f"[Rank {local_rank}] Epoch {epoch+1}/{args.epoch} | "
                                  f"Step {step+1}/{len(train_loader)} | "
                                  f"Loss {unscaled_loss:.4f} | "
                                  f"LR {optimizer.param_groups[0]['lr']:.3e} | "
                                  f"Elapsed {format_hms(elapsed)} | ETA {format_hms(eta)}")
                        try:
                            log_entry = f"{epoch+1},{step+1},{current_step},{unscaled_loss:.6f},{optimizer.param_groups[0]['lr']:.8f},{elapsed:.2f}\n"
                            with open(logfile_path, 'a') as f:
                                f.write(log_entry)
                        except Exception as e:
                            print(f"[Rank {local_rank}] ログ書き込み失敗: {e}")

                    # ======= 中間チェックポイント（全rankで呼ぶ）======
                    if args.save > 0 and (step + 1) % args.save == 0:
                        torch.cuda.synchronize()
                        dist.barrier()
                        save_mid_checkpoint(model, optimizer, epoch, step, CHECKPOINT_DIR, tokenizer)
                        dist.barrier()
                        torch.cuda.empty_cache()

            # ======= エポック末尾の保存（全rankで呼ぶ）======
            torch.cuda.synchronize()
            dist.barrier()
            save_final_checkpoint(model, optimizer, epoch, step, CHECKPOINT_DIR, tokenizer)
            dist.barrier()
            torch.cuda.empty_cache()

        dist.barrier()

    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass

if __name__ == "__main__":
    main()
