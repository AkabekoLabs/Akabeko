# train_gpt_ds.py
import os
import json
import time
import argparse
import datetime
from glob import glob
import math
import inspect
import shutil
from typing import List

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.akabeko_dataset_gpt import AkabekoDataset
from lib.utils import format_hms, get_optimizer
import bitsandbytes as bnb
import deepspeed


# ---------------- FlashAttention-2 compatibility ----------------
def _fa2_is_installed() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def _fa2_supports_s_aux() -> bool:
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
    except Exception:
        try:
            from flash_attn import flash_attn_varlen_func  # type: ignore
        except Exception:
            return False
    try:
        sig = inspect.signature(flash_attn_varlen_func)
        params = sig.parameters
        if "s_aux" in params:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    except Exception:
        return False


def _select_attn_impl(requested: str, strict: bool) -> str:
    req = requested.lower()
    if req == "sdpa":
        return "sdpa"
    if req == "flash2":
        if _fa2_is_installed() and _fa2_supports_s_aux():
            return "flash_attention_2"
        if strict:
            raise RuntimeError(
                "Requested flash2 but local flash-attn is incompatible with HF (missing 's_aux'). "
                "Upgrade flash-attn or use --attn_impl auto/sdpa."
            )
        print("⚠️ flash2 requested but flash-attn is incompatible. Falling back to SDPA.")
        return "sdpa"
    if _fa2_is_installed() and _fa2_supports_s_aux():
        return "flash_attention_2"
    return "sdpa"


def _build_sdpa_ctx():
    try:
        from torch.nn.attention import sdpa_kernel as _new_sdpa_kernel, SDPBackend

        def _ctx():
            return _new_sdpa_kernel(
                backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
            )

        return _ctx
    except Exception:
        def _ctx():
            return torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_mem_efficient=True, enable_math=True
            )
        return _ctx


sdpa_ctx = _build_sdpa_ctx()

# ---------------- paths ----------------
DATA_PATH = os.getcwd()
PT_DIR = os.path.join(DATA_PATH, "tokenized_dataset")
CHECKPOINT_DIR = os.path.join(DATA_PATH, "checkpoints")
LOG_DIR = os.path.join(DATA_PATH, "logs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------- collate ----------------
def custom_collate(batch, hard_cap=None):
    batch = [s for s in batch if s is not None]
    if not batch:
        return None
    ids = [s["input_ids"] for s in batch]
    amsk = [s["attention_mask"] for s in batch]
    labs = [s["labels"] for s in batch]
    ids = pad_sequence(ids, batch_first=True, padding_value=0)
    amsk = pad_sequence(amsk, batch_first=True, padding_value=0)
    labs = pad_sequence(labs, batch_first=True, padding_value=-100)
    if hard_cap is not None and ids.size(1) > hard_cap:
        L = hard_cap
        ids, amsk, labs = ids[:, :L], amsk[:, :L], labs[:, :L]
    return {"input_ids": ids, "attention_mask": amsk, "labels": labs}


# ---------------- save tokenizer config ----------------
def save_config(tokenizer, output_dir):
    path = os.path.join(output_dir, "config.json")
    if getattr(tokenizer, "is_fast", False):
        tokenizer_json_str = tokenizer.backend_tokenizer.to_str()
    else:
        tokenizer_json_str = tokenizer.to_json_string()
    cfg = {"model_name": tokenizer.name_or_path, "tokenizer_config": json.loads(tokenizer_json_str)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# ---------------- DS default config ----------------
def build_default_ds_config(micro_bs: int, grad_accum: int):
    zero3 = {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_scatter": True,
        "stage3_param_persistence_threshold": 0,
        "stage3_gather_16bit_weights_on_model_save": True,
        "allgather_partitions": True,
        "use_multi_rank_bucket_allreduce": True,
        "offload_param": {"device": "none"},
        "offload_optimizer": {"device": "none"},
    }
    return {
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": micro_bs,
        "gradient_accumulation_steps": grad_accum,
        "zero_optimization": zero3,
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False
        },
        "wall_clock_breakdown": False,
    }


# ---------------- simple cosine w/ warmup ----------------
class CosineWarmup:
    def __init__(self, optimizer, base_lr, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.warmup = max(1, int(warmup_steps))
        self.total = max(self.warmup + 1, int(total_steps))
        self.step_idx = 0
        self._apply_lr(0.0)

    def _lr_at(self, t):
        if t < self.warmup:
            return self.base_lr * (t / self.warmup)
        progress = (t - self.warmup) / max(1, self.total - self.warmup)
        return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _apply_lr(self, lr_val):
        for g in self.optimizer.param_groups:
            g["lr"] = lr_val

    def step(self):
        self.step_idx += 1
        self._apply_lr(self._lr_at(self.step_idx))

    @property
    def lr(self):
        return self.optimizer.param_groups[0]["lr"]


# ---------------- helper: normalize ds_config ----------------
def normalize_ds_config(ds_config: dict, use_bnb: bool):
    z = ds_config.setdefault("zero_optimization", {})
    if "zero_allow_untested_optimizer" in z:
        val = bool(z.pop("zero_allow_untested_optimizer"))
        ds_config["zero_allow_untested_optimizer"] = val
    if use_bnb:
        z["offload_param"] = {"device": "none"}
        z["offload_optimizer"] = {"device": "none"}
        ds_config["zero_allow_untested_optimizer"] = True
    return ds_config


def is_cpu_offload(zcfg: dict) -> bool:
    op = (zcfg.get("offload_param", {}) or {}).get("device", "none")
    oo = (zcfg.get("offload_optimizer", {}) or {}).get("device", "none")
    return (op == "cpu") or (oo == "cpu")


# ---------------- checkpoint helpers ----------------
def _bytes_to_gb(x: int) -> float:
    return x / (1024 ** 3)


def _get_free_gb(path: str) -> float:
    usage = shutil.disk_usage(path)
    return _bytes_to_gb(usage.free)


def _list_ckpt_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    items = [os.path.join(root, d) for d in os.listdir(root)]
    items = [d for d in items if os.path.isdir(d)]
    items.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return items


def _rotate_ckpts(root: str, keep: int, rank: int):
    if rank != 0:
        return
    dirs = _list_ckpt_dirs(root)
    for d in dirs[keep:]:
        try:
            print(f"[Rotate] remove old checkpoint: {d}")
            shutil.rmtree(d, ignore_errors=True)
        except Exception as e:
            print(f"[Rotate] failed to remove {d}: {e}")


def _save_sharded(engine, tokenizer, tag: str, out_root: str, keep: int, min_free_gb: float):
    rank = dist.get_rank()
    skip = torch.tensor([0], device=engine.device)
    if rank == 0:
        free_gb = _get_free_gb(out_root)
        if free_gb < min_free_gb:
            print(f"⚠️ Skip saving (free {free_gb:.1f}GB < min_free_gb {min_free_gb:.1f}GB)")
            skip.fill_(1)
    dist.broadcast(skip, src=0)
    dist.barrier()

    if int(skip.item()) == 0:
        out_dir = os.path.join(out_root, tag)
        os.makedirs(out_dir, exist_ok=True)
        engine.save_checkpoint(out_dir, tag=tag)
        if rank == 0:
            save_config(tokenizer, out_dir)
    dist.barrier()
    _rotate_ckpts(out_root, keep=keep, rank=rank)
    dist.barrier()


def _export_hf(engine, tokenizer, out_dir: str, tag: str, max_shard_gb: int = 10):
    t0 = time.time()
    try:
        os.makedirs(out_dir, exist_ok=True)
        state_dict = engine._zero3_consolidated_16bit_state_dict()
        if dist.get_rank() == 0:
            from transformers import AutoModelForCausalLM
            cfg = engine.module.config
            target = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
            target.to("cpu")
            target.load_state_dict(state_dict, strict=False)
            sub = os.path.join(out_dir, f"hf_{tag}")
            os.makedirs(sub, exist_ok=True)
            target.save_pretrained(sub, safe_serialization=True, max_shard_size=f"{max_shard_gb}GB")
            tokenizer.save_pretrained(sub)
            print(f"✅ HF snapshot saved at: {sub} ({time.time()-t0:.1f}s)")
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"HF export failed: {e}")
    finally:
        dist.barrier()


# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", type=str, default="pagedadamw8bit",
                   choices=["fusedadam", "adamw", "adamw8bit", "pagedadamw8bit", "muon"])
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--epoch", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--save", type=int, default=2000)
    p.add_argument("--save_keep", type=int, default=2)
    p.add_argument("--min_free_gb", type=float, default=150.0)
    p.add_argument("--save_hf_on_fail", action="store_true")
    p.add_argument("--hf_only_save", action="store_true")
    p.add_argument("--export_dir", type=str, default="")
    p.add_argument("--export_max_shard_gb", type=int, default=10)
    p.add_argument("--dataset_dir", type=str, default=PT_DIR)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--hf_model", type=str, required=True)
    p.add_argument("--max_len", type=int, default=896)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--deepspeed_config", type=str, default="")
    p.add_argument("--allow_cpu_offload", action="store_true")
    p.add_argument("--attn_impl", type=str, default="auto", choices=["auto", "sdpa", "flash2"])
    p.add_argument("--attn_strict", action="store_true")
    p.add_argument("--flush_interval", type=int, default=0)
    p.add_argument("--local_rank", type=int, default=0)
    args = p.parse_args()

    # ---- init distributed
    if not dist.is_initialized():
        deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # ---- matmul/cudnn knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # ---- tokenizer / dataset
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=False, trust_remote_code=True)
    pt_files = sorted(glob(os.path.join(args.dataset_dir, "*.pt")))
    raw = []
    if dist.get_rank() == 0:
        print(f"{args.dataset_dir} 内に {len(pt_files)} 個の .pt を検出")
    for f in pt_files:
        if dist.get_rank() == 0:
            print(f"読み込み中: {f}")
        data = torch.load(f, weights_only=False, mmap=True)
        if isinstance(data, dict):
            raw.append([data])
        elif isinstance(data, list):
            for s in data:
                raw.append([s] if isinstance(s, dict) else s)
    train_dataset = AkabekoDataset(raw, tokenizer)
    train_dataset = [s for s in train_dataset if len(s["input_ids"]) <= args.max_len]
    if dist.get_rank() == 0:
        print(f"max_len={args.max_len} 適用後サイズ: {len(train_dataset)}")

    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True, seed=0, drop_last=True
    )
    collate = (lambda b: custom_collate(b, hard_cap=args.max_len))
    num_workers = min(8, os.cpu_count() or 2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=collate,
    )

    # ---- DeepSpeed config
    use_bnb = args.optimizer in ("adamw8bit", "pagedadamw8bit")
    if args.deepspeed_config and os.path.isfile(args.deepspeed_config):
        with open(args.deepspeed_config, "r") as f:
            ds_config = json.load(f)
        ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
        ds_config["gradient_accumulation_steps"] = max(1, args.accumulation_steps)
    else:
        ds_config = build_default_ds_config(
            micro_bs=args.batch_size,
            grad_accum=max(1, args.accumulation_steps),
        )
    ds_config = normalize_ds_config(ds_config, use_bnb=use_bnb)
    z = ds_config.setdefault("zero_optimization", {})

    if is_cpu_offload(z) and not args.allow_cpu_offload and dist.get_rank() == 0:
        print("⚠️ DeepSpeed: CPU Offload が有効です（遅い）。--allow_cpu_offload を付けない限り解除します。")
        z["offload_param"] = {"device": "none"}
        z["offload_optimizer"] = {"device": "none"}

    if dist.get_rank() == 0:
        print(f"[DeepSpeed] offload_param={(z.get('offload_param') or {}).get('device','none')}, "
              f"offload_optimizer={(z.get('offload_optimizer') or {}).get('device','none')}")

    # ---- model & attention impl （★ partitioned initは使わない）
    choose_impl = _select_attn_impl(args.attn_impl, strict=args.attn_strict)
    if dist.get_rank() == 0:
        print(f"Attention impl: {choose_impl}  (requested={args.attn_impl}, strict={args.attn_strict})")
        if choose_impl != "flash_attention_2" and args.attn_impl in ("auto", "flash2") and _fa2_is_installed():
            print("⚠️ flash-attn は検出されましたが、Transformers と非互換のため SDPA にフォールバックしました。")

    # ここで CPU に実体化してから DeepSpeed.initialize へ
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=torch.bfloat16,
        attn_implementation=choose_impl,
        low_cpu_mem_usage=True,
        device_map={'': 'cpu'},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    # ---- optimizer
    client_optimizer = None
    if args.optimizer == "fusedadam":
        ds_config.pop("optimizer", None)
        ds_config["optimizer"] = {
            "type": "FusedAdam",
            "params": {"lr": args.lr, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": args.wd},
        }
        if dist.get_rank() == 0:
            print("Using DeepSpeed FusedAdam (GPU fused optimizer)")
    elif args.optimizer in ("pagedadamw8bit", "adamw8bit"):
        ds_config["zero_allow_untested_optimizer"] = True
        if args.optimizer == "pagedadamw8bit":
            if not hasattr(bnb.optim, "PagedAdamW8bit"):
                raise RuntimeError("bitsandbytes に PagedAdamW8bit がありません。")
            client_optimizer = bnb.optim.PagedAdamW8bit(
                model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95)
            )
            if dist.get_rank() == 0:
                print("Using bitsandbytes PagedAdamW8bit")
        else:
            client_optimizer = bnb.optim.AdamW8bit(
                model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95)
            )
            if dist.get_rank() == 0:
                print("Using bitsandbytes AdamW8bit")
    elif args.optimizer == "adamw":
        adamw_kwargs = dict(lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95), foreach=False)
        if "fused" in torch.optim.AdamW.__init__.__code__.co_varnames:
            adamw_kwargs["fused"] = False
        client_optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
        if dist.get_rank() == 0:
            print("Using torch.optim.AdamW (foreach=False)")
    elif args.optimizer == "muon":
        client_optimizer = get_optimizer("muon", model.parameters(), args.lr, args.wd)
        if dist.get_rank() == 0:
            print("Using Muon optimizer via get_optimizer")
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # ---- DeepSpeed initialize
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=client_optimizer,
        config=ds_config,
    )
    device = engine.device

    if dist.get_rank() == 0:
        try:
            print(f"[diag] world={dist.get_world_size()} dp={engine.dp_world_size} "
                  f"zero={engine.zero_optimization_stage}")
        except Exception:
            pass

    # ---- LR scheduler
    accum = max(1, args.accumulation_steps)
    steps_per_epoch_opt = max(1, (len(train_loader) + accum - 1) // accum)
    num_opt_steps = steps_per_epoch_opt * args.epoch
    warmup_steps = max(1, min(1000, int(0.03 * num_opt_steps)))
    lr_sched = CosineWarmup(engine.optimizer, base_lr=args.lr, warmup_steps=warmup_steps, total_steps=num_opt_steps)

    # ---- logging & throughput
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    logfile = os.path.join(LOG_DIR, f"{date_str}_{len(train_dataset)}samples_{args.optimizer}_ds.csv")
    total_start = time.time()
    total_steps = len(train_loader) * args.epoch
    oom_flag = torch.zeros(1, device=device)
    skip_window = False
    step_idx_global = -1

    running_tok = 0
    running_time = 0.0

    def _sync_empty_cache():
        dist.barrier()
        torch.cuda.empty_cache()
        dist.barrier()

    with sdpa_ctx():
        for epoch in range(args.epoch):
            sampler.set_epoch(epoch)
            for step, batch in enumerate(train_loader):
                step_idx_global = step
                if batch is None:
                    continue
                if (step % accum) == 0:
                    oom_flag.zero_()
                    skip_window = False

                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                if batch["attention_mask"].sum(dim=1).eq(0).all():
                    continue

                boundary = ((step + 1) % accum == 0)
                local_bad = False
                t0 = time.time()
                step_tokens = int(batch["attention_mask"].sum().item())

                try:
                    outputs = engine(**batch)
                    loss = outputs.loss / accum
                    if not torch.isfinite(loss):
                        local_bad = True
                        engine.zero_grad(set_to_none=True)
                        if dist.get_rank() == 0:
                            print(f"[Rank {local_rank}] NaN/Inf 検出: step {step+1}")
                    else:
                        engine.backward(loss)
                except torch.cuda.OutOfMemoryError:
                    local_bad = True
                    torch.cuda.empty_cache()
                    engine.zero_grad(set_to_none=True)
                    if dist.get_rank() == 0:
                        print(f"[Rank {local_rank}] OOM 検出: step {step+1}（この蓄積ウィンドウは破棄）")

                if local_bad:
                    oom_flag.fill_(1.0)
                dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)
                if oom_flag.item() > 0:
                    skip_window = True

                if boundary:
                    if skip_window:
                        if dist.get_rank() == 0:
                            ws, we = step - (accum - 1), step
                            print(f"[Rank 0] 協調スキップ: step {ws+1}〜{we+1} を破棄（OOM/NaN）")
                        engine.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                    else:
                        engine.step()
                        lr_sched.step()

                t1 = time.time()
                step_dt = (t1 - t0)
                step_tokens_t = torch.tensor([step_tokens], dtype=torch.long, device=device)
                dist.all_reduce(step_tokens_t, op=dist.ReduceOp.SUM)
                step_tokens_total = int(step_tokens_t.item())
                running_tok += step_tokens_total
                running_time += step_dt

                if args.flush_interval > 0 and ((step + 1) % args.flush_interval == 0):
                    _sync_empty_cache()

                if dist.get_rank() == 0 and not skip_window and 'loss' in locals():
                    cur = epoch * len(train_loader) + step + 1
                    elapsed = time.time() - total_start
                    eta = (elapsed / cur) * (total_steps - cur) if cur > 0 else 0.0
                    unscaled = loss.item() * accum
                    toks_per_s = (running_tok / running_time) if running_time > 0 else 0.0

                    if (step + 1) % args.log_interval == 0:
                        print(
                            f"[Rank {local_rank}] Epoch {epoch+1}/{args.epoch} | "
                            f"Step {step+1}/{len(train_loader)} | "
                            f"Loss {unscaled:.4f} | LR {lr_sched.lr:.3e} | "
                            f"Throughput {toks_per_s:.0f} tok/s | "
                            f"Elapsed {format_hms(elapsed)} | ETA {format_hms(eta)}"
                        )
                    try:
                        with open(logfile, "a") as f:
                            f.write(
                                f"{epoch+1},{step+1},{cur},{unscaled:.6f},"
                                f"{lr_sched.lr:.8f},{elapsed:.2f},{toks_per_s:.2f}\n"
                            )
                    except Exception as e:
                        print(f"[Rank {local_rank}] ログ書き込み失敗: {e}")

                if (not args.hf_only_save) and args.save > 0 and (step + 1) % args.save == 0:
                    tag = f"epoch{epoch}_step{step}"
                    try:
                        _save_sharded(
                            engine, tokenizer, tag, CHECKPOINT_DIR,
                            keep=args.save_keep, min_free_gb=args.min_free_gb
                        )
                    except Exception as e:
                        if dist.get_rank() == 0:
                            print(f"❌ ZeRO checkpoint save failed at tag={tag}: {e}")
                            print("推定原因：ディスク不足 or FS エラー。古い ckpt を削除するか、保存間隔/保存先を見直してください。")
                        if args.save_hf_on_fail:
                            fallback_dir = args.export_dir or os.path.join(DATA_PATH, "export_hf")
                            _export_hf(engine, tokenizer, fallback_dir, tag, max_shard_gb=args.export_max_shard_gb)

        # final save / export
        tag = f"final_epoch{args.epoch}_step{step_idx_global}"

        if not args.hf_only_save:
            if dist.get_rank() == 0:
                print("Saving final ZeRO checkpoint...")
            try:
                _save_sharded(
                    engine, tokenizer, tag, CHECKPOINT_DIR,
                    keep=args.save_keep, min_free_gb=args.min_free_gb
                )
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"❌ Final ZeRO checkpoint save failed: {e}")
                if args.save_hf_on_fail:
                    fallback_dir = args.export_dir or os.path.join(DATA_PATH, "export_hf")
                    _export_hf(engine, tokenizer, fallback_dir, tag, max_shard_gb=args.export_max_shard_gb)

        if args.hf_only_save or args.export_dir:
            export_dir = args.export_dir or os.path.join(DATA_PATH, "export_hf")
            if dist.get_rank() == 0:
                print(f"Exporting HF snapshot to {export_dir} ...")
            _export_hf(engine, tokenizer, export_dir, tag, max_shard_gb=args.export_max_shard_gb)

    dist.barrier()


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
