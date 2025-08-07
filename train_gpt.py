import os
import json
import time
import argparse
import datetime
from glob import glob
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from lib.akabeko_dataset_gpt import AkabekoDataset
from lib.utils import save_checkpoint, format_hms, get_optimizer
import bitsandbytes as bnb


# =========================================================
# 基本パスの設定（データ・チェックポイント・ログ出力先）
# =========================================================
DATA_PATH = os.getcwd()
PT_DIR = os.path.join(DATA_PATH, "tokenized_dataset")
CHECKPOINT_DIR = os.path.join(DATA_PATH, "checkpoints")
LOG_DIR = os.path.join(DATA_PATH, "logs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# =========================================================
# DataLoader 用の collate 関数（パディングと-100マスク）
# =========================================================
def custom_collate(batch):
    # None を除去（前処理で弾かれたサンプル等）
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None

    # バッチ内で sequence 長が異なるため、後段で pad する
    input_ids = [sample["input_ids"] for sample in batch]
    attention_mask = [sample["attention_mask"] for sample in batch]
    labels = [sample["labels"] for sample in batch]

    # PAD トークンは 0、損失計算を無視するラベルは -100 にするのが一般的
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded
    }


# =========================================================
# トークナイザ設定の保存（再現性確保のため）
# =========================================================
def save_config(tokenizer, output_dir):
    config_path = os.path.join(output_dir, "config.json")
    if getattr(tokenizer, "is_fast", False):
        tokenizer_json_str = tokenizer.backend_tokenizer.to_str()
    else:
        tokenizer_json_str = tokenizer.to_json_string()

    tokenizer_config = json.loads(tokenizer_json_str)
    config = {
        "model_name": tokenizer.name_or_path,
        "tokenizer_config": tokenizer_config
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# =========================================================
# メイン処理（DDPでの継続事前学習）
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    # Optimizer は 8bit AdamW をデフォルト（オプティマイザ状態でのVRAMを削減）
    parser.add_argument("--optimizer", type=str, default="adamw8bit", choices=["muon", "adamw", "adamw8bit"])
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save", type=int, default=2000)  # 何ステップごとに中間保存するか
    parser.add_argument("--dataset_dir", type=str, default=PT_DIR)
    parser.add_argument("--resume", action="store_true")  # ここでは未使用（拡張余地）
    parser.add_argument("--hf_model", type=str, required=True, help="例: openai/gpt-oss-20b")
    parser.add_argument("--max_len", type=int, default=1024)  # サンプル長の上限（長すぎるものを除外）
    parser.add_argument("--accumulation_steps", type=int, default=128)  # 勾配蓄積ステップ数（大バッチ相当）
    parser.add_argument("--log_interval", type=int, default=1)
    args = parser.parse_args()

    # ---------------------------
    # DDP 初期化
    # ---------------------------
    # torchrun が設定する LOCAL_RANK を用いて、このプロセスが担当する GPU を決める
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(hours=2))
    device = torch.device(f"cuda:{local_rank}")

    # ---------------------------
    # 高速化・安定化のヒント
    # ---------------------------
    # TF32 を許可（Ampere 以降で matmul/cudnn のスループットが向上、精度影響は小）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---------------------------
    # Tokenizer 読み込み
    # ---------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=False)

    # ---------------------------
    # 事前トークナイズ済みデータの読み込み（.pt）
    # ---------------------------
    dataset_pt_files = sorted(glob(os.path.join(args.dataset_dir, "*.pt")))
    raw_data = []
    if local_rank == 0:
        print(f"{args.dataset_dir} 内に {len(dataset_pt_files)} 個の .pt ファイルを検出しました")
    for pt_file in dataset_pt_files:
        if local_rank == 0:
            print(f"読み込み中: {pt_file}")
        # 大規模データを想定して mmap 読み込み（PyTorch バージョンによって挙動が異なる場合あり）
        data = torch.load(pt_file, weights_only=False, mmap=True)
        # 想定される形式（dict もしくは list[dict] など）に対応
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

    # AkabekoDataset はサンプル(dict)を受け取り、input_ids/attention_mask/labels を返す想定
    train_dataset = AkabekoDataset(raw_data, tokenizer)

    # 上限長でフィルタ（学習安定性とメモリ節約のため）
    MAX_LEN = args.max_len
    train_dataset = [s for s in train_dataset if len(s["input_ids"]) <= MAX_LEN]
    if local_rank == 0:
        print(f"max_len={MAX_LEN} 適用後のデータセットサイズ: {len(train_dataset)} サンプル")

    # ---------------------------
    # DataLoader（DDP用に DistributedSampler を使用）
    # ---------------------------
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True, seed=0, drop_last=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,           # I/O を並列化
        pin_memory=True,         # ホスト→GPU 転送の高速化
        persistent_workers=True, # ワーカーを使い回してオーバーヘッド削減
        collate_fn=custom_collate,
    )

    # ---------------------------
    # モデル読み込み（bf16 + FlashAttention 2 / SDPA）
    # ---------------------------
    # flash-attn が使える環境なら自動で有効化、なければ PyTorch SDPA を使う
    attn_impl = "sdpa"
    if local_rank == 0:
        print("FlashAttention を無効化し、SDPA を使用します")

    # DDP を使うため、device_map="auto" は使用しない
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=torch.bfloat16,      # 学習は bfloat16 で実行（安定・幅広いダイナミックレンジ）
        attn_implementation=attn_impl,   # attention 実装を指定
    )
    # 学習時はキャッシュを無効化（メモリ削減）
    model.config.use_cache = False
    # 勾配チェックポイント（メモリ削減）。use_reentrant=False の方が近年のHFで安定
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    model.train()

    # ---------------------------
    # DDP でモデルをラップ（1 プロセス = 1 GPU）
    # ---------------------------
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    # ---------------------------
    # Optimizer と Scheduler の用意
    # ---------------------------
    # get_optimizer は使わず bitsandbytes を直接指定（OOM対策で Paged を優先）
    if hasattr(bnb.optim, "PagedAdamW8bit"):
        optimizer = bnb.optim.PagedAdamW8bit(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
            betas=(0.9, 0.95),
        )
        if local_rank == 0:
            print("Using bitsandbytes PagedAdamW8bit")
    else:
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
            betas=(0.9, 0.95),
        )
        if local_rank == 0:
            print("Using bitsandbytes AdamW8bit")


    num_training_steps = max(1, len(train_loader) * args.epoch)
    # cosine with warmup。ウォームアップは全体の ~3% か 1000 ステップの小さい方
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=min(1000, int(0.03 * num_training_steps)),
        num_training_steps=num_training_steps,
        num_cycles=0.5,
    )

    # ---------------------------
    # ロギング用ファイルの準備
    # ---------------------------
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    logfile_name = f"{date_str}_{len(train_dataset)}samples_{args.optimizer}.csv"
    logfile_path = os.path.join(LOG_DIR, logfile_name)

    total_start_time = time.time()
    total_steps = num_training_steps
    accumulation_steps = max(1, args.accumulation_steps)
    step = -1  # エポック終了時のセーブで参照するために初期化

    # ---------------------------
    # 学習ループ
    # ---------------------------
    for epoch in range(args.epoch):
        # DDP では各エポックで seed を変えるため set_epoch が必要
        sampler.set_epoch(epoch)

        for step, batch in enumerate(train_loader):
            if batch is None:
                continue

            # non_blocking=True で HtoD 転送の並列性を高める
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            # すべてパディング（attention_mask==0）ならスキップ
            if batch["attention_mask"].sum(dim=1).eq(0).all():
                continue

            # 勾配蓄積中は no_sync で AllReduce を抑制（通信オーバーヘッド削減）
            sync_ctx = model.no_sync() if ((step + 1) % accumulation_steps != 0) else nullcontext()
            with sync_ctx:
                # bf16 autocast（安定かつ高速、AMP の grad scaler は bf16 では不要）
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(**batch)
                    # 勾配蓄積に合わせて loss をスケールダウン
                    loss = outputs.loss / accumulation_steps

                # NaN/Inf 検出で安全にスキップ
                if not torch.isfinite(loss):
                    if local_rank == 0:
                        print(f"[Rank {local_rank}] NaN/Inf を検出したため step {step} をスキップします")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # 逆伝播
                loss.backward()

            # 勾配蓄積ステップごとに更新
            if (step + 1) % accumulation_steps == 0:
                # 大規模モデル安定化のため勾配クリッピングを推奨
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Rank 0 のみログ出力・中間保存
            if local_rank == 0:
                current_step = epoch * len(train_loader) + step + 1
                elapsed = time.time() - total_start_time
                eta = (elapsed / current_step) * (total_steps - current_step) if current_step > 0 else 0.0

                # 表示上は非スケールの loss を見る（分かりやすさ重視）
                unscaled_loss = loss.item() * accumulation_steps
                if (step + 1) % args.log_interval == 0:
                    print(f"[Rank {local_rank}] Epoch {epoch+1}/{args.epoch} | "
                          f"Step {step+1}/{len(train_loader)} | "
                          f"Loss {unscaled_loss:.4f} | "
                          f"LR {optimizer.param_groups[0]['lr']:.6f} | "
                          f"Elapsed {format_hms(elapsed)} | ETA {format_hms(eta)}")

                # CSV ログ追記
                log_entry = f"{epoch+1},{step+1},{current_step},{unscaled_loss:.6f},{optimizer.param_groups[0]['lr']:.6f},{elapsed:.2f}\n"
                with open(logfile_path, 'a') as f:
                    f.write(log_entry)

                # 中間チェックポイント保存
                if args.save > 0 and (step + 1) % args.save == 0:
                    # DDP でラップされたモデルから実体を取り出す
                    unwrapped = model.module if isinstance(model, DDP) else model
                    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                    save_checkpoint(unwrapped, optimizer, epoch, step, checkpoint_dir=CHECKPOINT_DIR)
                    save_config(tokenizer, CHECKPOINT_DIR)

        # エポック終了時に「最後の」チェックポイントを保存
        if local_rank == 0:
            last_ckpt_dir = os.path.join(CHECKPOINT_DIR, "last")
            os.makedirs(last_ckpt_dir, exist_ok=True)
            unwrapped = model.module if isinstance(model, DDP) else model
            save_checkpoint(unwrapped, optimizer, epoch, step, checkpoint_dir=last_ckpt_dir, filename="final_checkpoint.pth")
            save_config(tokenizer, last_ckpt_dir)

    # 全プロセス同期・終了
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()