import os
import math
import torch
from loguru import logger
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from transformers import (
    Qwen3Config,
    Qwen3ForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import argparse
import datetime
from torch.utils.data import Sampler
import torch.distributed as dist
import json
import datetime
from lib.base_model import BaseModel
from lib.muon_optimizer import Muon
from lib.akabeko_dataset import AkabekoDataset
from lib.utils import (
    save_checkpoint,
    format_hms,
    get_traindata,
    get_optimizer,
    human_readable,
)

DATA_PATH = os.getcwd()
PT_DIR = os.path.join(DATA_PATH, "tokenized_dataset")
TOKENIZED_DATA_DIR = os.path.join(DATA_PATH, "tokenized_dataset")
CHECKPOINT_DIR = os.path.join(DATA_PATH, "checkpoints")
LOG_DIR = os.path.join(DATA_PATH, "logs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TOKENIZED_DATA_DIR, exist_ok=True)
os.makedirs(PT_DIR, exist_ok=True)
# FlashAttention優先 (cuDNN mathベースSDPAを無効化)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# TF32関連
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.nn.utils.rnn import pad_sequence

def custom_collate(batch):
    """改善されたcollate関数 - 最小限の修正"""
    # 1. 有効なサンプルのみフィルタリング
    valid_batch = []
    for sample in batch:
        if "input_ids" in sample and len(sample["input_ids"]) > 0:
            # 512トークンまでtruncate
            truncated_ids = sample["input_ids"][:512]
            if len(truncated_ids) > 0:  # 再度チェック
                valid_batch.append({
                    "input_ids": torch.LongTensor(truncated_ids)
                })
    
    # 2. 有効なバッチが空の場合はNoneを返す
    if len(valid_batch) == 0:
        return None
    
    # 3. パディング処理
    input_ids = [sample["input_ids"] for sample in valid_batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    
    # 4. attention_maskの作成と検証
    attention_mask = (input_ids_padded != 0).long()
    
    # 5. 各シーケンスが少なくとも1つの有効トークンを持つことを確認
    seq_lengths = attention_mask.sum(dim=1)
    if seq_lengths.min() == 0:
        # 空のシーケンスを除外
        valid_indices = seq_lengths > 0
        input_ids_padded = input_ids_padded[valid_indices]
        attention_mask = attention_mask[valid_indices]
        
        # それでも空になった場合
        if input_ids_padded.size(0) == 0:
            return None
    
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": input_ids_padded.clone(),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen", choices=["qwen", "llama"])
    parser.add_argument("--optimizer", type=str, default="muon", choices=["muon", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--size", type=str, default="1B", choices=["0.5B", "1B", "3B", "7B"])
    parser.add_argument("--save", type=int, default=5000)
    parser.add_argument("--dataset_dir", type=str, default=TOKENIZED_DATA_DIR)
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--hf_model", type=str, default=None, help="HuggingFace model path like Qwen/Qwen1.5-0.5B")
    args = parser.parse_args()

    # ----------------------------------------
    # まず分散環境を初期化
    # ----------------------------------------
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl', 
        init_method='env://',
        timeout=datetime.timedelta(hours=2)
    )
    device = torch.device(f"cuda:{local_rank}")
    
    # ----------------------------------------
    # rank=0だけデータセットを作り (なければ)、.pt に保存
    # ----------------------------------------
    dataset_pt_path = os.path.join(PT_DIR, "train_dataset.pt")

    if local_rank == 0 and not os.path.exists(dataset_pt_path):
        train_dataset = get_traindata(args.dataset_dir)
        torch.save(train_dataset, dataset_pt_path)
        print(f"Dataset is saved to: {dataset_pt_path}")
    else:
        if local_rank == 0:
            print(f"Dataset file {dataset_pt_path} already exists. Skip building dataset.")

    # 全rankで同期した後、.pt を読む
    dist.barrier(device_ids=[local_rank])

    # PTファイルからトレーニングデータを読み込み
    train_dataset = torch.load(dataset_pt_path, weights_only=False, mmap=True)
    train_dataset = [sample for sample in train_dataset if len(sample["input_ids"]) > 0]
    # GPU数の取得
    gpu_count = dist.get_world_size()
    
    # トークン数の取得
    total_tokens = sum(len(sample["input_ids"]) for sample in train_dataset)

    # 単位Bで表示
    token_str = human_readable(total_tokens)
    
    # 日付
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # ファイル名作成
    logfile_name = f"{date_str}_{args.size}_{token_str}_{args.optimizer}_{gpu_count}GPU.csv"
    logfile_path = os.path.join(LOG_DIR, logfile_name)

    # モデルの準備
    model = None
    start_epoch = 0
    start_step = 0
    resume_ckpt_path = os.path.join(CHECKPOINT_DIR, "last", "final_checkpoint.pth")

    if args.resume and os.path.exists(resume_ckpt_path):
        # ✅ ① checkpoint から継続学習
        print(f"Resuming from checkpoint: {resume_ckpt_path}")
        config = BaseModel.get_qwen3_config(args.size)
        config.use_sliding_window = False
        model = Qwen3ForCausalLM(config)
        ckpt = torch.load(resume_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        start_epoch = ckpt.get("epoch", 0)
        start_step = ckpt.get("step", 0)
        model = model.to(device=device, dtype=torch.bfloat16)

    elif args.hf_model:
        # ✅ ② HuggingFaceの学習済みモデルから継続学習
        print(f"Loading pretrained HF model: {args.hf_model}")
        model = Qwen3ForCausalLM.from_pretrained(
            args.hf_model,
            torch_dtype=torch.bfloat16
        ).to(device)

        if local_rank == 0:
            config_path = os.path.join(CHECKPOINT_DIR, "last", "config.json")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(model.config.to_dict(), f, indent=2, ensure_ascii=False)
    else:
        # ✅ ③ ゼロから初期化（configベース）
        print("Initializing model from scratch using config")
        if args.model == "qwen":
            config = BaseModel.get_qwen3_config(args.size)
            config.use_sliding_window = False  # 明示的に無効化
            model = Qwen3ForCausalLM(config).to(device=device, dtype=torch.bfloat16)

    model = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=False,     # 未使用パラメータがない前提で高速化
        gradient_as_bucket_view=True,     # 2.0以降で追加された高速化オプション
        static_graph=False,                 # 2.1以降、1つのグラフが固定なら更に高速化（まだ実験的）
        bucket_cap_mb=512,            # SXM GPUでは200～512MBが効果的（実測値に基づき調整）
        broadcast_buffers=False,      # 通常不要
    )

    model.train()

    # 分散用Samplerを作成
    from torch.utils.data.distributed import DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset, shuffle=True, seed=0, drop_last=True
    )
    torch.set_float32_matmul_precision('high')
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # DistributedSampler利用時はshuffle不要
        sampler=train_sampler,
        num_workers=os.cpu_count(),  # CPUを最大限利用
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=custom_collate,
    )
    print("DEBUG: len(train_dataset) =", len(train_dataset))
    total_steps = len(train_loader) * args.epoch
    # Optimizer & Scheduler
    optimizer = get_optimizer(args.optimizer, model, lr=args.lr, wd=args.wd)
    if args.resume and os.path.exists(resume_ckpt_path):
        optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * args.epoch,
        num_cycles=0.5,
    )

    # ----------------------------------------
    # トレーニングループ
    # ----------------------------------------
    total_start_time = time.time()

    # rank=0 だけが全体ステップ数を計算(表示用)
    if local_rank == 0:
        total_steps_per_rank = len(train_loader) * args.epoch
        print(f"Total steps per rank: {total_steps_per_rank}")

    for epoch in range(args.epoch):
        train_sampler.set_epoch(epoch)  # epochごとにShuffleシードを切り替え
        steps_per_rank = len(train_loader)

        for step, batch in enumerate(train_loader):
            step_start_time = time.time()
            if batch is None:
                print(f"[Rank:{local_rank}] Skipping empty batch at step {step}")
                continue

            batch = {k: v.to(device) for k, v in batch.items()}

            # バッチサイズとシーケンス長の確認
            if batch["input_ids"].size(0) == 0 or batch["input_ids"].size(1) == 0:
                print(f"[Rank:{local_rank}] Empty batch dimensions at step {step}")
                continue

            # attention_maskが全てゼロでないことを確認
            if batch["attention_mask"].sum() == 0:
                print(f"[Rank:{local_rank}] All-zero attention mask at step {step}")
                continue
            
            # 各シーケンスに少なくとも1つの有効トークンがあることを確認
            seq_lengths = batch["attention_mask"].sum(dim=1)
            if (seq_lengths == 0).any():
                print(f"[Rank:{local_rank}] Found empty sequences, filtering...")
                valid_mask = seq_lengths > 0
                batch = {k: v[valid_mask] for k, v in batch.items()}
                
                # フィルタリング後も空でないことを確認
                if batch["input_ids"].size(0) == 0:
                    print(f"[Rank:{local_rank}] Batch became empty after filtering")
                    continue
            
            try:
                outputs = model(**batch)  # use_flash_attn=False を削除（デフォルト値を使用）
                loss = outputs.loss
            except RuntimeError as e:
                print(f"[Rank:{local_rank}] Error at step {step}: {str(e)}")
                print(f"[Rank:{local_rank}] Batch shapes - input_ids: {batch['input_ids'].shape}, attention_mask: {batch['attention_mask'].shape}")
                print(f"[Rank:{local_rank}] Attention mask sum: {batch['attention_mask'].sum()}")
                continue
                
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # まず勾配をゼロに
            optimizer.zero_grad(set_to_none=True)

            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            elapsed = step_end_time - total_start_time
            
            if local_rank == 0:
                current_step = epoch * len(train_loader) + step + 1
                remaining_time = (elapsed / current_step) * (total_steps - current_step)

                log_entry = f"{epoch+1},{step},{current_step},{loss.item():.6f},{optimizer.param_groups[0]['lr']:.6f},{step_duration:.4f},{elapsed:.2f}"
                with open(logfile_path, 'a') as f:
                    f.write(log_entry + "\n")

                print(
                    f"[Rank:{local_rank}] Epoch: {epoch+1}/{args.epoch}, Step: {step}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, Step time: {step_duration:.3f}s, "
                    f"Elapsed: {format_hms(elapsed)}, ETA: {format_hms(remaining_time)}"
                )

            # checkpoint 保存（中間保存）
            if step % args.save == 1 and step > 1 and local_rank == 0:
                save_checkpoint(model, optimizer, epoch, step, checkpoint_dir=CHECKPOINT_DIR)
                torch.cuda.empty_cache()

    # 最終セーブ
    if local_rank == 0:
        last_ckpt_dir = os.path.join(CHECKPOINT_DIR, "last")
        os.makedirs(last_ckpt_dir, exist_ok=True)
        save_checkpoint(model, optimizer, epoch, step, checkpoint_dir=last_ckpt_dir, filename="final_checkpoint.pth")



if __name__ == "__main__":
    main()

