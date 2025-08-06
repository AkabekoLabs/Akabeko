import os
import torch
import argparse
import datetime
import time
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from lib.akabeko_dataset_gpt import AkabekoDataset
from lib.utils import save_checkpoint, format_hms, get_optimizer
from glob import glob
import json

DATA_PATH = os.getcwd()
PT_DIR = os.path.join(DATA_PATH, "tokenized_dataset")
CHECKPOINT_DIR = os.path.join(DATA_PATH, "checkpoints")
LOG_DIR = os.path.join(DATA_PATH, "logs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def custom_collate(batch):
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None

    input_ids = [sample["input_ids"] for sample in batch]
    attention_mask = [sample["attention_mask"] for sample in batch]
    labels = [sample["labels"] for sample in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded
    }

def save_config(tokenizer, output_dir):
    config_path = os.path.join(output_dir, "config.json")

    # Fast Tokenizerなら backend_tokenizer.to_str() で取得
    if getattr(tokenizer, "is_fast", False):
        tokenizer_json_str = tokenizer.backend_tokenizer.to_str()
    else:
        tokenizer_json_str = tokenizer.to_json_string()

    # dictとして読み直して、model_nameなど付加
    tokenizer_config = json.loads(tokenizer_json_str)
    config = {
        "model_name": tokenizer.name_or_path,
        "tokenizer_config": tokenizer_config
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="muon", choices=["muon", "adamw", "adamw8bit"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)  
    parser.add_argument("--save", type=int, default=5000)
    parser.add_argument("--dataset_dir", type=str, default=PT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hf_model", type=str, required=True, help="Model name/path like openai/gpt-oss-20b")
    args = parser.parse_args()

    # 分散環境初期化 (これは残す)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(hours=2))
    device = torch.device(f"cuda:{local_rank}")

    # Tokenizerロード
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=False)

    # Datasetロード
    dataset_pt_files = sorted(glob(os.path.join(args.dataset_dir, "*.pt")))
    raw_data = []
    for pt_file in dataset_pt_files:
        print(f"🔄 Loading {pt_file}")

        data = torch.load(pt_file, weights_only=False, mmap=True)

        if isinstance(data, dict):
            raw_data.append([data])  # dictならlistにwrapしてappend
        elif isinstance(data, list):
            for sample in data:
                if isinstance(sample, dict):
                    raw_data.append([sample])  # dictならlistにwrapしてappend
                elif isinstance(sample, list):
                    raw_data.append(sample)    # すでにlist of dictならそのままappend
        else:
            print(f"❌ Invalid data format in {pt_file}")


    print(f"✅ Loaded {len(raw_data)} samples from {len(dataset_pt_files)} files")
    train_dataset = AkabekoDataset(raw_data, tokenizer)
    print(f"📊 Total dataset size: {len(train_dataset)} samples")  # ← この行を追加！
    print(train_dataset[0])  # ← これで1サンプル中身見てみて

    # モデルロード (DDP無し・Accelerate使用)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.gradient_checkpointing_enable() 
    model.train()

    # DataLoader
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, seed=0, drop_last=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0, 
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=None,
        collate_fn=custom_collate,
    )

    # ログファイル準備
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    logfile_name = f"{date_str}_{len(train_dataset)}samples_{args.optimizer}.csv"
    logfile_path = os.path.join(LOG_DIR, logfile_name)

    # Optimizer & Scheduler
    optimizer = get_optimizer(args.optimizer, model, lr=args.lr, wd=args.wd)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * args.epoch,
        num_cycles=0.5,
    )

    # トレーニング開始
    total_start_time = time.time()
    total_steps = len(train_loader) * args.epoch
    accumulation_steps = 16  # batch_size=1 だけど accumulation_stepsで実質 0.5

    for epoch in range(args.epoch):
        sampler.set_epoch(epoch)

        for step, batch in enumerate(train_loader):
            if batch is None:
                continue

            batch = {k: v.to(device) for k, v in batch.items()}
            if batch["attention_mask"].sum(dim=1).eq(0).all():
                continue

            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps  # Scale loss by accumulation_steps
            loss.backward()

            if not torch.isfinite(loss):
                print(f"[Rank:{local_rank}] Skipping step {step} due to NaN loss")
                optimizer.zero_grad(set_to_none=True)
                continue

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Logging
            if local_rank == 0:
                current_step = epoch * len(train_loader) + step + 1
                elapsed = time.time() - total_start_time
                remaining_time = (elapsed / current_step) * (total_steps - current_step)

                log_entry = f"{epoch+1},{step},{current_step},{loss.item():.6f},{optimizer.param_groups[0]['lr']:.6f},{elapsed:.2f}\n"
                with open(logfile_path, 'a') as f:
                    f.write(log_entry)

                print(f"[Rank:{local_rank}] Epoch: {epoch+1}/{args.epoch}, Step: {step}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Elapsed: {format_hms(elapsed)}, ETA: {format_hms(remaining_time)}")

                # 中間checkpoint保存
                if step % args.save == 0 and step > 0 and local_rank == 0:
                    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                    save_checkpoint(model, optimizer, epoch, step, checkpoint_dir=CHECKPOINT_DIR)
                    save_config(tokenizer, CHECKPOINT_DIR)  # ← config.json 保存
                    torch.cuda.empty_cache()


        if local_rank == 0:
            last_ckpt_dir = os.path.join(CHECKPOINT_DIR, "last")
            os.makedirs(last_ckpt_dir, exist_ok=True)
            save_checkpoint(model, optimizer, epoch, step, checkpoint_dir=last_ckpt_dir, filename="final_checkpoint.pth")
            save_config(tokenizer, last_ckpt_dir)  # ← config.json 保存

if __name__ == "__main__":
    main()
