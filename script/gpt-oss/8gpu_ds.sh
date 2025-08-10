#!/usr/bin/env bash
set -euo pipefail

# 環境変数
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
# FA2/SDPAの安定化
export CUDA_DEVICE_MAX_CONNECTIONS=1

# プロジェクトルート想定（必要なら適宜変更）
cd ../../

# ⚠️ 行末の "\" の後ろに空白やコメントを置かないこと！
deepspeed --num_gpus=8 train_gpt_ds.py \
  --hf_model openai/gpt-oss-20b \
  --optimizer pagedadamw8bit \
  --lr 1e-5 --wd 0.01 \
  --batch_size 4 --accumulation_steps 1 \
  --max_len 640 \
  --deepspeed_config ./configs/ds_zero3_bnb.json \
  --attn_impl auto \
  --flush_interval 50 \
  --hf_only_save \
  --export_dir ./export_hf \
  --export_max_shard_gb 10