#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1

cd ../../

deepspeed --num_gpus=2 train_gpt_ds.py \
  --hf_model openai/gpt-oss-20b \
  --optimizer pagedadamw8bit \
  --lr 1e-5 \
  --wd 0.01 \
  --batch_size 4 \
  --accumulation_steps 1 \
  --max_len 640 \
  --save 1000 \
  --deepspeed_config ./configs/ds_zero3_bnb.json \
  --attn_impl auto \
  --flush_interval 50
