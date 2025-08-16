#!/usr/bin/env bash
set -euo pipefail

# 環境変数
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=16
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"
export PYTORCH_SDP_DISABLE_FLASH_ATTENTION=1
export PYTORCH_SDP_DISABLE_MEM_EFFICIENT=1
# 併せてスタック精度を上げる（原因特定用）
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1

# プロジェクトルート想定（必要なら適宜変更）
cd ../../
# ⚠️ 行末の "\" の後ろに空白やコメントを置かないこと！
deepspeed --num_gpus=4 train_gpt_ds.py \
  --hf_model openai/gpt-oss-20b \
  --optimizer adamw \
  --lr 1e-5 --wd 0.01 \
  --batch_size 4 --accumulation_steps 1 \
  --max_len 640 \
  --deepspeed_config ./configs/ds_zero3_bnb.json \
  --attn_impl auto \
  --flush_interval 50 \
  --hf_only_save \
  --export_dir ./export_hf \
  --export_max_shard_gb 10