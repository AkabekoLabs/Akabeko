#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
# 断片化対策。まずは256でOK。落ちるなら128に下げて再トライ。
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
# ↓NVLink/IBが無い環境で固まる時の保険（必要な時だけ有効化）
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_SHM_DISABLE=1

cd ../../
torchrun \
  --nproc_per_node=2 \
  train_gpt_fsdp.py \
  --hf_model openai/gpt-oss-20b \
  --optimizer pagedadamw8bit \
  --lr 1e-5 \
  --wd 0.01 \
  --batch_size 1 \
  --accumulation_steps 1 \
  --max_len 640 \
  --save 1000
