#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
# 断片化対策（落ちるなら128に下げる）
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
# ↓NVLink/IBが無い/不安定な時だけ有効化
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_SHM_DISABLE=1

cd ../../
deepspeed \
  --num_gpus=2 \
  train_gpt_ds.py \
  --hf_model openai/gpt-oss-20b \
  --optimizer adamw \
  --lr 1e-5 \
  --wd 0.01 \
  --batch_size 1 \
  --accumulation_steps 1 \
  --max_len 640 \
  --save 1000 \
  --deepspeed_config ./configs/ds_zero3_offload.json
