#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

torchrun \
  --nproc_per_node=4 \
  train_gpt.py \
  --hf_model openai/gpt-oss-20b \
  --optimizer agedadamw8bit \
  --lr 5e-5 \
  --wd 0.01 \
  --batch_size 1 \
  --save 1000