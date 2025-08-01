#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_IB_GDR_LEVEL=2
export NCCL_P2P_DISABLE=0
# 必要に応じて
# export NCCL_IB_HCA=mlx5_0
# export NCCL_SOCKET_IFNAME=eth0
# cuDNN SDPA実装を回避してFlashAttentionのみ使いたい場合
export CUBLAS_WORKSPACE_CONFIG=:16:8

torchrun \
    --nproc_per_node=2 \
    train.py \
    --model qwen \
    --optimizer adamw \
    --lr 1e-3 \
    --wd 0.1 \
    --size 1B \
    --save 1000 \
    --batch_size 16
