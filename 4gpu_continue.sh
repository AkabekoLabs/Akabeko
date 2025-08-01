#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#For CUDA 12.7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
#export NCCL_DEBUG=WARN
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# 必要に応じて
# export NCCL_IB_HCA=mlx5_0
# export NCCL_SOCKET_IFNAME=eth0
# cuDNN SDPA実装を回避してFlashAttentionのみ使いたい場合
export CUBLAS_WORKSPACE_CONFIG=:16:8
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# HPC系プラグインを無効化
export NCCL_NVLS_ENABLE=0
export NCCL_SHARP_DISABLE=1

torchrun \
    train.py \
    --nproc_per_node=4 \
    --hf_model Qwen/Qwen3-0.6B \
    --model qwen \
    --optimizer adamw \
    --lr 5e-5 \
    --wd 0.01 \
    --batch_size 16