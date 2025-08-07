#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
# HPC系プラグインを無効化
export NCCL_NVLS_ENABLE=0
export NCCL_SHARP_DISABLE=1

cd ../..
torchrun \
    --nproc_per_node=8 \
    train_gpt.py \
    --hf_model openai/gpt-oss-20b \
    --optimizer adamw8bit \
    --lr 5e-5 \
    --wd 0.01 \
    --batch_size 1
