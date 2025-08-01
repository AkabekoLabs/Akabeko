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
export PYTORCH_SDP_ALLOW_FLASH_ATTENTION=1
export PYTORCH_SDP_FORCE_FALLBACK=1
export PYTORCH_SDP_ATTENTION_MASK_FALLBACK=1
export PYTORCH_CUDA_ALLOW_SDP_BACKEND=1
torchrun \
    --nproc_per_node=4 \
    train.py \
    --model qwen \
    --optimizer muon \
    --lr 1e-3 \
    --wd 0.1 \
    --size 0.5B \
    --batch_size 16  # GPUメモリ不足解消のために小さめのバッチサイズを指定

