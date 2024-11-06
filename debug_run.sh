#!/bin/bash

# 检查 NVIDIA 驱动和 GPU 状态
echo "=== 检查 NVIDIA 驱动和 GPU 状态 ==="
nvidia-smi

# 检查 CUDA 版本
echo "=== 检查 CUDA 版本 ==="
cat /usr/local/cuda/version.txt
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2


# 检查 NCCL 版本（需要安装 NCCL 包）
echo "=== 检查 NCCL 版本 ==="
NCCL_VERSION=$(nccl_version=$(cat /usr/include/nccl.h 2>/dev/null | grep '#define NCCL_MAJOR' -A 2 | awk '{print $3}' | tr '\n' '.'); echo ${nccl_version%.})
if [ -z "$NCCL_VERSION" ]; then
    echo "无法找到 NCCL 版本，可能未安装或未在系统路径中"
else
    echo "NCCL 版本: $NCCL_VERSION"
fi

# 检查 Python 版本
echo "=== 检查 Python 版本 ==="
python3 --version

echo "=== check deepspeed ==="
#python3 -m deepspeed.env

# 设置环境变量（根据需要）
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#export MASTER_ADDR=127.0.0.1
#export MASTER_PORT=29500

# 可选：设置 NCCL 调试级别
export NCCL_DEBUG=INFO

# 启动训练
echo "=== 启动分布式训练 ==="

#torchrun --nproc_per_node=8 train_gpt2.py
#python3 train_gpt2.py
#python3 train_gpt2withgpt4tokenizer.py
#torchrun --nproc_per_node=8 train_gpt2withgpt4tokenizer.py
#torchrun --standalone  --nproc_per_node=8 train_fsdp_gqa.py 
#mlx worker  launch --gpu 8  --memory 64 -- bash /mnt/bn/bozhang41/build-nanogpt/debug_run.sh 
#torchrun --nproc_per_node=NUM_GPUS train_gpt2.py
#mlx worker  launch --memory 64  --gpu 2  --type h800 -- bash /mnt/bn/bozhang41/build-nanogpt/debug_run.sh 
#python3 minimal_gpt2_inference.py
#python3 minimal_gpt2_inference_ddp.py
#python3 minimal_gpt2_inference_1gpu_opt.py
#python3 minimal_gpt2_inference_generate1.py
python3 /mnt/bn/bozhang41/build-nanogpt/start_sleep.py
