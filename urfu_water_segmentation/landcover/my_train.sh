#!/bin/sh

source ../.venv/bin/activate

export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-11.6}  # или твой путь
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

GPU_ID="a100" # a100, a101, v100
GPU_COUNT=8

sbatch \
  -p hiperf --nodelist=tesla-${GPU_ID} \
  --nodes=1 --gres=gpu:${GPU_ID}:${GPU_COUNT} \
  --cpus-per-task=${GPU_COUNT} --mem=45G -t 9:59:59 -J pidnet-tree \
  --wrap="srun torchrun --nproc_per_node=${GPU_COUNT} --master_port=23456 \
          ./train.py ./configs_2025_05/config_trees_pidnet.py --launcher pytorch"
