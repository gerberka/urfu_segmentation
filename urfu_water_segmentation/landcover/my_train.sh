#!/bin/sh

source ../.venv/bin/activate

GPU_ID="v100" # a100, a101, v100
GPU_COUNT=8

sbatch \
  -p hiperf --nodelist=tesla-${GPU_ID} \
  --nodes=1 --gres=gpu:${GPU_ID}:${GPU_COUNT} \
  --cpus-per-task=${GPU_COUNT} --mem=45G -t 9:59:59 -J pidnet-tree \
  --wrap="srun torchrun --nproc_per_node=${GPU_COUNT} --master_port=23456 \
          ./train.py ./configs_2025_05/config_trees_pidnet.py --launcher pytorch"
