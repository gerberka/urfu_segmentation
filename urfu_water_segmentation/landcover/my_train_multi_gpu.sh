#!/bin/bash
#SBATCH -p hiperf
#SBATCH --nodelist=tesla-a100
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G
#SBATCH -t 02:00:00
#SBATCH -J mmsegm-tree

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Диагностика DDP/NCCL (можно выключить позже)
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1           # если нет Infiniband
# export NCCL_SOCKET_IFNAME=lo     # single-node иногда помогает

source ../.venv/bin/activate
cd landcover

GPUS=2
PORT=$((12000 + RANDOM % 20000))

# важное: именно torchrun и launcher=pytorch
srun --gpu-bind=closest \
  torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
  ./train.py ./configs_2025_05/config_trees_mask2former.py \
  --launcher pytorch
