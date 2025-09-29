#!/bin/sh

sbatch \
  -p hiperf --nodelist=tesla-a100 \
  --nodes=1 --gres=gpu:a100:8 \
  --cpus-per-task=8 --mem=45G -t 02:00:00 -J mmsegm-tree \
  --wrap="srun torchrun --nproc_per_node=8 --master_port=23456 \
          ./train.py ./configs_2025_05/config_trees_mask2former.py --launcher pytorch"
