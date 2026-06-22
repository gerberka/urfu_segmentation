#!/bin/sh

sbatch \
  -p hiperf --nodelist=tesla-v100 \
  --nodes=1 --gres=gpu:v100:8 \
  --cpus-per-task=8 --mem=0 -t 9:59:59 -J mmsegm-tree \
  --wrap="srun torchrun --nproc_per_node=8 --master_port=23456 \
          ./train.py ./trees/config_trees_knet_ce_togray_augm.py --launcher pytorch"


