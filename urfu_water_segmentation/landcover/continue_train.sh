#!/bin/sh

WORKDIR="logs/KNet_SwinL_TreesDS_CE_2cls_512crop_AdamW_200ep_v2"
CONFIG="./trees/config_trees_knet_ce.py"

sbatch \
  -p hiperf --nodelist=tesla-v100 \
  --nodes=1 --gres=gpu:v100:8 \
  --cpus-per-task=8 --mem=0 -t 9:59:59 -J mmsegm-tree \
  --wrap="cd \$SLURM_SUBMIT_DIR && \
          srun torchrun --nproc_per_node=8 --master_port=23456 \
          ./train.py ${CONFIG} --launcher pytorch --resume --work-dir ${WORKDIR}"

