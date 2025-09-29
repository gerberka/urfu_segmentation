#!/bin/sh

source ../.venv/bin/activate

GPUS=1
NODE_PARAMS="-p hiperf --gres=gpu:a100:$GPUS --nodelist=tesla-a100 -t 02:00:00"

sbatch -n1 \
    --cpus-per-task=8 \
    --mem=45000 \
    $NODE_PARAMS \
    --job-name=mmsegm-tree \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS} \
    --wrap="python ./train.py ./configs_2025_05/config_trees_mask2former.py --launcher slurm"