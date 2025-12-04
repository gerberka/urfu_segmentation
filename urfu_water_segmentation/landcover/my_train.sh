#!/bin/sh

source ../.venv/bin/activate

GPUS=1
NODE_PARAMS="-p hiperf --gres=gpu:v100:$GPUS --nodelist=tesla-v100 -t 10:00:00"

sbatch -n1 \
    --cpus-per-task=8 \
    --mem=45000 \
    $NODE_PARAMS \
    --job-name=mmsegm-tree \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS} \
    --wrap="python ./train.py ./trees/config_trees_knet_min.py --launcher slurm"