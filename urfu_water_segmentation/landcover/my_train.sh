#!/bin/sh

source ../.venv/bin/activate

GPUS=1
NODE_PARAMS="-p hiperf --gres=gpu:a100:$GPUS --nodelist=tesla-a101,tesla-v100 -t 00:15:00"

sbatch -n1 \
    --cpus-per-task=8 \
    --mem=45000 \
    $NODE_PARAMS \
    --job-name=mmsegm-water \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS} \
    --wrap="python ./train.py ./roads_config/config_landcover_mask2_baseline.py --launcher slurm"
