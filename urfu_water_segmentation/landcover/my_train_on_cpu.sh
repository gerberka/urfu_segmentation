srun -n1  --cpus-per-task=12 --mem=40000 -p apollo  -t 10:00:00 --job-name=seg python ./train.py ./trees/config_trees_knet_focal_loss.py --launcher none
