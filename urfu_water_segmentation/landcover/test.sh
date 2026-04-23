export PYTHONPATH=/misc/home6/s0039/urfu_segmentation:/misc/home6/s0039/urfu_segmentation/urfu_water_segmentation/landcover:$PYTHONPATH

python ../../tools/test.py \
  trees/config_trees_knet_focal_loss.py \
  logs/KNet_SwinL_TreesDS_Focal_2cls_512crop_AdamW_80000iter/iter_5028.pth