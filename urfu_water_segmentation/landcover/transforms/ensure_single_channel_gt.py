import numpy as np
from mmseg.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class EnsureSingleChannelGT(BaseTransform):
    """Convert 3-channel GT to 1-channel by taking channel 0."""

    def __init__(self, ignore_index=255, **kwargs):
        self.ignore_index = ignore_index

    def transform(self, results):
        gt = results.get("gt_seg_map", None)
        if gt is None:
            return results

        if isinstance(gt, np.ndarray) and gt.ndim == 3:
            gt = gt[:, :, 0]

        results["gt_seg_map"] = gt
        return results
