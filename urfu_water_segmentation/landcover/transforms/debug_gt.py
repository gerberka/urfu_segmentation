import numpy as np
from mmseg.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

@TRANSFORMS.register_module()
class DebugGT(BaseTransform):
    def transform(self, results):
        gt = results.get("gt_seg_map")
        if gt is None:
            return results
        print("GT shape:", gt.shape, "dtype:", gt.dtype, "unique:", np.unique(gt)[:50])
        return results
