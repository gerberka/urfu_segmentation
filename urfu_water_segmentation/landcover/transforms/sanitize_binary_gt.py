import numpy as np
from mmseg.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

@TRANSFORMS.register_module()
class SanitizeBinaryGT(BaseTransform):
    def __init__(self, ignore_index=255, **kwargs):
        self.ignore_index = ignore_index

    def transform(self, results):
        gt = results.get("gt_seg_map", None)
        if gt is None:
            return results

        # На всякий случай
        if gt.ndim == 3:
            gt = gt[:, :, 0]

        gt = gt.astype(np.uint8)

        # Разрешаем только 0, 1 и ignore
        valid = np.isin(gt, [0, 1, self.ignore_index])

        if not valid.all():
            gt = gt.copy()
            # Всё неизвестное — в background (0)
            gt[~valid] = 0

        results["gt_seg_map"] = gt
        return results
