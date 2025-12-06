import numpy as np
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class EnsureSingleChannelGT:
    """Convert 3-channel GT to 1-channel by taking channel 0.
    Works for RGB masks where channels are identical (grayscale stored as RGB).
    """

    def transform(self, results):
        gt = results.get("gt_seg_map", None)
        if gt is None:
            return results

        if isinstance(gt, np.ndarray) and gt.ndim == 3:
            gt = gt[:, :, 0]

        results["gt_seg_map"] = gt
        return results

    def __repr__(self):
        return self.__class__.__name__
