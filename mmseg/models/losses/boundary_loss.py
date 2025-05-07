# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy.ndimage import distance_transform_edt
from mmseg.registry import MODELS
from typing import List

def compute_distance_map(gt: Tensor, idc: List[int]) -> Tensor:
    """Вычисляет карты расстояний до границы для заданных классов.

    Args:
        gt (Tensor): Ground truth (B, 1, H, W)
        idc (List[int]): Индексы классов

    Returns:
        Tensor: Distance maps (B, C, H, W), где C = len(idc)
    """
    gt_np = gt.squeeze(1).cpu().numpy()  # (B, H, W)
    B, H, W = gt_np.shape
    C = len(idc)
    dist_maps = np.zeros((B, C, H, W), dtype=np.float32)

    for b in range(B):
        for i, cls in enumerate(idc):
            posmask = gt_np[b] == cls
            negmask = ~posmask
            dist_out = distance_transform_edt(negmask)
            dist_in = distance_transform_edt(posmask)
            dist_maps[b, i] = dist_out - dist_in

    return torch.from_numpy(dist_maps).to(gt.device)

@MODELS.register_module()
class BoundaryLoss(nn.Module):
    """Boundary loss.

    This function is modified from
    `PIDNet <https://github.com/XuJiacong/PIDNet/blob/main/utils/criterion.py#L122>`_.  # noqa
    Licensed under the MIT License.


    Args:
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        loss_name (str): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_boundary'.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 idc: List[int] = [1],
                 loss_name: str = 'loss_boundary'):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name
        self.idc = idc

    def forward(self, bd_pre: Tensor, bd_gt: Tensor) -> Tensor:
        """Forward function.
        Args:
            bd_pre (Tensor): Predictions of the boundary head.
            bd_gt (Tensor): Ground truth of the boundary.

        Returns:
            Tensor: Loss tensor.
        """
        dist_maps = compute_distance_map(bd_gt, self.idc)

        pc = bd_pre[:, self.idc, ...].type(torch.float32)
        dc = dist_maps.type(torch.float32)

        multipled = pc * dc
        loss = multipled.mean()

        return loss

    @property
    def loss_name(self):
        return self.loss_name_
