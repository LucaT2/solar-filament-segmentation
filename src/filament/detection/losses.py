"""Mask-head losses for the instance models.

torchvision trains the mask head with per-pixel binary cross-entropy against each detection's ground-truth mask,
cropped and resized to the M x M mask grid. ``mask_loss`` adds an optional soft Dice term per instance (as for the
U-Net): Dice measures the overlap of the whole mask, so thin parts of a filament count as much as its bulk, and it
is closer to the IoU that PQ rewards than an average over pixels.

``DiceMaskRoIHeads`` swaps this loss into torchvision's standard ``RoIHeads`` without copying its forward pass; the
module and its weights are unchanged, so checkpoints stay compatible.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.models.detection import roi_heads as tv_roi_heads
from torchvision.models.detection.roi_heads import RoIHeads, maskrcnn_loss, project_masks_on_boxes

_TORCHVISION_MASK_LOSS = maskrcnn_loss


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """1 - Dice per instance (``logits``, ``targets``: (N, M, M)), averaged over instances."""
    p = torch.sigmoid(logits.float()).flatten(1)
    t = targets.float().flatten(1)
    dice = (2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)
    return (1 - dice).mean()


def mask_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs, dice_weight: float = 0.0):
    """torchvision's mask BCE, plus ``dice_weight`` x soft Dice on the same targets."""
    bce = _TORCHVISION_MASK_LOSS(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs)
    if dice_weight <= 0:
        return bce
    m = mask_logits.shape[-1]
    labels = torch.cat([lab[idx] for lab, idx in zip(gt_labels, mask_matched_idxs)])
    if labels.numel() == 0:
        return bce
    targets = torch.cat([project_masks_on_boxes(g, p, i, m) for g, p, i in zip(gt_masks, proposals, mask_matched_idxs)])
    logits = mask_logits[torch.arange(labels.shape[0], device=labels.device), labels]
    return bce + dice_weight * soft_dice_loss(logits, targets)


class DiceMaskRoIHeads(RoIHeads):
    """torchvision's RoIHeads, trained with ``mask_loss`` (BCE + Dice) instead of BCE alone."""

    dice_weight: float = 1.0

    def forward(self, features, proposals, image_shapes, targets=None):
        if not self.training:
            return super().forward(features, proposals, image_shapes, targets)
        weight = self.dice_weight
        tv_roi_heads.maskrcnn_loss = lambda *args: mask_loss(*args, dice_weight=weight)
        try:
            return super().forward(features, proposals, image_shapes, targets)
        finally:
            tv_roi_heads.maskrcnn_loss = _TORCHVISION_MASK_LOSS


def use_dice_mask_loss(roi_heads: RoIHeads, dice_weight: float) -> None:
    """Switch a standard RoIHeads to BCE + ``dice_weight`` x Dice (same module and weights)."""
    roi_heads.__class__ = DiceMaskRoIHeads
    roi_heads.dice_weight = dice_weight
