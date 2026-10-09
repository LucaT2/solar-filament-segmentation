"""Segmentation model, loss, and full-image prediction."""

from __future__ import annotations

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_model(arch: str = "Unet", encoder: str = "resnet34", pretrained: bool = True) -> nn.Module:
    """Single-channel input (grayscale Hα), single-channel output (filament logit).

    smp adapts the pretrained RGB stem to 1 input channel by summing its weights.
    """
    return smp.create_model(
        arch, encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, in_channels=1, classes=1
    )


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1.0):
        super().__init__()
        self.bce_weight, self.dice_weight, self.eps = bce_weight, dice_weight, eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum()
        dice = 1 - (2 * inter + self.eps) / (prob.sum() + target.sum() + self.eps)  # batch-level soft Dice
        return self.bce_weight * bce + self.dice_weight * dice


class SegLoss(nn.Module):
    """BCE + Dice on the mask, plus (optionally) Skeleton Recall (Kirchhoff et al., ECCV 2024).

    Skeleton Recall = 1 - (sum of predicted probability on the tubed skeleton) / (skeleton size), over the batch.
    It only rewards *covering* each filament's centre line, so dropping a faint middle stretch of a filament (which
    splits it into pieces) is penalized, while the mask loss still controls the outline.
    ``target`` is (B, 1, H, W) [mask] or (B, 2, H, W) [mask, tubed skeleton].
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, skel_weight: float = 0.0, eps: float = 1.0):
        super().__init__()
        self.mask_loss = BCEDiceLoss(bce_weight, dice_weight, eps)
        self.skel_weight, self.eps = skel_weight, eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.mask_loss(logits, target[:, :1])
        if self.skel_weight > 0:
            skel = target[:, 1:2]
            prob = torch.sigmoid(logits.float())
            recall = ((prob * skel).sum() + self.eps) / (skel.sum() + self.eps)
            loss = loss + self.skel_weight * (1 - recall)
        return loss


def _blend_window(size: int, overlap: int, device) -> torch.Tensor:
    """2-D weight that ramps linearly from ~0 to 1 over ``overlap`` px at each tile border."""
    ramp = torch.ones(size, device=device)
    if overlap > 0:
        r = torch.linspace(1.0 / overlap, 1.0, overlap, device=device)
        ramp[:overlap], ramp[-overlap:] = r, r.flip(0)
    return ramp[:, None] * ramp[None, :]


def _forward_tiled(model: nn.Module, t: torch.Tensor, tile: int, overlap: int) -> torch.Tensor:
    """Sigmoid output of ``model`` on (1, 1, H, W) via overlapping tiles with linear blending."""
    _, _, h, w = t.shape
    stride = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    acc = torch.zeros((1, 1, h, w), device=t.device)
    norm = torch.zeros_like(acc)
    weight = _blend_window(tile, overlap, t.device)
    for y in ys:
        for x in xs:
            out = torch.sigmoid(model(t[:, :, y : y + tile, x : x + tile]).float())
            acc[:, :, y : y + tile, x : x + tile] += out * weight
            norm[:, :, y : y + tile, x : x + tile] += weight
    return acc / norm.clamp_min(1e-6)


@torch.no_grad()
def predict_full(
    models: list[nn.Module],
    x: np.ndarray,
    device: torch.device,
    tta: bool = False,
    amp_dtype: torch.dtype | None = torch.float16,
    tile: int | None = None,
    overlap: int = 128,
) -> np.ndarray:
    """Average sigmoid probability of ``models`` on one normalized (H, W) image, optionally with flip TTA.

    The networks are fully convolutional, so by default the whole 2048 x 2048 image is processed in one pass
    (H and W must be divisible by 32). Models too large for that can use overlapping ``tile`` x ``tile`` windows
    blended linearly over ``overlap`` px.
    """
    t = torch.from_numpy(x)[None, None].to(device)
    flips = [(), (-1,), (-2,), (-2, -1)] if tta else [()]
    acc = torch.zeros_like(t, dtype=torch.float32)
    for model in models:
        for dims in flips:
            inp = torch.flip(t, dims) if dims else t
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
                if tile:
                    out = _forward_tiled(model, inp, tile, overlap)
                else:
                    out = torch.sigmoid(model(inp).float())
            acc += torch.flip(out, dims) if dims else out
    return (acc / (len(models) * len(flips)))[0, 0].cpu().numpy()
