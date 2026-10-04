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


@torch.no_grad()
def predict_full(
    models: list[nn.Module],
    x: np.ndarray,
    device: torch.device,
    tta: bool = False,
    amp_dtype: torch.dtype | None = torch.float16,
) -> np.ndarray:
    """Average sigmoid probability of ``models`` on one normalized (H, W) image, optionally with flip TTA.

    The networks are fully convolutional, so the whole 2048 x 2048 image is processed in one pass
    (H and W must be divisible by 32).
    """
    t = torch.from_numpy(x)[None, None].to(device)
    flips = [(), (-1,), (-2,), (-2, -1)] if tta else [()]
    acc = torch.zeros_like(t, dtype=torch.float32)
    for model in models:
        for dims in flips:
            inp = torch.flip(t, dims) if dims else t
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
                out = torch.sigmoid(model(inp).float())
            acc += torch.flip(out, dims) if dims else out
    return (acc / (len(models) * len(flips)))[0, 0].cpu().numpy()
