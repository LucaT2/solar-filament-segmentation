"""Segmentation model, loss, and full-image prediction."""

from __future__ import annotations

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_float(obj):
    if torch.is_tensor(obj):
        return obj.float()
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_float(o) for o in obj)
    return obj


def _fp32_forward(forward):
    """Run a module outside autocast in float32 (e.g. a decoder whose attention overflows in fp16)."""

    def wrapped(*args, **kwargs):
        device_type = next((a.device.type for a in _flatten(args) if torch.is_tensor(a)), "cuda")
        with torch.autocast(device_type, enabled=False):
            return forward(*_to_float(args), **{k: _to_float(v) for k, v in kwargs.items()})

    return wrapped


def _flatten(obj):
    if isinstance(obj, (list, tuple)):
        for o in obj:
            yield from _flatten(o)
    else:
        yield obj


def build_model(arch: str = "Unet", encoder: str = "resnet34", pretrained: bool = True,
                decoder_fp32: bool = False, classes: int = 1) -> nn.Module:
    """Single-channel input (grayscale Hα); output channel 0 is the filament logit (``classes`` = 2 adds a boundary
    logit, used only as a training signal: prediction reads channel 0).

    smp adapts the pretrained RGB stem to 1 input channel by summing its weights. ``decoder_fp32`` keeps the decoder
    in float32 under mixed precision (the encoder still runs in fp16), for decoders whose activations overflow fp16.
    """
    model = smp.create_model(
        arch, encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, in_channels=1, classes=classes
    )
    if decoder_fp32:
        model.decoder.forward = _fp32_forward(model.decoder.forward)
    return model


def n_outputs(cfg: dict) -> int:
    """Output channels of a model trained with config ``cfg`` (2 with the boundary head)."""
    return 2 if cfg.get("boundary_weight", 0) > 0 else 1


def dilate(m: torch.Tensor, k: int) -> torch.Tensor:
    return F.max_pool2d(m, k, stride=1, padding=k // 2)


def erode(m: torch.Tensor, k: int) -> torch.Tensor:
    return -F.max_pool2d(-m, k, stride=1, padding=k // 2)


def boundary_target(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    """The band where a k x k dilation and erosion of ``mask`` differ: (k - 1) px across each outline, and the whole
    of any structure narrower than k px."""
    return dilate(mask, k) - erode(mask, k)


def thin_target(mask: torch.Tensor, k: int = 7) -> torch.Tensor:
    """Mask pixels removed by a k x k opening: parts narrower than ~k px (barbs, thin ends)."""
    return (mask - dilate(erode(mask, k), k)).clamp_min(0)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1.0):
        super().__init__()
        self.bce_weight, self.dice_weight, self.eps = bce_weight, dice_weight, eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        """``weight``: optional per-pixel weights (same shape), applied to both terms."""
        logits = logits.float()
        w = torch.ones_like(target) if weight is None else weight
        bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * w).sum() / w.sum()
        prob = torch.sigmoid(logits)
        inter = (w * prob * target).sum()
        dice = 1 - (2 * inter + self.eps) / ((w * prob).sum() + (w * target).sum() + self.eps)  # batch-level soft Dice
        return self.bce_weight * bce + self.dice_weight * dice


class SegLoss(nn.Module):
    """BCE + Dice on the mask, plus (optionally) Skeleton Recall (Kirchhoff et al., ECCV 2024), a boundary head and
    extra weight on thin structures.

    Skeleton Recall = 1 - (sum of predicted probability on the tubed skeleton) / (skeleton size), over the batch.
    It only rewards *covering* each filament's centre line, so dropping a faint middle stretch of a filament (which
    splits it into pieces) is penalized, while the mask loss still controls the outline.
    ``boundary_weight`` > 0: logits channel 1 predicts ``boundary_target`` of the mask (BCE + Dice); the band covers
    whole barbs, so the shared features must resolve outlines and thin structures.
    ``thin_weight`` > 0: mask pixels in ``thin_target`` count 1 + thin_weight times in the mask loss.
    ``target`` is (B, 1, H, W) [mask] or (B, 2, H, W) [mask, tubed skeleton].
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, skel_weight: float = 0.0, eps: float = 1.0,
                 boundary_weight: float = 0.0, boundary_width: int = 3, thin_weight: float = 0.0, thin_kernel: int = 7):
        super().__init__()
        self.mask_loss = BCEDiceLoss(bce_weight, dice_weight, eps)
        self.skel_weight, self.eps = skel_weight, eps
        self.boundary_weight, self.boundary_width = boundary_weight, boundary_width
        self.thin_weight, self.thin_kernel = thin_weight, thin_kernel

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target[:, :1]
        weight = 1 + self.thin_weight * thin_target(mask, self.thin_kernel) if self.thin_weight > 0 else None
        loss = self.mask_loss(logits[:, :1], mask, weight)
        if self.skel_weight > 0:
            skel = target[:, 1:2]
            prob = torch.sigmoid(logits[:, :1].float())
            recall = ((prob * skel).sum() + self.eps) / (skel.sum() + self.eps)
            loss = loss + self.skel_weight * (1 - recall)
        if self.boundary_weight > 0:
            loss = loss + self.boundary_weight * self.mask_loss(logits[:, 1:2], boundary_target(mask, self.boundary_width))
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
            out = torch.sigmoid(model(t[:, :, y : y + tile, x : x + tile])[:, :1].float())
            acc[:, :, y : y + tile, x : x + tile] += out * weight
            norm[:, :, y : y + tile, x : x + tile] += weight
    return acc / norm.clamp_min(1e-6)


def tta_transforms(mode) -> list[tuple]:
    """(forward, inverse) pairs on (B, C, H, W) tensors.

    False / "none": identity only; True / "flip4": identity + horizontal, vertical and both flips;
    "d8": all 8 symmetries of the square (4 rotations x optional mirror flip; needs H == W).
    """
    ident = (lambda t: t, lambda t: t)
    if mode in (False, None, "none"):
        return [ident]
    if mode in (True, "flip4"):
        return [ident] + [(lambda t, d=d: torch.flip(t, d), lambda t, d=d: torch.flip(t, d))
                          for d in [(-1,), (-2,), (-2, -1)]]
    if mode == "d8":
        out = []
        for k in range(4):
            for flip in (False, True):
                def fwd(t, k=k, flip=flip):
                    t = torch.rot90(t, k, dims=(-2, -1))
                    return torch.flip(t, (-1,)) if flip else t

                def inv(t, k=k, flip=flip):
                    t = torch.flip(t, (-1,)) if flip else t
                    return torch.rot90(t, -k, dims=(-2, -1))

                out.append((fwd, inv))
        return out
    raise ValueError(f"unknown TTA mode {mode!r}")


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
    transforms = tta_transforms(tta)
    acc = torch.zeros_like(t, dtype=torch.float32)
    for model in models:
        for fwd, inv in transforms:
            inp = fwd(t)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
                if tile:
                    out = _forward_tiled(model, inp, tile, overlap)
                else:
                    out = torch.sigmoid(model(inp)[:, :1].float())
            acc += inv(out)
    return (acc / (len(models) * len(transforms)))[0, 0].cpu().numpy()
