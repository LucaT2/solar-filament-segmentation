"""Mask2Former (Hugging Face ``transformers``) behind the same interface as the torchvision detectors.

* COCO instance-segmentation weights (Swin-T by default, ``checkpoint``), the class head replaced for one class
  (filament) + "no object". 100 queries, masked-attention decoder, as published.
* Training: ``model(images, targets)`` with torchvision-style targets returns a loss dict, like Mask R-CNN. The
  network runs under the caller's autocast; the Hungarian matching and the losses run in fp32 (the matcher sums
  point-sampled costs over 12,544 points, which overflows fp16). The 9 auxiliary decoder layers' losses are summed
  into ``loss_aux``.
* Prediction: each query's score is its class probability x the mean mask probability inside its mask (the paper's
  instance score). Masks are predicted at 1/4 of the input resolution; each query's logits are cut out around its box
  and resampled to ``mask_size`` x ``mask_size`` (``roi_align``), so the predictions are ``Instances`` like Mask R-CNN's
  and TTA, tuning, fusion and submissions work unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from filament.detection.instances import Instances

DEFAULT_CHECKPOINT = "facebook/mask2former-swin-tiny-coco-instance"


def _recomputed(forward):
    """``forward`` with gradient checkpointing while training (activations recomputed in the backward pass)."""
    from torch.utils.checkpoint import checkpoint

    def wrapped(*args, **kwargs):
        if torch.is_grad_enabled():
            return checkpoint(forward, *args, use_reentrant=False, **kwargs)
        return forward(*args, **kwargs)

    return wrapped


class Mask2FormerDetector(nn.Module):
    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, pretrained: bool = True, mask_size: int = 128,
                 box_threshold: float = 0.3, min_score: float = 0.01):
        """``box_threshold``: mask probability that defines a query's box (below the lowest mask threshold that will be
        tuned, so the box holds the whole mask). Queries scoring below ``min_score`` are not returned."""
        super().__init__()
        from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

        labels = dict(num_labels=1, id2label={0: "filament"}, label2id={"filament": 0})
        if pretrained:
            self.net = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint, ignore_mismatched_sizes=True,
                                                                           **labels)
        else:  # architecture only (the weights come from a checkpoint of ours); no checkpoint: the default (Swin-T)
            config = Mask2FormerConfig.from_pretrained(checkpoint, **labels) if checkpoint else Mask2FormerConfig(**labels)
            self.net = Mask2FormerForUniversalSegmentation(config)
        self.mask_size, self.box_threshold, self.min_score = mask_size, box_threshold, min_score
        # the pixel decoder's deformable attention (plain PyTorch) keeps large sampling tensors for the backward pass:
        # recomputing them halves the training memory (6.0 -> 3.6 GiB per 1024 px crop) for ~15% more time
        for layer in self.net.model.pixel_level_module.decoder.encoder.layers:
            layer.forward = _recomputed(layer.forward)

    @property
    def encoder(self) -> nn.Module:
        """The pretrained image encoder (Swin / ResNet), trained at ``backbone_lr_mult`` x the learning rate."""
        return self.net.model.pixel_level_module.encoder

    def forward(self, images: list[torch.Tensor], targets: list[dict]) -> dict[str, torch.Tensor]:
        out = self.net(pixel_values=torch.stack(images), output_auxiliary_logits=True)
        with torch.autocast(images[0].device.type, enabled=False):
            losses = self.net.get_loss_dict(
                out.masks_queries_logits.float(), out.class_queries_logits.float(),
                [t["masks"].float() for t in targets],
                [t["labels"] - 1 for t in targets],  # torchvision's label 1 (filament) is class 0 here
                [{k: v.float() for k, v in aux.items()} for aux in out.auxiliary_logits],
            )
        main = {k: v for k, v in losses.items() if not k[-1].isdigit()}
        return {**main, "loss_aux": sum(v for k, v in losses.items() if k[-1].isdigit())}

    @torch.no_grad()
    def predict(self, inp: torch.Tensor, amp_dtype=None) -> Instances:
        """Detections for one (3, H, W) input, in its pixel coordinates."""
        with torch.autocast(inp.device.type, dtype=amp_dtype, enabled=amp_dtype is not None and inp.is_cuda):
            out = self.net(pixel_values=inp[None])
        return queries_to_instances(out.class_queries_logits[0].float(), out.masks_queries_logits[0].float(),
                                    tuple(inp.shape[-2:]), self.mask_size, self.box_threshold, self.min_score)


def queries_to_instances(class_logits: torch.Tensor, mask_logits: torch.Tensor, shape: tuple[int, int],
                         mask_size: int = 128, box_threshold: float = 0.3, min_score: float = 0.01) -> Instances:
    """(Q, 2) class logits (filament, no object) and (Q, h, w) low-resolution mask logits of an image of ``shape``
    -> scored detections: box around the mask's pixels above ``box_threshold`` (+ one low-resolution pixel), the
    logits inside it resampled to ``mask_size`` x ``mask_size`` probabilities."""
    from torchvision.ops import roi_align

    prob = mask_logits.sigmoid()
    fg = prob > 0.5
    score = class_logits.softmax(-1)[:, 0] * (prob * fg).sum((1, 2)) / fg.sum((1, 2)).clamp(min=1)
    inside = prob > box_threshold
    keep = torch.nonzero((score >= min_score) & inside.flatten(1).any(1))[:, 0]
    if not len(keep):
        return Instances.empty(mask_size)
    rows, cols = inside[keep].any(2), inside[keep].any(1)  # (K, h), (K, w)
    first = lambda m: m.float().argmax(1)
    last = lambda m: m.shape[1] - 1 - m.flip(1).float().argmax(1)
    # box in low-resolution pixels (pixel i covers [i, i + 1)), one pixel of margin
    lr = torch.stack([first(cols) - 1, first(rows) - 1, last(cols) + 2, last(rows) + 2], 1).float()
    lr[:, 0::2] = lr[:, 0::2].clamp(0, mask_logits.shape[2])
    lr[:, 1::2] = lr[:, 1::2].clamp(0, mask_logits.shape[1])
    index = torch.arange(len(keep), device=lr.device, dtype=lr.dtype)[:, None]
    crops = roi_align(mask_logits[keep][:, None], torch.cat([index, lr], 1), mask_size, spatial_scale=1.0,
                      sampling_ratio=2, aligned=True)[:, 0]
    stride = torch.tensor([shape[1] / mask_logits.shape[2], shape[0] / mask_logits.shape[1]] * 2, device=lr.device)
    return Instances((lr * stride).cpu().numpy().astype(np.float32), score[keep].cpu().numpy().astype(np.float32),
                     crops.sigmoid().cpu().numpy().astype(np.float16))
