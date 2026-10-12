"""Mask R-CNN and Cascade Mask R-CNN (torchvision) set up for 1-channel, full-resolution solar images.

* COCO-pretrained ResNet50-FPN v2, with the box and mask heads replaced for 2 classes (background, filament);
  optionally with cascade box heads (``arch="cascade"``).
* Images are already normalized per image (``filament.segmentation.dataset.normalize_with``) and repeated to 3
  channels, so torchvision's own normalization is switched off (mean 0, std 1).
* torchvision resizes every image to ``min_size`` (800 px by default), which would blur 8 px filaments away. The input
  size is set to the real size instead: the crop size for training, 2048 for full images (``set_input_size``).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from filament.detection.instances import Instances


def build_maskrcnn(pretrained: bool = True, trainable_layers: int = 3, detections_per_img: int = 100,
                   score_thresh: float = 0.01, mask_size: int = 28, size: int = 1024,
                   arch: str = "maskrcnn") -> nn.Module:
    """``arch``: "maskrcnn" (one box stage) or "cascade" (three box stages, ``filament.detection.cascade``)."""
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None,
        weights_backbone=None,
        trainable_backbone_layers=trainable_layers if pretrained else None,
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        min_size=size,
        max_size=size,
        box_detections_per_img=detections_per_img,
        box_score_thresh=score_thresh,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    if mask_size != 28:  # the mask head predicts at 2x its RoI size; a larger RoI gives finer masks
        from torchvision.ops import MultiScaleRoIAlign

        model.roi_heads.mask_roi_pool = MultiScaleRoIAlign(["0", "1", "2", "3"], mask_size // 2, sampling_ratio=2)
    if arch == "cascade":
        from filament.detection.cascade import CascadeRoIHeads

        model.roi_heads = CascadeRoIHeads(model.roi_heads)
    elif arch != "maskrcnn":
        raise ValueError(f"unknown detection arch {arch!r}")
    return model


def set_input_size(model: nn.Module, size: int) -> None:
    """Make torchvision feed images at their native size (no resizing) for images of side ``size``."""
    model.transform.min_size = (size,)
    model.transform.max_size = size


def make_input(x: np.ndarray, guide: np.ndarray | None = None) -> np.ndarray:
    """(3, H, W) model input from a normalized image and, optionally, a segmentation model's probability map.

    Without a guide, the gray image fills all three channels (what the COCO weights expect: three similar channels).
    With a guide, the middle channel is the probability map, scaled to about the image's range ([0, 1] -> [-2, 2]),
    so the detector starts from the U-Net's per-pixel answer and learns to group it into filaments and to score them.
    """
    x = x.astype(np.float32)
    middle = x if guide is None else (guide.astype(np.float32) - 0.5) * 4
    return np.ascontiguousarray(np.stack([x, middle, x]))


def to_input(x: np.ndarray, device) -> torch.Tensor:
    """(H, W) normalized image or (3, H, W) ``make_input`` output -> (3, H, W) tensor on ``device``."""
    x = make_input(x) if x.ndim == 2 else x
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)


@torch.no_grad()
def predict_instances(model: nn.Module, x: np.ndarray, device, amp_dtype=None) -> Instances:
    """Raw detections for one image: boxes, scores and the M x M soft masks (not pasted).

    ``x`` is a normalized (H, W) image or a (3, H, W) ``make_input`` array (with a guide map).
    Runs the model's stages directly instead of ``model(images)``, which would paste every mask into a full-size
    float image (100 detections x 2048 x 2048 floats).
    """
    model.eval()
    h, w = x.shape[-2:]
    set_input_size(model, max(h, w))
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
        images, _ = model.transform([to_input(x, device)])
        features = model.backbone(images.tensors)
        proposals, _ = model.rpn(images, features)
        det = model.roi_heads(features, proposals, images.image_sizes)[0][0]
    ih, iw = images.image_sizes[0]
    boxes = det["boxes"].float() * torch.tensor([w / iw, h / ih, w / iw, h / ih], device=det["boxes"].device)
    masks = det["masks"][:, 0]
    if len(det["scores"]) == 0:
        return Instances.empty(masks.shape[-1] if masks.ndim == 3 else 28)
    return Instances(boxes.cpu().numpy().astype(np.float32), det["scores"].float().cpu().numpy(),
                     masks.float().cpu().numpy().astype(np.float16))
