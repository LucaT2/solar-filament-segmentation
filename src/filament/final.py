"""The submitted pipeline in one place: a GONG H-alpha image -> filament masks.

One Mask R-CNN (ConvNeXt-T backbone with DINOv3 weights, FPN, 56 x 56 mask head) trained on all 707 training images:

1. Eight views of the image (flips and transposes). Detections of the same filament are merged across views, and a
   filament's score is the sum of its scores divided by the number of views, so one seen in few views scores low.
2. Each soft mask is pasted into the image at the mask threshold, highest score first, so instances never overlap.
3. Instances below the score cut-off are dropped.

The tuned settings are committed in ``configs/final/``; the checkpoint is a release asset
(``scripts/final/download_weights.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from filament.data.images import disk_mask
from filament.detection.dataset import full_image_input
from filament.detection.instances import Instances, to_components
from filament.detection.model import predict_instances
from filament.detection.train import build_from_config
from filament.detection.tta import predict_tta
from filament.paths import REPO_ROOT, resolve_exp_dir
from filament.postprocess.components import Components
from filament.postprocess.pipeline import PostConfig, load_postprocess
from filament.segmentation.dataset import ImageInfo, image_info

EXPERIMENT = "maskrcnn_dinov3_convnext_t_m56"
CHECKPOINT = "full_s42.pt"
VIEWS = 8
FINAL_CONFIG_DIR = REPO_ROOT / "configs" / "final"


@dataclass
class FinalPipeline:
    """The trained model with its tuned settings; ``segment`` goes from an image file to filament masks."""

    model: torch.nn.Module
    cfg: PostConfig
    device: torch.device

    @classmethod
    def load(cls, device: torch.device | str | None = None) -> "FinalPipeline":
        """The checkpoint (from ``scripts/final/download_weights.py``) and ``configs/final/postprocess.json``."""
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(resolve_exp_dir(EXPERIMENT) / CHECKPOINT, map_location=device)
        model = build_from_config(state["config"], pretrained=False)
        model.load_state_dict(state["model"])
        return cls(model.to(device).eval(), load_postprocess(FINAL_CONFIG_DIR / "postprocess.json"), device)

    def detect(self, path: Path, info: ImageInfo | None = None) -> Instances:
        """All detections of one image, merged across the views."""
        x = full_image_input(info or image_info(path))
        return predict_tta(lambda v: predict_instances(self.model, v, self.device, torch.float16), x, VIEWS)

    def components(self, detections: Instances, info: ImageInfo) -> Components:
        """Detections as non-overlapping instances on the solar disk, before the score cut-off."""
        region = disk_mask((2048, 2048), info.disk, self.cfg.disk_scale)
        return to_components(detections, self.cfg.threshold, region)

    def segment(self, path: Path) -> list[np.ndarray]:
        """Binary masks of the filaments found in one image."""
        info = image_info(path)
        comps = self.components(self.detect(path, info), info)
        return [comps.labels == i for i in comps.keep(self.cfg.min_area, self.cfg.min_score)]
