"""The submitted pipeline in one place: a GONG H-alpha image -> filament masks.

Two complementary model families vote per pixel, then a classifier decides per filament:

1. U-Net ensemble: ResNet34 and ConvNeXt-T encoders, 5 folds each, 8 views (d8 TTA): a good outline of every candidate.
2. Mask R-CNN ensemble: 5 folds, 8 views, each model's views merged per filament, then drawn as score-weighted soft
   masks: object-level confidence, and neighbours kept apart.
3. Weighted map average (0.4 / 0.6), threshold, connected components, size filter.
4. Component classifier (gradient boosting on area, shape, contrast, position, site and mean probability) removes
   components unlikely to match an annotated filament.

The tuned settings and the classifier are committed in ``configs/final/``; the 15 checkpoints are release assets
(``scripts/final/download_weights.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from filament.data.images import read_image
from filament.detection.dataset import full_image_input
from filament.detection.instances import to_prob_map
from filament.detection.model import predict_instances
from filament.detection.train import build_from_config
from filament.detection.tta import predict_tta
from filament.paths import REPO_ROOT, resolve_exp_dir
from filament.postprocess.pipeline import PostConfig, load_postprocess
from filament.postprocess.rescoring import Rescorer, component_table
from filament.segmentation.dataset import ImageInfo, image_info, normalize_with
from filament.segmentation.model import build_model, n_outputs, predict_full

UNETS = ("unet_r34_skel050", "unet_convnext_t_skel050")
DETECTOR = "maskrcnn_ftall_5f"
FOLDS = range(5)
UNET_WEIGHT, DETECTOR_WEIGHT = 0.4, 0.6
VIEWS = 8
FINAL_CONFIG_DIR = REPO_ROOT / "configs" / "final"


def load_unets(names, folds, device) -> list[torch.nn.Module]:
    out = []
    for name in names:
        for k in folds:
            state = torch.load(resolve_exp_dir(name) / f"fold{k}.pt", map_location=device)
            cfg = state["config"]
            m = build_model(cfg["arch"], cfg["encoder"], pretrained=False, decoder_fp32=cfg.get("decoder_fp32", False),
                            classes=n_outputs(cfg))
            m.load_state_dict(state["model"])
            out.append(m.to(device).eval())
    return out


def load_detectors(folds, device) -> list[torch.nn.Module]:
    out = []
    for k in folds:
        state = torch.load(resolve_exp_dir(DETECTOR) / f"fold{k}.pt", map_location=device)
        m = build_from_config(state["config"], pretrained=False)
        m.load_state_dict(state["model"])
        out.append(m.to(device).eval())
    return out


def detections(models, x, device, views: int) -> list:
    """Per model: ``views`` views merged per filament (one Instances per model)."""
    return [predict_tta(lambda v, m=m: predict_instances(m, v, device, torch.float16), x, views) for m in models]


@dataclass
class FinalPipeline:
    unets: list[torch.nn.Module]
    detectors: list[torch.nn.Module]
    cfg: PostConfig
    rescorer: Rescorer
    device: torch.device

    @classmethod
    def load(cls, device: torch.device | str | None = None) -> "FinalPipeline":
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        return cls(load_unets(UNETS, FOLDS, device), load_detectors(FOLDS, device),
                   load_postprocess(FINAL_CONFIG_DIR / "postprocess.json"),
                   Rescorer.load(FINAL_CONFIG_DIR / "rescorer.pkl"), device)

    def maps(self, path: Path, info: ImageInfo | None = None) -> tuple[np.ndarray, np.ndarray]:
        """The two ensembles' probability maps of one image: (U-Net, Mask R-CNN)."""
        info = info or image_info(path)
        unet = predict_full(self.unets, normalize_with(read_image(path), info), self.device, "d8")
        per_model = detections(self.detectors, full_image_input(info), self.device, VIEWS)
        return unet, np.mean([to_prob_map(i) for i in per_model], axis=0)

    def segment(self, path: Path) -> list[np.ndarray]:
        """Binary masks of the filaments found in one image."""
        info = image_info(path)
        unet, detector = self.maps(path, info)
        prob = UNET_WEIGHT * unet + DETECTOR_WEIGHT * detector
        comps, table = component_table(path.stem, prob, read_image(path), self.cfg, info.disk)
        return [comps.labels == i for i in self.rescorer.keep_ids(table)]
