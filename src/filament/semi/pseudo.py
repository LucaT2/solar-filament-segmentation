"""Teacher for pseudo-labels: the same recipe as our best submission, applied to unlabelled frames.

probability map = 0.5 * mean(U-Net maps, 4-flip TTA) + 0.5 * Mask R-CNN detections drawn as a map;
components (tuned threshold, minimum area) are scored by the component classifier, giving P(match) per candidate.
The teacher's settings (``postprocess.json``, ``rescorer.pkl``) are tuned on fold-0 OOF maps of exactly these models,
all trained without fold 0, so a student validated on fold 0 stays honest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from filament.data.images import read_image
from filament.detection.instances import to_prob_map
from filament.detection.model import predict_instances
from filament.postprocess.components import PostConfig
from filament.postprocess.maps import prob_to_uint8, uint8_to_prob
from filament.postprocess.rescoring import Rescorer, component_table
from filament.segmentation.dataset import ImageInfo, image_info, normalize_with
from filament.segmentation.model import predict_full


@dataclass
class Teacher:
    unets: list  # loaded segmentation models (each predicted separately, maps averaged)
    unet_cfgs: list[dict]
    detector: torch.nn.Module
    det_cfg: dict
    cfg: PostConfig
    rescorer: Rescorer
    device: torch.device
    weights: tuple[float, float] = (0.5, 0.5)

    @torch.no_grad()
    def prob_map(self, x: np.ndarray) -> np.ndarray:
        maps = []
        for model, c in zip(self.unets, self.unet_cfgs):
            maps.append(predict_full([model], x, self.device, tta="flip4", amp_dtype=torch.float16,
                                     tile=c.get("infer_tile"), overlap=c.get("infer_overlap", 128)))
        # quantize like the saved PNG maps the teacher was tuned on
        unet = np.mean([uint8_to_prob(prob_to_uint8(m)) for m in maps], axis=0)
        det = to_prob_map(predict_instances(self.detector, x, self.device, torch.float16,
                                            self.det_cfg.get("scale", 1.0)))
        det = uint8_to_prob(prob_to_uint8(det))
        return self.weights[0] * unet + self.weights[1] * det

    def label(self, path: Path) -> tuple[np.ndarray, list[dict], ImageInfo]:
        """Label map (uint16, 0 = background) of the size-filtered candidates and one row per candidate."""
        info = image_info(path)
        img = read_image(path)
        prob = uint8_to_prob(prob_to_uint8(self.prob_map(normalize_with(img, info))))
        comps, table = component_table(path.stem, prob, img, self.cfg, info.disk)
        kept = table[table.kept.astype(bool)] if len(table) else table
        labels = np.zeros(prob.shape, np.uint16)
        rows = []
        if len(kept):
            p = self.rescorer.probabilities(kept)
            for new_id, (comp, area, pm) in enumerate(zip(kept.comp, kept.area, p), start=1):
                labels[comps.labels == comp] = new_id
                rows.append({"stem": path.stem, "id": new_id, "area": int(area), "p": float(pm)})
        return labels, rows, info
