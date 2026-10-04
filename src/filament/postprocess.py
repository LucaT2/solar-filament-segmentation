"""Probability map -> filament instances.

Stage 1 showed ground-truth filaments almost never touch, so instances are the connected components of the
thresholded map. Components are then filtered by size and confidence, the knobs tuned on out-of-fold PQ.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

# Note: probability maps are cached as 8-bit PNGs, so thresholds are effectively quantized to steps of 1/255.


@dataclass(frozen=True)
class PostConfig:
    threshold: float = 0.5  # on the probability map (in [0, 1]); sets the outline width
    min_area: int = 0  # drop components smaller than this (px)
    min_score: float = 0.0  # drop components whose mean probability is below this
    disk_scale: float = 1.0  # keep only components whose pixels are inside scale * R_sun (0 = no disk mask)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Components:
    """Connected components of one thresholded probability map."""

    labels: np.ndarray  # int32 (H, W), 0 = background, 1..n = components
    area: np.ndarray  # (n + 1,) pixel counts, index 0 = background
    mean_prob: np.ndarray  # (n + 1,) mean probability inside each component

    @property
    def n(self) -> int:
        return len(self.area) - 1

    def keep(self, min_area: int = 0, min_score: float = 0.0) -> np.ndarray:
        """Ids (1-based) of components passing the size and confidence filters."""
        ids = np.arange(1, self.n + 1)
        ok = (self.area[1:] >= min_area) & (self.mean_prob[1:] >= min_score)
        return ids[ok]


def find_components(prob: np.ndarray, threshold: float, region: np.ndarray | None = None) -> Components:
    """Threshold ``prob`` (float in [0, 1]) inside ``region`` and label 8-connected components."""
    binary = prob > threshold
    if region is not None:
        binary &= region
    n, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
    flat = labels.ravel()
    area = np.bincount(flat, minlength=n)
    psum = np.bincount(flat, weights=prob.ravel(), minlength=n)
    return Components(labels, area, psum / np.maximum(area, 1))


def instances(prob: np.ndarray, cfg: PostConfig, region: np.ndarray | None = None) -> list[np.ndarray]:
    """Final instance masks for one image."""
    comps = find_components(prob, cfg.threshold, region)
    return [comps.labels == i for i in comps.keep(cfg.min_area, cfg.min_score)]


def prob_to_uint8(prob: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(prob * 255), 0, 255).astype(np.uint8)


def uint8_to_prob(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0
