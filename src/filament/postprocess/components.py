"""Probability map -> filament instances.

Stage 1 showed ground-truth filaments almost never touch, so instances are the connected components of the
thresholded map. Components are then filtered by size and confidence, the knobs tuned on out-of-fold PQ.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from filament.data.images import disk_mask, find_disk, read_image


@dataclass(frozen=True)
class PostConfig:
    """Post-processing settings, tuned on out-of-fold predictions (``scripts/postprocess/tune_postprocess.py``)."""

    threshold: float = 0.5  # on the probability map (in [0, 1]); sets the outline width
    min_area: int = 0  # drop components smaller than this (px)
    min_score: float = 0.0  # drop components whose mean probability is below this
    disk_scale: float = 1.0  # keep only components whose pixels are inside scale * R_sun (0 = no disk mask)
    link_px: int = 0  # pieces closer than about 2 * link_px px become one instance (each keeps its own pixels)

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


def find_components(prob: np.ndarray, threshold: float, region: np.ndarray | None = None, link_px: int = 0) -> Components:
    """Threshold ``prob`` (float in [0, 1]) inside ``region`` and label 8-connected components.

    With ``link_px > 0``, components are found on the mask dilated by ``link_px`` (so nearby pieces of a fragmented
    filament share one label), but each instance keeps only its original thresholded pixels.
    """
    binary = prob > threshold
    if region is not None:
        binary &= region
    seed = binary.astype(np.uint8)
    if link_px > 0:
        seed = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * link_px + 1, 2 * link_px + 1)))
    n, labels = cv2.connectedComponents(seed, connectivity=8, ltype=cv2.CV_32S)
    if link_px > 0:
        labels[~binary] = 0
    flat = labels.ravel()
    area = np.bincount(flat, minlength=n)
    psum = np.bincount(flat, weights=prob.ravel(), minlength=n)
    return Components(labels, area, psum / np.maximum(area, 1))


def instances(prob: np.ndarray, cfg: PostConfig, region: np.ndarray | None = None) -> list[np.ndarray]:
    """Final instance masks for one image."""
    comps = find_components(prob, cfg.threshold, region, cfg.link_px)
    return [comps.labels == i for i in comps.keep(cfg.min_area, cfg.min_score)]


def region_for(stem: str, image_dir: Path, scale: float, cache: dict) -> np.ndarray | None:
    """Disk mask (``scale`` * R_sun) for an image, with the detected disk cached per stem; None if ``scale`` <= 0."""
    if scale <= 0:
        return None
    if stem not in cache:
        cache[stem] = find_disk(read_image(Path(image_dir) / f"{stem}.jpeg"))
    return disk_mask((2048, 2048), cache[stem], scale)
