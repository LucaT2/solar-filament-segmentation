"""Image loading, solar-disk geometry, and input normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    """Read a GONG JPEG as a 2-D uint8 array."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


@dataclass(frozen=True)
class Disk:
    cx: float
    cy: float
    radius: float


def find_disk(img: np.ndarray, thresh: int | None = None) -> Disk:
    """Solar disk center and radius from the largest bright blob (area-equivalent radius)."""
    if thresh is None:
        thresh = max(10, int(np.percentile(img, 50) * 0.3))
    _, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return Disk(float(centroids[k][0]), float(centroids[k][1]), float(np.sqrt(stats[k, cv2.CC_STAT_AREA] / np.pi)))


def disk_mask(shape: tuple[int, int], disk: Disk, scale: float = 1.0) -> np.ndarray:
    """Boolean mask of pixels within ``scale * radius`` of the disk center."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, (round(disk.cx), round(disk.cy)), round(disk.radius * scale), 1, thickness=-1)
    return mask.astype(bool)


def normalize(img: np.ndarray, disk: Disk | None = None) -> np.ndarray:
    """uint8 image -> float32 standardized with on-disk statistics (off-disk pixels are not informative)."""
    x = img.astype(np.float32)
    ref = x[disk_mask(img.shape, disk, 0.95)] if disk is not None else x
    return (x - ref.mean()) / (ref.std() + 1e-6)
