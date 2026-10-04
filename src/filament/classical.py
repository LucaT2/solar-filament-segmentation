"""Classical (no learning) filament detector: local darkness relative to a smooth background.

score = (background - image) / background, where the background is a large median filter. This removes limb
darkening and site-to-site brightness differences in one step. Filaments are elongated dark structures, so the
score is high on them, but also on sunspots and dark fibrils; size filtering in post-processing handles part of
that. The score is mapped to [0, 1] so the shared post-processing and tuning code applies unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np

from filament.preprocess import Disk, disk_mask

SCORE_SCALE = 0.25  # relative darkness mapped to probability 1.0 (filaments are ~12% darker than surroundings)


def darkness_score(img: np.ndarray, disk: Disk, background_px: int = 101, smooth_sigma: float = 1.5) -> np.ndarray:
    """Relative darkness in [0, 1] (0 = as bright as the local background or brighter), zero off the disk."""
    on_disk = disk_mask(img.shape, disk, 1.0)
    # fill the off-disk area with the median disk value so the background filter isn't pulled down at the limb
    filled = img.copy()
    filled[~on_disk] = int(np.median(img[on_disk]))
    background = cv2.medianBlur(filled, background_px | 1).astype(np.float32)
    x = cv2.GaussianBlur(img.astype(np.float32), (0, 0), smooth_sigma) if smooth_sigma > 0 else img.astype(np.float32)
    score = (background - x) / np.maximum(background, 1.0)
    score = np.clip(score / SCORE_SCALE, 0.0, 1.0)
    score[~on_disk] = 0.0
    return score
