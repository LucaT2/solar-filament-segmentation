"""Probability maps on disk: one 8-bit PNG per image stem (``<dir>/<stem>.png``).

Every model (U-Net, ensembles, the classical baseline) writes its per-pixel filament probability here, and all
post-processing, tuning and scoring tools read from here. Thresholds are therefore quantized to steps of 1/255.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def prob_to_uint8(prob: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(prob * 255), 0, 255).astype(np.uint8)


def uint8_to_prob(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def save_prob(prob_dir: Path, stem: str, prob) -> None:
    prob_dir = Path(prob_dir)
    prob_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(prob_dir / f"{stem}.png"), prob_to_uint8(prob))


def load_prob(prob_dir: Path, stem: str) -> np.ndarray:
    img = cv2.imread(str(Path(prob_dir) / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(Path(prob_dir) / f"{stem}.png")
    return uint8_to_prob(img)
