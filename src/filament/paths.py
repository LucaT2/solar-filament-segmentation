"""Locate the competition data and output directories, both locally and on Kaggle.

Resolution order for the data root (the ``MAGFiLO_1.0_Kaggle_2026`` folder):
    1. the ``FILAMENT_DATA`` environment variable,
    2. a search under ``/kaggle/input`` (Kaggle notebooks),
    3. ``<repo>/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026`` (local download).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DATASET_DIR_NAME = "MAGFiLO_1.0_Kaggle_2026"
ANNOTATION_FILE = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

REPO_ROOT = Path(__file__).resolve().parents[2]
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def on_kaggle() -> bool:
    return KAGGLE_INPUT.is_dir()


def find_data_root() -> Path:
    env = os.environ.get("FILAMENT_DATA")
    if env:
        root = Path(env)
        if not root.is_dir():
            raise FileNotFoundError(f"FILAMENT_DATA points to a missing directory: {root}")
        return root

    if on_kaggle():
        # The mount path differs between notebook versions, so search for the dataset folder.
        matches = sorted(KAGGLE_INPUT.glob(f"**/{DATASET_DIR_NAME}"))
        if matches:
            return matches[0]

    local = REPO_ROOT / "filament-segmentation-2026" / DATASET_DIR_NAME
    if local.is_dir():
        return local

    raise FileNotFoundError(
        f"Could not find {DATASET_DIR_NAME}. Set FILAMENT_DATA to the folder containing train/ and test/."
    )


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def train_images(self) -> Path:
        return self.root / "train" / "train_images"

    @property
    def test_images(self) -> Path:
        return self.root / "test" / "test_images"

    @property
    def annotations(self) -> Path:
        return self.root / "train" / ANNOTATION_FILE


def get_data_paths() -> DataPaths:
    return DataPaths(find_data_root())


def get_output_dir(name: str = "") -> Path:
    """Directory for checkpoints, predictions and submissions (git-ignored locally)."""
    base = KAGGLE_WORKING if on_kaggle() else REPO_ROOT / "outputs"
    out = base / name if name else base
    out.mkdir(parents=True, exist_ok=True)
    return out
