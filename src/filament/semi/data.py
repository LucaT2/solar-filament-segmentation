"""Pseudo-labelled frames as a training source (student side of semi-supervised training).

A folder written by scripts/semi/pseudo_label.py (+ scripts/semi/finalize_pseudo.py): ``frames.csv`` (disk and
brightness per frame), ``candidates.csv`` (teacher candidates: id, P(match), centre), ``labels/<stem>.png`` (uint16
label maps) and ``frames/<stem>.jpg``. Per candidate:

* P >= ``conf``           a training instance (the teacher is confident)
* ``unsure`` <= P < ``conf``  unsure: crops containing it are rejected (it is neither a reliable filament nor
                          reliable background, so it must not teach either)
* P < ``unsure``          background (very likely a false candidate)

The archive frames carry text and a logo outside the disk, which the competition images do not: everything beyond
1.02 solar radii is set to the competition images' typical off-disk value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from filament.data.images import Disk, disk_mask, read_image
from filament.segmentation.dataset import ImageInfo

OFF_DISK_VALUE = 17  # mean off-disk grey level of the competition images (archive frames: ~24, with text)


@dataclass
class PseudoFrame:
    stem: str
    info: ImageInfo
    confident: set[int]
    unsure: set[int]
    centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # (cy, cx) of confident candidates


def find_pseudo_dir(name: str) -> Path:
    """Folder with frames.csv + candidates.csv: a path, outputs/<name>, or (on Kaggle) an input containing ``name``."""
    from filament.paths import KAGGLE_INPUT, REPO_ROOT, on_kaggle

    for cand in (Path(name), REPO_ROOT / "outputs" / name):
        if (cand / "frames.csv").exists():
            return cand
    if on_kaggle():
        hits = [p.parent for p in KAGGLE_INPUT.glob("**/frames.csv") if name in str(p)]
        if hits:
            return min(hits, key=lambda p: len(str(p)))
    raise FileNotFoundError(f"no pseudo-label folder (frames.csv) found for {name!r}")


def frames_dir(folder: Path) -> Path:
    from filament.paths import REPO_ROOT

    for cand in (folder / "frames", REPO_ROOT / "outputs" / "unlabelled" / "frames"):
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"no frames/ folder next to {folder}")


class PseudoFrames:
    """Samples crops (image, instance masks) from pseudo-labelled frames, avoiding unsure candidates."""

    def __init__(self, folder: Path, conf: float = 0.6, unsure: float = 0.3, cache_size: int = 8):
        folder = Path(folder)
        self.folder, self.frames_dir = folder, frames_dir(folder)
        frames = pd.read_csv(folder / "frames.csv")
        frames = frames[frames.ok.astype(bool)]
        cand = pd.read_csv(folder / "candidates.csv")
        self.items: list[PseudoFrame] = []
        by_stem = dict(tuple(cand.groupby("stem"))) if len(cand) else {}
        for r in frames.itertuples():
            g = by_stem.get(r.stem)
            disk = Disk(float(r.cx), float(r.cy), float(r.radius))
            info = ImageInfo(r.stem, self.frames_dir / f"{r.stem}.jpg", disk, float(r.mean), float(r.std))
            if g is None:
                self.items.append(PseudoFrame(r.stem, info, set(), set()))
                continue
            conf_rows = g[g.p >= conf]
            centers = conf_rows[["cy", "cx"]].to_numpy(float) if "cy" in g else np.zeros((0, 2))
            self.items.append(PseudoFrame(r.stem, info, set(conf_rows.id.astype(int)),
                                          set(g[(g.p >= unsure) & (g.p < conf)].id.astype(int)), centers))
        self.cache_size = cache_size
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, it: PseudoFrame) -> tuple[np.ndarray, np.ndarray]:
        if it.stem not in self._cache:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            img = read_image(it.info.path)
            img[~disk_mask(img.shape, it.info.disk, 1.02)] = OFF_DISK_VALUE
            labels = cv2.imread(str(self.folder / "labels" / f"{it.stem}.png"), cv2.IMREAD_UNCHANGED)
            self._cache[it.stem] = (img, labels)
        return self._cache[it.stem]

    def sample(self, rng: np.random.Generator, size: int, pos_fraction: float, tries: int = 10):
        """(image crop, list of instance masks, ImageInfo) of ``size`` native px, or None after ``tries`` rejections."""
        for _ in range(tries):
            it = self.items[rng.integers(len(self.items))]
            img, labels = self._load(it)
            d = it.info.disk
            if rng.random() < pos_fraction and len(it.centers):
                cy, cx = it.centers[rng.integers(len(it.centers))]
                cy, cx = cy + rng.integers(-size // 3, size // 3 + 1), cx + rng.integers(-size // 3, size // 3 + 1)
            else:
                r, t = d.radius * np.sqrt(rng.random()), rng.random() * 2 * np.pi
                cy, cx = d.cy + r * np.sin(t), d.cx + r * np.cos(t)
            h, w = img.shape
            y0, x0 = int(np.clip(cy - size // 2, 0, h - size)), int(np.clip(cx - size // 2, 0, w - size))
            lab = labels[y0:y0 + size, x0:x0 + size]
            ids = set(np.unique(lab[lab > 0]).astype(int).tolist())
            if ids & it.unsure:
                continue  # an unsure candidate in the crop: neither filament nor background is safe to teach
            layers = [(lab == i).astype(np.uint8) for i in sorted(ids & it.confident)]
            return img[y0:y0 + size, x0:x0 + size], layers, it.info
        return None
