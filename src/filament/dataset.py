"""Training data: random full-resolution crops from (image, one annotator's mask) pairs.

Every annotator-image is a separate sample (the organizers allow this). Sampling is uniform over annotator-images,
so an image with 3 annotators is seen 3x as often, which mirrors its 3x weight in the metric. Crops are centered
on a filament with probability ``pos_fraction``, otherwise on a random point of the solar disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from filament.preprocess import Disk, find_disk, read_image
from filament.rle import rle_to_mask, rle_union


@dataclass
class ImageInfo:
    stem: str
    path: Path
    disk: Disk
    mean: float  # on-disk intensity statistics for normalization
    std: float


def image_info(path: Path) -> ImageInfo:
    img = read_image(path)
    disk = find_disk(img)
    yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
    on_disk = (xx - disk.cx) ** 2 + (yy - disk.cy) ** 2 < (0.95 * disk.radius) ** 2
    vals = img[on_disk][::7].astype(np.float32)
    return ImageInfo(path.stem, path, disk, float(vals.mean()), float(vals.std()))


def normalize_with(img: np.ndarray, info: ImageInfo) -> np.ndarray:
    return (img.astype(np.float32) - info.mean) / max(info.std, 1.0)


def build_augment(strength: str = "base") -> A.Compose:
    if strength == "none":
        return A.Compose([])
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(85, 115), p=0.3),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.Defocus(radius=(1, 3))], p=0.2),  # Mauna-Loa-like blur
            A.GaussNoise(std_range=(0.01, 0.04), p=0.2),
        ]
    )


class CropDataset(Dataset):
    def __init__(
        self,
        ann: pd.DataFrame,
        infos: dict[str, ImageInfo],
        stems,
        crop: int = 512,
        samples_per_epoch: int = 4000,
        pos_fraction: float = 0.7,
        augment: A.Compose | None = None,
        seed: int = 0,
    ):
        stems = set(stems)
        sub = ann[ann.stem.isin(stems)]
        self.infos = infos
        self.crop = crop
        self.samples_per_epoch = samples_per_epoch
        self.pos_fraction = pos_fraction
        self.augment = augment
        self.seed = seed
        self.items = []
        for aid, g in sub.groupby("annotator_image", sort=True):
            centers = np.stack([g.bbox_y + g.bbox_h / 2, g.bbox_x + g.bbox_w / 2], axis=1)
            self.items.append({"stem": g.stem.iloc[0], "union": rle_union(g.rle.tolist()), "centers": centers})
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _image(self, stem: str) -> np.ndarray:
        # small per-worker cache: consecutive samples often come from the same few images
        if stem not in self._cache:
            if len(self._cache) >= 16:
                self._cache.pop(next(iter(self._cache)))
            self._cache[stem] = read_image(self.infos[stem].path)
        return self._cache[stem]

    def _center(self, rng: np.random.Generator, item: dict, info: ImageInfo) -> tuple[int, int]:
        if rng.random() < self.pos_fraction and len(item["centers"]):
            cy, cx = item["centers"][rng.integers(len(item["centers"]))]
            jitter = rng.integers(-self.crop // 3, self.crop // 3 + 1, 2)
            return int(cy + jitter[0]), int(cx + jitter[1])
        r = info.disk.radius * np.sqrt(rng.random())
        t = rng.random() * 2 * np.pi
        return int(info.disk.cy + r * np.sin(t)), int(info.disk.cx + r * np.cos(t))

    def __getitem__(self, index: int):
        # torch's RNG is seeded per worker and per epoch by the DataLoader, so crops differ across epochs
        rng = np.random.default_rng(int(torch.randint(0, 2**62, (1,)).item()))
        item = self.items[rng.integers(len(self.items))]
        info = self.infos[item["stem"]]
        img = self._image(item["stem"])
        cy, cx = self._center(rng, item, info)
        h, w = img.shape
        y0 = int(np.clip(cy - self.crop // 2, 0, h - self.crop))
        x0 = int(np.clip(cx - self.crop // 2, 0, w - self.crop))
        sl = (slice(y0, y0 + self.crop), slice(x0, x0 + self.crop))
        image = img[sl]
        mask = rle_to_mask(item["union"])[sl].astype(np.uint8)
        if self.augment is not None:
            out = self.augment(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        x = normalize_with(image, info)[None]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(mask[None])).float()
