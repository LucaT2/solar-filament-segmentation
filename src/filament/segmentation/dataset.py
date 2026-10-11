"""Training data: random full-resolution crops from (image, one annotator's mask) pairs.

Every annotator-image is a separate sample (the organizers allow this). Sampling is uniform over annotator-images,
so an image with 3 annotators is seen 3x as often, which mirrors its 3x weight in the metric. Crops are centered
on a filament with probability ``pos_fraction``, otherwise on a random point of the solar disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from skimage.morphology import skeletonize
from torch.utils.data import Dataset

from filament.data.images import Disk, find_disk, read_image
from filament.data.rle import rle_to_mask, rle_union


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


def tubed_skeleton(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """Centre line of every filament, thickened to a thin tube (Skeleton Recall loss target, Kirchhoff et al. 2024).

    The skeleton is dilated with a diamond kernel of ``radius`` so the loss also rewards pixels right next to the
    exact centre line, which makes it less brittle than a 1-px skeleton.
    """
    if not mask.any():
        return np.zeros_like(mask, dtype=np.uint8)
    skel = skeletonize(mask > 0).astype(np.uint8)
    if radius > 0:
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        diamond = (np.abs(yy) + np.abs(xx) <= radius).astype(np.uint8)
        skel = cv2.dilate(skel, diamond)
    return skel


class CropDataset(Dataset):
    """Returns ``(x, target)``: x is (1, crop, crop) normalized image; target is (1, crop, crop) mask, or
    (2, crop, crop) [mask, tubed skeleton] when ``skeleton=True``.

    Images are drawn with probability proportional to their number of annotators (their weight in the metric).
    ``target_mode`` decides what the model learns from a drawn image:
        "random"     one annotator's mask, picked at random (binary target)
        "consensus"  the average of all annotators' masks (soft target: 0, 1/3, 1/2, 2/3, 1), and the average of
                     their tubed skeletons, so disputed pixels are taught as "maybe" instead of alternately yes/no
    """

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
        skeleton: bool = False,
        skel_radius: int = 2,
        target_mode: str = "random",
    ):
        if target_mode not in ("random", "consensus"):
            raise ValueError(f"unknown target_mode {target_mode!r}")
        self.skeleton = skeleton
        self.skel_radius = skel_radius
        self.target_mode = target_mode
        stems = set(stems)
        sub = ann[ann.stem.isin(stems)]
        self.infos = infos
        self.crop = crop
        self.samples_per_epoch = samples_per_epoch
        self.pos_fraction = pos_fraction
        self.augment = augment
        self.seed = seed
        self.items = []  # one per image: every annotator's union mask and filament centres
        for stem, gs in sub.groupby("stem", sort=True):
            unions, centers = [], []
            for _, g in gs.groupby("annotator_image", sort=True):
                unions.append(rle_union(g.rle.tolist()))
                centers.append(np.stack([g.bbox_y + g.bbox_h / 2, g.bbox_x + g.bbox_w / 2], axis=1))
            self.items.append({"stem": stem, "unions": unions, "centers": centers})
        n_ann = np.array([len(it["unions"]) for it in self.items], dtype=float)
        self.item_p = n_ann / n_ann.sum()  # == uniform over annotator-images
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

    def _center(self, rng: np.random.Generator, centers: np.ndarray, info: ImageInfo) -> tuple[int, int]:
        if rng.random() < self.pos_fraction and len(centers):
            cy, cx = centers[rng.integers(len(centers))]
            jitter = rng.integers(-self.crop // 3, self.crop // 3 + 1, 2)
            return int(cy + jitter[0]), int(cx + jitter[1])
        r = info.disk.radius * np.sqrt(rng.random())
        t = rng.random() * 2 * np.pi
        return int(info.disk.cy + r * np.sin(t)), int(info.disk.cx + r * np.cos(t))

    def __getitem__(self, index: int):
        # torch's RNG is seeded per worker and per epoch by the DataLoader, so crops differ across epochs
        rng = np.random.default_rng(int(torch.randint(0, 2**62, (1,)).item()))
        item = self.items[rng.choice(len(self.items), p=self.item_p)]
        if self.target_mode == "random":
            a = int(rng.integers(len(item["unions"])))
            unions, centers = [item["unions"][a]], item["centers"][a]
        else:
            unions, centers = item["unions"], np.concatenate(item["centers"])
        info = self.infos[item["stem"]]
        img = self._image(item["stem"])
        cy, cx = self._center(rng, centers, info)
        h, w = img.shape
        y0 = int(np.clip(cy - self.crop // 2, 0, h - self.crop))
        x0 = int(np.clip(cx - self.crop // 2, 0, w - self.crop))
        sl = (slice(y0, y0 + self.crop), slice(x0, x0 + self.crop))
        image = img[sl]
        masks = np.stack([rle_to_mask(u)[sl] for u in unions], axis=-1).astype(np.uint8)  # (crop, crop, n)
        if self.augment is not None:  # one call, so every annotator's mask gets the same geometric transform
            out = self.augment(image=image, mask=masks)
            image, masks = out["image"], out["mask"]
        masks = masks.reshape(self.crop, self.crop, -1)
        x = normalize_with(image, info)[None]
        mask = masks.mean(axis=-1, dtype=np.float32)
        target = mask[None]
        if self.skeleton:  # computed after augmentation so it always matches the transformed masks
            skel = np.mean([tubed_skeleton(masks[..., k], self.skel_radius) for k in range(masks.shape[-1])],
                           axis=0, dtype=np.float32)
            target = np.stack([mask, skel])
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(target)).float()
