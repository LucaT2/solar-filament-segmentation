"""Training data for instance models: random crops with one annotator's filaments as separate instances.

Sampling follows the segmentation ``CropDataset``: uniform over annotator-images (an image with 3 annotators is drawn
3x as often, like its weight in the metric), crops centred on a filament with probability ``pos_fraction``, the same
augmentations. Each sample is one annotator's filaments, so the model sees every annotator's way of splitting and
outlining filaments.

Filaments cut by the crop edge are kept if at least ``min_instance_area`` pixels remain; boxes are recomputed from the
(augmented) masks.
"""

from __future__ import annotations

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from filament.data.images import read_image
from filament.data.rle import rle_to_mask
from filament.segmentation.dataset import ImageInfo, normalize_with


def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """(n, H, W) binary masks -> (n, 4) boxes x0 y0 x1 y1 (x1, y1 exclusive, so every box has positive size)."""
    boxes = np.zeros((len(masks), 4), np.float32)
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m.any(axis=1))[0], np.nonzero(m.any(axis=0))[0]
        boxes[i] = xs[0], ys[0], xs[-1] + 1, ys[-1] + 1
    return boxes


class InstanceCropDataset(Dataset):
    """Returns ``(x, target)``: x is a (3, crop, crop) normalized image (the gray channel repeated), target is the
    torchvision detection target ``{"boxes": (n, 4), "labels": (n,), "masks": (n, crop, crop)}``."""

    def __init__(
        self,
        ann: pd.DataFrame,
        infos: dict[str, ImageInfo],
        stems,
        crop: int = 1024,
        samples_per_epoch: int = 2000,
        pos_fraction: float = 0.8,
        augment: A.Compose | None = None,
        min_instance_area: int = 30,
    ):
        sub = ann[ann.stem.isin(set(stems))]
        self.infos = infos
        self.crop = crop
        self.samples_per_epoch = samples_per_epoch
        self.pos_fraction = pos_fraction
        self.augment = augment
        self.min_instance_area = min_instance_area
        self.items = []  # one per annotator-image: its filaments' RLEs and centres
        for (stem, _), g in sub.groupby(["stem", "annotator_image"], sort=True):
            centers = np.stack([g.bbox_y + g.bbox_h / 2, g.bbox_x + g.bbox_w / 2], axis=1)
            self.items.append({"stem": stem, "rles": g.rle.tolist(), "centers": centers})
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _image(self, stem: str) -> np.ndarray:
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
        rng = np.random.default_rng(int(torch.randint(0, 2**62, (1,)).item()))
        item = self.items[rng.integers(len(self.items))]
        info = self.infos[item["stem"]]
        img = self._image(item["stem"])
        cy, cx = self._center(rng, item["centers"], info)
        h, w = img.shape
        y0 = int(np.clip(cy - self.crop // 2, 0, h - self.crop))
        x0 = int(np.clip(cx - self.crop // 2, 0, w - self.crop))
        sl = (slice(y0, y0 + self.crop), slice(x0, x0 + self.crop))
        image = img[sl]
        masks = [m for m in (rle_to_mask(r)[sl] for r in item["rles"]) if m.any()]
        stack = np.stack(masks, axis=-1).astype(np.uint8) if masks else np.zeros((self.crop, self.crop, 0), np.uint8)
        if self.augment is not None:
            if stack.shape[-1]:
                out = self.augment(image=image, mask=stack)
                image, stack = out["image"], out["mask"].reshape(self.crop, self.crop, -1)
            else:
                image = self.augment(image=image)["image"]
        inst = np.moveaxis(stack, -1, 0)
        inst = inst[inst.sum(axis=(1, 2)) >= self.min_instance_area]
        x = normalize_with(image, info)
        target = {
            "boxes": torch.from_numpy(masks_to_boxes(inst)).reshape(-1, 4),
            "labels": torch.ones(len(inst), dtype=torch.int64),
            "masks": torch.from_numpy(np.ascontiguousarray(inst)),
        }
        return torch.from_numpy(np.ascontiguousarray(x))[None].expand(3, -1, -1).contiguous(), target


def collate(batch):
    """Detection models take lists of images and targets (instance counts differ per image)."""
    return [b[0] for b in batch], [b[1] for b in batch]
