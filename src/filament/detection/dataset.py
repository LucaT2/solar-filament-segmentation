"""Training data for instance models: random crops with one annotator's filaments as separate instances.

Sampling follows the segmentation ``CropDataset``: uniform over annotator-images (an image with 3 annotators is drawn
3x as often, like its weight in the metric), crops centred on a filament with probability ``pos_fraction``, the same
augmentations. Each sample is one annotator's filaments, so the model sees every annotator's way of splitting and
outlining filaments.

Filaments cut by the crop edge are kept if at least ``min_instance_area`` pixels remain; boxes are recomputed from the
(augmented) masks.

With ``guide_dir``, a segmentation model's out-of-fold probability map of the same image is cropped and augmented
together with it and becomes the second input channel (see ``filament.detection.model.make_input``).
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from filament.data.images import read_image
from filament.data.rle import rle_to_mask
from filament.detection.model import make_input
from filament.segmentation.dataset import ImageInfo, NeighbourSampler, normalize_with


def full_image_input(info: ImageInfo, guide_dir: Path | None = None) -> np.ndarray:
    """(3, 2048, 2048) model input for a whole image, with the guide map from ``guide_dir`` if given."""
    guide = None
    if guide_dir is not None:
        path = Path(guide_dir) / f"{info.stem}.png"
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"guide map missing: {path}")
        guide = raw.astype(np.float32) / 255
    return make_input(normalize_with(read_image(info.path), info), guide)


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
        guide_dir: Path | None = None,
        neighbours: dict[str, list[Path]] | None = None,
        neighbour_p: float = 0.5,
        scale: float = 1.0,
    ):
        sub = ann[ann.stem.isin(set(stems))]
        # scale > 1: a crop of crop / scale native pixels is upscaled to ``crop`` (filaments appear ``scale`` x larger)
        self.scale = scale
        self.native = int(round(crop / scale))
        self.swap = NeighbourSampler(neighbours, neighbour_p)  # real neighbouring frames, same labels
        self.guide_dir = Path(guide_dir) if guide_dir is not None else None
        if self.guide_dir is not None:
            missing = sorted(set(sub.stem) - {p.stem for p in self.guide_dir.glob("*.png")})
            if missing:
                raise FileNotFoundError(f"{len(missing)} training images have no guide map in {self.guide_dir}, "
                                        f"e.g. {missing[:3]}")
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

    def _image(self, stem: str) -> tuple[np.ndarray, np.ndarray | None]:
        """The image and (with a guide) its uint8 probability map, cached per worker."""
        if stem not in self._cache:
            if len(self._cache) >= 16:
                self._cache.pop(next(iter(self._cache)))
            guide = None
            if self.guide_dir is not None:
                guide = cv2.imread(str(self.guide_dir / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            self._cache[stem] = (read_image(self.infos[stem].path), guide)
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
        img, guide = self._image(item["stem"])
        img, info = self.swap(item["stem"], img, info, rng)
        cy, cx = self._center(rng, item["centers"], info)
        h, w = img.shape
        n = self.native
        y0 = int(np.clip(cy - n // 2, 0, h - n))
        x0 = int(np.clip(cx - n // 2, 0, w - n))
        sl = (slice(y0, y0 + n), slice(x0, x0 + n))
        image = img[sl]
        layers = [m for m in (rle_to_mask(r)[sl] for r in item["rles"]) if m.any()]
        n_inst = len(layers)
        if guide is not None:  # rides along as one more mask layer: same flips/rotations, no intensity changes
            layers.append(guide[sl])
        stack = np.stack(layers, axis=-1).astype(np.uint8) if layers else np.zeros((n, n, 0), np.uint8)
        if n != self.crop:  # upscale: image smoothly, masks with nearest neighbour (stay binary)
            image = cv2.resize(image, (self.crop, self.crop), interpolation=cv2.INTER_CUBIC)
            stack = np.stack([cv2.resize(stack[..., k], (self.crop, self.crop), interpolation=cv2.INTER_NEAREST)
                              for k in range(stack.shape[-1])], axis=-1) if stack.shape[-1] else \
                np.zeros((self.crop, self.crop, 0), np.uint8)
        if self.augment is not None:
            if stack.shape[-1]:
                out = self.augment(image=image, mask=stack)
                image, stack = out["image"], out["mask"].reshape(self.crop, self.crop, -1)
            else:
                image = self.augment(image=image)["image"]
        inst = np.moveaxis(stack[..., :n_inst], -1, 0)
        inst = inst[inst.sum(axis=(1, 2)) >= self.min_instance_area]
        prob = stack[..., n_inst].astype(np.float32) / 255 if guide is not None else None
        x = make_input(normalize_with(image, info), prob)
        target = {
            "boxes": torch.from_numpy(masks_to_boxes(inst)).reshape(-1, 4),
            "labels": torch.ones(len(inst), dtype=torch.int64),
            "masks": torch.from_numpy(np.ascontiguousarray(inst)),
        }
        return torch.from_numpy(x), target


def collate(batch):
    """Detection models take lists of images and targets (instance counts differ per image)."""
    return [b[0] for b in batch], [b[1] for b in batch]
