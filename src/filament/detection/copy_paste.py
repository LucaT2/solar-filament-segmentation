"""Copy-paste augmentation for filaments (Ghiasi et al. 2021, "Simple Copy-Paste is a Strong Data Augmentation Method
for Instance Segmentation"), adapted to H-alpha images.

A filament is not pasted as raw pixels but as its darkness relative to the local background of its own image: the
source background is the image with the filament inpainted and smoothed, and the ratio image / background inside the
slightly grown, feathered mask multiplies the target crop. The target's brightness and limb darkening stay as they
are; the filament brings its own contrast, texture and outline.

Placement: the whole filament lies on the disk (< 0.97 R) at about the same distance from the disk centre as in its
own image (filaments near the limb look different), and its box plus a margin overlaps no filament box of any
annotator of the target image and no earlier paste. Sources are the training annotations of the same observing site
(similar sharpness and contrast), so fold-clean. The pasted filament becomes one more instance of the crop's target.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from filament.data.rle import rle_to_mask

_GROW = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 2 px: the dark edge just outside the outline
_HOLE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # inpainted for the background estimate


class FilamentPaster:
    """``items``: dicts with ``stem``, ``rles`` and ``boxes`` ((n, 4) x0 y0 x1 y1, full-image px) per annotator-image;
    ``infos``: ``ImageInfo`` per stem (for the disk). Each call pastes 1..``max_paste`` filaments with probability ``p``."""

    def __init__(self, items, infos, p: float = 0.5, max_paste: int = 3, margin: int = 10, radial_tol: float = 0.15,
                 tries: int = 15, pad: int = 12, max_ratio: float = 0.95):
        # by observing site (stem[14]): a sharp filament pasted into a blurry Mauna Loa image would look pasted
        self.sources: dict[str, list] = {}
        for it in items:
            self.sources.setdefault(it["stem"][14], []).extend(
                (it["stem"], r, b) for r, b in zip(it["rles"], it["boxes"]))
        self.infos = infos
        self.p, self.max_paste, self.margin, self.radial_tol, self.tries, self.pad = p, max_paste, margin, radial_tol, \
            tries, pad
        self.max_ratio = max_ratio  # sources whose mean intensity / background inside the mask is above this are skipped

    def source(self, rng: np.random.Generator, load: Callable[[str], np.ndarray], site: str):
        """(multiplicative effect patch, binary mask patch, distance of the filament from the disk centre / R) of a
        random training filament from ``site``."""
        pool = self.sources[site]
        stem, rle, (bx0, by0, bx1, by1) = pool[rng.integers(len(pool))]
        img = load(stem)
        h, w = img.shape
        ya, yb = max(int(by0) - self.pad, 0), min(int(np.ceil(by1)) + self.pad, h)
        xa, xb = max(int(bx0) - self.pad, 0), min(int(np.ceil(bx1)) + self.pad, w)
        mask = np.ascontiguousarray(rle_to_mask(rle)[ya:yb, xa:xb], dtype=np.uint8)
        patch = np.ascontiguousarray(img[ya:yb, xa:xb])
        bg = cv2.inpaint(patch, cv2.dilate(mask, _HOLE), 5, cv2.INPAINT_TELEA).astype(np.float32)
        bg = cv2.GaussianBlur(bg, (0, 0), 3)
        alpha = cv2.GaussianBlur(cv2.dilate(mask, _GROW).astype(np.float32), (0, 0), 1.0)
        effect = 1 + alpha * (patch.astype(np.float32) / np.maximum(bg, 1.0) - 1)
        d = self.infos[stem].disk
        r = np.hypot((by0 + by1) / 2 - d.cy, (bx0 + bx1) / 2 - d.cx) / d.radius
        k = int(rng.integers(4))
        effect, mask = np.rot90(effect, k), np.rot90(mask, k)
        if rng.random() < 0.5:
            effect, mask = effect[:, ::-1], mask[:, ::-1]
        return effect, mask, r

    def __call__(self, crop: np.ndarray, stem: str, origin: tuple[int, int], disk, taken: np.ndarray,
                 rng: np.random.Generator, load: Callable[[str], np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
        """Paste into ``crop`` (uint8, native px, top-left ``origin`` in image ``stem``; left unchanged, a copy is
        returned). ``taken``: (m, 4) boxes x0 y0 x1 y1 (full-image px) of all filaments of the crop's image. Returns the
        crop and the pasted filaments' masks (crop-sized uint8)."""
        if stem[14] not in self.sources or rng.random() >= self.p:
            return crop, []
        n_y, n_x = crop.shape
        oy, ox = origin
        m = self.margin
        boxes = [b for b in (np.asarray(taken, np.float32).reshape(-1, 4) - [ox, oy, ox, oy])
                 if b[2] > -m and b[3] > -m and b[0] < n_x + m and b[1] < n_y + m]
        out, layers = None, []
        for _ in range(int(rng.integers(1, self.max_paste + 1))):
            effect, mask, r_src = self.source(rng, load, stem[14])
            ph, pw = mask.shape
            if ph >= n_y or pw >= n_x:
                continue
            ys, xs = np.nonzero(mask)
            if not len(ys) or effect[ys, xs].mean() > self.max_ratio:
                continue  # empty, or too faint to be visible in another place: a pasted label would only be noise
            box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
            sy, sx = ys[::4], xs[::4]
            for _ in range(self.tries):
                py, px = int(rng.integers(0, n_y - ph + 1)), int(rng.integers(0, n_x - pw + 1))
                b = box + [px, py, px, py]
                if any(b[0] < o[2] + m and o[0] - m < b[2] and b[1] < o[3] + m and o[1] - m < b[3] for o in boxes):
                    continue
                rr = np.hypot(sy + py + oy - disk.cy, sx + px + ox - disk.cx) / disk.radius
                if rr.max() >= 0.97 or abs(np.hypot(*(np.array([b[1] + b[3], b[0] + b[2]]) / 2 + [oy, ox]
                                                      - [disk.cy, disk.cx])) / disk.radius - r_src) > self.radial_tol:
                    continue
                if out is None:
                    out = crop.astype(np.float32)
                out[py:py + ph, px:px + pw] *= effect
                layer = np.zeros((n_y, n_x), np.uint8)
                layer[py:py + ph, px:px + pw] = mask
                layers.append(layer)
                boxes.append(b)
                break
        if out is None:
            return crop, []
        return np.clip(np.rint(out), 0, 255).astype(np.uint8), layers
