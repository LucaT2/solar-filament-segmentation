"""Scored instance predictions of one image, as saved by the detection models, and their conversion to components.

Mask R-CNN predicts, per detection, a box, a score and a small soft mask (28 x 28 probabilities in the box). These raw
outputs are saved per image (``<dir>/<stem>.npz``) so the mask threshold and the score cut-off can be tuned later
without re-running the model, exactly like the probability maps of the segmentation models.

``to_components`` pastes the masks into the full image and resolves overlaps: detections are processed from the
highest score down, each pixel belongs to the first detection that claims it, and a detection that loses most of its
pixels to higher-scoring ones is dropped as a duplicate. Because higher-scoring detections never depend on lower
ones, a score cut-off can be applied afterwards (``Components.keep(min_score=...)``) with identical results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from filament.postprocess.components import Components


@dataclass
class Instances:
    """Detections of one image: boxes, scores and soft masks in box coordinates, as Mask R-CNN outputs them."""

    boxes: np.ndarray  # (n, 4) float32, x0 y0 x1 y1 in image pixels
    scores: np.ndarray  # (n,) float32
    masks: np.ndarray  # (n, M, M) float16 mask probabilities inside each box

    def __len__(self) -> int:
        return len(self.scores)

    @staticmethod
    def empty(m: int = 28) -> "Instances":
        return Instances(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros((0, m, m), np.float16))


def save_instances(inst_dir: Path, stem: str, inst: Instances) -> None:
    inst_dir = Path(inst_dir)
    inst_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(inst_dir / f"{stem}.npz", boxes=inst.boxes.astype(np.float32),
                        scores=inst.scores.astype(np.float32), masks=inst.masks.astype(np.float16))


def load_instances(inst_dir: Path, stem: str) -> Instances:
    with np.load(Path(inst_dir) / f"{stem}.npz") as f:
        return Instances(f["boxes"], f["scores"], f["masks"])


def paste_mask(mask: np.ndarray, box: np.ndarray, threshold: float, shape: tuple[int, int]):
    """Binary mask of one detection in image coordinates, as ``(y0, x0, patch)``; None if it falls outside.

    Follows torchvision's ``paste_masks_in_image``: the M x M mask is padded by one pixel and the box enlarged to match,
    then the mask is resized bilinearly to the box and thresholded.
    """
    m = mask.shape[0]
    padded = np.pad(mask.astype(np.float32), 1)
    scale = (m + 2) / m
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = (box[2] - box[0]) / 2 * scale, (box[3] - box[1]) / 2 * scale
    x0, y0, x1, y1 = int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh)
    w, h = max(x1 - x0 + 1, 1), max(y1 - y0 + 1, 1)
    patch = cv2.resize(padded, (w, h), interpolation=cv2.INTER_LINEAR) > threshold
    # clip to the image
    top, left = max(-y0, 0), max(-x0, 0)
    bottom, right = max(y0 + h - shape[0], 0), max(x0 + w - shape[1], 0)
    patch = patch[top: h - bottom, left: w - right]
    if patch.size == 0:
        return None
    return y0 + top, x0 + left, patch


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, 4) x (m, 4) boxes (x0 y0 x1 y1) -> (n, m) IoU."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def swap_outlines(det: Instances, outline: Instances, min_iou: float = 0.5) -> tuple[Instances, float]:
    """``det``'s detections (which and how confident) with ``outline``'s masks where a detection of ``outline``
    overlaps them clearly (box IoU >= ``min_iou``); box and mask are taken together, the score stays ``det``'s.
    Returns the merged instances and the share of detections whose outline was replaced."""
    if not len(det) or not len(outline):
        return det, 0.0
    iou = box_iou(det.boxes, outline.boxes)
    best = iou.argmax(1)
    swap = iou[np.arange(len(det)), best] >= min_iou
    boxes, masks = det.boxes.copy(), det.masks.copy()
    if outline.masks.shape[1:] != masks.shape[1:]:
        raise ValueError("mask sizes differ")
    boxes[swap] = outline.boxes[best[swap]]
    masks[swap] = outline.masks[best[swap]]
    return Instances(boxes, det.scores.copy(), masks), float(swap.mean())


def to_prob_map(inst: Instances, shape: tuple[int, int] = (2048, 2048), min_score: float = 0.05) -> np.ndarray:
    """A probability map from detections: each soft mask resized into its box and weighted by the detection score;
    the maximum where detections overlap. Lets instance models join probability-map ensembles."""
    out = np.zeros(shape, np.float32)
    m = inst.masks.shape[-1] if len(inst) else 28
    for box, score, mask in zip(inst.boxes, inst.scores, inst.masks):
        if score < min_score:
            continue
        padded = np.pad(mask.astype(np.float32), 1)
        scale = (m + 2) / m
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        hw, hh = (box[2] - box[0]) / 2 * scale, (box[3] - box[1]) / 2 * scale
        x0, y0, x1, y1 = int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh)
        w, h = max(x1 - x0 + 1, 1), max(y1 - y0 + 1, 1)
        soft = cv2.resize(padded, (w, h), interpolation=cv2.INTER_LINEAR) * float(score)
        ty, tx = max(-y0, 0), max(-x0, 0)
        by, bx = max(y0 + h - shape[0], 0), max(x0 + w - shape[1], 0)
        soft = soft[ty: h - by, tx: w - bx]
        if soft.size:
            sl = (slice(y0 + ty, y0 + ty + soft.shape[0]), slice(x0 + tx, x0 + tx + soft.shape[1]))
            np.maximum(out[sl], soft, out=out[sl])
    return out


def to_components(inst: Instances, threshold: float, region: np.ndarray | None = None,
                  shape: tuple[int, int] = (2048, 2048), max_overlap: float = 0.5) -> Components:
    """Non-overlapping instances as a component label map; ``mean_prob`` holds each instance's detection score."""
    labels = np.zeros(shape, np.int32)
    area, score = [0], [0.0]
    for i in np.argsort(-inst.scores, kind="stable"):
        pasted = paste_mask(inst.masks[i], inst.boxes[i], threshold, shape)
        if pasted is None:
            continue
        y0, x0, patch = pasted
        sl = (slice(y0, y0 + patch.shape[0]), slice(x0, x0 + patch.shape[1]))
        if region is not None:
            patch = patch & region[sl]
        own = int(patch.sum())
        free = patch & (labels[sl] == 0)
        n_free = int(free.sum())
        if own == 0 or n_free == 0 or n_free < (1 - max_overlap) * own:
            continue  # empty, or mostly covered by higher-scoring detections: a duplicate
        labels[sl][free] = len(area)
        area.append(n_free)
        score.append(float(inst.scores[i]))
    return Components(labels, np.asarray(area, np.int64), np.asarray(score, np.float64))
