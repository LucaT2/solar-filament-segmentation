"""Test-time augmentation for the detection models: predict flipped / transposed copies of the image, map each
copy's detections back to the original orientation and merge the copies' detections of the same filament.

A view is ``(transpose, flip_x, flip_y)``, applied to the image in that order; 4 views are the flips, 8 views add the
transposed copies (the dihedral group of the square, as in the U-Net TTA).

Merging (``merge_views``): detections are taken from the highest score down; each one collects at most one detection
from every other view (the best box overlap, IoU >= ``min_iou``). The merged detection gets the score-weighted mean
box, the mean of the members' soft masks (resampled onto that box) and the score summed over the members and divided
by the number of views, so a filament found in only some views is scored down.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from filament.detection.instances import Instances, box_iou

View = tuple[bool, bool, bool]  # transpose, flip_x, flip_y
VIEWS = {
    1: [(False, False, False)],
    2: [(False, False, False), (False, True, False)],
    4: [(False, False, False), (False, True, False), (False, False, True), (False, True, True)],
}
VIEWS[8] = VIEWS[4] + [(True, f, g) for _, f, g in VIEWS[4]]


def view_image(x: np.ndarray, view: View) -> np.ndarray:
    """The image (``(H, W)`` or ``(C, H, W)``) as seen in ``view``."""
    t, fx, fy = view
    if t:
        x = np.swapaxes(x, -1, -2)
    if fx:
        x = x[..., ::-1]
    if fy:
        x = x[..., ::-1, :]
    return np.ascontiguousarray(x)


def unview_instances(inst: Instances, view: View, shape: tuple[int, int]) -> Instances:
    """Detections predicted on ``view_image(x, view)`` in the coordinates of ``x`` (of size ``shape`` = (H, W))."""
    t, fx, fy = view
    h, w = (shape[1], shape[0]) if t else shape  # the view's size
    boxes, masks = inst.boxes.astype(np.float32).copy(), inst.masks
    if fy:
        boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        masks = masks[:, ::-1, :]
    if fx:
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        masks = masks[:, :, ::-1]
    if t:
        boxes = boxes[:, [1, 0, 3, 2]]
        masks = np.swapaxes(masks, 1, 2)
    return Instances(boxes, inst.scores.copy(), np.ascontiguousarray(masks))


def resample_mask(mask: np.ndarray, box: np.ndarray, target: np.ndarray) -> np.ndarray:
    """An M x M soft mask over ``box`` resampled onto the M x M grid of ``target`` (zero outside ``box``)."""
    m = mask.shape[0]
    centres = (np.arange(m, dtype=np.float32) + 0.5) / m
    tx = target[0] + centres * (target[2] - target[0])  # target cell centres in image coordinates
    ty = target[1] + centres * (target[3] - target[1])
    u = (tx - box[0]) / max(box[2] - box[0], 1e-6) * m - 0.5  # ... as indices into ``mask``
    v = (ty - box[1]) / max(box[3] - box[1], 1e-6) * m - 0.5
    map_x, map_y = np.meshgrid(u, v)
    return cv2.remap(mask.astype(np.float32), map_x.astype(np.float32), map_y.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def merge_views(views: list[Instances], min_iou: float = 0.55) -> Instances:
    """One list of detections from the same image's views (all in the original coordinates)."""
    n_views = len(views)
    if n_views == 1:
        return views[0]
    m = next((v.masks.shape[-1] for v in views if len(v)), 28)
    boxes = np.concatenate([v.boxes for v in views]) if views else np.zeros((0, 4), np.float32)
    if not len(boxes):
        return Instances.empty(m)
    scores = np.concatenate([v.scores for v in views])
    masks = np.concatenate([v.masks for v in views])
    origin = np.concatenate([np.full(len(v), k) for k, v in enumerate(views)])
    iou = box_iou(boxes, boxes)
    used = np.zeros(len(scores), bool)
    out_boxes, out_scores, out_masks = [], [], []
    for i in np.argsort(-scores, kind="stable"):
        if used[i]:
            continue
        members = [i]
        used[i] = True
        for k in range(n_views):
            if k == origin[i]:
                continue
            cand = np.flatnonzero((origin == k) & ~used & (iou[i] >= min_iou))
            if len(cand):
                j = cand[iou[i, cand].argmax()]
                members.append(j)
                used[j] = True
        s = scores[members].astype(np.float64)
        box = (boxes[members] * s[:, None]).sum(0) / s.sum()
        mask = np.mean([resample_mask(masks[j], boxes[j], box) for j in members], axis=0)
        out_boxes.append(box)
        out_scores.append(s.sum() / n_views)
        out_masks.append(mask)
    return Instances(np.asarray(out_boxes, np.float32), np.asarray(out_scores, np.float32),
                     np.asarray(out_masks, np.float16))


def predict_tta(predict: Callable[[np.ndarray], Instances], x: np.ndarray, n_views: int = 8,
                min_iou: float = 0.55) -> Instances:
    """``predict`` (image -> detections) on ``n_views`` views of ``x``, merged."""
    shape = x.shape[-2:]
    return merge_views([unview_instances(predict(view_image(x, v)), v, shape) for v in VIEWS[n_views]], min_iou)
