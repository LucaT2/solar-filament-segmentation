"""Fusion of several models' predictions: one pool of candidate instances, one classifier, one selection.

Candidates come from probability maps (connected components, e.g. a U-Net) and from scored instances (pasted
detections, e.g. Mask R-CNN). Every candidate gets the usual component features plus *agreement* features (how well
the best-overlapping candidate of each other source overlaps it, and that candidate's confidence). A classifier
predicts P(match an annotator's filament, IoU > 0.5). Selection then places candidates from the most to the least
probable; each pixel goes to the first candidate that claims it, and a candidate that loses most of its pixels is
dropped as a duplicate, so the result has no overlaps (as the metric expects).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from filament.data.images import Disk
from filament.detection.instances import Instances, paste_mask
from filament.postprocess.components import Components, find_components

SITES = "BCLMTU"
SIZE = 2048


@dataclass
class Candidate:
    source: str
    y0: int
    x0: int
    patch: np.ndarray  # bool mask inside its box
    conf: float  # the source model's own confidence: mean probability (maps) or detection score (instances)

    @property
    def area(self) -> int:
        return int(self.patch.sum())

    def flat(self) -> np.ndarray:
        ys, xs = np.nonzero(self.patch)
        return np.sort((ys + self.y0) * SIZE + xs + self.x0)


def candidates_from_map(prob: np.ndarray, threshold: float, min_area: int, region: np.ndarray | None,
                        source: str) -> list[Candidate]:
    from scipy.ndimage import find_objects

    comps = find_components(prob, threshold, region)
    out = []
    for i, sl in enumerate(find_objects(comps.labels, max_label=comps.n), start=1):
        if sl is None or comps.area[i] < min_area:
            continue
        out.append(Candidate(source, sl[0].start, sl[1].start, comps.labels[sl] == i, float(comps.mean_prob[i])))
    return out


def candidates_from_instances(inst: Instances, mask_threshold: float, min_score: float, min_area: int,
                              region: np.ndarray | None, source: str) -> list[Candidate]:
    out = []
    for box, score, mask in zip(inst.boxes, inst.scores, inst.masks):
        if score < min_score:
            continue
        pasted = paste_mask(mask, box, mask_threshold, (SIZE, SIZE))
        if pasted is None:
            continue
        y0, x0, patch = pasted
        if region is not None:
            patch = patch & region[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]]
        if patch.sum() >= min_area:
            out.append(Candidate(source, y0, x0, patch, float(score)))
    return out


def _iou_matrix(a: list[np.ndarray], b: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((len(a), len(b)))
    for i, fa in enumerate(a):
        for j, fb in enumerate(b):
            inter = np.intersect1d(fa, fb, assume_unique=True).size
            if inter:
                out[i, j] = inter / (fa.size + fb.size - inter)
    return out


def _shape_contrast(c: Candidate, img: np.ndarray, ring: int = 15) -> tuple[float, float, float]:
    """(length, elongation, contrast) of one candidate; contrast = (ring median - inside median) / ring median."""
    pts = cv2.findNonZero(c.patch.astype(np.uint8))
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts) if pts is not None and len(pts) > 2 else ((0, 0), (1, 1), 0)
    h, w = c.patch.shape
    y0, x0 = max(c.y0 - ring, 0), max(c.x0 - ring, 0)
    y1, x1 = min(c.y0 + h + ring, SIZE), min(c.x0 + w + ring, SIZE)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    m[c.y0 - y0:c.y0 - y0 + h, c.x0 - x0:c.x0 - x0 + w] = c.patch
    around = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1,) * 2)).astype(bool) & ~m.astype(bool)
    crop = img[y0:y1, x0:x1]
    contrast = np.nan
    if around.any() and m.any():
        bg = float(np.median(crop[around]))
        contrast = (bg - float(np.median(crop[m.astype(bool)]))) / max(bg, 1.0)
    return max(rw, rh), max(rw, rh) / max(min(rw, rh), 1.0), contrast


def candidate_table(stem: str, cands: list[Candidate], img: np.ndarray, disk: Disk, sources: list[str],
                    gt_filaments: list[np.ndarray] | None = None) -> pd.DataFrame:
    """Features of every candidate (and, with ground truth, its best IoU with any annotator's filament)."""
    if not cands:
        return pd.DataFrame()
    flats = [c.flat() for c in cands]
    iou = _iou_matrix(flats, flats)
    rows = []
    for i, c in enumerate(cands):
        ys, xs = np.nonzero(c.patch)
        length, elong, contrast = _shape_contrast(c, img)
        row = {"stem": stem, "cand": i, "source": c.source, "conf": c.conf, "area": c.area,
               "length": length, "elongation": elong, "contrast": contrast,
               "r_norm": np.hypot(c.x0 + xs.mean() - disk.cx, c.y0 + ys.mean() - disk.cy) / disk.radius,
               "site": stem[14]}
        # duplicate signal: overlap with a more confident candidate of the same source (detections overlap a lot)
        better = [j for j, o in enumerate(cands) if o.source == c.source and o.conf > c.conf]
        row["dup_iou"] = max((iou[i, j] for j in better), default=0.0)
        row["conf_rank"] = len([j for j in better if iou[i, j] > 0.1])  # more confident overlapping same-source
        for s in sources:  # agreement with every other source
            if s == c.source:
                continue
            others = [j for j, o in enumerate(cands) if o.source == s]
            best = max(others, key=lambda j: iou[i, j], default=None)
            row[f"iou_{s}"] = iou[i, best] if best is not None else 0.0
            row[f"conf_{s}"] = cands[best].conf if best is not None and iou[i, best] > 0 else 0.0
        rows.append(row)
    df = pd.DataFrame(rows)
    if gt_filaments is not None:
        df["best_gt_iou"] = _iou_matrix(flats, gt_filaments).max(axis=1) if gt_filaments else 0.0
    return df


def feature_matrix(df: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    for s in sources:
        x[f"is_{s}"] = (df.source == s).astype(float)
    x["conf"] = df.conf.astype(float)
    x["log_area"] = np.log10(df.area.astype(float))
    for col in ("length", "elongation", "contrast", "r_norm", "dup_iou", "conf_rank"):
        x[col] = df[col].astype(float)
    x["site_code"] = df.site.map({s: i for i, s in enumerate(SITES)}).astype(float)
    for col in df.columns:
        if col.startswith(("iou_", "conf_")):
            x[col] = df[col].astype(float)
    return x


def select(cands: list[Candidate], p: np.ndarray, min_p: float, max_overlap: float = 0.5,
           min_area: int = 0) -> Components:
    """Non-overlapping instances from the most to the least probable candidate; ``mean_prob`` holds P(match)."""
    labels = np.zeros((SIZE, SIZE), np.int32)
    area, score = [0], [0.0]
    for i in np.argsort(-p, kind="stable"):
        if p[i] < min_p:
            break
        c = cands[i]
        sl = (slice(c.y0, c.y0 + c.patch.shape[0]), slice(c.x0, c.x0 + c.patch.shape[1]))
        free = c.patch & (labels[sl] == 0)
        n_free = int(free.sum())
        if n_free == 0 or n_free < (1 - max_overlap) * c.area or n_free < min_area:
            continue
        labels[sl][free] = len(area)
        area.append(n_free)
        score.append(float(p[i]))
    return Components(labels, np.asarray(area, np.int64), np.asarray(score))
