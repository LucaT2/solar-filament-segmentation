"""Exploratory analyses used by notebooks/analysis/01_eda.ipynb."""

from __future__ import annotations

from itertools import permutations

import cv2
import numpy as np
import pandas as pd

from filament.data.images import read_image
from filament.data.rle import mask_to_rle, rle_iou, rle_to_mask, rle_union
from filament.evaluation.metric import ImageScore, PQResult, match_degrees, score_images

# ----------------------------------------------------------------------------------------------
# Annotator agreement
# ----------------------------------------------------------------------------------------------


def _rles_by_annotator_image(ann: pd.DataFrame) -> dict[str, list[dict]]:
    return {k: g["rle"].tolist() for k, g in ann.groupby("annotator_image", sort=False)}


def pairwise_agreement(ann: pd.DataFrame, ai: pd.DataFrame) -> PQResult:
    """Score every annotator against every other annotator of the same image (ordered pairs: A=GT, B=pred).

    PQ is symmetric under swapping GT and prediction, so ordered pairs only double-count; they keep
    per-filament statistics (e.g. recall of A's filaments) easy to read off.
    """
    rles = _rles_by_annotator_image(ann)
    images = []
    for _, group in ai[ai.n_annotators > 1].groupby("stem"):
        for a, b in permutations(group["annotator_image"], 2):
            gt, pred = rles.get(a, []), rles.get(b, [])
            images.append(ImageScore(f"{a}|{b}", len(gt), len(pred), rle_iou(gt, pred)))
    return score_images(images)


def transformed_agreement(ann: pd.DataFrame, ai: pd.DataFrame, transform) -> PQResult:
    """Like ``pairwise_agreement`` but the 'prediction' annotator's filaments are first passed through
    ``transform(list_of_masks) -> list_of_masks`` (e.g. dilation, or re-instancing by connected components).

    Answers questions like "would a model that draws 1 px wider outlines than annotator B agree better?".
    Scoring every annotator in turn as the submission against the others is exactly how the official
    metric treats a multi-annotated image.
    """
    rles = _rles_by_annotator_image(ann)
    cache: dict[str, list[dict]] = {}
    images = []
    for _, group in ai[ai.n_annotators > 1].groupby("stem"):
        for a, b in permutations(group["annotator_image"], 2):
            if b not in cache:
                masks = [rle_to_mask(r) for r in rles.get(b, [])]
                cache[b] = [mask_to_rle(m) for m in transform(masks) if m.any()]
            gt, pred = rles.get(a, []), cache[b]
            images.append(ImageScore(f"{a}|{b}", len(gt), len(pred), rle_iou(gt, pred)))
    return score_images(images)


def morph(masks: list[np.ndarray], k: int) -> list[np.ndarray]:
    """Dilate (k > 0) or erode (k < 0) each mask by |k| px."""
    if k == 0:
        return masks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(k) + 1, 2 * abs(k) + 1))
    op = cv2.dilate if k > 0 else cv2.erode
    return [op(m.astype(np.uint8), kernel).astype(bool) for m in masks]


def reinstance_by_components(masks: list[np.ndarray], link_px: int = 0) -> list[np.ndarray]:
    """Throw away instance identities: union all masks, then split into connected components.

    With ``link_px > 0`` components closer than ~2*link_px are linked (via dilation) into one instance,
    but each instance keeps only its original (undilated) pixels.
    """
    if not masks:
        return []
    union = np.any(masks, axis=0).astype(np.uint8)
    seed = union
    if link_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * link_px + 1, 2 * link_px + 1))
        seed = cv2.dilate(union, kernel)
    n, labels = cv2.connectedComponents(seed, connectivity=8)
    return [(labels == i) & union.astype(bool) for i in range(1, n)]


def per_filament_match(ann: pd.DataFrame, ai: pd.DataFrame) -> pd.DataFrame:
    """For each filament of a multi-annotated image: its best IoU with each other annotator's filaments
    and whether it overlaps several of them (split) or shares its best partner with others (merge)."""
    rles = _rles_by_annotator_image(ann)
    ann_idx = {k: g.index.tolist() for k, g in ann.groupby("annotator_image", sort=False)}
    rows = []
    for _, group in ai[ai.n_annotators > 1].groupby("stem"):
        for a, b in permutations(group["annotator_image"], 2):
            gt, pred = rles.get(a, []), rles.get(b, [])
            if not gt:
                continue
            iou = rle_iou(gt, pred)
            for i, idx in enumerate(ann_idx[a]):
                row_iou = iou[i] if pred else np.zeros(0)
                n_overlap = int((row_iou > 0).sum())
                best = float(row_iou.max()) if row_iou.size else 0.0
                # how many of A's filaments overlap B's best partner of this filament
                partner_degree = int((iou[:, row_iou.argmax()] > 0).sum()) if best > 0 else 0
                rows.append(
                    {
                        "idx": idx,
                        "gt": a,
                        "other": b,
                        "best_iou": best,
                        "n_overlap": n_overlap,
                        "partner_degree": partner_degree,
                    }
                )
    out = pd.DataFrame(rows)
    out = out.join(ann[["category", "area"]], on="idx")
    out["outcome"] = np.select(
        [
            out.best_iou > 0.5,
            out.n_overlap == 0,
            out.n_overlap > 1,
            out.partner_degree > 1,
        ],
        ["match", "missing", "split", "merged"],
        default="low_iou",
    )
    return out


def union_dice(ann: pd.DataFrame, ai: pd.DataFrame) -> pd.DataFrame:
    """Pixel-level Dice between annotators' union masks (instance grouping ignored)."""
    rles = _rles_by_annotator_image(ann)
    rows = []
    for stem, group in ai[ai.n_annotators > 1].groupby("stem"):
        ids = group["annotator_image"].tolist()
        unions = {k: rle_union(rles[k]) for k in ids if rles.get(k)}
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if a in unions and b in unions:
                    iou = rle_iou([unions[a]], [unions[b]])[0, 0]
                    rows.append({"stem": stem, "a": a, "b": b, "iou": iou, "dice": 2 * iou / (1 + iou)})
    return pd.DataFrame(rows)


def degree_summary(result: PQResult) -> pd.DataFrame:
    gt_deg, pred_deg = match_degrees(result.images)
    return pd.DataFrame(
        {
            "GT filament overlaps k preds": pd.Series(gt_deg).value_counts().sort_index(),
            "pred filament overlaps k GTs": pd.Series(pred_deg).value_counts().sort_index(),
        }
    ).fillna(0).astype(int)


# ----------------------------------------------------------------------------------------------
# Instance geometry and spacing within one annotator-image
# ----------------------------------------------------------------------------------------------


def shape_stats(ann: pd.DataFrame) -> pd.DataFrame:
    """Per-filament shape descriptors computed from the decoded mask (cropped to its bbox)."""
    rows = []
    for idx, rle in ann["rle"].items():
        x, y, w, h = (int(round(v)) for v in ann.loc[idx, ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]])
        mask = rle_to_mask(rle)[max(y - 1, 0) : y + h + 2, max(x - 1, 0) : x + w + 2].astype(np.uint8)
        area = int(mask.sum())
        n_comp = cv2.connectedComponents(mask, connectivity=8)[0] - 1
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        hull_area = cv2.contourArea(cv2.convexHull(np.vstack(contours))) if contours else 0
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.vstack(contours)) if contours else ((0, 0), (0, 0), 0)
        rows.append(
            {
                "idx": idx,
                "mask_area": area,
                "n_components": n_comp,
                "max_half_width": float(dist.max()),  # ~ half the thickest part, in px
                "mean_width": 2 * area / perimeter if perimeter else 0.0,  # ~ average thickness for thin shapes
                "length": max(rw, rh),
                "elongation": max(rw, rh) / max(min(rw, rh), 1.0),
                "solidity": area / hull_area if hull_area else 0.0,
                "cx": x + w / 2,
                "cy": y + h / 2,
            }
        )
    return pd.DataFrame(rows).set_index("idx")


def instance_spacing(ann: pd.DataFrame, max_gap: int = 40) -> pd.DataFrame:
    """For every pair of filaments in the same annotator-image: overlap and the gap between them (px).

    Gap is the smallest distance between the two masks, measured up to ``max_gap`` (larger -> inf).
    """
    rows = []
    for key, group in ann.groupby("annotator_image", sort=False):
        items = list(group[["rle", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]].itertuples())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                # bbox prefilter
                if (
                    a.bbox_x > b.bbox_x + b.bbox_w + max_gap
                    or b.bbox_x > a.bbox_x + a.bbox_w + max_gap
                    or a.bbox_y > b.bbox_y + b.bbox_h + max_gap
                    or b.bbox_y > a.bbox_y + a.bbox_h + max_gap
                ):
                    continue
                x0 = int(min(a.bbox_x, b.bbox_x)) - max_gap
                y0 = int(min(a.bbox_y, b.bbox_y)) - max_gap
                x1 = int(max(a.bbox_x + a.bbox_w, b.bbox_x + b.bbox_w)) + max_gap + 1
                y1 = int(max(a.bbox_y + a.bbox_h, b.bbox_y + b.bbox_h)) + max_gap + 1
                x0, y0 = max(x0, 0), max(y0, 0)
                ma = rle_to_mask(a.rle)[y0:y1, x0:x1]
                mb = rle_to_mask(b.rle)[y0:y1, x0:x1]
                overlap = int((ma & mb).sum())
                dist_to_a = cv2.distanceTransform((~ma).astype(np.uint8), cv2.DIST_L2, 5)
                gap = float(dist_to_a[mb].min()) if mb.any() else np.inf
                rows.append(
                    {
                        "annotator_image": key,
                        "i": a.Index,
                        "j": b.Index,
                        "overlap_px": overlap,
                        "gap": 0.0 if overlap else (gap if gap <= max_gap else np.inf),
                    }
                )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Image properties
# ----------------------------------------------------------------------------------------------


read_gray = read_image  # name used by the EDA notebook


def disk_geometry(img: np.ndarray, thresh: int | None = None) -> dict:
    """Estimate solar disk center and radius from the largest bright blob."""
    if thresh is None:
        thresh = max(10, int(np.percentile(img, 50) * 0.3))
    _, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[k, cv2.CC_STAT_AREA]
    return {"cx": centroids[k][0], "cy": centroids[k][1], "radius": float(np.sqrt(area / np.pi))}


def radial_profile(img: np.ndarray, cx: float, cy: float, radius: float, n_bins: int = 60) -> np.ndarray:
    """Median intensity in rings of normalized radius r/R in [0, 1.1]."""
    yy, xx = np.indices(img.shape)
    r = np.hypot(xx - cx, yy - cy) / radius
    bins = np.linspace(0, 1.1, n_bins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    vals = img.ravel()
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = vals[idx == b]
        if sel.size:
            out[b] = np.median(sel)
    return out


def image_stats(paths: list, sample_px: int = 4) -> pd.DataFrame:
    """Per-image global stats on the disk: geometry, brightness, contrast, sharpness."""
    rows = []
    for path in paths:
        img = read_gray(path)
        geo = disk_geometry(img)
        yy, xx = np.indices(img.shape)
        on_disk = np.hypot(xx - geo["cx"], yy - geo["cy"]) < 0.95 * geo["radius"]
        disk = img[on_disk][::sample_px]
        lap = cv2.Laplacian(img, cv2.CV_32F)[on_disk][::sample_px]
        rows.append(
            {
                "stem": path.stem,
                **geo,
                "channels": cv2.imread(str(path), cv2.IMREAD_UNCHANGED).ndim,
                "disk_mean": float(disk.mean()),
                "disk_std": float(disk.std()),
                "disk_p01": float(np.percentile(disk, 1)),
                "disk_p99": float(np.percentile(disk, 99)),
                "sharpness": float(lap.var()),
                "frac_saturated": float((disk >= 250).mean()),
            }
        )
    return pd.DataFrame(rows)


def filament_contrast(img: np.ndarray, masks: list[np.ndarray], ring: int = 15) -> list[dict]:
    """Intensity inside each filament vs a surrounding ring, and how 'dark' the outline is."""
    out = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    all_fil = np.any(masks, axis=0) if masks else np.zeros(img.shape, bool)
    for m in masks:
        around = cv2.dilate(m.astype(np.uint8), kernel).astype(bool) & ~all_fil
        inside = img[m]
        bg = img[around]
        if inside.size == 0 or bg.size == 0:
            continue
        bg_med = float(np.median(bg))
        out.append(
            {
                "inside_median": float(np.median(inside)),
                "ring_median": bg_med,
                "contrast": (bg_med - float(np.median(inside))) / max(bg_med, 1.0),
                # share of mask pixels that are darker than the local background -> outline tightness
                "frac_darker_than_ring": float((inside < bg_med).mean()),
                # share of the ring that is as dark as the filament core -> missed dark pixels
                "frac_ring_dark": float((bg < np.median(inside)).mean()),
            }
        )
    return out
