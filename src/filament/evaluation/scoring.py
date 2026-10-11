"""Fast out-of-fold scoring and post-processing search on saved predictions (probability maps or instances).

Instead of encoding every candidate component as RLE, IoUs are computed directly from the component label map and
each ground-truth filament's pixel indices. This gives the same IoU matrices as ``filament.evaluation.metric`` (and
the official code) at a fraction of the cost, so a whole grid of post-processing settings can be scored in minutes.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from filament.data.rle import rle_to_mask
from filament.evaluation.metric import ImageScore, score_images
from filament.postprocess.components import Components, region_for
from filament.postprocess.predictions import component_builder


class GroundTruth:
    """Per image stem: for each annotator-image, the flat pixel indices of every filament."""

    def __init__(self, ann: pd.DataFrame, stems):
        self.by_stem: dict[str, dict[str, list[np.ndarray]]] = {}
        sub = ann[ann.stem.isin(set(stems))]
        for (stem, aid), g in sub.groupby(["stem", "annotator_image"], sort=False):
            self.by_stem.setdefault(stem, {})[aid] = [np.flatnonzero(rle_to_mask(r)) for r in g.rle]

    @property
    def stems(self) -> list[str]:
        return list(self.by_stem)


def score_components(stem: str, comps: Components, keep: np.ndarray, gt: GroundTruth) -> list[ImageScore]:
    """One ImageScore per annotator-image of ``stem`` for the kept components (exact official IoU)."""
    flat = comps.labels.ravel()
    pred_area = comps.area[keep]
    out = []
    for aid, filaments in gt.by_stem[stem].items():
        iou = np.zeros((len(filaments), len(keep)))
        if len(keep):
            col = np.full(comps.n + 1, -1)
            col[keep] = np.arange(len(keep))
            for i, idx in enumerate(filaments):
                hit = col[flat[idx]]
                inter = np.bincount(hit[hit >= 0], minlength=len(keep))
                iou[i] = inter / (len(idx) + pred_area - inter)
        out.append(ImageScore(aid, len(filaments), len(keep), iou))
    return out


def _grid_one_image(args) -> dict[tuple, list[ImageScore]]:
    stem, stem_gt, pred_dir, image_dir, thresholds, min_areas, min_scores, disk_scale, link_pxs = args
    gt = GroundTruth.__new__(GroundTruth)
    gt.by_stem = {stem: stem_gt}
    components = component_builder(pred_dir, stem, region_for(stem, image_dir, disk_scale, {}))
    out: dict[tuple, list[ImageScore]] = {}
    for thr, link in itertools.product(thresholds, link_pxs):
        comps = components(thr, link)
        for min_area, min_score in itertools.product(min_areas, min_scores):
            keep = comps.keep(min_area, min_score)
            out[(thr, min_area, min_score, link)] = score_components(stem, comps, keep, gt)
    return out


def grid_search(
    pred_dir,
    gt: GroundTruth,
    image_dir,
    thresholds,
    min_areas=(0,),
    min_scores=(0.0,),
    disk_scale: float = 1.0,
    workers: int = 8,
    link_pxs=(0,),
) -> pd.DataFrame:
    """PQ for every (threshold, min_area, min_score, link_px) combination; components are computed once per
    (threshold, link_px).

    Images are processed in parallel (``workers`` processes; 0 = in-process).
    """
    jobs = [(s, gt.by_stem[s], pred_dir, image_dir, list(thresholds), list(min_areas), list(min_scores), disk_scale,
             list(link_pxs)) for s in gt.stems]
    if workers > 0:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(_grid_one_image, jobs, chunksize=8))
    else:
        results = [_grid_one_image(j) for j in jobs]
    per_combo: dict[tuple, list[ImageScore]] = {}
    for res in results:
        for combo, scores in res.items():
            per_combo.setdefault(combo, []).extend(scores)
    rows = []
    for (thr, min_area, min_score, link), images in per_combo.items():
        r = score_images(images)
        rows.append({"threshold": thr, "min_area": min_area, "min_score": min_score, "link_px": link,
                     "pq": r.pq, "sq": r.sq, "rq": r.rq, "tp": r.tp, "fp": r.fp, "fn": r.fn})
    return pd.DataFrame(rows).sort_values("pq", ascending=False).reset_index(drop=True)
