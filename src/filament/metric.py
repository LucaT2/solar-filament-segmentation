"""Local re-implementation of the official Panoptic Quality metric.

Mirrors ``self-evaluation-notebook.ipynb`` from the organizers:
    * ground truth is grouped per annotator-image (``filament_id`` before ``_`` = ``<annotator>-<stem>``),
      predictions per image stem; every annotator-image is scored against the predictions for its stem;
    * predictions for stems without any ground truth are ignored;
    * every GT/prediction pair with IoU > 0.5 is a true positive (no one-to-one assignment step);
    * FP = predictions hitting no GT, FN = GT hitting no prediction;
    * PQ = sum(TP IoU) / (|TP| + 0.5 |FP| + 0.5 |FN|).

IoU is computed on RLEs with pycocotools instead of decoding dense 2048 x 2048 stacks, which is
much faster but numerically equivalent up to float precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from filament.rle import counts_to_rle, rle_iou

IOU_THRESHOLD = 0.5


@dataclass
class ImageScore:
    key: str
    n_gt: int
    n_pred: int
    iou: np.ndarray  # (n_gt, n_pred)

    @property
    def hits(self) -> np.ndarray:
        return self.iou > IOU_THRESHOLD

    @property
    def tp_ious(self) -> np.ndarray:
        return self.iou[self.hits]

    @property
    def fp(self) -> int:
        return int((self.hits.sum(axis=0) == 0).sum()) if self.n_gt else self.n_pred

    @property
    def fn(self) -> int:
        return int((self.hits.sum(axis=1) == 0).sum()) if self.n_pred else self.n_gt


@dataclass
class PQResult:
    pq: float
    sq: float  # mean IoU of true positives (segmentation quality)
    rq: float  # |TP| / (|TP| + 0.5|FP| + 0.5|FN|) (recognition quality)
    tp: int
    fp: int
    fn: int
    images: list[ImageScore]

    def per_image(self) -> pd.DataFrame:
        rows = []
        for s in self.images:
            tp = s.tp_ious
            denom = len(tp) + 0.5 * s.fp + 0.5 * s.fn
            rows.append(
                {
                    "key": s.key,
                    "n_gt": s.n_gt,
                    "n_pred": s.n_pred,
                    "tp": len(tp),
                    "fp": s.fp,
                    "fn": s.fn,
                    "pq": tp.sum() / denom if denom else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"PQResult(pq={self.pq:.4f}, sq={self.sq:.4f}, rq={self.rq:.4f}, "
            f"tp={self.tp}, fp={self.fp}, fn={self.fn}, n_images={len(self.images)})"
        )


def score_images(images: list[ImageScore]) -> PQResult:
    tp_ious = np.concatenate([s.tp_ious for s in images]) if images else np.zeros(0)
    tp, fp, fn = len(tp_ious), sum(s.fp for s in images), sum(s.fn for s in images)
    denom = tp + 0.5 * fp + 0.5 * fn
    return PQResult(
        pq=float(tp_ious.sum() / denom) if denom else 0.0,
        sq=float(tp_ious.mean()) if tp else 0.0,
        rq=float(tp / denom) if denom else 0.0,
        tp=tp,
        fp=fp,
        fn=fn,
        images=images,
    )


def pq_from_rles(gt: dict[str, list[dict]], pred: dict[str, list[dict]]) -> PQResult:
    """Score already-grouped RLE dicts: ``gt[key]`` vs ``pred[key]`` for every key in ``gt``."""
    images = []
    for key, gt_rles in gt.items():
        pred_rles = pred.get(key, [])
        images.append(ImageScore(key, len(gt_rles), len(pred_rles), rle_iou(gt_rles, pred_rles)))
    return score_images(images)


def pq_score(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> PQResult:
    """Score two tables in submission format (``filament_id``, ``segmentation_rle``), like the official notebook."""
    gt_keys = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_keys = pred_df["filament_id"].str.split("_", n=1).str[0]

    pred_groups: dict[str, list[dict]] = {}
    for key, counts in zip(pred_keys, pred_df["segmentation_rle"]):
        pred_groups.setdefault(key, []).append(counts_to_rle(counts))

    images = []
    for annotator_image, group in gt_df.groupby(gt_keys, sort=False):
        stem = annotator_image.split("-", 1)[1]
        gt_rles = [counts_to_rle(c) for c in group["segmentation_rle"]]
        pred_rles = pred_groups.get(stem, [])
        images.append(ImageScore(annotator_image, len(gt_rles), len(pred_rles), rle_iou(gt_rles, pred_rles)))
    return score_images(images)


def match_degrees(images: list[ImageScore]) -> tuple[np.ndarray, np.ndarray]:
    """Number of partners with IoU > 0 for each GT (1:n) and each prediction (n:1), as in the official plots."""
    gt_deg, pred_deg = [], []
    for s in images:
        overlap = s.iou > 0
        gt_deg.extend(overlap.sum(axis=1).tolist() if s.n_pred else [0] * s.n_gt)
        pred_deg.extend(overlap.sum(axis=0).tolist() if s.n_gt else [0] * s.n_pred)
    return np.asarray(gt_deg), np.asarray(pred_deg)
