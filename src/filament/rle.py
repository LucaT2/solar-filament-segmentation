"""COCO run-length encoding helpers in the competition's submission format.

The submission stores only the compressed RLE ``counts`` string; the size is implicitly 2048 x 2048.
"""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_util

IMAGE_SIZE = 2048


def mask_to_rle(mask: np.ndarray) -> dict:
    """Binary (H, W) mask -> compressed COCO RLE dict with ``counts`` as str."""
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def mask_to_counts(mask: np.ndarray) -> str:
    """Binary (H, W) mask -> RLE counts string, as written to ``segmentation_rle``."""
    return mask_to_rle(mask)["counts"]


def counts_to_rle(counts: str, height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> dict:
    return {"size": [height, width], "counts": counts}


def rle_to_mask(rle: dict) -> np.ndarray:
    return mask_util.decode(rle).astype(bool)


def counts_to_mask(counts: str, height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> np.ndarray:
    return rle_to_mask(counts_to_rle(counts, height, width))


def polygons_to_rle(polygons: list[list[float]], height: int, width: int) -> dict:
    """COCO polygon list -> single compressed RLE (pieces merged), ``counts`` as str."""
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles) if isinstance(rles, list) else rles
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _as_bytes(rle: dict) -> dict:
    counts = rle["counts"]
    return {"size": rle["size"], "counts": counts.encode("ascii") if isinstance(counts, str) else counts}


def rle_area(rles: list[dict]) -> np.ndarray:
    return mask_util.area([_as_bytes(r) for r in rles])


def rle_iou(gt: list[dict], pred: list[dict]) -> np.ndarray:
    """Pairwise IoU matrix of shape (len(gt), len(pred)), computed directly on RLEs."""
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)), dtype=np.float64)
    iou = mask_util.iou([_as_bytes(r) for r in gt], [_as_bytes(r) for r in pred], [0] * len(pred))
    return np.asarray(iou, dtype=np.float64).reshape(len(gt), len(pred))


def rle_union(rles: list[dict]) -> dict:
    merged = mask_util.merge([_as_bytes(r) for r in rles], intersect=False)
    merged["counts"] = merged["counts"].decode("ascii")
    return merged
