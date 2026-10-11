"""Component rescoring: a small classifier that decides which predicted components to keep.

Stage 4 showed that per-component features separate real filaments from false blobs much better than size alone
(fold 0: PQ 0.391 -> 0.419). The same feature code is used for the OOF analysis, for fitting, and for test
prediction, so the classifier always sees identically computed inputs.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from filament.data.images import Disk, disk_mask, find_disk
from filament.postprocess.components import Components, PostConfig, find_components

FEATURES = ["log_area", "mean_prob", "contrast", "elongation", "length", "r_norm", "site_code"]
SITES = "BCLMTU"


def _shape(labels: np.ndarray, n: int) -> pd.DataFrame:
    from scipy.ndimage import find_objects

    rows = []
    for i, sl in enumerate(find_objects(labels, max_label=n), start=1):
        crop = (labels[sl] == i).astype(np.uint8)
        ys, xs = np.nonzero(crop)
        pts = cv2.findNonZero(crop)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts) if len(xs) > 2 else ((0, 0), (1, 1), 0)
        rows.append({"cx": sl[1].start + xs.mean(), "cy": sl[0].start + ys.mean(), "length": max(rw, rh),
                     "elongation": max(rw, rh) / max(min(rw, rh), 1.0),
                     "y0": sl[0].start, "y1": sl[0].stop, "x0": sl[1].start, "x1": sl[1].stop})
    return pd.DataFrame(rows, index=np.arange(1, n + 1))


def _contrast(img: np.ndarray, labels: np.ndarray, shape: pd.DataFrame, ring: int = 15) -> np.ndarray:
    """(median of a ring around the component - median inside) / median of the ring."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    out = np.full(len(shape), np.nan)
    h, w = labels.shape
    boxes = shape[["y0", "y1", "x0", "x1"]].to_numpy(dtype=int)
    for k, (i, (by0, by1, bx0, bx1)) in enumerate(zip(shape.index, boxes)):
        y0, y1, x0, x1 = max(by0 - ring, 0), min(by1 + ring, h), max(bx0 - ring, 0), min(bx1 + ring, w)
        lab = labels[y0:y1, x0:x1]
        m = lab == i
        around = cv2.dilate(m.astype(np.uint8), kernel).astype(bool) & (lab == 0)
        if around.any():
            crop = img[y0:y1, x0:x1]
            bg = float(np.median(crop[around]))
            out[k] = (bg - float(np.median(crop[m]))) / max(bg, 1.0)
    return out


def component_table(stem: str, prob: np.ndarray, img: np.ndarray, cfg: PostConfig,
                    disk: Disk | None = None) -> tuple[Components, pd.DataFrame]:
    """All thresholded components of one image with their features and whether the size/score filter keeps them."""
    disk = disk or find_disk(img)
    region = disk_mask(prob.shape, disk, cfg.disk_scale) if cfg.disk_scale > 0 else None
    comps = find_components(prob, cfg.threshold, region, cfg.link_px)
    n = comps.n
    if n == 0:
        return comps, pd.DataFrame(columns=["stem", "comp", "area", "mean_prob", "kept", *FEATURES])
    kept = set(comps.keep(cfg.min_area, cfg.min_score).tolist())
    shape = _shape(comps.labels, n)
    df = pd.DataFrame({
        "stem": stem,
        "comp": np.arange(1, n + 1),
        "area": comps.area[1:].astype(int),
        "mean_prob": comps.mean_prob[1:],
        "kept": [i in kept for i in range(1, n + 1)],
        "contrast": _contrast(img, comps.labels, shape),
        "length": shape.length.to_numpy(),
        "elongation": shape.elongation.to_numpy(),
        "r_norm": np.hypot(shape.cx - disk.cx, shape.cy - disk.cy).to_numpy() / disk.radius,
        "site": stem[14],
        **{k: shape[k].to_numpy() for k in ("y0", "y1", "x0", "x1")},
    })
    return comps, df


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["log_area"] = np.log10(df.area.astype(float))
    x["site_code"] = df.site.map({s: i for i, s in enumerate(SITES)}).astype(float)
    for col in ("mean_prob", "contrast", "elongation", "length", "r_norm"):
        x[col] = df[col].astype(float)
    return x[FEATURES]


def make_classifier(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=seed)


def oof_probabilities(df: pd.DataFrame, labels: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """P(match) for every component from classifiers that never saw that component's image group."""
    from sklearn.model_selection import GroupKFold

    x, p = feature_matrix(df), np.zeros(len(df))
    for tr, va in GroupKFold(min(n_splits, len(np.unique(groups)))).split(x, labels, groups):
        p[va] = make_classifier().fit(x.iloc[tr], labels[tr]).predict_proba(x.iloc[va])[:, 1]
    return p


class Rescorer:
    """Fitted classifier + the cut-off on P(match). Components need to pass the size filter *and* the cut-off."""

    def __init__(self, model, min_p: float):
        self.model, self.min_p = model, min_p

    def probabilities(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(feature_matrix(df))[:, 1] if len(df) else np.zeros(0)

    def keep_ids(self, df: pd.DataFrame) -> np.ndarray:
        if not len(df):
            return np.zeros(0, dtype=int)
        ok = df.kept.to_numpy(bool) & (self.probabilities(df) >= self.min_p)
        return df.comp.to_numpy()[ok]

    def save(self, path: Path) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "Rescorer":
        with open(path, "rb") as f:
            return _Unpickler(f).load()


class _Unpickler(pickle.Unpickler):
    """Reads classifiers saved before the 2026-10-11 reorganisation, when this module was ``filament.rescoring``."""

    def find_class(self, module, name):
        return super().find_class(__name__ if module == "filament.rescoring" else module, name)
