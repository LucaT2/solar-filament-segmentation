"""Fuse a probability-map model and an instance model with one classifier; honest OOF evaluation of the variants.

Candidates: the map model's connected components (at its tuned threshold, area >= --map-min-area) and the instance
model's detections (at its tuned mask threshold, score >= --det-min-score). Variants, all scored the same way:

    <maps>  only      classifier on the map model's candidates (like fit_rescorer.py)
    <inst>  only      classifier on the instance model's detections
    <inst>  score     no classifier: the detection score itself (like tune_postprocess.py)
    fused             classifier on both pools, with agreement features

The classifier's P(match) is out-of-fold over images (GroupKFold). The cut-off on P(match) and the minimum area are
chosen by a nested split (on the other images) and applied to held-out images, so every PQ is honest. Paired bootstrap
intervals compare the variants on the same images.

Usage:
    python scripts/postprocess/fuse.py --maps ens_d8_r34_convnext_f0 \
        --instances outputs/kaggle_runs/solar-filament-v2_maskrcnn/maskrcnn_r50_v2_m56
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from filament.data.annotations import annotation_table, load_coco
from filament.data.folds import load_folds
from filament.data.images import disk_mask, find_disk, read_image
from filament.detection.instances import load_instances
from filament.evaluation.metric import score_images
from filament.evaluation.scoring import GroundTruth, score_components
from filament.paths import get_data_paths, resolve_exp_dir
from filament.postprocess.fusion import (candidate_table, candidates_from_instances, candidates_from_map,
                                         feature_matrix, select)
from filament.postprocess.maps import load_prob
from filament.postprocess.pipeline import load_postprocess
from filament.postprocess.predictions import available_stems
from filament.postprocess.rescoring import make_classifier

MIN_P = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
MIN_AREA = [0, 100, 200, 300, 400]


def oof_p(df: pd.DataFrame, sources: list[str], y: np.ndarray, n_splits: int = 5) -> np.ndarray:
    x = feature_matrix(df, sources)
    x = x[[c for c in x.columns if not c.startswith(("iou_", "conf_")) or c.split("_", 1)[1] in sources]]
    p = np.zeros(len(df))
    for tr, va in GroupKFold(n_splits).split(x, y, df.stem):
        p[va] = make_classifier().fit(x.iloc[tr], y[tr]).predict_proba(x.iloc[va])[:, 1]
    return p


def per_image_scores(stems, cands, df, p, gt):
    """ImageScores per (min_p, min_area) and image; selection runs once per image (lower P never displaces higher)."""
    out = {(t, a): {} for t in MIN_P for a in MIN_AREA}
    for s in stems:
        idx = np.flatnonzero(df.stem.to_numpy() == s)
        comps = select([cands[s][df.cand.iat[i]] for i in idx], p[idx], min(MIN_P))
        for t in MIN_P:
            for a in MIN_AREA:
                out[(t, a)][s] = score_components(s, comps, comps.keep(a, t), gt)
    return out


def nested(per_image, stems, n_splits=5):
    """Pick (min_p, min_area) on the other images, apply to the held-out ones; per-image totals of the result."""
    stems = np.asarray(stems)
    rows, picked = [], []
    for tr, te in GroupKFold(n_splits).split(stems, groups=stems):
        best = max(per_image, key=lambda k: score_images([x for s in stems[tr] for x in per_image[k][s]]).pq)
        picked.append(best)
        for s in stems[te]:
            sc = per_image[best][s]
            rows.append({"stem": s, "iou_sum": sum(float(x.tp_ious.sum()) for x in sc),
                         "tp": sum(len(x.tp_ious) for x in sc), "fp": sum(x.fp for x in sc),
                         "fn": sum(x.fn for x in sc)})
    return pd.DataFrame(rows).set_index("stem").loc[list(stems)], picked


def pq(t: pd.DataFrame | np.ndarray, w: np.ndarray | None = None):
    arr = t[["iou_sum", "tp", "fp", "fn"]].to_numpy(float) if isinstance(t, pd.DataFrame) else t
    s = arr.sum(0) if w is None else w @ arr
    denom = s[..., 1] + 0.5 * s[..., 2] + 0.5 * s[..., 3]
    return s[..., 0] / denom, s[..., 0] / np.maximum(s[..., 1], 1), s[..., 1] / denom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maps", required=True, help="experiment with OOF probability maps")
    parser.add_argument("--instances", required=True, help="experiment with OOF instances")
    parser.add_argument("--map-min-area", type=int, default=100)
    parser.add_argument("--det-min-score", type=float, default=0.05)
    parser.add_argument("--det-min-area", type=int, default=50)
    parser.add_argument("--folds", type=int, nargs="*", default=[0])
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    t0 = time.time()
    paths = get_data_paths()
    dm, di = resolve_exp_dir(args.maps), resolve_exp_dir(args.instances)
    cm, ci = load_postprocess(dm / "postprocess.json"), load_postprocess(di / "postprocess.json")
    folds = load_folds()
    common = available_stems(dm / "oof") & available_stems(di / "oof")
    stems = [s for s in folds.stem[folds.fold.isin(args.folds)] if s in common]
    print(f"{len(stems)} images | maps {dm.name} @ thr {cm.threshold} | instances {di.name} @ mask thr {ci.threshold}")

    gt = GroundTruth(annotation_table(load_coco(paths.annotations)), stems)
    cands, tables = {}, []
    sources = ["unet", "mrcnn"]
    for s in stems:
        img = read_image(paths.train_images / f"{s}.jpeg")
        disk = find_disk(img)
        region = disk_mask((2048, 2048), disk, 1.0)
        cands[s] = (candidates_from_map(load_prob(dm / "oof", s), cm.threshold, args.map_min_area, region, "unet")
                    + candidates_from_instances(load_instances(di / "oof", s), ci.threshold, args.det_min_score,
                                                args.det_min_area, region, "mrcnn"))
        gt_flat = [f for fil in gt.by_stem[s].values() for f in fil]
        tables.append(candidate_table(s, cands[s], img, disk, sources, gt_flat))
    df = pd.concat(tables, ignore_index=True)
    y = (df.best_gt_iou > 0.5).to_numpy(int)
    print(f"candidates: {df.source.value_counts().to_dict()}, {y.mean():.1%} match an annotator "
          f"({time.time() - t0:.0f}s)")

    variants = {}
    for name, srcs, use_clf in (("U-Net + classifier", ["unet"], True), ("Mask R-CNN + classifier", ["mrcnn"], True),
                                ("Mask R-CNN score only", ["mrcnn"], False), ("fused + classifier", sources, True)):
        sub = df[df.source.isin(srcs)].reset_index(drop=True)
        p = oof_p(sub, srcs, y[df.source.isin(srcs).to_numpy()]) if use_clf else sub.conf.to_numpy()
        totals, picked = nested(per_image_scores(stems, cands, sub, p, gt), stems)
        variants[name] = totals
        v_pq, v_sq, v_rq = pq(totals)
        print(f"{name:24s} nested PQ {v_pq:.4f}  SQ {v_sq:.4f}  RQ {v_rq:.4f}  FP {int(totals.fp.sum()):4d}  "
              f"FN {int(totals.fn.sum()):4d}  picked (min_p, min_area): {sorted(set(picked))}")

    rng = np.random.default_rng(0)
    w = rng.multinomial(len(stems), np.full(len(stems), 1 / len(stems)), size=args.n_boot).astype(float)
    print("\npaired differences (95% bootstrap CI over images):")
    base = "fused + classifier"
    for other in [k for k in variants if k != base]:
        d = pq(variants[base], w)[0] - pq(variants[other], w)[0]
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  fused - {other:24s}: {pq(variants[base])[0] - pq(variants[other])[0]:+.4f}  [{lo:+.4f}, {hi:+.4f}]"
              f"  P(fused better) {(d > 0).mean():.2f}")
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
