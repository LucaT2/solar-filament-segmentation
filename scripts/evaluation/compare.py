"""Paired comparison of two experiments on the same out-of-fold images: ΔPQ with a bootstrap confidence interval.

Both experiments (probability maps or instances) are post-processed (threshold, size filter; no classifier) and scored with the exact PQ on the
images they have in common. Resampling *images* (with all their annotator-images) 2,000 times gives the uncertainty
of the difference. Because both are scored on the same images, image-to-image difficulty cancels out and the interval
is much tighter than comparing two separate scores.

--post own : each experiment uses its own postprocess.json (default)
--post a   : both use A's settings (removes any advantage from tuning B on the compared images)

Usage:
    python scripts/evaluation/compare.py --a unet_r34 --b unet_r34_c1024_skel                  # all common OOF images
    python scripts/evaluation/compare.py --a unet_r34 --b unet_r34_c1024_skel --folds 0 --post a
    python scripts/evaluation/compare.py --a unet_r34 --b outputs/kaggle_runs/<run>/unet_r34_c1024_skel   # paths work too
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from filament.data.annotations import annotation_table, load_coco
from filament.data.folds import load_folds
from filament.evaluation.scoring import GroundTruth, score_components
from filament.paths import REPO_ROOT, get_data_paths
from filament.postprocess.components import region_for
from filament.postprocess.pipeline import load_postprocess
from filament.postprocess.predictions import available_stems, load_components


def resolve(exp: str) -> Path:
    p = Path(exp)
    if p.is_dir():
        return p
    return REPO_ROOT / "outputs" / exp


def per_image_totals(exp_dir: Path, stems, gt: GroundTruth, cfg, image_dir) -> pd.DataFrame:
    """Per image stem: summed TP IoU, TP, FP, FN over its annotator-images (everything PQ needs)."""
    rows, disks = [], {}
    for stem in stems:
        comps = load_components(exp_dir / "oof", stem, cfg.threshold,
                                region_for(stem, image_dir, cfg.disk_scale, disks), cfg.link_px)
        scores = score_components(stem, comps, comps.keep(cfg.min_area, cfg.min_score), gt)
        rows.append({"stem": stem, "iou_sum": sum(float(s.tp_ious.sum()) for s in scores),
                     "tp": sum(len(s.tp_ious) for s in scores), "fp": sum(s.fp for s in scores),
                     "fn": sum(s.fn for s in scores)})
    return pd.DataFrame(rows).set_index("stem")


def pq(t: pd.DataFrame | np.ndarray, weights: np.ndarray | None = None):
    """PQ, SQ, RQ from per-image totals; ``weights`` (n_boot, n_images) gives bootstrap replicates."""
    arr = t[["iou_sum", "tp", "fp", "fn"]].to_numpy(float) if isinstance(t, pd.DataFrame) else t
    s = arr.sum(0) if weights is None else weights @ arr
    iou, tp, fp, fn = (s[..., i] for i in range(4))
    denom = tp + 0.5 * fp + 0.5 * fn
    return iou / denom, iou / np.maximum(tp, 1), tp / denom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="baseline experiment (name in outputs/ or a directory)")
    parser.add_argument("--b", required=True, help="candidate experiment")
    parser.add_argument("--folds", type=int, nargs="*", help="restrict to these folds (default: all common images)")
    parser.add_argument("--post", choices=["own", "a"], default="own")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = get_data_paths()
    dir_a, dir_b = resolve(args.a), resolve(args.b)
    folds = load_folds()
    if args.folds:
        folds = folds[folds.fold.isin(args.folds)]
    common = available_stems(dir_a / "oof") & available_stems(dir_b / "oof")
    stems = [s for s in folds.stem if s in common]
    if not stems:
        raise SystemExit("no common OOF images between the two experiments")
    cfg_a = load_postprocess(dir_a / "postprocess.json")
    cfg_b = cfg_a if args.post == "a" else load_postprocess(dir_b / "postprocess.json")
    print(f"A = {dir_a.name}: {cfg_a}\nB = {dir_b.name}: {cfg_b}")
    print(f"{len(stems)} common images (folds {sorted(folds[folds.stem.isin(stems)].fold.unique().tolist())})\n")

    gt = GroundTruth(annotation_table(load_coco(paths.annotations)), stems)
    ta = per_image_totals(dir_a, stems, gt, cfg_a, paths.train_images)
    tb = per_image_totals(dir_b, stems, gt, cfg_b, paths.train_images)

    rng = np.random.default_rng(args.seed)
    w = rng.multinomial(len(stems), np.full(len(stems), 1 / len(stems)), size=args.n_boot).astype(float)
    (pa, sa, ra), (pb, sb, rb) = pq(ta), pq(tb)
    delta = pq(tb, w)[0] - pq(ta, w)[0]
    lo, hi = np.percentile(delta, [2.5, 97.5])

    table = pd.DataFrame({"A": [pa, sa, ra, *ta.sum()[["tp", "fp", "fn"]]],
                          "B": [pb, sb, rb, *tb.sum()[["tp", "fp", "fn"]]]},
                         index=["PQ", "SQ", "RQ", "TP", "FP", "FN"])
    table["B - A"] = table.B - table.A
    print(table.round(4).to_string())
    print(f"\nΔPQ = {pb - pa:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   P(B better) = {(delta > 0).mean():.2f}")

    site = pd.Series([s[14] for s in stems], index=stems)
    rows = []
    for g, idx in site.groupby(site).groups.items():
        rows.append({"site": g, "n": len(idx), "PQ A": pq(ta.loc[idx])[0], "PQ B": pq(tb.loc[idx])[0]})
    by_site = pd.DataFrame(rows).assign(delta=lambda d: d["PQ B"] - d["PQ A"])
    print("\nby site:\n" + by_site.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
