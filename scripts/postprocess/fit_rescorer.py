"""Fit the component rescoring classifier on out-of-fold components and choose its cut-off on PQ.

1. Every thresholded component of every OOF image gets features (filament.postprocess.rescoring) and a label: does it match a
   filament of at least one annotator (IoU > 0.5)?
2. Out-of-fold P(match) for each component (grouped by fold, or by image when only one fold exists) -> the cut-off
   ``min_p`` that maximizes OOF PQ. Choosing the cut-off on the same data biases that PQ upwards, so a nested
   estimate (cut-off chosen on the other groups, applied to the held-out group) is reported as ``oof_pq_nested``.
3. A final classifier on all components + that cut-off -> outputs/<exp>/rescorer.pkl, applied automatically by
   scripts/postprocess/make_submission.py.

Usage:
    python scripts/postprocess/fit_rescorer.py --exp unet_r34
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from filament.data.annotations import annotation_table, load_coco
from filament.data.folds import load_folds
from filament.evaluation.error_analysis import build_tables
from filament.evaluation.metric import score_images
from filament.evaluation.scoring import GroundTruth, score_components
from filament.paths import get_data_paths, resolve_exp_dir
from filament.postprocess.components import find_components, region_for
from filament.postprocess.maps import load_prob
from filament.postprocess.pipeline import load_postprocess
from filament.postprocess.rescoring import Rescorer, feature_matrix, make_classifier, oof_probabilities
from filament.tracking import start_run


def per_image_scores(prob_dir, image_dir, gt, cfg, pmap, cutoffs) -> dict:
    """ImageScores for every cut-off and image (components computed once per image)."""
    out = {t: {} for t in cutoffs}
    disks: dict = {}
    for stem in gt.stems:
        comps = find_components(load_prob(prob_dir, stem), cfg.threshold,
                                region_for(stem, image_dir, cfg.disk_scale, disks), cfg.link_px)
        keep = comps.keep(cfg.min_area, cfg.min_score)
        for t in cutoffs:
            k = keep[np.array([pmap[(stem, int(i))] >= t for i in keep], dtype=bool)] if len(keep) else keep
            out[t][stem] = score_components(stem, comps, k, gt)
    return out


def nested_estimate(per_image, stems, groups_by_stem, cutoffs, n_splits=5):
    """Choose the cut-off on the other groups, apply it to the held-out group: PQ without selection bias."""
    from sklearn.model_selection import GroupKFold

    stems = np.asarray(stems)
    groups = np.array([groups_by_stem[s] for s in stems])
    held, picked = [], []
    for tr, te in GroupKFold(min(n_splits, len(np.unique(groups)))).split(stems, groups=groups):
        best = max(cutoffs, key=lambda t: score_images([x for s in stems[tr] for x in per_image[t][s]]).pq)
        picked.append(best)
        held += [x for s in stems[te] for x in per_image[best][s]]
    return score_images(held).pq, picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--min-p", type=float, nargs="+", default=[0, .1, .15, .2, .25, .3, .35, .4, .5])
    parser.add_argument("--wandb", default="auto", choices=["auto", "on", "offline", "off"])
    parser.add_argument("--no-save", action="store_true", help="report only; keep the existing rescorer.pkl")
    args = parser.parse_args()

    paths = get_data_paths()
    exp_dir = resolve_exp_dir(args.exp)  # name or folder path
    cfg = load_postprocess(exp_dir / "postprocess.json")
    folds = load_folds()
    available = {p.stem for p in (exp_dir / "oof").glob("*.png")}
    stems = [s for s in folds.stem if s in available]
    fold_of = folds.set_index("stem").fold
    n_folds = fold_of[stems].nunique()
    print(f"{cfg}\n{len(stems)} OOF images from {n_folds} fold(s)")

    ann = annotation_table(load_coco(paths.annotations))
    _, comps = build_tables(exp_dir / "oof", paths.train_images, ann, stems, cfg)
    y = comps.matched_any.astype(int).to_numpy()
    groups = (fold_of[comps.stem].to_numpy() if n_folds >= 3 else comps.stem.to_numpy())
    comps["p"] = oof_probabilities(comps, y, groups)
    print(f"{len(comps)} components, {y.mean():.1%} match an annotator")

    gt = GroundTruth(ann, stems)
    pmap = {(s, c): v for s, c, v in zip(comps.stem, comps.comp, comps.p)}
    per_image = per_image_scores(exp_dir / "oof", paths.train_images, gt, cfg, pmap, args.min_p)
    rows = []
    for t in args.min_p:
        r = score_images([x for s in stems for x in per_image[t][s]])
        rows.append({"min_p": t, "pq": r.pq, "sq": r.sq, "rq": r.rq, "tp": r.tp, "fp": r.fp, "fn": r.fn})
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    best = table.loc[table.pq.idxmax()]
    nested_pq, picked = nested_estimate(per_image, stems, groups_by_stem={s: fold_of[s] if n_folds >= 3 else s
                                                                         for s in stems}, cutoffs=args.min_p)
    print(f"honest (nested) OOF PQ with rescoring: {nested_pq:.4f}  (cut-offs picked per split: {picked})")

    if args.no_save:
        return
    model = make_classifier().fit(feature_matrix(comps), y)
    Rescorer(model, float(best.min_p)).save(exp_dir / "rescorer.pkl")
    summary = {"min_p": float(best.min_p), "oof_pq": float(best.pq), "oof_pq_nested": float(nested_pq),
               "oof_pq_without": float(table.pq.iloc[0]),
               "n_images": len(stems), "n_folds": int(n_folds), "n_components": len(comps)}
    (exp_dir / "rescorer.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {exp_dir / 'rescorer.pkl'}: {summary}")

    tracker = start_run(args.wandb, name=f"{args.exp}-rescorer", group=args.exp, job_type="rescore",
                        config={"postprocess": cfg.to_dict(), "min_p_grid": args.min_p, "n_folds": int(n_folds)})
    tracker.summary(summary)
    tracker.table("min_p_grid", table)
    tracker.finish()


if __name__ == "__main__":
    main()
