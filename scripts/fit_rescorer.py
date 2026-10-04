"""Fit the component rescoring classifier on out-of-fold components and choose its cut-off on PQ.

1. Every thresholded component of every OOF image gets features (filament.rescoring) and a label: does it match a
   filament of at least one annotator (IoU > 0.5)?
2. Out-of-fold P(match) for each component (grouped by fold, or by image when only one fold exists) -> the cut-off
   ``min_p`` that maximizes OOF PQ.
3. A final classifier on all components + that cut-off -> outputs/<exp>/rescorer.pkl, applied automatically by
   scripts/make_submission.py.

Usage:
    python scripts/fit_rescorer.py --exp unet_r34
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from filament.data import annotation_table, load_coco
from filament.error_analysis import build_tables, pq_variant
from filament.evaluation import GroundTruth
from filament.folds import load_folds
from filament.inference import load_postprocess
from filament.paths import get_data_paths, get_output_dir
from filament.rescoring import Rescorer, feature_matrix, make_classifier, oof_probabilities
from filament.tracking import start_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--min-p", type=float, nargs="+", default=[0, .1, .15, .2, .25, .3, .35, .4, .5])
    parser.add_argument("--wandb", default="auto", choices=["auto", "on", "offline", "off"])
    args = parser.parse_args()

    paths = get_data_paths()
    exp_dir = get_output_dir(args.exp)
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
    rows = []
    for t in args.min_p:
        r = pq_variant(exp_dir / "oof", paths.train_images, gt, cfg,
                       keep_fn=lambda s, ids, t=t: [pmap[(s, int(i))] >= t for i in ids])
        rows.append({"min_p": t, "pq": r.pq, "sq": r.sq, "rq": r.rq, "tp": r.tp, "fp": r.fp, "fn": r.fn})
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    best = table.loc[table.pq.idxmax()]

    model = make_classifier().fit(feature_matrix(comps), y)
    Rescorer(model, float(best.min_p)).save(exp_dir / "rescorer.pkl")
    summary = {"min_p": float(best.min_p), "oof_pq": float(best.pq), "oof_pq_without": float(table.pq.iloc[0]),
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
