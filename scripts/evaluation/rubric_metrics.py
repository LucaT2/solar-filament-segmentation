"""The competition's quantitative criteria for final candidates, on all out-of-fold images: PQ (SQ, RQ), Dice and IoU
distributions, one-to-many / many-to-one relations, pixel Dice, and breakdowns by site, annotator count and filament
size. Optionally the human-vs-human reference (first annotator vs the others, multi-annotated images).

Each experiment is scored as its submission would be: its tuned post-processing (postprocess.json) and, if it has a
component classifier (rescorer.json), that classifier's cut-off applied to **out-of-fold** probabilities (classifiers
fitted on the other folds, as in fit_rescorer.py), so nothing is scored in-sample except the cut-off itself.

Usage:
    python scripts/evaluation/rubric_metrics.py --exp final_w40 final_w60 det5f_tta --human --out outputs/rubric
Outputs (--out): summary.csv (one row per experiment), by_group.csv, distributions.csv and <exp>.npz (per-pair IoU,
per-filament best IoU / area, degrees) for the report's figures.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from filament.data.annotations import annotation_table, annotator_image_table, load_coco
from filament.data.folds import load_folds
from filament.data.rle import rle_iou, rle_to_mask
from filament.evaluation.metric import ImageScore, match_degrees, overlap_scores, pq_by_group, score_images
from filament.evaluation.scoring import GroundTruth, score_components
from filament.paths import get_data_paths, resolve_exp_dir
from filament.postprocess.components import region_for
from filament.postprocess.pipeline import load_postprocess
from filament.postprocess.predictions import available_stems, load_components

SIZE_BINS = [0, 800, 3200, np.inf]  # small / medium / large filaments (px), as in the error analysis
SIZE_NAMES = ["small <800", "medium 800-3200", "large >3200"]


def _tables(args):
    from filament.evaluation.error_analysis import build_tables

    pred_dir, image_dir, ann, stems, cfg = args
    return build_tables(pred_dir, image_dir, ann, stems, cfg)[1]


def oof_kept(exp_dir: Path, ann, stems, cfg, workers: int) -> dict[str, set[int]] | None:
    """Per image, the component ids the classifier keeps at its cut-off, from out-of-fold P(match); None without one."""
    if not (exp_dir / "rescorer.json").exists():
        return None
    from filament.postprocess.rescoring import oof_probabilities

    min_p = json.loads((exp_dir / "rescorer.json").read_text())["min_p"]
    chunks = [stems[i::workers] for i in range(workers)]
    image_dir = get_data_paths().train_images
    with ProcessPoolExecutor(workers) as pool:
        comps = pd.concat(pool.map(_tables, [(exp_dir / "oof", image_dir, ann[ann.stem.isin(set(c))], c, cfg)
                                             for c in chunks]), ignore_index=True)
    fold_of = load_folds().set_index("stem").fold
    comps["p"] = oof_probabilities(comps, comps.matched_any.astype(int).to_numpy(), fold_of[comps.stem].to_numpy())
    keep = comps[comps.kept.astype(bool) & (comps.p >= min_p)]
    print(f"  classifier: {len(comps)} components, cut-off {min_p} keeps {len(keep)} (out-of-fold P)")
    return {s: set(g.comp.astype(int)) for s, g in keep.groupby("stem")}


def _score_image(args):
    stem, stem_gt, pred_dir, cfg, allowed = args
    paths = get_data_paths()
    comps = load_components(pred_dir, stem, cfg.threshold, region_for(stem, paths.train_images, cfg.disk_scale, {}),
                            cfg.link_px)
    keep = comps.keep(cfg.min_area, cfg.min_score)
    if allowed is not None:
        keep = np.array([i for i in keep if i in allowed], dtype=int)
    gt = GroundTruth.__new__(GroundTruth)
    gt.by_stem = {stem: stem_gt}
    scores = score_components(stem, comps, keep, gt)
    pred_px = np.flatnonzero(np.isin(comps.labels.ravel(), keep)) if len(keep) else np.zeros(0, np.int64)
    return scores, pixel_dice(stem_gt, pred_px)


def pixel_dice(stem_gt: dict[str, list[np.ndarray]], pred_px: np.ndarray) -> list[tuple[str, float]]:
    """Dice of the union masks (all predicted vs all of one annotator's filaments), per annotator-image."""
    out = []
    for aid, fils in stem_gt.items():
        g = np.unique(np.concatenate(fils)) if fils else np.zeros(0, np.int64)
        denom = len(g) + len(pred_px)
        out.append((aid, 2 * np.intersect1d(g, pred_px, assume_unique=True).size / denom if denom else 1.0))
    return out


def score_experiment(name: str, ann, gt: GroundTruth, workers: int):
    exp_dir = resolve_exp_dir(name)
    cfg = load_postprocess(exp_dir / "postprocess.json")
    stems = [s for s in gt.stems if s in available_stems(exp_dir / "oof")]
    print(f"{name}: {cfg}; {len(stems)} images")
    allowed = oof_kept(exp_dir, ann, stems, cfg, workers)
    jobs = [(s, gt.by_stem[s], exp_dir / "oof", cfg, None if allowed is None else allowed.get(s, set()))
            for s in stems]
    with ProcessPoolExecutor(workers) as pool:
        res = list(pool.map(_score_image, jobs, chunksize=8))
    return [x for r in res for x in r[0]], dict(x for r in res for x in r[1])


def score_human(ann, ai, stems):
    """First annotator of each multi-annotated image vs the other annotators (the metric's reference point)."""
    multi = ai[(ai.n_annotators > 1) & ai.stem.isin(stems)]
    first = set(multi.drop_duplicates("stem").annotator_image)
    images, dice = [], {}
    for _, g in ann[ann.stem.isin(set(multi.stem))].groupby("stem"):
        pred = g[g.annotator_image.isin(first)].rle.tolist()
        pred_px = np.flatnonzero(np.any([rle_to_mask(r) for r in pred], axis=0)) if pred else np.zeros(0, np.int64)
        for aid, o in g[~g.annotator_image.isin(first)].groupby("annotator_image"):
            images.append(ImageScore(aid, len(o), len(pred), rle_iou(o.rle.tolist(), pred)))
            dice.update(pixel_dice({aid: [np.flatnonzero(rle_to_mask(r)) for r in o.rle]}, pred_px))
    return images, dice


def describe(name: str, images: list[ImageScore], dice: dict[str, float], areas: dict[str, list[int]], n_ann,
             out: Path):
    r = score_images(images)
    iou, pair_dice = overlap_scores(images)
    tp_iou = np.concatenate([s.tp_ious for s in images]) if images else np.zeros(0)
    gt_deg, pred_deg = match_degrees(images)
    best = np.concatenate([s.iou.max(axis=1) if s.n_pred else np.zeros(s.n_gt) for s in images])
    area = np.concatenate([areas[s.key] for s in images])
    pdice = np.array([dice[s.key] for s in images])
    row = {
        "experiment": name, "annotator_images": len(images), "pq": r.pq, "sq": r.sq, "rq": r.rq,
        "tp": r.tp, "fp": r.fp, "fn": r.fn,
        "precision": r.tp / max(r.tp + r.fp, 1), "recall": r.tp / max(r.tp + r.fn, 1),
        "matched_mean_iou": float(tp_iou.mean()) if r.tp else 0.0,
        "matched_mean_dice": float((2 * tp_iou / (1 + tp_iou)).mean()) if r.tp else 0.0,
        "overlap_pairs": len(iou), "overlap_mean_iou": float(iou.mean()) if len(iou) else 0.0,
        "overlap_mean_dice": float(pair_dice.mean()) if len(iou) else 0.0,
        "overlap_median_dice": float(np.median(pair_dice)) if len(iou) else 0.0,
        "pixel_dice_mean": float(pdice.mean()), "pixel_dice_median": float(np.median(pdice)),
        "gt_missed": float((gt_deg == 0).mean()), "gt_1to1": float((gt_deg == 1).mean()),
        "gt_split_1ton": float((gt_deg > 1).mean()), "pred_unmatched": float((pred_deg == 0).mean()),
        "pred_1to1": float((pred_deg == 1).mean()), "pred_merge_nto1": float((pred_deg > 1).mean()),
    }
    groups = []
    for label, fn in [("site", lambda k: k.split("-", 1)[1][14]), ("annotators", lambda k: n_ann[k])]:
        groups.append(pq_by_group(r, fn).assign(experiment=name, by=label))
    size = pd.cut(area, SIZE_BINS, labels=SIZE_NAMES, right=False)
    found = pd.DataFrame({"size": size, "found": best > 0.5, "best_iou": best}).groupby("size", observed=True).agg(
        n=("found", "size"), found=("found", "mean"), mean_best_iou=("best_iou", "mean")).reset_index()
    groups.append(found.rename(columns={"size": "group"}).assign(experiment=name, by="gt size"))
    dist = []
    for what, values in [("overlap iou", iou), ("overlap dice", pair_dice), ("matched iou", tp_iou),
                         ("pixel dice", pdice)]:
        hist, _ = np.histogram(values, bins=np.linspace(0, 1, 11))
        dist.append({"experiment": name, "what": what, "n": len(values),
                     **{f"q{int(q * 100)}": float(np.quantile(values, q)) if len(values) else np.nan
                        for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
                     **{f"bin{i / 10:.1f}": int(h) for i, h in enumerate(hist)}})
    for what, deg in [("gt degree", gt_deg), ("pred degree", pred_deg)]:
        counts = np.bincount(np.minimum(deg, 4), minlength=5)
        dist.append({"experiment": name, "what": what, "n": len(deg),
                     **{f"deg{k}" if k < 4 else "deg4+": int(c) for k, c in enumerate(counts)}})
    np.savez_compressed(out / f"{name}.npz", overlap_iou=iou, matched_iou=tp_iou, gt_best_iou=best, gt_area=area,
                        gt_degree=gt_deg, pred_degree=pred_deg, pixel_dice=pdice)
    return row, pd.concat(groups, ignore_index=True), pd.DataFrame(dist)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", nargs="+", default=[])
    parser.add_argument("--human", action="store_true", help="add the human-vs-human reference")
    parser.add_argument("--out", default="outputs/rubric")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    coco = load_coco(get_data_paths().annotations)
    ann, ai = annotation_table(coco), annotator_image_table(coco)
    stems = load_folds().stem.tolist()
    gt = GroundTruth(ann, stems)
    areas = {aid: [len(i) for i in fils] for s in gt.by_stem.values() for aid, fils in s.items()}
    n_ann = ai.set_index("annotator_image").n_annotators.to_dict()

    rows, groups, dists = [], [], []
    runs = [(e, lambda e=e: score_experiment(e, ann, gt, args.workers)) for e in args.exp]
    if args.human:
        runs.append(("human", lambda: score_human(ann, ai, stems)))
    for name, run in runs:
        images, dice = run()
        row, g, d = describe(name, images, dice, areas, n_ann, out)
        print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()),
              flush=True)
        rows.append(row)
        groups.append(g)
        dists.append(d)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    pd.concat(groups, ignore_index=True).to_csv(out / "by_group.csv", index=False)
    pd.concat(dists, ignore_index=True).to_csv(out / "distributions.csv", index=False)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
