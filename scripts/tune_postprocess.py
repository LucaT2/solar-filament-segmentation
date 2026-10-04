"""Grid-search post-processing (threshold, min component area, min mean probability) on out-of-fold maps.

Reads outputs/<exp>/oof/<stem>.png, scores every combination with the exact PQ, and writes the best setting to
outputs/<exp>/postprocess.json (plus the full grid to grid.csv).

Usage:
    python scripts/tune_postprocess.py --exp classical
    python scripts/tune_postprocess.py --exp unet_r34 --thresholds 0.3 0.4 0.5 0.6 --min-areas 0 50 100 200
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from filament.data import annotation_table, load_coco
from filament.evaluation import GroundTruth, grid_search
from filament.folds import load_folds
from filament.inference import save_postprocess
from filament.paths import get_data_paths, get_output_dir
from filament.postprocess import PostConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(np.round(np.arange(0.2, 0.81, 0.05), 2)))
    parser.add_argument("--min-areas", type=int, nargs="+", default=[0, 25, 50, 100, 150, 200, 300])
    parser.add_argument("--min-scores", type=float, nargs="+", default=[0.0])
    parser.add_argument("--disk-scale", type=float, default=1.0)
    parser.add_argument("--fold", type=int, nargs="*", help="restrict to these folds (default: all images)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    paths = get_data_paths()
    exp_dir = get_output_dir(args.exp)
    folds = load_folds()
    if args.fold:
        folds = folds[folds.fold.isin(args.fold)]
    available = {p.stem for p in (exp_dir / "oof").glob("*.png")}
    stems = [s for s in folds.stem if s in available]
    if len(stems) < len(folds):
        print(f"note: OOF maps found for {len(stems)} of {len(folds)} images; scoring those only")

    t0 = time.time()
    gt = GroundTruth(annotation_table(load_coco(paths.annotations)), stems)
    grid = grid_search(exp_dir / "oof", gt, paths.train_images, args.thresholds, args.min_areas, args.min_scores,
                       args.disk_scale, args.workers)
    grid.to_csv(exp_dir / "grid.csv", index=False)
    best = grid.iloc[0]
    cfg = PostConfig(float(best.threshold), int(best.min_area), float(best.min_score), args.disk_scale)
    save_postprocess(exp_dir / "postprocess.json", cfg, {"oof_pq": float(best.pq), "n_images": len(stems)})

    print(grid.head(10).round(4).to_string(index=False))
    print("\nPQ by threshold (best min_area / min_score for each):")
    print(grid.groupby("threshold").pq.max().round(4).to_string())
    print(f"\nbest: {cfg}  OOF PQ = {best.pq:.4f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
