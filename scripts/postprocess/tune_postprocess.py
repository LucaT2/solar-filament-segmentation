"""Grid-search post-processing (threshold, min component area, min mean probability) on out-of-fold maps.

Reads outputs/<exp>/oof/<stem>.png, scores every combination with the exact PQ, and writes the best setting to
outputs/<exp>/postprocess.json (plus the full grid to grid.csv).

Usage:
    python scripts/postprocess/tune_postprocess.py --exp classical
    python scripts/postprocess/tune_postprocess.py --exp unet_r34 --thresholds 0.3 0.4 0.5 0.6 --min-areas 0 50 100 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from filament.data.annotations import annotation_table, load_coco
from filament.data.folds import load_folds
from filament.evaluation.scoring import GroundTruth, grid_search
from filament.paths import get_data_paths, get_output_dir, resolve_exp_dir
from filament.postprocess.components import PostConfig
from filament.postprocess.pipeline import save_postprocess
from filament.tracking import start_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(np.round(np.arange(0.2, 0.81, 0.05), 2)))
    parser.add_argument("--min-areas", type=int, nargs="+", default=[0, 25, 50, 100, 150, 200, 300])
    parser.add_argument("--min-scores", type=float, nargs="+", default=[0.0])
    parser.add_argument("--disk-scale", type=float, default=1.0)
    parser.add_argument("--link-px", type=int, nargs="+", default=[0], help="link fragments closer than ~2*k px")
    parser.add_argument("--fold", type=int, nargs="*", help="restrict to these folds (default: all images)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-save", action="store_true", help="analysis only: don't overwrite postprocess.json/grid.csv")
    parser.add_argument("--wandb", default="auto", choices=["auto", "on", "offline", "off"])
    args = parser.parse_args()

    paths = get_data_paths()
    exp_dir = resolve_exp_dir(args.exp)  # name or folder path
    folds = load_folds()
    if args.fold:
        folds = folds[folds.fold.isin(args.fold)]
    available = {p.stem for p in (exp_dir / "oof").glob("*.png")}
    stems = [s for s in folds.stem if s in available]
    if not stems:
        raise SystemExit(f"no OOF maps found in {exp_dir / 'oof'}")
    if len(stems) < len(folds):
        print(f"note: OOF maps found for {len(stems)} of {len(folds)} images; scoring those only")

    t0 = time.time()
    gt = GroundTruth(annotation_table(load_coco(paths.annotations)), stems)
    grid = grid_search(exp_dir / "oof", gt, paths.train_images, args.thresholds, args.min_areas, args.min_scores,
                       args.disk_scale, args.workers, args.link_px)
    best = grid.iloc[0]
    cfg = PostConfig(float(best.threshold), int(best.min_area), float(best.min_score), args.disk_scale,
                     int(best.link_px))
    if not args.no_save:
        grid.to_csv(exp_dir / "grid.csv", index=False)
        save_postprocess(exp_dir / "postprocess.json", cfg, {"oof_pq": float(best.pq), "n_images": len(stems)})

    print(grid.head(10).round(4).to_string(index=False))
    print("\nPQ by threshold (best min_area / min_score for each):")
    print(grid.groupby("threshold").pq.max().round(4).to_string())
    print(f"\nbest: {cfg}  OOF PQ = {best.pq:.4f}  ({time.time() - t0:.0f}s)")

    folds_used = sorted(folds[folds.stem.isin(stems)].fold.unique().tolist())
    tracker = start_run(args.wandb, name=f"{args.exp}-tune-f{''.join(map(str, folds_used))}", group=args.exp,
                        job_type="tune", config={"exp": args.exp, "folds": folds_used, "n_images": len(stems),
                                                 "thresholds": args.thresholds, "min_areas": args.min_areas,
                                                 "min_scores": args.min_scores, "disk_scale": args.disk_scale,
                                                 "link_px": args.link_px})
    tracker.summary({"oof_pq": float(best.pq), "oof_sq": float(best.sq), "oof_rq": float(best.rq),
                     **{f"best/{k}": v for k, v in cfg.to_dict().items()}})
    tracker.table("grid", grid)
    tracker.finish()


if __name__ == "__main__":
    main()
