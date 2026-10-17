"""Download unlabelled GONG H-alpha frames from new days (filament.data.unlabelled), for pseudo-labelling.

Days at least --margin-days away from every training and test image, spread evenly over 2011-2022; per day one
frame for each site (its middle frame of that day). Resumable (cached listings, existing files skipped). For every
frame a small quality record: disk found, on-disk mean/std, sharpness (Laplacian variance).

Writes <out>/frames/<stem>.jpg and <out>/manifest.csv (stem, site, day, disk_r, mean, std, sharpness, ok).

Usage:
    python scripts/data/download_unlabelled.py --days 600                     # ~3,000 frames into outputs/unlabelled
    python scripts/data/download_unlabelled.py --days 5 --out /tmp/test      # quick test
"""

from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from filament.data.folds import load_folds
from filament.data.images import disk_mask, find_disk, read_image
from filament.data.neighbours import download_frame, stem_time
from filament.data.unlabelled import candidate_days, day_listing, middle_frames, sample_days
from filament.paths import get_data_paths, get_output_dir


def quality(path: Path) -> dict:
    try:
        img = read_image(path)
        disk = find_disk(img)
        on = disk_mask(img.shape, disk, 0.9)
        vals = img[on].astype(np.float32)
        lap = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F)[on]
        return {"disk_r": disk.radius, "mean": float(vals.mean()), "std": float(vals.std()),
                "sharpness": float(lap.var())}
    except Exception:
        return {"disk_r": np.nan, "mean": np.nan, "std": np.nan, "sharpness": np.nan}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=600, help="number of days to sample")
    parser.add_argument("--margin-days", type=int, default=2, help="distance from every training/test image")
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between downloads")
    parser.add_argument("--out", help="output folder (default outputs/unlabelled)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out) if args.out else get_output_dir("unlabelled")
    paths = get_data_paths()
    avoid = [stem_time(s) for s in load_folds().stem] + [stem_time(p.stem) for p in paths.test_images.glob("*.jpeg")]
    days = candidate_days(date.fromisoformat(args.start), date.fromisoformat(args.end), avoid, args.margin_days)
    chosen = sample_days(days, args.days, args.seed)
    print(f"{len(days)} candidate days (>= {args.margin_days} days from {len(avoid)} training/test images); "
          f"sampling {len(chosen)}", flush=True)

    t0, stems = time.time(), []
    for d in tqdm(chosen, desc="list"):
        try:
            stems += middle_frames(day_listing(d, out / "listings"))
        except Exception as exc:  # no observations that day, or archive hiccup
            print(f"  {d}: {exc}")
    print(f"{len(stems)} frames selected ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for s in tqdm(stems, desc="download"):
        try:
            path = download_frame(s, out / "frames", args.pause)
        except Exception as exc:
            print(f"  {s}: {exc}")
            continue
        rows.append({"stem": s, "site": s[14], "day": s[:8], **quality(path)})
    table = pd.DataFrame(rows)
    # plausible full-disk frames: disk found at the usual size, not too dark/bright, some contrast
    table["ok"] = (table.disk_r.between(850, 1050) & table["mean"].between(60, 200) & (table["std"] > 5)).astype(bool)
    table.to_csv(out / "manifest.csv", index=False)
    print(f"{len(table)} frames downloaded, {int(table.ok.sum())} pass the quality check; per site: "
          f"{table[table.ok].site.value_counts().to_dict()} -> {out}")


if __name__ == "__main__":
    main()
