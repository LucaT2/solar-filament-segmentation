"""Select and download neighbouring GONG frames of the training images (filament.data.neighbours).

1. For each training image, list its day in the GONG archive and pick the same-site frames nearest to the wanted
   offsets (default -5, -2, +2, +5 minutes). Frames within 30 min of any test image are excluded.
2. Download them (resumable: existing files are skipped), pausing between requests.

Writes outputs/neighbours/manifest.csv (train_stem, stem, offset_min) and outputs/neighbours/raw/<stem>.jpg.

Usage:
    python scripts/data/download_neighbours.py --limit 5            # test on 5 training images
    python scripts/data/download_neighbours.py                      # all training images
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
from tqdm import tqdm

from filament.data.folds import load_folds
from filament.data.neighbours import day_frames, download_frame, pick_neighbours, stem_time
from filament.paths import get_data_paths, get_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offsets", type=float, nargs="+", default=[-5, -2, 2, 5], help="minutes")
    parser.add_argument("--exclude-window", type=float, default=30.0, help="minutes around every test image")
    parser.add_argument("--limit", type=int, help="only the first N training images (testing)")
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between downloads")
    parser.add_argument("--select-only", action="store_true", help="write the manifest, download nothing")
    args = parser.parse_args()

    out = get_output_dir("neighbours")
    stems = load_folds().stem.tolist()[: args.limit]
    test_times = [stem_time(p.stem) for p in get_data_paths().test_images.glob("*.jpeg")]

    t0, rows = time.time(), []
    for s in tqdm(stems, desc="select"):
        try:
            frames = day_frames(s[:8], out / "listings")
        except Exception as exc:  # a missing day listing only costs this image's neighbours
            print(f"  {s}: no listing ({exc})")
            continue
        rows += [n.__dict__ for n in pick_neighbours(s, frames, args.offsets, exclude_times=test_times,
                                                     exclude_window=args.exclude_window)]
    manifest = pd.DataFrame(rows, columns=["train_stem", "stem", "offset_min"])
    manifest.to_csv(out / "manifest.csv", index=False)
    per_image = manifest.groupby("train_stem").size().reindex(stems, fill_value=0)
    print(f"{len(manifest)} neighbours for {len(stems)} training images ({time.time() - t0:.0f}s); per image: "
          f"{per_image.value_counts().sort_index().to_dict()}")
    if args.select_only:
        return

    t0, size = time.time(), 0
    failed = []
    for stem in tqdm(manifest.stem, desc="download"):
        try:
            size += download_frame(stem, out / "raw", args.pause).stat().st_size
        except Exception as exc:
            failed.append(stem)
            print(f"  {stem}: {exc}")
    dt = time.time() - t0
    print(f"downloaded/kept {len(manifest) - len(failed)} frames, {size / 2**20:.0f} MB in {dt:.0f}s "
          f"({len(manifest) / max(dt, 1):.2f} frames/s); failed: {len(failed)}")


if __name__ == "__main__":
    main()
