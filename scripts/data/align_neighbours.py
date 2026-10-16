"""Align downloaded neighbour frames onto their training images and check their quality.

For each frame in outputs/neighbours/manifest.csv whose download exists (outputs/neighbours/raw/<stem>.jpg):
* detect its solar disk, scale and shift it onto the training image's disk, copy the training image's off-disk area
* score it against the training image on the disk: correlation, brightness ratio, sharpness ratio, disk geometry
* keep it if it looks like the same scene (thresholds below), save it as outputs/neighbours/aligned/<stem>.jpeg

Writes outputs/neighbours/aligned/manifest.csv (one row per frame, with the scores and ``keep``) and a contact sheet
(outputs/neighbours/contact_sheet.png): crops of training images next to their kept neighbours.

Usage:
    python scripts/data/align_neighbours.py
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from filament.data.annotations import annotation_table, load_coco
from filament.data.images import find_disk, read_image
from filament.data.neighbours import align_to, frame_quality
from filament.paths import get_data_paths, get_output_dir

OUT = get_output_dir("neighbours")


def process(args) -> list[dict]:
    train_stem, rows, min_corr, max_shift, max_dr = args
    train = read_image(get_data_paths().train_images / f"{train_stem}.jpeg")
    tdisk = find_disk(train)
    out = []
    for stem, offset in rows:
        raw = OUT / "raw" / f"{stem}.jpg"
        rec = {"train_stem": train_stem, "stem": stem, "offset_min": offset, "keep": False}
        if not raw.exists():
            out.append({**rec, "reason": "not downloaded"})
            continue
        frame = read_image(raw)
        fdisk = find_disk(frame)
        shift, dr = float(np.hypot(fdisk.cx - tdisk.cx, fdisk.cy - tdisk.cy)), abs(fdisk.radius - tdisk.radius)
        aligned = align_to(frame, train, fdisk, tdisk)
        q = frame_quality(aligned, train, tdisk)
        reason = ("disk moved" if shift > max_shift else "disk size" if dr > max_dr
                  else "low correlation" if q["corr"] < min_corr
                  else "brightness" if not 0.7 <= q["mean_ratio"] <= 1.4 else "")
        if not reason:
            cv2.imwrite(str(OUT / "aligned" / f"{stem}.jpeg"), aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])
        out.append({**rec, **q, "shift_px": shift, "dr_px": dr, "keep": not reason, "reason": reason})
    return out


def contact_sheet(table: pd.DataFrame, n_images: int = 6, crop: int = 300) -> None:
    """Crops centred on each training image's largest filament, next to the same crop of its kept neighbours."""
    kept = table[table.keep]
    stems = kept.train_stem.drop_duplicates().head(n_images).tolist()
    cols = 1 + kept.groupby("train_stem").size().max()
    ann = annotation_table(load_coco(get_data_paths().annotations))
    fig, axes = plt.subplots(len(stems), cols, figsize=(2.6 * cols, 2.6 * len(stems)), squeeze=False)
    for r, ts in enumerate(stems):
        train = read_image(get_data_paths().train_images / f"{ts}.jpeg")
        f = ann[ann.stem == ts].sort_values("area").iloc[-1]
        cy, cx = f.bbox_y + f.bbox_h / 2, f.bbox_x + f.bbox_w / 2
        y0, x0 = int(np.clip(cy - crop / 2, 0, 2048 - crop)), int(np.clip(cx - crop / 2, 0, 2048 - crop))
        panels = [(train, f"{ts[8:12]} train")] + [
            (cv2.imread(str(OUT / "aligned" / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE), f"{o:+.0f} min, r={c:.2f}")
            for s, o, c in kept[kept.train_stem == ts][["stem", "offset_min", "corr"]].itertuples(index=False)]
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(panels):
                ax.imshow(panels[c][0][y0:y0 + crop, x0:x0 + crop], cmap="gray", vmin=0, vmax=255)
                ax.set_title(panels[c][1], fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "contact_sheet.png", dpi=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-corr", type=float, default=0.9,
                        help="on-disk correlation with the training image, both smoothed (filament-scale structure)")
    parser.add_argument("--max-shift", type=float, default=30.0, help="max disk-centre difference before alignment (px)")
    parser.add_argument("--max-dr", type=float, default=15.0, help="max disk-radius difference (px)")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    (OUT / "aligned").mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(OUT / "manifest.csv")
    jobs = [(ts, list(zip(g.stem, g.offset_min)), args.min_corr, args.max_shift, args.max_dr)
            for ts, g in manifest.groupby("train_stem")]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for r in tqdm(pool.map(process, jobs), total=len(jobs), desc="align"):
            rows += r
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "aligned" / "manifest.csv", index=False)
    done = table[table.reason != "not downloaded"]
    print(f"{len(done)} frames aligned and scored, {int(table.keep.sum())} kept; rejected: "
          f"{done[~done.keep].reason.value_counts().to_dict()}")
    if len(done):
        print(done[["corr", "corr_raw", "mean_ratio", "sharp_ratio", "shift_px", "dr_px"]].describe().round(3).to_string())
        print("kept per training image:", table[table.keep].groupby("train_stem").size().value_counts().sort_index()
              .to_dict())
        contact_sheet(table)


if __name__ == "__main__":
    main()
