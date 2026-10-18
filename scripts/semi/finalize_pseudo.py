"""Finish a pseudo-label folder: add each candidate's centre (from the label maps) to candidates.csv, report the
confidence distribution, and optionally copy the frames next to the labels (for a Kaggle dataset).

Usage:
    python scripts/semi/finalize_pseudo.py --pseudo outputs/pseudo
    python scripts/semi/finalize_pseudo.py --pseudo outputs/pseudo --package outputs/kaggle_upload/solar-filament-pseudo/pseudo
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from tqdm import tqdm

from filament.semi.data import frames_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pseudo", default="outputs/pseudo")
    parser.add_argument("--package", help="copy frames.csv, candidates.csv, labels/ and frames/ (ok frames) here")
    args = parser.parse_args()

    folder = Path(args.pseudo)
    cand = pd.read_csv(folder / "candidates.csv").drop_duplicates(["stem", "id"])
    frames = pd.read_csv(folder / "frames.csv").drop_duplicates("stem", keep="last")
    centers = []
    for stem, g in tqdm(cand.groupby("stem"), desc="centres"):
        lab = cv2.imread(str(folder / "labels" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        com = center_of_mass(np.ones_like(lab), lab, g.id.tolist())
        centers += [{"stem": stem, "id": i, "cy": c[0], "cx": c[1]} for i, c in zip(g.id, com)]
    cand = cand.drop(columns=[c for c in ("cy", "cx") if c in cand]).merge(pd.DataFrame(centers), on=["stem", "id"])
    cand.to_csv(folder / "candidates.csv", index=False)
    frames.to_csv(folder / "frames.csv", index=False)

    ok = frames[frames.ok.astype(bool)]
    p = cand.p
    print(f"{len(frames)} frames, {len(ok)} ok; {len(cand)} candidates: P >= 0.6 {int((p >= 0.6).sum())}, "
          f"0.3-0.6 {int(((p >= 0.3) & (p < 0.6)).sum())}, < 0.3 {int((p < 0.3).sum())}")
    print("confident candidates per ok frame:",
          cand[cand.p >= 0.6].groupby("stem").size().reindex(ok.stem, fill_value=0).describe().round(2).to_dict())

    if args.package:
        dst = Path(args.package)
        (dst / "labels").mkdir(parents=True, exist_ok=True)
        (dst / "frames").mkdir(parents=True, exist_ok=True)
        src_frames = frames_dir(folder)
        for s in tqdm(ok.stem, desc="package"):
            shutil.copy2(folder / "labels" / f"{s}.png", dst / "labels" / f"{s}.png")
            shutil.copy2(src_frames / f"{s}.jpg", dst / "frames" / f"{s}.jpg")
        ok.to_csv(dst / "frames.csv", index=False)
        cand[cand.stem.isin(set(ok.stem))].to_csv(dst / "candidates.csv", index=False)
        size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 2**20
        print(f"packaged {len(ok)} frames -> {dst} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
