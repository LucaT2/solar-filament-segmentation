"""Combine two instance models: detections (which filaments, how confident) from one, outlines from another.

For every image, each detection of --det keeps its score but takes the box and mask of the --outline detection that
overlaps it clearly (box IoU >= --min-iou), if there is one (filament.detection.instances.swap_outlines). Example: a
1x model detects better, a 2x model draws sharper outlines.

Usage:
    python scripts/detection/swap_outlines.py --det <exp>/oof --outline <exp2>/oof --out outputs/<new>/oof
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from filament.detection.instances import load_instances, save_instances, swap_outlines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--det", required=True, help="folder of .npz detections that decide which filaments")
    parser.add_argument("--outline", required=True, help="folder of .npz detections whose masks are used")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--min-iou", type=float, default=0.5)
    args = parser.parse_args()

    det_dir, out_dir = Path(args.det), Path(args.out)
    stems = sorted(p.stem for p in det_dir.glob("*.npz"))
    shares = []
    for s in tqdm(stems, desc="swap"):
        merged, share = swap_outlines(load_instances(det_dir, s), load_instances(args.outline, s), args.min_iou)
        save_instances(out_dir, s, merged)
        shares.append(share)
    print(f"{len(stems)} images -> {out_dir}; outlines replaced for {np.mean(shares):.1%} of detections on average")


if __name__ == "__main__":
    main()
