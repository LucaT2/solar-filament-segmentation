"""Turn an instance experiment's saved detections into probability maps (score-weighted soft masks, max over overlaps),
so the instance model can join probability-map ensembles (scripts/postprocess/ensemble_maps.py).

Usage:
    python scripts/detection/instances_to_maps.py --exp <instance exp> --split oof --out mrcnn_maps
    python scripts/detection/instances_to_maps.py --exp <instance exp> --split test_<name> --out mrcnn_maps \
        --out-split test
"""

from __future__ import annotations

import argparse

from tqdm import tqdm

from filament.detection.instances import load_instances, to_prob_map
from filament.paths import get_output_dir, resolve_exp_dir
from filament.postprocess.maps import save_prob


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True, help="instance experiment (name or folder)")
    parser.add_argument("--split", default="oof", help="sub-folder with the .npz detections")
    parser.add_argument("--out", required=True, help="new map experiment (outputs/<out>)")
    parser.add_argument("--out-split", help="sub-folder to write (default: same as --split)")
    parser.add_argument("--min-score", type=float, default=0.05)
    args = parser.parse_args()

    src = resolve_exp_dir(args.exp) / args.split
    dst = get_output_dir(args.out) / (args.out_split or args.split)
    stems = sorted(p.stem for p in src.glob("*.npz"))
    for stem in tqdm(stems, desc="maps"):
        save_prob(dst, stem, to_prob_map(load_instances(src, stem), min_score=args.min_score))
    print(f"{len(stems)} maps -> {dst}")


if __name__ == "__main__":
    main()
