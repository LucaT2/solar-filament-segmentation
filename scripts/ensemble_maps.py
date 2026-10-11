"""Average the probability maps of several experiments into a new experiment folder (an ensemble).

Only images present in every input are written. The result is a normal experiment folder, so the usual tools work
on it: tune_postprocess.py, fit_rescorer.py, compare.py, make_submission.py.

Usage:
    python scripts/ensemble_maps.py --exps unet_r34_skel050 unet_convnext_t_skel050 --out ens_r34_convnext --split oof
    python scripts/ensemble_maps.py --exps A B --weights 0.4 0.6 --out ens_AB --split test
"""

from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from filament.evaluation import load_prob
from filament.inference import save_prob
from filament.paths import get_output_dir, resolve_exp_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exps", nargs="+", required=True, help="experiment names or folders")
    parser.add_argument("--weights", type=float, nargs="*", help="one per experiment (default: equal)")
    parser.add_argument("--out", required=True, help="name of the ensemble experiment (outputs/<out>)")
    parser.add_argument("--split", default="oof", help="map sub-folder to average: oof, test, ...")
    args = parser.parse_args()

    dirs = [resolve_exp_dir(e) / args.split for e in args.exps]
    weights = np.asarray(args.weights or [1.0] * len(dirs), dtype=np.float32)
    if len(weights) != len(dirs):
        parser.error("--weights needs one value per experiment")
    weights /= weights.sum()
    stems = sorted(set.intersection(*[{p.stem for p in d.glob("*.png")} for d in dirs]))
    if not stems:
        raise SystemExit("no images common to all experiments")
    out = get_output_dir(args.out) / args.split
    for stem in tqdm(stems, desc="ensemble"):
        save_prob(out, stem, sum(w * load_prob(d, stem) for w, d in zip(weights, dirs)))
    print(f"{len(stems)} averaged maps ({', '.join(f'{e}: {w:.2f}' for e, w in zip(args.exps, weights))}) -> {out}")


if __name__ == "__main__":
    main()
