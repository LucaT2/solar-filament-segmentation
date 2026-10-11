"""Turn an experiment's probability maps into a submission CSV, using its tuned post-processing.

Usage:
    python scripts/make_submission.py --exp classical                 # test maps -> outputs/classical/submission.csv
    python scripts/make_submission.py --exp classical --split oof     # OOF CSV, checkable with scripts/evaluate.py
"""

from __future__ import annotations

import argparse

from filament.inference import load_postprocess, probs_to_submission
from filament.paths import get_data_paths, get_output_dir, resolve_exp_dir
from filament.rescoring import Rescorer
from filament.submission import validate_submission, write_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--split", choices=["test", "oof"], default="test")
    parser.add_argument("--no-rescore", action="store_true", help="ignore outputs/<exp>/rescorer.pkl if present")
    parser.add_argument("--out", help="output file name (default: submission.csv / oof_submission.csv)")
    parser.add_argument("--maps", help="sub-folder of outputs/<exp>/ with the probability maps (default: the split)")
    args = parser.parse_args()

    paths = get_data_paths()
    exp_dir = resolve_exp_dir(args.exp)  # name or folder path
    cfg = load_postprocess(exp_dir / "postprocess.json")
    rescorer = None
    if (exp_dir / "rescorer.pkl").exists() and not args.no_rescore:
        rescorer = Rescorer.load(exp_dir / "rescorer.pkl")
        print(f"using component rescorer (min P(match) = {rescorer.min_p})")
        if args.split == "oof":
            print("WARNING: the rescorer was fitted on these OOF components; scoring this CSV is in-sample (optimistic).\n"
                  "         Use --no-rescore for an honest CSV check; the honest PQ with rescoring is oof_pq_nested "
                  "in rescorer.json.")
    image_dir = paths.test_images if args.split == "test" else paths.train_images
    stems = sorted(p.stem for p in image_dir.glob("*.jpeg"))
    maps = exp_dir / (args.maps or args.split)
    if args.split == "oof":  # OOF maps may exist for only some folds so far
        available = {p.stem for p in maps.glob("*.png")}
        if len(available) < len(stems):
            print(f"note: OOF maps for {len(available)} of {len(stems)} training images; writing those only")
        stems = [s for s in stems if s in available]

    sub = probs_to_submission(maps, stems, cfg, image_dir, rescorer)
    problems = validate_submission(sub, expected_stems=stems)
    name = args.out or ("submission.csv" if args.split == "test" else "oof_submission.csv")
    path = write_submission(sub, exp_dir / name)
    n_img = sub.filament_id.str.rsplit("_", n=1).str[0].nunique()
    print(f"{cfg}\nwrote {path}: {len(sub)} filaments on {n_img}/{len(stems)} images")
    print("validation:", "OK" if not problems else problems)


if __name__ == "__main__":
    main()
