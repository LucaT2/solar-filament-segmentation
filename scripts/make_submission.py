"""Turn an experiment's probability maps into a submission CSV, using its tuned post-processing.

Usage:
    python scripts/make_submission.py --exp classical                 # test maps -> outputs/classical/submission.csv
    python scripts/make_submission.py --exp classical --split oof     # OOF CSV, checkable with scripts/evaluate.py
"""

from __future__ import annotations

import argparse

from filament.inference import load_postprocess, probs_to_submission
from filament.paths import get_data_paths, get_output_dir
from filament.submission import validate_submission, write_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--split", choices=["test", "oof"], default="test")
    args = parser.parse_args()

    paths = get_data_paths()
    exp_dir = get_output_dir(args.exp)
    cfg = load_postprocess(exp_dir / "postprocess.json")
    image_dir = paths.test_images if args.split == "test" else paths.train_images
    stems = sorted(p.stem for p in image_dir.glob("*.jpeg"))
    if args.split == "oof":  # OOF maps may exist for only some folds so far
        available = {p.stem for p in (exp_dir / "oof").glob("*.png")}
        if len(available) < len(stems):
            print(f"note: OOF maps for {len(available)} of {len(stems)} training images; writing those only")
        stems = [s for s in stems if s in available]

    sub = probs_to_submission(exp_dir / args.split, stems, cfg, image_dir)
    problems = validate_submission(sub, expected_stems=stems)
    name = "submission.csv" if args.split == "test" else "oof_submission.csv"
    path = write_submission(sub, exp_dir / name)
    n_img = sub.filament_id.str.rsplit("_", n=1).str[0].nunique()
    print(f"{cfg}\nwrote {path}: {len(sub)} filaments on {n_img}/{len(stems)} images")
    print("validation:", "OK" if not problems else problems)


if __name__ == "__main__":
    main()
