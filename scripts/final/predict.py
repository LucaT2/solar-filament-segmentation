"""Run the submitted pipeline (filament.final) on a folder of images and write the submission CSV.

Usage:
    python scripts/final/download_weights.py                             # once: the checkpoint into outputs/
    python scripts/final/predict.py --out outputs/submission_final.csv   # all test images, ~2 s per image on a GPU
    python scripts/final/predict.py --limit 3                            # a quick check on 3 images
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from filament.data.submission import COLUMNS, instances_to_rows, validate_submission, write_submission
from filament.final import FinalPipeline
from filament.paths import get_data_paths, get_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, help="folder of .jpeg images (default: the competition test images)")
    parser.add_argument("--out", type=Path, default=get_output_dir() / "submission_final.csv")
    parser.add_argument("--limit", type=int, help="only the first N images")
    args = parser.parse_args()

    paths = sorted((args.images or get_data_paths().test_images).glob("*.jpeg"))[:args.limit]
    pipeline = FinalPipeline.load()
    rows = []
    for path in tqdm(paths, desc="images"):
        rows.extend(instances_to_rows(path.stem, pipeline.segment(path)))
    df = pd.DataFrame(rows, columns=COLUMNS)
    problems = validate_submission(df, [p.stem for p in paths])
    if problems:
        raise SystemExit("invalid submission:\n" + "\n".join(problems))
    write_submission(df, args.out)
    print(f"{len(df)} filaments on {len(paths)} images -> {args.out}")


if __name__ == "__main__":
    main()
