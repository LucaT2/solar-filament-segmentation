"""Classical baseline: write darkness-score maps for all train and test images.

Nothing is learned, so the maps of all 707 training images are already "out-of-fold".
Output: outputs/classical/oof/<stem>.png and outputs/classical/test/<stem>.png

Usage:
    python scripts/run_classical.py [--workers 8]
then
    python scripts/tune_postprocess.py --exp classical
    python scripts/make_submission.py --exp classical
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from filament.classical import darkness_score
from filament.inference import save_prob
from filament.paths import get_data_paths, get_output_dir
from filament.preprocess import find_disk, read_image


def process(args: tuple[Path, Path]) -> None:
    image_path, out_dir = args
    img = read_image(image_path)
    save_prob(out_dir, image_path.stem, darkness_score(img, find_disk(img)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    paths = get_data_paths()
    out = get_output_dir("classical")
    jobs = [(p, out / "oof") for p in sorted(paths.train_images.glob("*.jpeg"))]
    jobs += [(p, out / "test") for p in sorted(paths.test_images.glob("*.jpeg"))]
    with ProcessPoolExecutor(args.workers) as pool:
        list(tqdm(pool.map(process, jobs, chunksize=4), total=len(jobs), desc="darkness maps"))
    print(f"wrote {len(jobs)} maps to {out}")


if __name__ == "__main__":
    main()
