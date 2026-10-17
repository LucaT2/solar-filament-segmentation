"""Package the aligned neighbour frames (kept ones + manifest) as a private Kaggle dataset.

Copies outputs/neighbours/aligned/{manifest.csv, <stem>.jpeg for kept frames} into
outputs/kaggle_upload/<slug>/neighbours/ with a dataset-metadata.json; uploaded with --dir-mode zip, the folder goes up
as one archive and Kaggle unpacks it, so the training code finds manifest.csv and the frames together
(filament.data.neighbours.find_neighbour_dir("neighbours")).

Usage:
    python scripts/kaggle/prepare_neighbours_dataset.py --user tasadanluca
    py -3.14 -m kaggle datasets create -p outputs/kaggle_upload/solar-filament-neighbours --dir-mode zip
"""

from __future__ import annotations

import argparse
import json
import shutil

import pandas as pd

from filament.paths import REPO_ROOT, get_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", required=True, help="Kaggle username")
    parser.add_argument("--slug", default="solar-filament-neighbours")
    args = parser.parse_args()

    src = get_output_dir("neighbours") / "aligned"
    table = pd.read_csv(src / "manifest.csv")
    kept = table[table.keep.astype(bool)]
    dst = REPO_ROOT / "outputs" / "kaggle_upload" / args.slug
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "neighbours").mkdir(parents=True)
    kept.to_csv(dst / "neighbours" / "manifest.csv", index=False)
    for stem in kept.stem:
        shutil.copy2(src / f"{stem}.jpeg", dst / "neighbours" / f"{stem}.jpeg")
    (dst / "dataset-metadata.json").write_text(json.dumps({
        "title": args.slug, "id": f"{args.user}/{args.slug}", "licenses": [{"name": "other"}],
        "subtitle": "Aligned neighbouring GONG H-alpha frames of the training images",
    }, indent=2))
    size = sum(p.stat().st_size for p in (dst / "neighbours").iterdir()) / 2**20
    print(f"{dst}: {len(kept)} frames for {kept.train_stem.nunique()} training images, {size:.0f} MB")
    print(f'upload: py -3.14 -m kaggle datasets create -p "{dst}" --dir-mode zip')


if __name__ == "__main__":
    main()
