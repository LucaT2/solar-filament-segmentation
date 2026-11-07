"""Create configs/folds.csv: 5 folds grouped by image stem, stratified by site x number of annotators.

Usage:
    python scripts/setup/make_folds.py
"""

import argparse
import pandas as pd

from filament.data.annotations import annotator_image_table, load_coco
from filament.data.folds import make_folds, save_folds
from filament.paths import get_data_paths


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    ai = annotator_image_table(load_coco(get_data_paths().annotations))
    folds = make_folds(ai)
    path = save_folds(folds)
    print(f"wrote {path}")
    print(pd.crosstab(folds.fold, folds.site, margins=True))
    print(pd.crosstab(folds.fold, folds.n_annotators, margins=True))
    ann_per_fold = ai.merge(folds[["stem", "fold"]], on="stem").groupby("fold").size()
    print("annotator-images per fold:", ann_per_fold.to_dict())


if __name__ == "__main__":
    main()
