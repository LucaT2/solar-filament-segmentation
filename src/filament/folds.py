"""Cross-validation folds.

Folds are grouped by image stem: all annotators of one image land in the same fold, so the model never trains on
one annotator's outline of an image it is validated on. Stage 1 showed this reproduces the train -> test time
structure (test images are as close in time to training images as training images are to each other).
Folds are stratified by GONG site x number of annotators.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from filament.paths import REPO_ROOT

FOLDS_PATH = REPO_ROOT / "configs" / "folds.csv"
N_FOLDS = 5
SEED = 42


def make_folds(ai: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED) -> pd.DataFrame:
    """One row per image stem: ``stem, site, n_annotators, fold``.

    ``ai`` is the annotator-image table from ``filament.data.annotator_image_table``. Because every stem is a
    single row, a stratified split over stems is automatically grouped.
    """
    stems = ai.drop_duplicates("stem")[["stem", "n_annotators"]].sort_values("stem").reset_index(drop=True)
    stems["site"] = stems["stem"].str[14]
    strata = stems["site"] + "_" + stems["n_annotators"].astype(str)
    stems["fold"] = -1
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(skf.split(stems, strata)):
        stems.loc[val_idx, "fold"] = fold
    return stems[["stem", "site", "n_annotators", "fold"]]


def save_folds(folds: pd.DataFrame, path: Path = FOLDS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(path, index=False)
    return path


def load_folds(path: Path = FOLDS_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"stem": str})
