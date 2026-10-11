import pandas as pd

from filament.data.folds import FOLDS_PATH, N_FOLDS, load_folds, make_folds


def test_every_image_in_exactly_one_fold(ai):
    folds = make_folds(ai)
    assert folds.stem.is_unique
    assert set(folds.stem) == set(ai.stem)
    assert set(folds.fold) == set(range(N_FOLDS))


def test_annotations_of_one_image_share_a_fold(ai):
    """No image may be in training under one annotator and in validation under another."""
    merged = ai.merge(load_folds()[["stem", "fold"]], on="stem", how="left")
    assert merged.fold.notna().all()
    assert (merged.groupby("stem").fold.nunique() == 1).all()


def test_folds_are_balanced(ai):
    folds = make_folds(ai)
    sizes = folds.fold.value_counts()
    assert sizes.max() - sizes.min() <= 2
    # each site's share per fold stays close to its overall share
    share = pd.crosstab(folds.fold, folds.site, normalize="index")
    overall = folds.site.value_counts(normalize=True)
    assert (share - overall).abs().max().max() < 0.05


def test_folds_are_deterministic(ai):
    pd.testing.assert_frame_equal(make_folds(ai), make_folds(ai))


def test_saved_folds_match_generator(ai):
    """configs/folds.csv is the single source of truth; it must not drift from the generator."""
    assert FOLDS_PATH.exists(), "run: python scripts/setup/make_folds.py"
    saved = load_folds()
    pd.testing.assert_frame_equal(saved.reset_index(drop=True), make_folds(ai).reset_index(drop=True), check_dtype=False)
