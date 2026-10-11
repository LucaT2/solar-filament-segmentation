import numpy as np
import pandas as pd
import pytest

from filament.data.annotations import ground_truth_table
from filament.evaluation.metric import pq_score
from filament.data.rle import (
    counts_to_mask,
    mask_to_counts,
    polygons_to_rle,
    rle_area,
    rle_iou,
    rle_to_mask,
)
from filament.data.submission import build_submission, read_submission, validate_submission, write_submission


def random_mask(rng, size=2048, n_blobs=3):
    mask = np.zeros((size, size), dtype=bool)
    for _ in range(n_blobs):
        y, x = rng.integers(0, size - 60, 2)
        h, w = rng.integers(3, 60, 2)
        mask[y : y + h, x : x + w] = True
    return mask


def test_mask_counts_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(5):
        mask = random_mask(rng)
        assert np.array_equal(counts_to_mask(mask_to_counts(mask)), mask)


def test_edge_pixels_roundtrip():
    # pixels touching the image border and the corners must survive (column-major RLE order)
    mask = np.zeros((2048, 2048), dtype=bool)
    mask[0, :5] = mask[-1, -5:] = mask[:5, 0] = mask[-5:, -1] = True
    assert np.array_equal(counts_to_mask(mask_to_counts(mask)), mask)


def test_polygon_area_matches_coco_area(ann):
    # the area field in the JSON was computed from the same polygons
    sample = ann.sample(200, random_state=0)
    ours = rle_area(sample["rle"].tolist())
    rel_err = np.abs(ours - sample["area"].to_numpy()) / sample["area"].to_numpy()
    assert np.median(rel_err) < 0.01


def test_polygon_to_rle_x_y_orientation():
    # a horizontal bar in polygon (x, y) coordinates must be wide in the mask, not tall
    rle = polygons_to_rle([[100, 10, 300, 10, 300, 20, 100, 20]], 2048, 2048)
    ys, xs = np.nonzero(rle_to_mask(rle))
    assert xs.max() - xs.min() > 150 and ys.max() - ys.min() < 15


def test_ground_truth_as_submission_scores_one(ann, ai, tmp_path):
    """Single-annotator images: writing GT through the submission pipeline and reading it back scores PQ = 1."""
    single = ai[ai.n_annotators == 1].stem.tolist()[:60]
    sub_ann = ann[ann.stem.isin(single)]
    preds = {stem: [r["counts"] for r in g.rle] for stem, g in sub_ann.groupby("stem")}
    path = write_submission(build_submission(preds), tmp_path / "submission.csv")
    sub = read_submission(path)

    assert validate_submission(sub, expected_stems=single) == []
    assert '"' not in path.read_text()
    result = pq_score(ground_truth_table(ann, single), sub)
    assert result.pq == pytest.approx(1.0)
    assert result.fp == result.fn == 0


def test_submission_from_masks_matches_counts(ann):
    g = ann[ann.annotator_image == ann.annotator_image.iloc[0]]
    masks = [rle_to_mask(r) for r in g.rle]
    stem = g.stem.iloc[0]
    from_masks = build_submission({stem: masks})
    from_counts = build_submission({stem: [r["counts"] for r in g.rle]})
    pd.testing.assert_frame_equal(from_masks, from_counts)


def test_validate_submission_catches_problems():
    good = mask_to_counts(random_mask(np.random.default_rng(1)))
    empty = mask_to_counts(np.zeros((2048, 2048), bool))
    df = pd.DataFrame(
        {
            "filament_id": ["20110120105534Ch_1", "20110120105534Ch_1", "badid", "20110120105534Ch_3"],
            "segmentation_rle": [good, good, good, empty],
        }
    )
    problems = " | ".join(validate_submission(df, expected_stems=["20110120105534Ch"]))
    assert "duplicated" in problems
    assert "malformed" in problems
    assert "empty masks" in problems

    assert validate_submission(df.rename(columns={"segmentation_rle": "rle"}))
    quoted = pd.DataFrame({"filament_id": ["20110120105534Ch_1"], "segmentation_rle": [f'"{good}"']})
    assert any("quotes" in p for p in validate_submission(quoted))


def test_rle_iou_matches_dense():
    rng = np.random.default_rng(2)
    a, b = random_mask(rng), random_mask(rng)
    b |= a & (rng.random(a.shape) < 0.5)
    dense = (a & b).sum() / (a | b).sum()
    from filament.data.rle import mask_to_rle

    assert rle_iou([mask_to_rle(a)], [mask_to_rle(b)])[0, 0] == pytest.approx(dense)
