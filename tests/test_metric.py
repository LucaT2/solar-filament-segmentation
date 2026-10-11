"""Our RLE-based PQ must reproduce the official metric exactly, including its quirks."""

import cv2
import numpy as np
import pandas as pd
import pytest

from filament.data.annotations import ground_truth_table
from filament.data.rle import mask_to_counts, rle_to_mask
from filament.evaluation.metric import pq_by_group, pq_score, summarize

SIZE = 2048


def rect(y, x, h, w):
    m = np.zeros((SIZE, SIZE), dtype=bool)
    m[y : y + h, x : x + w] = True
    return m


def table(rows):
    """rows: [(filament_id, mask), ...] -> submission-format DataFrame."""
    return pd.DataFrame({"filament_id": [r[0] for r in rows], "segmentation_rle": [mask_to_counts(r[1]) for r in rows]})


# ---------------------------------------------------------------------------------------------------------------
# Hand-checked cases for each rule of the official implementation
# ---------------------------------------------------------------------------------------------------------------


def test_perfect_and_partial_match():
    gt = table([("000001-20110120105534Ch_1", rect(100, 100, 10, 100))])
    exact = pq_score(gt, table([("20110120105534Ch_1", rect(100, 100, 10, 100))]))
    assert (exact.pq, exact.tp, exact.fp, exact.fn) == (pytest.approx(1.0), 1, 0, 0)

    # 10 x 120 prediction over a 10 x 100 filament: IoU = 1000 / 1200
    wider = pq_score(gt, table([("20110120105534Ch_1", rect(100, 100, 10, 120))]))
    assert wider.pq == pytest.approx(1000 / 1200)


def test_fragmentation_example():
    """A filament split into two halves below IoU 0.5 -> 0 TP, 2 FP, 1 FN (the worked example in the notes)."""
    gt = table([("000001-20110120105534Ch_1", rect(100, 100, 10, 100))])
    halves = table([("20110120105534Ch_1", rect(100, 100, 10, 48)), ("20110120105534Ch_2", rect(100, 152, 10, 48))])
    r = pq_score(gt, halves)
    assert (r.tp, r.fp, r.fn, r.pq) == (0, 2, 1, 0.0)


def test_iou_exactly_half_is_not_a_match():
    gt = table([("000001-20110120105534Ch_1", rect(100, 100, 10, 100))])
    half = table([("20110120105534Ch_1", rect(100, 100, 10, 50))])  # IoU = 500 / 1000 = 0.5, threshold is "> 0.5"
    assert pq_score(gt, half).tp == 0


def test_duplicate_predictions_both_count_as_tp():
    """Official quirk: no one-to-one assignment, every pair with IoU > 0.5 is a TP."""
    gt = table([("000001-20110120105534Ch_1", rect(100, 100, 10, 100))])
    dup = table([("20110120105534Ch_1", rect(100, 100, 10, 100)), ("20110120105534Ch_2", rect(100, 100, 10, 100))])
    r = pq_score(gt, dup)
    assert (r.tp, r.fp, r.fn) == (2, 0, 0)


def test_empty_prediction_and_unknown_stem():
    gt = table([("000001-20110120105534Ch_1", rect(100, 100, 10, 100)), ("000001-20110120105534Ch_2", rect(500, 500, 10, 100))])
    # predictions only for an image that has no ground truth -> ignored; the GT image gets 2 FN
    other = table([("20110130110334Ch_1", rect(100, 100, 10, 100))])
    r = pq_score(gt, other)
    assert (r.tp, r.fp, r.fn, r.pq) == (0, 0, 2, 0.0)


def test_each_annotator_scored_separately():
    """One prediction, two annotators -> scored twice; FN/FP accumulate per annotator-image."""
    a = rect(100, 100, 10, 100)
    gt = table([("000001-20110120105534Ch_1", a), ("000002-20110120105534Ch_1", a), ("000002-20110120105534Ch_2", rect(800, 800, 10, 100))])
    r = pq_score(gt, table([("20110120105534Ch_1", a)]))
    assert len(r.images) == 2
    assert (r.tp, r.fp, r.fn) == (2, 0, 1)
    assert r.pq == pytest.approx(2 / (2 + 0.5))


# ---------------------------------------------------------------------------------------------------------------
# Equivalence with the organizers' implementation on realistic, perturbed predictions
# ---------------------------------------------------------------------------------------------------------------


def perturbed_predictions(ann, ai, stems, seed=0):
    """Damage one annotator's masks per image in all the ways a model might: wider/thinner, shifted, dropped,
    split, merged, duplicated, plus false blobs. Also an extra image with predictions but no GT."""
    rng = np.random.default_rng(seed)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    rows = []
    for stem in stems:
        aid = ai[ai.stem == stem].annotator_image.iloc[0]
        masks = [rle_to_mask(r) for r in ann[ann.annotator_image == aid].rle]
        out = []
        for i, m in enumerate(masks):
            op = rng.choice(["keep", "dilate", "erode", "shift", "drop", "split", "dup"])
            mu8 = m.astype(np.uint8)
            if op == "keep":
                out.append(m)
            elif op == "dilate":
                out.append(cv2.dilate(mu8, kernel).astype(bool))
            elif op == "erode":
                e = cv2.erode(mu8, kernel).astype(bool)
                out.append(e if e.any() else m)
            elif op == "shift":
                out.append(np.roll(m, (3, -4), axis=(0, 1)))
            elif op == "split":
                ys, xs = np.nonzero(m)
                cut = np.median(xs)
                out += [p for p in (m & (np.arange(SIZE)[None, :] < cut), m & (np.arange(SIZE)[None, :] > cut + 2)) if p.any()]
            elif op == "dup":
                out += [m, m]
        if len(masks) >= 2:  # merge the first two filaments into one prediction
            out.append(masks[0] | masks[1])
        out.append(rect(*rng.integers(300, 1700, 2), 8, 60))  # false blob
        rows += [(f"{stem}_{k}", m) for k, m in enumerate(out, start=1)]
    rows.append(("20990101000000Bh_1", rect(10, 10, 10, 10)))  # stem without GT -> ignored by both
    return table(rows)


@pytest.fixture(scope="module")
def scenario(ann, ai):
    # a mix of single- and multi-annotated images, kept small because the official code decodes dense stacks
    multi = ai[ai.n_annotators == 3].stem.drop_duplicates().iloc[:2].tolist()
    single = ai[ai.n_annotators == 1].stem.drop_duplicates().iloc[:3].tolist()
    empty_pred = ai[ai.n_annotators == 2].stem.drop_duplicates().iloc[:1].tolist()  # GT but no predictions
    stems = multi + single
    gt = ground_truth_table(ann, stems + empty_pred)
    pred = perturbed_predictions(ann, ai, stems)
    return gt, pred


def test_matches_official_pq(official, scenario):
    gt, pred = scenario
    expected = official.get_pq_score(official.get_overlap_df(gt, pred))
    ours = pq_score(gt, pred)
    assert 0.05 < ours.pq < 0.95  # the scenario must be non-trivial
    assert ours.pq == pytest.approx(expected, abs=1e-6)


def test_matches_official_overlap_matrices(official, scenario):
    gt, pred = scenario
    off = official.get_overlap_df(gt, pred).set_index("annotator_image")
    ours = {s.key: s for s in pq_score(gt, pred).images}
    assert set(off.index) == set(ours)
    for key, row in off.iterrows():
        assert (row.n_gt, row.n_pred) == (ours[key].n_gt, ours[key].n_pred)
        np.testing.assert_allclose(row.iou_matrix.numpy(), ours[key].iou, atol=1e-6)


def test_report_helpers(scenario):
    gt, pred = scenario
    r = pq_score(gt, pred)
    s = summarize(r)
    assert s["pq"] == pytest.approx(s["sq"] * s["rq"])
    by_site = pq_by_group(r, lambda k: k.split("-", 1)[1][14])
    assert by_site.tp.sum() == r.tp and by_site.fp.sum() == r.fp and by_site.fn.sum() == r.fn
