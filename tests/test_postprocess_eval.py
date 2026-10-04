"""The fast label-map scorer used for tuning must agree with the verified RLE metric."""

import cv2
import numpy as np
import pytest

from filament.evaluation import GroundTruth, score_components
from filament.metric import pq_from_rles, score_images
from filament.postprocess import PostConfig, find_components, instances
from filament.rle import mask_to_rle, rle_to_mask


def fake_prob(ann, aid, seed=0):
    """A plausible probability map: blurred, slightly shifted GT of one annotator plus noise blobs."""
    rng = np.random.default_rng(seed)
    union = np.zeros((2048, 2048), np.float32)
    for r in ann[ann.annotator_image == aid].rle:
        union = np.maximum(union, rle_to_mask(r).astype(np.float32))
    prob = cv2.GaussianBlur(np.roll(union, (2, -1), axis=(0, 1)), (0, 0), 2.0)
    for _ in range(15):
        y, x = rng.integers(100, 1900, 2)
        prob[y : y + rng.integers(3, 25), x : x + rng.integers(3, 40)] = rng.uniform(0.3, 0.9)
    return np.clip(prob, 0, 1)


@pytest.mark.parametrize(
    "cfg", [PostConfig(0.5), PostConfig(0.3, 40), PostConfig(0.6, 10, 0.7), PostConfig(0.5, 40, link_px=4)]
)
def test_fast_scorer_matches_metric(ann, ai, cfg):
    stems = ai[ai.n_annotators == 3].stem.drop_duplicates().iloc[:2].tolist()
    gt = GroundTruth(ann, stems)
    fast, pred_rles = [], {}
    for stem in stems:
        prob = fake_prob(ann, ai[ai.stem == stem].annotator_image.iloc[0])
        comps = find_components(prob, cfg.threshold, link_px=cfg.link_px)
        fast += score_components(stem, comps, comps.keep(cfg.min_area, cfg.min_score), gt)
        pred_rles[stem] = [mask_to_rle(m) for m in instances(prob, cfg)]

    gt_rles = {aid: ann[ann.annotator_image == aid].rle.tolist() for stem in stems for aid in gt.by_stem[stem]}
    pred_by_aid = {aid: pred_rles[aid.split("-", 1)[1]] for aid in gt_rles}
    reference = pq_from_rles(gt_rles, pred_by_aid)
    ours = score_images(fast)
    assert (ours.tp, ours.fp, ours.fn) == (reference.tp, reference.fp, reference.fn)
    assert ours.pq == pytest.approx(reference.pq, abs=1e-9)


def test_link_keeps_original_pixels():
    prob = np.zeros((2048, 2048), np.float32)
    prob[100:110, 100:150] = 0.9
    prob[100:110, 154:200] = 0.9  # 4 px gap
    unlinked, linked = find_components(prob, 0.5), find_components(prob, 0.5, link_px=3)
    assert (unlinked.n, linked.n) == (2, 1)
    assert linked.area[1] == unlinked.area[1:].sum()  # gap pixels are not added to the instance


def test_component_stats():
    prob = np.zeros((2048, 2048), np.float32)
    prob[10:20, 10:20] = 0.8
    prob[100:102, 100:103] = 0.6
    comps = find_components(prob, 0.5)
    assert comps.n == 2
    assert sorted(comps.area[1:]) == [6, 100]
    assert sorted(comps.mean_prob[1:].round(3).tolist()) == pytest.approx([0.6, 0.8])
    assert len(comps.keep(min_area=10)) == 1
    assert len(comps.keep(min_score=0.7)) == 1
