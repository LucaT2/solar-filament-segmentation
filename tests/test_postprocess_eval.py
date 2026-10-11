"""The fast label-map scorer used for tuning must agree with the verified RLE metric."""

import cv2
import numpy as np
import pytest

from filament.data.rle import mask_to_rle, rle_to_mask
from filament.evaluation.metric import pq_from_rles, score_images
from filament.evaluation.scoring import GroundTruth, score_components
from filament.postprocess.components import PostConfig, find_components, instances


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


def test_component_table_handles_empty_and_normal_maps():
    from filament.data.images import Disk
    from filament.postprocess.rescoring import Rescorer, component_table

    img = np.full((2048, 2048), 130, np.uint8)
    disk = Disk(1024.0, 1024.0, 950.0)
    comps, table = component_table("20110120105534Ch", np.zeros((2048, 2048), np.float32), img, PostConfig(0.5), disk)
    assert comps.n == 0 and len(table) == 0
    assert len(Rescorer(None, 0.3).keep_ids(table)) == 0

    prob = np.zeros((2048, 2048), np.float32)
    prob[1000:1010, 900:1100] = 0.9
    img[1000:1010, 900:1100] = 100
    comps, table = component_table("20110120105534Ch", prob, img, PostConfig(0.5, 100), disk)
    row = table.iloc[0]
    assert (comps.n, row.area, bool(row.kept), row.site) == (1, 2000, True, "C")
    assert row.contrast == pytest.approx(30 / 130) and row.elongation > 15


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
