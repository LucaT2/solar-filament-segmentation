"""Skeleton target, losses and tiled prediction."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from filament.dataset import tubed_skeleton
from filament.model import BCEDiceLoss, SegLoss, predict_full


def test_tubed_skeleton_is_thin_and_inside_the_mask():
    mask = np.zeros((128, 128), np.uint8)
    mask[60:70, 10:120] = 1  # a horizontal 10 px wide bar
    skel = tubed_skeleton(mask, radius=2)
    ys, xs = np.nonzero(skel)
    assert skel.sum() > 0
    assert ys.max() - ys.min() <= 6  # centre line (1 px) + 2 px on each side, not the full 10 px
    assert ((skel > 0) & (mask == 0)).sum() < 0.05 * skel.sum()  # (only the bar's tips may poke out)
    assert tubed_skeleton(np.zeros((64, 64), np.uint8)).sum() == 0


def test_skeleton_recall_rewards_covering_the_centre_line():
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., 30:34, 5:60] = 1
    skel = torch.zeros_like(mask)
    skel[..., 31:33, 6:59] = 1
    target = torch.cat([mask, skel], 1)
    loss = SegLoss(skel_weight=1.0)
    covered = torch.where(mask > 0, 8.0, -8.0)  # confident, correct prediction
    gap = covered.clone()
    gap[..., 20:40] = -8.0  # the model drops the middle stretch of the filament
    assert loss(gap, target) - loss(covered, target) > 0.3  # the gap costs noticeably more with Skeleton Recall
    # with skel_weight = 0 the loss equals plain BCE + Dice on the mask channel
    assert SegLoss(skel_weight=0)(gap, target).item() == pytest.approx(BCEDiceLoss()(gap, mask).item())


@pytest.fixture(scope="module")
def three_annotator_dataset(ann, ai):
    from filament.dataset import CropDataset, image_info
    from filament.paths import get_data_paths

    stem = ai[ai.n_annotators == 3].stem.iloc[0]
    infos = {stem: image_info(get_data_paths().train_images / f"{stem}.jpeg")}

    def make(mode, skeleton=False):
        return CropDataset(ann, infos, [stem], crop=512, samples_per_epoch=30, pos_fraction=1.0,
                           skeleton=skeleton, skel_radius=1, target_mode=mode)

    return make


def test_random_mode_gives_binary_one_annotator_targets(three_annotator_dataset):
    ds = three_annotator_dataset("random")
    assert len(ds.items) == 1 and len(ds.items[0]["unions"]) == 3
    for i in range(10):
        _, y = ds[i]
        assert set(torch.unique(y).tolist()) <= {0.0, 1.0}


def test_consensus_mode_averages_annotators(three_annotator_dataset):
    ds = three_annotator_dataset("consensus", skeleton=True)
    seen = set()
    for i in range(15):
        x, y = ds[i]
        assert y.shape == (2, 512, 512)
        seen |= {round(v, 3) for v in torch.unique(y[0]).tolist()}
        assert float(y[1].max()) <= 1.0 and float(y[1].min()) >= 0.0
    assert seen <= {0.0, 0.333, 0.667, 1.0}
    assert {0.333, 0.667} & seen  # the annotators disagree somewhere, so partial values must occur


def test_sampling_weights_follow_annotator_count(ann, ai):
    from filament.dataset import CropDataset

    stems = ai.drop_duplicates("stem").groupby("n_annotators").stem.first().tolist()  # one image each with 1, 2, 3
    ds = CropDataset(ann, {}, stems, samples_per_epoch=1)
    weights = {len(it["unions"]): p for it, p in zip(ds.items, ds.item_p)}
    assert weights[2] == pytest.approx(2 * weights[1]) and weights[3] == pytest.approx(3 * weights[1])


class _Pointwise(nn.Module):
    """A 'network' with no spatial context: tiled and full-image predictions must agree exactly."""

    def forward(self, x):
        return 3 * x - 1


def test_tiled_prediction_matches_full_image():
    x = np.random.default_rng(0).normal(size=(512, 512)).astype(np.float32)
    dev = torch.device("cpu")
    full = predict_full([_Pointwise()], x, dev, amp_dtype=None)
    tiled = predict_full([_Pointwise()], x, dev, amp_dtype=None, tile=192, overlap=64)
    np.testing.assert_allclose(tiled, full, atol=1e-6)
