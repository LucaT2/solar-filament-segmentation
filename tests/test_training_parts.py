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
