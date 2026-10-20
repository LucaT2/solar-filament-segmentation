"""Skeleton target, losses and tiled prediction."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from filament.segmentation.dataset import tubed_skeleton
from filament.segmentation.model import BCEDiceLoss, SegLoss, predict_full


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
    from filament.paths import get_data_paths
    from filament.segmentation.dataset import CropDataset, image_info

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
    from filament.segmentation.dataset import CropDataset

    stems = ai.drop_duplicates("stem").groupby("n_annotators").stem.first().tolist()  # one image each with 1, 2, 3
    ds = CropDataset(ann, {}, stems, samples_per_epoch=1)
    weights = {len(it["unions"]): p for it, p in zip(ds.items, ds.item_p)}
    assert weights[2] == pytest.approx(2 * weights[1]) and weights[3] == pytest.approx(3 * weights[1])


def test_decoder_fp32_runs_decoder_in_float32_under_autocast():
    from filament.segmentation.model import build_model

    seen = {}
    model = build_model("Unet", "resnet34", pretrained=False, decoder_fp32=True)
    model.segmentation_head.register_forward_hook(lambda m, i, o: seen.setdefault("dtype", i[0].dtype))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        model(torch.randn(1, 1, 64, 64))
    assert seen["dtype"] == torch.float32  # decoder output stayed float32 although autocast was on
    plain = build_model("Unet", "resnet34", pretrained=False)
    plain.segmentation_head.register_forward_hook(lambda m, i, o: seen.__setitem__("plain", i[0].dtype))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        plain(torch.randn(1, 1, 64, 64))
    assert seen["plain"] == torch.bfloat16


class _Pointwise(nn.Module):
    """A 'network' with no spatial context: tiled and full-image predictions must agree exactly."""

    def forward(self, x):
        return 3 * x - 1


@pytest.mark.parametrize("mode", ["none", "flip4", "d8"])
def test_tta_transforms_are_exact_inverses(mode):
    from filament.segmentation.model import tta_transforms

    t = torch.randn(1, 1, 16, 16)
    transforms = tta_transforms(mode)
    assert len(transforms) == {"none": 1, "flip4": 4, "d8": 8}[mode]
    outputs = []
    for fwd, inv in transforms:
        assert torch.equal(inv(fwd(t)), t)
        outputs.append(fwd(t))
    # all transformed versions are distinct (no symmetry counted twice) for a random image
    assert len({tuple(o.flatten().tolist()) for o in outputs}) == len(transforms)


def test_tiled_prediction_matches_full_image():
    x = np.random.default_rng(0).normal(size=(512, 512)).astype(np.float32)
    dev = torch.device("cpu")
    full = predict_full([_Pointwise()], x, dev, amp_dtype=None)
    tiled = predict_full([_Pointwise()], x, dev, amp_dtype=None, tile=192, overlap=64)
    np.testing.assert_allclose(tiled, full, atol=1e-6)


def _bar_with_barb():
    m = torch.zeros(1, 1, 64, 64)
    m[..., 20:36, 8:56] = 1  # a 16 px thick body
    m[..., 36:50, 30:33] = 1  # a 3 px wide barb hanging off it
    return m


def test_boundary_and_thin_targets():
    from filament.segmentation.model import boundary_target, thin_target
    m = _bar_with_barb()
    band = boundary_target(m, 3)
    assert band[0, 0, 28, 30] == 0 and band[0, 0, 20, 30] == 1 and band[0, 0, 19, 30] == 1  # interior / edge / outside
    assert band[0, 0, 40:50, 30:33].sum(-1).min() == 2  # a 3 px barb keeps its centre line with a 3 x 3 band ...
    assert boundary_target(m, 5)[0, 0, 40:50, 30:33].min() == 1  # ... and is boundary throughout with 5 x 5
    thin = thin_target(m, 7)
    assert thin[0, 0, 40:50, 30:33].min() == 1 and thin[0, 0, 28, 30] == 0  # barb thin, body interior not
    assert thin[..., 20:36, 8:56].sum() < 0.05 * 16 * 48  # (only the body's corners)
    assert ((thin > 0) & (m == 0)).sum() == 0


def test_boundary_head_and_thin_weight_losses():
    from filament.segmentation.model import boundary_target
    m = _bar_with_barb()
    target = torch.cat([m, m], 1)  # [mask, skeleton] (skeleton unused here)
    perfect_mask = (m * 2 - 1) * 20
    plain = SegLoss()(perfect_mask, target)
    assert SegLoss(thin_weight=4.0)(perfect_mask, target).item() == pytest.approx(plain.item(), abs=1e-4)
    with_head = torch.cat([perfect_mask, (boundary_target(m) * 2 - 1) * 20], 1)
    assert SegLoss(boundary_weight=0.5)(with_head, target).item() == pytest.approx(plain.item(), abs=1e-3)
    wrong_head = torch.cat([perfect_mask, -20 * torch.ones_like(m)], 1)
    assert SegLoss(boundary_weight=0.5)(wrong_head, target) > plain + 0.4
    no_barb = perfect_mask.clone()
    no_barb[..., 36:50, 30:33] = -20  # missing the barb costs more with thin weighting
    assert SegLoss(thin_weight=4.0)(no_barb, target) > SegLoss()(no_barb, target) * 2


def test_two_channel_model_predicts_the_mask_channel():
    from filament.segmentation.model import build_model, n_outputs
    model = build_model("Unet", "resnet34", pretrained=False, classes=n_outputs({"boundary_weight": 0.5})).eval()
    assert model(torch.zeros(1, 1, 64, 64)).shape[1] == 2
    prob = predict_full([model], np.zeros((64, 64), np.float32), torch.device("cpu"), amp_dtype=None)
    assert prob.shape == (64, 64)
