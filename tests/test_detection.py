"""Mask R-CNN parts: mask pasting, overlap resolution, the shared prediction loader, training data, model I/O."""

import numpy as np
import pytest
import torch

from filament.data.rle import mask_to_rle
from filament.detection.instances import Instances, load_instances, paste_mask, save_instances, to_components
from filament.evaluation.metric import pq_from_rles
from filament.postprocess.predictions import available_stems, load_components, prediction_kind


def box_instance(boxes, scores, m: int = 28) -> Instances:
    """Detections whose soft masks fill their whole box."""
    return Instances(np.asarray(boxes, np.float32), np.asarray(scores, np.float32),
                     np.ones((len(scores), m, m), np.float16))


def test_full_mask_pastes_onto_its_box():
    y0, x0, patch = paste_mask(np.ones((28, 28)), np.array([10, 20, 50, 60]), 0.5, (128, 128))
    # torchvision pads the mask by one pixel and enlarges the box to match; a full mask covers about the box
    assert abs(y0 - 20) <= 2 and abs(x0 - 10) <= 2
    assert abs(patch.sum() - 40 * 40) < 0.15 * 40 * 40


def test_paste_clips_at_the_image_border():
    y0, x0, patch = paste_mask(np.ones((28, 28)), np.array([-10, 100, 30, 140]), 0.5, (128, 128))
    assert y0 >= 0 and x0 == 0
    assert y0 + patch.shape[0] <= 128 and patch.shape[1] <= 128


def test_overlapping_duplicates_are_dropped_and_higher_score_wins():
    inst = box_instance([[10, 10, 60, 60], [12, 12, 62, 62], [80, 80, 120, 120]], [0.6, 0.9, 0.5])
    comps = to_components(inst, 0.5, shape=(128, 128))
    assert comps.n == 2  # the 0.6 duplicate of the 0.9 detection is gone
    assert sorted(comps.mean_prob[1:].round(2).tolist()) == [0.5, 0.9]
    # a score cut-off applied afterwards equals filtering the detections first
    kept = comps.keep(min_score=0.7)
    assert len(kept) == 1 and comps.mean_prob[kept[0]] == pytest.approx(0.9)


def test_partial_overlap_keeps_both_with_disjoint_pixels():
    inst = box_instance([[10, 10, 60, 60], [40, 10, 100, 60]], [0.9, 0.8])
    comps = to_components(inst, 0.5, shape=(128, 128))
    assert comps.n == 2
    assert comps.area[1:].sum() == (comps.labels > 0).sum()  # every pixel belongs to exactly one instance


def test_region_restricts_instances():
    region = np.zeros((128, 128), bool)
    region[:, :64] = True
    comps = to_components(box_instance([[10, 10, 50, 50], [80, 80, 120, 120]], [0.9, 0.8]), 0.5, region, (128, 128))
    assert comps.n == 1 and not (comps.labels[:, 64:] > 0).any()


def test_instances_render_as_score_weighted_probability_map():
    from filament.detection.instances import to_prob_map

    prob = to_prob_map(box_instance([[10, 10, 60, 60], [40, 40, 100, 100]], [0.9, 0.6]), shape=(128, 128))
    assert prob.max() == pytest.approx(0.9, abs=0.01)  # the stronger detection wins where they overlap
    assert prob[80, 80] == pytest.approx(0.6, abs=0.01) and prob[120, 5] == 0
    assert to_prob_map(Instances.empty(), shape=(64, 64)).sum() == 0


def test_saved_instances_load_through_the_shared_loader(tmp_path):
    inst = box_instance([[10, 10, 60, 60], [80, 80, 120, 120]], [0.9, 0.4])
    save_instances(tmp_path, "img", inst)
    back = load_instances(tmp_path, "img")
    np.testing.assert_allclose(back.boxes, inst.boxes)
    assert prediction_kind(tmp_path) == "instances" and available_stems(tmp_path) == {"img"}
    comps = load_components(tmp_path, "img", 0.5)
    assert comps.n == 2
    # scoring the components against themselves as ground truth gives PQ 1
    masks = [comps.labels == i for i in comps.keep()]
    rles = {"img": [mask_to_rle(m) for m in masks]}
    assert pq_from_rles(rles, rles).pq == pytest.approx(1.0)


def test_instance_dataset_targets_are_valid(ann, ai):
    from filament.detection.dataset import InstanceCropDataset
    from filament.paths import get_data_paths
    from filament.segmentation.dataset import build_augment, image_info

    stem = ai[ai.n_annotators == 3].stem.iloc[0]
    infos = {stem: image_info(get_data_paths().train_images / f"{stem}.jpeg")}
    ds = InstanceCropDataset(ann, infos, [stem], crop=512, samples_per_epoch=12, pos_fraction=1.0,
                             augment=build_augment("base"))
    assert len(ds.items) == 3  # one item per annotator-image
    n_with_instances = 0
    for i in range(12):
        x, t = ds[i]
        assert x.shape == (3, 512, 512) and x.dtype == torch.float32
        n = len(t["labels"])
        assert t["boxes"].shape == (n, 4) and t["masks"].shape == (n, 512, 512)
        if n:
            n_with_instances += 1
            b = t["boxes"]
            assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all()
            for box, m in zip(b.int().tolist(), t["masks"]):  # each box is the tight box of its mask
                ys, xs = torch.nonzero(m, as_tuple=True)
                assert [xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1] == box
    assert n_with_instances >= 8  # crops are centred on filaments

    empty = InstanceCropDataset(ann, infos, [stem], crop=512, samples_per_epoch=1, pos_fraction=1.0,
                                min_instance_area=10**9)  # every piece too small: a crop without instances
    x, t = empty[0]
    assert t["boxes"].shape == (0, 4) and t["masks"].shape == (0, 512, 512) and len(t["labels"]) == 0


def test_guide_map_stays_aligned_with_the_instances(ann, ai, tmp_path):
    """Use the ground truth itself as the 'guide': after cropping and augmentation, the guide channel must still
    sit exactly on the instance masks."""
    import cv2

    from filament.data.rle import rle_to_mask
    from filament.detection.dataset import InstanceCropDataset, full_image_input
    from filament.paths import get_data_paths
    from filament.segmentation.dataset import build_augment, image_info

    stem = ai[ai.n_annotators == 1].stem.iloc[0]  # one annotator: the guide equals every sample's instances
    union = np.zeros((2048, 2048), np.uint8)
    for r in ann[ann.stem == stem].rle:
        union |= rle_to_mask(r).astype(np.uint8)
    cv2.imwrite(str(tmp_path / f"{stem}.png"), union * 255)
    info = image_info(get_data_paths().train_images / f"{stem}.jpeg")
    ds = InstanceCropDataset(ann, {stem: info}, [stem], crop=512, samples_per_epoch=8, pos_fraction=1.0,
                             augment=build_augment("base"), min_instance_area=1, guide_dir=tmp_path)
    for i in range(8):
        x, t = ds[i]
        guide = x[1] > 0  # guide 1 -> +2, guide 0 -> -2
        inst = t["masks"].any(0) if len(t["masks"]) else torch.zeros_like(guide)
        assert torch.equal(guide, inst)
        assert torch.equal(x[0], x[2])  # the image channels stay identical
    full = full_image_input(info, tmp_path)
    assert full.shape == (3, 2048, 2048) and (full[1] > 0).sum() == union.sum()


@pytest.fixture(scope="module")
def cascade_model():
    from filament.detection.model import build_maskrcnn

    torch.manual_seed(0)
    return build_maskrcnn(pretrained=False, size=256, mask_size=56, arch="cascade")


def _one_filament_target():
    masks = torch.zeros(1, 256, 256, dtype=torch.uint8)
    masks[0, 100:120, 40:200] = 1
    return {"boxes": torch.tensor([[40.0, 100.0, 200.0, 120.0]]), "labels": torch.ones(1, dtype=torch.int64),
            "masks": masks}


def test_cascade_trains_every_stage(cascade_model):
    from filament.detection.model import set_input_size

    model = cascade_model
    model.train()
    set_input_size(model, 256)
    losses = model([torch.randn(3, 256, 256), torch.randn(3, 256, 256)],
                   [_one_filament_target(), {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64),
                                             "masks": torch.zeros(0, 256, 256, dtype=torch.uint8)}])
    stages = model.roi_heads.num_stages
    assert stages == 3
    assert {f"loss_classifier_s{t}" for t in range(stages)} | {f"loss_box_reg_s{t}" for t in range(stages)} | \
        {"loss_mask", "loss_objectness", "loss_rpn_box_reg"} <= set(losses)
    assert all(torch.isfinite(v) for v in losses.values())
    model.zero_grad()
    sum(losses.values()).backward()
    for t in range(stages):  # every stage's head and predictor receives gradient
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.roi_heads.box_predictors[t].parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.roi_heads.box_heads[t].parameters())


def test_cascade_stages_label_with_stricter_iou(cascade_model):
    heads = cascade_model.roi_heads
    gt = [torch.tensor([[0.0, 0.0, 100.0, 100.0]])]
    # proposals with IoU 1.0, 0.64 and 0.55 with the ground-truth box
    props = [torch.tensor([[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 64.0], [0.0, 0.0, 100.0, 55.0]])]
    labels = [heads._label(t, props, gt, [torch.ones(1, dtype=torch.int64)])[0][0].tolist() for t in range(3)]
    assert labels == [[1, 1, 1], [1, 1, 0], [1, 0, 0]]  # thresholds 0.5, 0.6, 0.7


def test_cascade_zero_regression_keeps_boxes(cascade_model):
    heads = cascade_model.roi_heads
    props = [torch.tensor([[10.0, 20.0, 60.0, 90.0]])]
    refined = heads._refine(1, torch.zeros(1, 8), props, [(256, 256)])
    torch.testing.assert_close(refined[0], props[0])


def test_cascade_predicts_raw_instances(cascade_model):
    from filament.detection.model import predict_instances

    inst = predict_instances(cascade_model, np.random.default_rng(0).normal(size=(256, 256)).astype(np.float32),
                             torch.device("cpu"))
    assert inst.boxes.shape == (len(inst), 4) and inst.masks.shape[1:] == (56, 56)
    assert ((inst.scores >= 0) & (inst.scores <= 1)).all()


def test_soft_dice_loss():
    from filament.detection.losses import soft_dice_loss

    t = torch.zeros(2, 8, 8)
    t[:, 2:6, 2:6] = 1
    perfect = torch.where(t > 0, 20.0, -20.0)
    assert soft_dice_loss(perfect, t).item() == pytest.approx(0.0, abs=1e-3)
    assert soft_dice_loss(-perfect, t).item() > 0.9  # everything wrong


def test_dice_mask_loss_adds_to_bce_and_restores_torchvision():
    from torchvision.models.detection import roi_heads as tv_roi_heads

    from filament.detection.losses import DiceMaskRoIHeads
    from filament.detection.model import build_maskrcnn, set_input_size

    original = tv_roi_heads.maskrcnn_loss
    losses = {}
    for w in (0.0, 1.0):
        torch.manual_seed(0)
        model = build_maskrcnn(pretrained=False, size=256, mask_size=56, mask_dice_weight=w)
        assert isinstance(model.roi_heads, DiceMaskRoIHeads) == (w > 0)
        model.train()
        set_input_size(model, 256)
        torch.manual_seed(1)
        losses[w] = model([torch.randn(3, 256, 256)], [_one_filament_target()])
    # identical model and data: the box losses agree, the mask loss grows by the Dice term
    assert losses[1.0]["loss_classifier"].item() == pytest.approx(losses[0.0]["loss_classifier"].item(), rel=1e-4)
    assert losses[1.0]["loss_mask"].item() > losses[0.0]["loss_mask"].item() + 0.1
    assert tv_roi_heads.maskrcnn_loss is original  # torchvision's function is restored after the forward pass


def test_upscaled_crops_keep_masks_on_the_filaments(ann, ai):
    from filament.detection.dataset import InstanceCropDataset
    from filament.paths import get_data_paths
    from filament.segmentation.dataset import image_info

    stem = ai[ai.n_annotators == 1].stem.iloc[0]
    infos = {stem: image_info(get_data_paths().train_images / f"{stem}.jpeg")}
    contrast = {}
    for scale in (1.0, 2.0):
        ds = InstanceCropDataset(ann, infos, [stem], crop=512, samples_per_epoch=10, pos_fraction=1.0, scale=scale)
        assert ds.native == int(512 / scale)
        vals = []
        for i in range(10):
            x, t = ds[i]
            assert x.shape == (3, 512, 512) and t["masks"].shape[1:] == (512, 512)
            if len(t["masks"]):
                inside = t["masks"].any(0)
                vals.append(float(x[0][~inside].mean() - x[0][inside].mean()))
        contrast[scale] = np.mean(vals)
    assert contrast[2.0] > 0.5 * contrast[1.0] > 0  # masks still sit on the dark filaments after upscaling


def test_upscaled_prediction_returns_original_coordinates():
    from filament.detection.model import build_maskrcnn, predict_instances

    torch.manual_seed(0)
    model = build_maskrcnn(pretrained=False, size=256)
    x = np.random.default_rng(0).normal(size=(128, 128)).astype(np.float32)
    inst = predict_instances(model, x, torch.device("cpu"), scale=2.0)  # runs on a 256 px input
    assert len(inst) == 0 or (inst.boxes[:, 2:] <= 128 + 1e-3).all()


def test_maskrcnn_trains_and_predicts_raw_instances():
    from filament.detection.model import build_maskrcnn, predict_instances, set_input_size

    torch.manual_seed(0)
    model = build_maskrcnn(pretrained=False, size=256)
    masks = torch.zeros(1, 256, 256, dtype=torch.uint8)
    masks[0, 100:120, 40:200] = 1
    target = {"boxes": torch.tensor([[40.0, 100.0, 200.0, 120.0]]), "labels": torch.ones(1, dtype=torch.int64),
              "masks": masks}
    model.train()
    set_input_size(model, 256)
    losses = model([torch.randn(3, 256, 256)], [target])
    assert {"loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"} <= set(losses)
    assert all(torch.isfinite(v) for v in losses.values())

    inst = predict_instances(model, np.random.default_rng(0).normal(size=(256, 256)).astype(np.float32),
                             torch.device("cpu"))
    assert inst.boxes.shape == (len(inst), 4) and inst.masks.shape[1:] == (28, 28)
    assert inst.masks.dtype == np.float16 and (inst.boxes[:, 2:] <= 256 + 1e-3).all()


def test_mask2former_queries_become_instances_on_their_masks():
    from filament.detection.instances import paste_mask
    from filament.detection.mask2former import queries_to_instances

    logits = torch.full((3, 64, 64), -10.0)  # an image of 256 px, masks at 1/4 resolution
    logits[0, 10:14, 5:45] = 10  # a filament: rows 40-55, columns 20-179 at full resolution
    logits[1, 30:40, 30:40] = 10  # confident mask, but "no object"
    classes = torch.tensor([[5.0, -5.0], [-5.0, 5.0], [5.0, -5.0]])  # query 2: no mask at all
    inst = queries_to_instances(classes, logits, (256, 256), mask_size=64)
    assert len(inst) == 1 and inst.masks.shape == (1, 64, 64) and inst.scores[0] > 0.99
    np.testing.assert_allclose(inst.boxes[0], [16, 36, 184, 60])  # the mask's box + one low-resolution pixel
    y0, x0, patch = paste_mask(inst.masks[0], inst.boxes[0], 0.5, (256, 256))
    full = np.zeros((256, 256), bool)
    full[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]] = patch
    want = np.zeros((256, 256), bool)
    want[40:56, 20:180] = True
    assert (full & want).sum() / (full | want).sum() > 0.85


def test_mask2former_trains_on_empty_and_filled_crops_and_predicts():
    pytest.importorskip("transformers")
    from filament.detection.mask2former import Mask2FormerDetector
    from filament.detection.model import backbone_body, predict_instances

    torch.manual_seed(0)
    model = Mask2FormerDetector(checkpoint=None, pretrained=False, mask_size=32)  # random Swin-T, no download
    masks = torch.zeros(1, 128, 128, dtype=torch.uint8)
    masks[0, 50:58, 20:100] = 1
    filled = {"labels": torch.ones(1, dtype=torch.int64), "masks": masks}
    empty = {"labels": torch.ones(0, dtype=torch.int64), "masks": torch.zeros(0, 128, 128, dtype=torch.uint8)}
    model.train()
    losses = model([torch.randn(3, 128, 128), torch.randn(3, 128, 128)], [filled, empty])
    assert {"loss_cross_entropy", "loss_mask", "loss_dice", "loss_aux"} == set(losses)
    sum(losses.values()).backward()
    assert all(torch.isfinite(v) for v in losses.values())
    assert any(p.grad is not None for p in backbone_body(model).parameters())

    inst = predict_instances(model, np.random.default_rng(0).normal(size=(64, 64)).astype(np.float32),
                             torch.device("cpu"), scale=2.0)  # runs on a 128 px input
    assert inst.masks.shape[1:] == (32, 32) and inst.masks.dtype == np.float16
    assert len(inst) == 0 or (inst.boxes[:, 2:] <= 64 + 1e-3).all()


def _fake_detector(view):
    """Detects the bright pixels of a (view of an) image as one instance, as Mask R-CNN would: box + 28 x 28 mask."""
    from filament.detection.instances import Instances
    import cv2 as _cv2
    ys, xs = np.nonzero(view > 0.5)
    box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
    crop = view[int(box[1]):int(box[3]), int(box[0]):int(box[2])].astype(np.float32)
    mask = _cv2.resize(crop, (28, 28), interpolation=_cv2.INTER_AREA)
    return Instances(box[None], np.array([0.9], np.float32), mask[None].astype(np.float16))


@pytest.mark.parametrize("n_views", [1, 2, 4, 8])
def test_tta_views_map_back_to_the_original_image(n_views):
    from filament.detection.tta import VIEWS, predict_tta, unview_instances, view_image
    x = np.zeros((96, 128), np.float32)
    x[10:30, 20:80] = 1  # an asymmetric "L": a bar and a leg
    x[30:70, 20:35] = 1
    ref = _fake_detector(x)
    for v in VIEWS[n_views]:
        back = unview_instances(_fake_detector(view_image(x, v)), v, x.shape)
        np.testing.assert_allclose(back.boxes, ref.boxes, atol=1e-4)
        np.testing.assert_allclose(back.masks.astype(np.float32), ref.masks.astype(np.float32), atol=2e-3)
    merged = predict_tta(_fake_detector, x, n_views)
    assert len(merged) == 1 and merged.scores[0] == pytest.approx(0.9, abs=1e-6)
    np.testing.assert_allclose(merged.boxes, ref.boxes, atol=1e-3)
    np.testing.assert_allclose(merged.masks.astype(np.float32), ref.masks.astype(np.float32), atol=2e-2)


def test_tta_scores_down_detections_missing_from_some_views():
    from filament.detection.instances import Instances
    from filament.detection.tta import merge_views
    a = _fake_detector(np.pad(np.ones((20, 40), np.float32), 20))
    far = Instances(a.boxes + 500, a.scores.copy(), a.masks.copy())
    merged = merge_views([a, a, far, Instances.empty()])
    assert sorted(np.round(merged.scores, 4)) == [0.225, 0.45]
