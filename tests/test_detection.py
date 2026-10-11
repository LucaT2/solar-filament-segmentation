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
