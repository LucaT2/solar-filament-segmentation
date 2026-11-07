"""Cascade Mask R-CNN heads for torchvision (Cai & Vasconcelos, "Cascade R-CNN", CVPR 2018 / TPAMI 2019).

The box head runs in several stages. Each stage refines the boxes of the previous one and is trained with a stricter
IoU threshold for "this proposal is a filament" (0.5, 0.6, 0.7), so later stages learn to produce tight boxes and
well-ranked scores. Following the paper and detectron2's ``CascadeROIHeads``:

* stage t has its own box head and predictor, matcher (IoU threshold) and box-regression weights; its loss is scaled
  by ``stage_weights[t]``; boxes passed to the next stage are detached (no gradient between stages)
* stage 0 samples proposals (512 per image) as in Mask R-CNN; later stages relabel the refined boxes, no resampling
* the mask head is trained on the positive stage-0 proposals and, at inference, runs on the final boxes
* final score = mean of the stages' class probabilities; final box = last stage's refinement

Every stage's box head starts from the COCO-pretrained Mask R-CNN box head; the predictors start fresh.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss, maskrcnn_inference
from torchvision.ops import boxes as box_ops

from filament.detection.losses import mask_loss

IOUS = (0.5, 0.6, 0.7)
REG_WEIGHTS = ((10.0, 10.0, 5.0, 5.0), (20.0, 20.0, 10.0, 10.0), (30.0, 30.0, 15.0, 15.0))
STAGE_WEIGHTS = (1.0, 0.5, 0.25)


class CascadeRoIHeads(RoIHeads):
    def __init__(self, base: RoIHeads, ious=IOUS, reg_weights=REG_WEIGHTS, stage_weights=STAGE_WEIGHTS,
                 dice_weight: float = 0.0):
        super().__init__(
            box_roi_pool=base.box_roi_pool, box_head=base.box_head, box_predictor=base.box_predictor,
            fg_iou_thresh=ious[0], bg_iou_thresh=ious[0],
            batch_size_per_image=base.fg_bg_sampler.batch_size_per_image,
            positive_fraction=base.fg_bg_sampler.positive_fraction, bbox_reg_weights=reg_weights[0],
            score_thresh=base.score_thresh, nms_thresh=base.nms_thresh, detections_per_img=base.detections_per_img,
            mask_roi_pool=base.mask_roi_pool, mask_head=base.mask_head, mask_predictor=base.mask_predictor,
        )
        n = len(ious)
        self.box_heads = nn.ModuleList([base.box_head] + [copy.deepcopy(base.box_head) for _ in range(n - 1)])
        self.box_predictors = nn.ModuleList(
            [base.box_predictor] + [copy.deepcopy(base.box_predictor) for _ in range(n - 1)])
        for p in self.box_predictors[1:]:  # fresh predictors for the later stages
            for m in p.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01 if m.out_features == 2 else 0.001)
                    nn.init.zeros_(m.bias)
        del self.box_head, self.box_predictor  # replaced by the per-stage lists
        self.matchers = [det_utils.Matcher(t, t, allow_low_quality_matches=False) for t in ious]
        self.coders = [det_utils.BoxCoder(w) for w in reg_weights]
        self.stage_weights = tuple(stage_weights)
        self.dice_weight = dice_weight  # optional Dice term in the mask loss (filament.detection.losses)

    @property
    def num_stages(self) -> int:
        return len(self.box_heads)

    def _stage(self, t, features, proposals, image_shapes):
        x = self.box_roi_pool(features, proposals, image_shapes)
        return self.box_predictors[t](self.box_heads[t](x))

    def _refine(self, t, deltas, proposals, image_shapes) -> list[torch.Tensor]:
        """Stage t's refined foreground boxes (float32, clipped to the image), one tensor per image."""
        counts = [len(p) for p in proposals]
        boxes = self.coders[t].decode(deltas.float(), [p.float() for p in proposals])[:, 1]  # class 1 = filament
        return [box_ops.clip_boxes_to_image(b, s) for b, s in zip(boxes.split(counts), image_shapes)]

    def _label(self, t, proposals, gt_boxes, gt_labels):
        """Labels and box-regression targets of stage t (IoU threshold IOUS[t]) for the given boxes."""
        labels, targets = [], []
        for props, boxes, lab in zip(proposals, gt_boxes, gt_labels):
            if boxes.numel() == 0:
                labels.append(torch.zeros(len(props), dtype=torch.int64, device=props.device))
                matched = torch.zeros((len(props), 4), dtype=props.dtype, device=props.device)
            else:
                idx = self.matchers[t](box_ops.box_iou(boxes, props))
                labels.append(torch.where(idx >= 0, lab[idx.clamp(min=0)], torch.zeros_like(idx)).to(torch.int64))
                matched = boxes[idx.clamp(min=0)]
            targets.append(matched)
        return labels, self.coders[t].encode(targets, proposals)

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            # stage 0: sample proposals exactly as Mask R-CNN does (IoU 0.5, 25% positives, ground-truth boxes added)
            proposals, matched_idxs, labels, reg_targets = self.select_training_samples(proposals, targets)
            mask_props, mask_idxs = [], []
            for p, m, lab in zip(proposals, matched_idxs, labels):
                pos = torch.where(lab > 0)[0]
                mask_props.append(p[pos])
                mask_idxs.append(m[pos])
            gt_boxes = [t["boxes"].to(proposals[0].dtype) for t in targets]
            gt_labels = [t["labels"] for t in targets]
            losses, refined = {}, proposals
            for t in range(self.num_stages):
                if t > 0:
                    proposals = [b.detach() for b in refined]
                    labels, reg_targets = self._label(t, proposals, gt_boxes, gt_labels)
                logits, deltas = self._stage(t, features, proposals, image_shapes)
                loss_cls, loss_box = fastrcnn_loss(logits.float(), deltas.float(), labels, reg_targets)
                losses[f"loss_classifier_s{t}"] = loss_cls * self.stage_weights[t]
                losses[f"loss_box_reg_s{t}"] = loss_box * self.stage_weights[t]
                if t < self.num_stages - 1:
                    refined = self._refine(t, deltas, proposals, image_shapes)
            mask_logits = self.mask_predictor(self.mask_head(self.mask_roi_pool(features, mask_props, image_shapes)))
            losses["loss_mask"] = mask_loss(mask_logits, mask_props, [t["masks"] for t in targets], gt_labels,
                                            mask_idxs, dice_weight=self.dice_weight)
            return [], losses

        # inference: run every stage on the previous stage's boxes, average the class probabilities
        probs = []
        for t in range(self.num_stages):
            logits, deltas = self._stage(t, features, proposals, image_shapes)
            probs.append(F.softmax(logits.float(), -1))
            proposals = self._refine(t, deltas, proposals, image_shapes)
        scores = torch.stack(probs).mean(0)[:, 1].split([len(p) for p in proposals])
        result = []
        for boxes, sc in zip(proposals, scores):
            keep = torch.where(sc > self.score_thresh)[0]
            boxes, sc = boxes[keep], sc[keep]
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, sc = boxes[keep], sc[keep]
            keep = box_ops.nms(boxes, sc, self.nms_thresh)[: self.detections_per_img]
            result.append({"boxes": boxes[keep], "scores": sc[keep],
                           "labels": torch.ones(len(keep), dtype=torch.int64, device=boxes.device)})
        mask_logits = self.mask_predictor(self.mask_head(
            self.mask_roi_pool(features, [r["boxes"] for r in result], image_shapes)))
        for r, m in zip(result, maskrcnn_inference(mask_logits, [r["labels"] for r in result])):
            r["masks"] = m
        return result, {}
