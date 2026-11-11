"""Error analysis of out-of-fold predictions.

Two tables are built from cached probability maps:

* ``gt``   one row per ground-truth filament (per annotator-image): how the prediction treated it
           (match / low_iou / missing / split / merged, same categories as the human-vs-human analysis), plus
           size, category, position on the disk, and the model's probability inside it.
* ``pred`` one row per predicted component (thresholded, *before* the size/confidence filter): whether it
           matches a filament of at least one annotator, its features (area, mean/max probability, shape,
           position, contrast), and whether post-processing keeps it.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from filament.data.images import find_disk, read_image
from filament.evaluation.scoring import GroundTruth
from filament.postprocess.components import PostConfig, find_components
from filament.postprocess.maps import load_prob
from filament.postprocess.rescoring import component_table, feature_matrix, oof_probabilities

OUTCOMES = ["match", "low_iou", "missing", "split", "merged"]


def analyze_image(stem: str, prob: np.ndarray, img: np.ndarray, gt: GroundTruth, cfg: PostConfig,
                  ann_meta: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    disk = find_disk(img)
    comps, table = component_table(stem, prob, img, cfg, disk)  # same features as the production rescorer
    n = comps.n
    kept = set(table.comp[table.kept].tolist())
    flat = comps.labels.ravel()
    pflat = prob.ravel()

    gt_rows: list[dict] = []
    best_iou_any = np.zeros(n + 1)  # best IoU of each component with any annotator's filament
    overlaps_any = np.zeros(n + 1, dtype=bool)
    matched_by = np.zeros(n + 1, dtype=int)  # number of annotators for whom the component is a TP
    n_ann = len(gt.by_stem[stem])
    kept_ids = np.array(sorted(kept), dtype=int)

    for aid, filaments in gt.by_stem[stem].items():
        # IoU against *kept* components (what the metric sees) and against all components (diagnostics)
        inter_all = np.zeros((len(filaments), n + 1))
        for i, idx in enumerate(filaments):
            inter_all[i] = np.bincount(flat[idx], minlength=n + 1)
        areas = np.array([len(idx) for idx in filaments], dtype=float)[:, None]
        iou_all = inter_all / (areas + comps.area[None, :] - inter_all)
        iou_all[:, 0] = 0.0
        best_iou_any = np.maximum(best_iou_any, iou_all.max(axis=0))
        overlaps_any |= (inter_all > 0).any(axis=0)
        matched_by += (iou_all > 0.5).any(axis=0)

        iou_kept = iou_all[:, kept_ids] if len(kept_ids) else np.zeros((len(filaments), 0))
        for i, idx in enumerate(filaments):
            row_iou = iou_kept[i]
            n_overlap = int((inter_all[i, kept_ids] > 0).sum()) if len(kept_ids) else 0
            best = float(row_iou.max()) if row_iou.size else 0.0
            partner_degree = 0
            if best > 0:
                partner = kept_ids[int(row_iou.argmax())]
                partner_degree = int((inter_all[:, partner] > 0).sum())
            outcome = ("match" if best > 0.5 else "missing" if n_overlap == 0 else "split" if n_overlap > 1
                       else "merged" if partner_degree > 1 else "low_iou")
            meta = ann_meta.loc[(aid, i)]
            ys, xs = np.unravel_index(idx, prob.shape)
            gt_rows.append({
                "stem": stem, "annotator_image": aid, "n_annotators": n_ann, "category": meta.category,
                "area": len(idx), "best_iou": best, "n_overlap": n_overlap, "outcome": outcome,
                "prob_max": float(pflat[idx].max()), "prob_mean": float(pflat[idx].mean()),
                "covered_by_any_component": bool((flat[idx] > 0).any()),
                "r_norm": float(np.hypot(xs.mean() - disk.cx, ys.mean() - disk.cy) / disk.radius),
                "y0": int(ys.min()), "y1": int(ys.max()) + 1, "x0": int(xs.min()), "x1": int(xs.max()) + 1,
            })

    table = table.drop(columns=["site"], errors="ignore").assign(
        n_annotators=n_ann,
        best_iou_any=best_iou_any[1:],
        matched_any=matched_by[1:] > 0,
        matched_share=matched_by[1:] / n_ann,
        overlaps_any=overlaps_any[1:],
    )
    return gt_rows, table.to_dict("records")


def build_tables(prob_dir, image_dir, ann: pd.DataFrame, stems, cfg: PostConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt = GroundTruth(ann, stems)
    sub = ann[ann.stem.isin(set(stems))].copy()
    sub["k"] = sub.groupby("annotator_image").cumcount()
    ann_meta = sub.set_index(["annotator_image", "k"])[["category"]]
    gt_rows, pred_rows = [], []
    for stem in gt.stems:
        g, p = analyze_image(stem, load_prob(prob_dir, stem), read_image(f"{image_dir}/{stem}.jpeg"), gt, cfg,
                             ann_meta)
        gt_rows += g
        pred_rows += p
    gt_df, pred_df = pd.DataFrame(gt_rows), pd.DataFrame(pred_rows)
    for df in (gt_df, pred_df):
        df["site"] = df.stem.str[14]
    return gt_df, pred_df


def pq_variant(prob_dir, image_dir, gt: GroundTruth, cfg: PostConfig, morph_px: int = 0,
               keep_fn=None, use_size_filter: bool = True):
    """Exact PQ of a modified post-processing.

    ``morph_px``  grow (> 0) or shrink (< 0) every kept instance by that many pixels (outline-width test)
    ``keep_fn``   ``keep_fn(stem, comp_ids) -> boolean array``: an extra per-component filter (rescoring test)
    """
    from filament.evaluation.metric import score_images
    from filament.evaluation.scoring import score_components
    from filament.postprocess.components import Components, region_for

    images, disks = [], {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(morph_px) + 1,) * 2) if morph_px else None
    for stem in gt.stems:
        prob = load_prob(prob_dir, stem)
        comps = find_components(prob, cfg.threshold, region_for(stem, image_dir, cfg.disk_scale, disks), cfg.link_px)
        keep = comps.keep(cfg.min_area if use_size_filter else 0, cfg.min_score)
        if keep_fn is not None and len(keep):
            keep = keep[np.asarray(keep_fn(stem, keep), dtype=bool)]
        if morph_px:
            lab = np.where(np.isin(comps.labels, keep), comps.labels, 0).astype(np.float32)
            if morph_px > 0:  # grey dilation spreads each label into the background around it
                lab = cv2.dilate(lab, kernel)
            else:
                lab *= cv2.erode((lab > 0).astype(np.uint8), kernel)
            lab = lab.astype(np.int32)
            area = np.bincount(lab.ravel(), minlength=comps.n + 1)
            comps = Components(lab, area, comps.mean_prob)
            keep = keep[area[keep] > 0]
        images += score_components(stem, comps, keep, gt)
    return score_images(images)


component_features = feature_matrix  # kept for the notebook


def rescoring_oof(pred: pd.DataFrame, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold P(component matches at least one annotator), grouped by image."""
    return oof_probabilities(pred, pred.matched_any.astype(int).to_numpy(), pred.stem.to_numpy(), n_splits)


def pred_category(pred: pd.DataFrame) -> pd.Series:
    """Kept components: TP for all annotators / TP for some (annotators disagree) / overlaps but no match /
    pure false positive (touches no annotator's filament)."""
    return np.select(
        [pred.matched_share >= 1, pred.matched_any, pred.overlaps_any],
        ["match_all", "match_some", "overlap_no_match"],
        default="pure_fp",
    )
