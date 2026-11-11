"""Figures and 95% intervals for the technical report.

* pipeline.pdf   the submitted pipeline on one test image, stage by stage (runs the model: needs the checkpoint from
                 scripts/final/download_weights.py; a GPU helps)
* examples.pdf   three held-out fold-0 images: the annotators' outlines and the fold-0 model's, which never saw them
                 (needs outputs/mrcnn_dino_f0_tta8/oof_submission.csv: the fold-0 model's out-of-fold predictions)
* metrics.pdf    PQ by site, IoU of overlapping pairs, pixel Dice per annotator-image and found rate by filament size,
                 model vs human agreement, with 95% intervals
* intervals.csv  the numbers of the report's table, with 95% intervals

Intervals: PQ, SQ, RQ, precision, recall and pixel Dice resample *images* (with all their annotator-images) 2,000
times; per-filament rates (found, 1:n, n:1) use Wilson intervals. The metric inputs are in results/dino_fold0/
(scripts/evaluation/rubric_metrics.py --exp mrcnn_dino_f0_tta8 --human).

Usage:
    python scripts/figures/report_figures.py --out outputs/report_figures
    python scripts/figures/report_figures.py --only metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

from filament.data.annotations import SITE_NAMES, annotation_table, load_coco
from filament.data.images import read_image
from filament.data.rle import counts_to_mask, rle_to_mask
from filament.paths import REPO_ROOT, get_data_paths, resolve_exp_dir

RESULTS = REPO_ROOT / "results" / "dino_fold0"
MODEL, HUMAN = "mrcnn_dino_f0_tta8", "human"
PIPELINE_IMAGE = "20140420195854Bh"  # the walk-through notebook's example
SIZE_BINS, SIZE_NAMES = [0, 800, 3200, np.inf], ["small\n<800", "medium\n800-3200", "large\n>3200"]

# reference categorical palette, slots 1-4 in order (model first); text stays in ink colours
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#d9d8d2"
COLORS = {MODEL: BLUE, HUMAN: ORANGE}
LABELS = {MODEL: "model (fold 0, held out)", HUMAN: "human vs human"}

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
    "savefig.dpi": 300,
})


# ---------------------------------------------------------------- intervals

def per_stem(images: pd.DataFrame) -> pd.DataFrame:
    """Per-image sums of the per-annotator-image counts, so whole images can be resampled."""
    images = images.assign(stem=images.annotator_image.str.split("-", n=1).str[1])
    return images.groupby("stem").agg(tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
                                      iou=("iou_sum", "sum"), dice=("pixel_dice", "sum"),
                                      n=("pixel_dice", "size")).reset_index()


def stats(w: np.ndarray, s: pd.DataFrame) -> dict[str, np.ndarray]:
    """The image-level metrics for weight vectors w (one row per bootstrap sample, one column per image)."""
    tp, fp, fn, iou, dice, n = (w @ s[c].to_numpy(float) for c in ("tp", "fp", "fn", "iou", "dice", "n"))
    return {"PQ": iou / (tp + 0.5 * fp + 0.5 * fn), "SQ": iou / tp, "RQ": tp / (tp + 0.5 * fp + 0.5 * fn),
            "precision": tp / (tp + fp), "recall": tp / (tp + fn), "pixel Dice (mean)": dice / n}


def bootstrap(s: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    k = len(s)
    w = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(n_boot)]).astype(float)
    point, boot = stats(np.ones((1, k)), s), stats(w, s)
    return pd.DataFrame([{"metric": m, "value": point[m][0], "lo": np.percentile(boot[m], 2.5),
                          "hi": np.percentile(boot[m], 97.5)} for m in point])


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float, float]:
    p = k / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (
        1 + z * z / n)
    return p, centre - half, centre + half


def load_results():
    images = {e: pd.read_csv(RESULTS / f"{e}_images.csv") for e in (MODEL, HUMAN)}
    arrays = {e: np.load(RESULTS / f"{e}.npz") for e in (MODEL, HUMAN)}
    return images, arrays


def intervals(images, arrays) -> pd.DataFrame:
    rows = []
    for e in (MODEL, HUMAN):
        rows.append(bootstrap(per_stem(images[e])).assign(experiment=e))
        a = arrays[e]
        pair = 2 * a["overlap_iou"] / (1 + a["overlap_iou"])
        extra = [("median pair Dice", np.median(pair), np.nan, np.nan)]
        for name, k, n in [("GT missed", (a["gt_degree"] == 0).sum(), len(a["gt_degree"])),
                           ("1:n (GT split)", (a["gt_degree"] > 1).sum(), len(a["gt_degree"])),
                           ("n:1 (predictions merged)", (a["pred_degree"] > 1).sum(), len(a["pred_degree"]))]:
            extra.append((name, *wilson(k, n)))
        rows.append(pd.DataFrame(extra, columns=["metric", "value", "lo", "hi"]).assign(experiment=e))
    out = pd.concat(rows, ignore_index=True)
    return out.pivot(index="metric", columns="experiment", values=["value", "lo", "hi"]).swaplevel(axis=1).sort_index(
        axis=1)


# ---------------------------------------------------------------- metrics figure

def site_pq(images: pd.DataFrame) -> pd.DataFrame:
    s = per_stem(images)
    s["site"] = s.stem.str[14]
    return pd.concat([bootstrap(g.reset_index(drop=True)).query("metric == 'PQ'").assign(site=site)
                      for site, g in s.groupby("site")], ignore_index=True)


def metrics_figure(images, arrays, out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.75), gridspec_kw={"wspace": 0.42})

    ax = axes[0]  # PQ by site
    sites = sorted(SITE_NAMES)
    for j, e in enumerate((MODEL, HUMAN)):
        d = site_pq(images[e]).set_index("site").loc[sites]
        x = np.arange(len(sites)) + (j - 0.5) * 0.3
        ax.errorbar(x, d.value, yerr=[d.value - d.lo, d.hi - d.value], fmt="o", ms=3.2, color=COLORS[e],
                    elinewidth=1.0, capsize=0, label=LABELS[e])
    ax.set_xticks(np.arange(len(sites)), sites)
    ax.set_ylim(0.15, 0.6)
    ax.set_ylabel("PQ")
    ax.set_title("(a) PQ by site", loc="left")
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)

    bins = np.linspace(0, 1, 21)
    for k, (title, key, vline) in enumerate([("(b) IoU, overlapping pairs", "overlap_iou", 0.5),
                                              ("(c) pixel Dice per image", "pixel_dice", None)]):
        ax = axes[1 + k]
        for e in (MODEL, HUMAN):
            v = arrays[e][key]
            ax.hist(v, bins=bins, density=True, histtype="step", lw=1.4, color=COLORS[e])
            ax.axvline(np.median(v), color=COLORS[e], lw=0.8, ls=(0, (2, 2)))
        if vline is not None:
            ax.axvline(vline, color=MUTED, lw=0.6)
            ax.text(vline - 0.02, ax.get_ylim()[1] * 0.95, "match\n(IoU > 0.5)", ha="right", va="top", fontsize=6,
                    color=MUTED)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("IoU" if key == "overlap_iou" else "Dice")
        ax.set_title(title, loc="left")

    ax = axes[3]  # found rate by size
    for j, e in enumerate((MODEL, HUMAN)):
        a = arrays[e]
        size = np.digitize(a["gt_area"], SIZE_BINS[1:-1])
        vals = [wilson((a["gt_best_iou"][size == b] > 0.5).sum(), int((size == b).sum())) for b in range(3)]
        p, lo, hi = map(np.array, zip(*vals))
        x = np.arange(3) + (j - 0.5) * 0.3
        ax.errorbar(x, p, yerr=[p - lo, hi - p], fmt="o", ms=3.2, color=COLORS[e], elinewidth=1.0, capsize=0)
    ax.set_xticks(range(3), SIZE_NAMES, fontsize=6)
    ax.set_ylim(0.3, 0.8)
    ax.set_ylabel("found (best IoU > 0.5)")
    ax.set_title("(d) found, by size (px)", loc="left")
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)

    handles = axes[0].get_legend_handles_labels()
    fig.legend(*handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.16))
    fig.savefig(out / "metrics.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------- image figures

def outline(ax, mask: np.ndarray, color: str, lw: float = 0.7, ls="-") -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        c = c[:, 0, :]
        ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color=color, lw=lw, ls=ls)


def best_window(weight: np.ndarray, size: int, stride: int = 32) -> tuple[int, int]:
    """Top-left corner of the size x size window with the most weight."""
    ii = np.pad(weight.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, corner = -1.0, (0, 0)
    for y in range(0, weight.shape[0] - size + 1, stride):
        for x in range(0, weight.shape[1] - size + 1, stride):
            s = ii[y + size, x + size] - ii[y, x + size] - ii[y + size, x] + ii[y, x]
            if s > best:
                best, corner = s, (y, x)
    return corner


def show(ax, img: np.ndarray) -> None:
    lo, hi = np.percentile(img, [1, 99.5])
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def pipeline_figure(out: Path) -> None:
    import torch

    from filament.detection import tta
    from filament.detection.dataset import full_image_input
    from filament.detection.instances import to_prob_map
    from filament.detection.model import predict_instances
    from filament.final import VIEWS, FinalPipeline
    from filament.segmentation.dataset import image_info

    pipe = FinalPipeline.load()
    path = get_data_paths().test_images / f"{PIPELINE_IMAGE}.jpeg"
    info, img = image_info(path), read_image(path)
    x = full_image_input(info)
    per_view = []
    for view in tta.VIEWS[VIEWS]:
        found = predict_instances(pipe.model, tta.view_image(x, view), pipe.device, torch.float16)
        per_view.append(tta.unview_instances(found, view, x.shape[-2:]))
    det = tta.merge_views(per_view)
    comps = pipe.components(det, info)
    keep = comps.keep(pipe.cfg.min_area, pipe.cfg.min_score)
    prob = to_prob_map(det)

    n = 600  # the crop shown in panels 2-5: the window with the most kept filament pixels
    y0, x0 = best_window(np.isin(comps.labels, keep), n)
    win = (slice(y0, y0 + n), slice(x0, x0 + n))
    crop = img[win]

    fig = plt.figure(figsize=(7.0, 1.55))
    h = 0.78  # panel height (figure fraction); panel widths follow from the aspect ratio
    aspect = 1.55 / 7.0
    w = h * aspect
    lefts = [0.0, w + 0.045, 3 * w + 0.09, 4 * w + 0.135, 5 * w + 0.18]
    bottom = 0.04
    titles = ["1  input (2048 x 2048)", "2  eight views (flips, transposes)", "3  Mask R-CNN, one view",
              "4  merged over 8 views", "5  pasted at 0.45, score ≥ 0.8"]

    ax = fig.add_axes([lefts[0], bottom, w, h])
    show(ax, img)
    ax.add_patch(Rectangle((x0, y0), n, n, fill=False, ec=YELLOW, lw=0.9))
    ax.set_title(titles[0], loc="left", pad=3)

    sub = 2 * w / 4
    for i, view in enumerate(tta.VIEWS[VIEWS]):
        t, fx, fy = view
        v = crop.T if t else crop
        v = v[:, ::-1] if fx else v
        v = v[::-1] if fy else v
        ax = fig.add_axes([lefts[1] + (i % 4) * sub, bottom + (1 - i // 4) * h / 2, sub * 0.96, h / 2 * 0.96])
        show(ax, v)
        if i == 0:
            ax.set_title(titles[1], loc="left", pad=3)

    ax = fig.add_axes([lefts[2], bottom, w, h])
    show(ax, crop)
    d0 = per_view[0]
    for (bx0, by0, bx1, by1), s in zip(d0.boxes, d0.scores, strict=True):
        if bx1 > x0 and bx0 < x0 + n and by1 > y0 and by0 < y0 + n:
            ax.add_patch(Rectangle((bx0 - x0, by0 - y0), bx1 - bx0, by1 - by0, fill=False, ec=BLUE,
                                   lw=0.4 + 0.8 * float(s), alpha=0.35 + 0.65 * float(s)))
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_title(titles[2], loc="left", pad=3)

    ax = fig.add_axes([lefts[3], bottom, w, h])
    ax.imshow(prob[win], cmap="magma", vmin=0, vmax=1, interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titles[3], loc="left", pad=3)

    ax = fig.add_axes([lefts[4], bottom, w, h])
    show(ax, crop)
    kept = set(keep.tolist())
    for i in range(1, comps.n + 1):
        m = comps.labels[win] == i
        if m.any():
            outline(ax, m, BLUE if i in kept else RED, lw=0.8, ls="-" if i in kept else (0, (2, 1.5)))
    ax.set_title(titles[4], loc="left", pad=3)

    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        xa = lefts[a] + (2 * w if a == 1 else w) + 0.004
        fig.patches.append(FancyArrowPatch((xa, bottom + h / 2), (lefts[b] - 0.004, bottom + h / 2),
                                           transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7,
                                           color=MUTED, lw=0.8))
    fig.savefig(out / "pipeline.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"pipeline: {len(det)} merged detections, {comps.n} instances on the disk, {len(keep)} kept")


def examples_figure(images, out: Path) -> None:
    sub = pd.read_csv(resolve_exp_dir(MODEL) / "oof_submission.csv", dtype=str, keep_default_na=False)
    sub["stem"] = sub.filament_id.str.rsplit("_", n=1).str[0]
    ann = annotation_table(load_coco(get_data_paths().annotations))

    s = per_stem(images[MODEL])
    s = s[(s.n == 3) & (s.tp + s.fn >= 3 * 6)]  # three annotators, at least ~6 filaments each
    s["pq"] = s.iou / (s.tp + 0.5 * s.fp + 0.5 * s.fn)
    s = s.sort_values("pq").reset_index(drop=True)
    picks = [s.iloc[int(q * (len(s) - 1))] for q in (0.85, 0.5, 0.15)]

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.55), gridspec_kw={"wspace": 0.03, "hspace": 0.08})
    n = 720
    for col, row in enumerate(picks):
        stem = row.stem
        img = read_image(get_data_paths().train_images / f"{stem}.jpeg")
        gt = ann[ann.stem == stem]
        by_annotator = [[rle_to_mask(r) for r in g.rle] for _, g in gt.groupby("annotator_image")]
        pred = [counts_to_mask(c) for c in sub[sub.stem == stem].segmentation_rle]
        weight = sum((np.any(m, axis=0) for m in by_annotator if m), np.zeros(img.shape, bool).astype(int))
        y0, x0 = best_window(weight, n)
        win = (slice(y0, y0 + n), slice(x0, x0 + n))
        for r, (ax, masks_sets) in enumerate(zip(axes[:, col], [by_annotator, [pred]], strict=True)):
            show(ax, img[win])
            for masks, color in zip(masks_sets, [ORANGE, AQUA, YELLOW] if r == 0 else [BLUE], strict=False):
                for m in masks:
                    if m[win].any():
                        outline(ax, m[win], color, lw=0.8)
        site = SITE_NAMES[stem[14]]
        axes[0, col].set_title(f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}, {site}", loc="left", pad=3)
        axes[1, col].text(0.02, 0.03, f"PQ {row.pq:.2f} against the 3 annotators", transform=axes[1, col].transAxes,
                          color="white", fontsize=6.5, va="bottom",
                          bbox={"fc": "black", "alpha": 0.55, "lw": 0, "pad": 1.5})
    axes[0, 0].set_ylabel("annotators", labelpad=3)
    axes[1, 0].set_ylabel("model", labelpad=3)
    fig.savefig(out / "examples.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("examples:", ", ".join(f"{r.stem} (PQ {r.pq:.3f})" for r in picks))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="outputs/report_figures")
    parser.add_argument("--only", nargs="*", choices=["metrics", "pipeline", "examples"])
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = set(args.only or ["metrics", "pipeline", "examples"])

    images, arrays = load_results()
    table = intervals(images, arrays)
    table.to_csv(out / "intervals.csv")
    with pd.option_context("display.width", 160, "display.precision", 4):
        print(table)
    if "metrics" in todo:
        metrics_figure(images, arrays, out)
    if "examples" in todo:
        examples_figure(images, out)
    if "pipeline" in todo:
        pipeline_figure(out)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
