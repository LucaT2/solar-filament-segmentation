"""Draw a training image with every annotated filament colored as a separate instance.

Usage:
    python scripts/show_example.py                       # default example image
    python scripts/show_example.py --stem 20111116063134Lh --out outputs/example.png
    python scripts/show_example.py --random 3            # three random images

Figure 1 (per image): raw full disk | colored instances | zoom raw | zoom with instances (+ category labels).
Figure 2 (multi-annotated images only): the same zoom for every annotator side by side.
"""

from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from filament.data import annotation_table, annotator_image_table, load_coco
from filament.eda import read_gray
from filament.paths import get_data_paths, get_output_dir
from filament.rle import rle_to_mask

CATEGORY_SHORT = {"Left": "L", "Right": "R", "Unidentifiable": "U", "Ambiguous": "A"}


def instance_colors(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hues = (np.arange(n) * 0.618034 + rng.random()) % 1.0  # golden-ratio spacing -> distinct neighbours
    hsv = np.stack([hues * 179, np.full(n, 230), np.full(n, 255)], axis=1).astype(np.uint8)[None]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0]


def overlay(img: np.ndarray, masks: list[np.ndarray], alpha: float = 0.45, thickness: int = 2) -> np.ndarray:
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32)
    for mask, color in zip(masks, instance_colors(len(masks))):
        vis[mask] = (1 - alpha) * vis[mask] + alpha * color
    vis = vis.astype(np.uint8)
    for mask, color in zip(masks, instance_colors(len(masks))):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, contours, -1, tuple(int(c) for c in color), thickness)
    return vis


def busiest_window(masks: list[np.ndarray], size: int) -> tuple[int, int]:
    """Top-left corner of the size x size window containing the most filament pixels."""
    union = np.any(masks, axis=0).astype(np.float32)
    small = cv2.resize(union, (256, 256), interpolation=cv2.INTER_AREA)
    k = max(1, size * 256 // union.shape[0])
    score = cv2.boxFilter(small, -1, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    scale = union.shape[0] / 256
    y0 = int(np.clip(cy * scale - size / 2, 0, union.shape[0] - size))
    x0 = int(np.clip(cx * scale - size / 2, 0, union.shape[1] - size))
    return y0, x0


def plot_image(stem, ann, ai, image_dir, out_dir, zoom: int = 512):
    img = read_gray(image_dir / f"{stem}.jpeg")
    annotators = ai[ai.stem == stem].annotator_image.tolist()
    first = ann[ann.annotator_image == annotators[0]]
    masks = [rle_to_mask(r) for r in first.rle]
    y0, x0 = busiest_window(masks, zoom)
    sl = (slice(y0, y0 + zoom), slice(x0, x0 + zoom))
    vis = overlay(img, masks, thickness=4)
    vis_zoom = overlay(img[sl], [m[sl] for m in masks])

    fig, axes = plt.subplots(1, 4, figsize=(24, 6.6))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"Input: {stem}.jpeg  (2048×2048, grayscale)")
    axes[1].imshow(vis)
    axes[1].set_title(f"Target: {len(masks)} filament instances ({annotators[0]})")
    axes[1].add_patch(plt.Rectangle((x0, y0), zoom, zoom, fill=False, ec="yellow", lw=1.5))
    axes[2].imshow(img[sl], cmap="gray", vmin=img.min(), vmax=img.max())
    axes[2].set_title(f"Zoom {zoom}×{zoom} px at full resolution")
    axes[3].imshow(vis_zoom)
    axes[3].set_title("Zoom with instances (L/R = chirality, U = unidentifiable)")
    for mask, cat in zip(masks, first.category):
        ys, xs = np.nonzero(mask[sl])
        if len(xs):
            axes[3].text(xs.mean(), ys.mean(), CATEGORY_SHORT.get(cat, "?"), color="white", fontsize=11,
                         ha="center", va="center", weight="bold",
                         bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5, lw=0))
    for a in axes:
        a.axis("off")
    plt.tight_layout()
    path1 = out_dir / f"{stem}_instances.png"
    fig.savefig(path1, dpi=100, bbox_inches="tight")
    plt.close(fig)
    paths = [path1]

    if len(annotators) > 1:
        fig, axes = plt.subplots(1, len(annotators) + 1, figsize=(6.5 * (len(annotators) + 1), 6.8))
        axes[0].imshow(img[sl], cmap="gray")
        axes[0].set_title("Zoom (raw)")
        for ax, aid in zip(axes[1:], annotators):
            g = ann[ann.annotator_image == aid]
            ms = [rle_to_mask(r)[sl] for r in g.rle]
            ax.imshow(overlay(img[sl], [m for m in ms if m.any()]))
            ax.set_title(f"annotator {aid.split('-')[0]}: {len(g)} filaments on the whole disk")
        for a in axes:
            a.axis("off")
        plt.tight_layout()
        fig.suptitle(f"{stem}: the same region drawn independently by {len(annotators)} annotators", fontsize=14, y=1.03)
        path2 = out_dir / f"{stem}_annotators.png"
        fig.savefig(path2, dpi=100, bbox_inches="tight")
        plt.close(fig)
        paths.append(path2)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stem", default="20111116063134Lh")
    parser.add_argument("--random", type=int, default=0, help="plot N random training images instead")
    parser.add_argument("--zoom", type=int, default=512)
    parser.add_argument("--out", default=None, help="output directory (default: outputs/examples)")
    args = parser.parse_args()

    paths = get_data_paths()
    coco = load_coco(paths.annotations)
    ann, ai = annotation_table(coco), annotator_image_table(coco)
    out_dir = get_output_dir("examples") if args.out is None else get_output_dir(args.out)
    stems = ai.stem.drop_duplicates().sample(args.random).tolist() if args.random else [args.stem]
    for stem in stems:
        for p in plot_image(stem, ann, ai, paths.train_images, out_dir, args.zoom):
            print(p)


if __name__ == "__main__":
    main()
