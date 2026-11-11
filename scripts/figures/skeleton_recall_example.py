"""Figure for the report: how the Skeleton Recall loss sees a filament.

Image crop, one annotator's mask (BCE + Dice target), its tubed skeleton (Skeleton Recall target), and an OOF
probability map with the skeleton overlaid. Uses the largest filament of a fold-0 image.

Usage:
    python scripts/figures/skeleton_recall_example.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mu

from filament.data.images import read_image
from filament.paths import get_data_paths
from filament.postprocess.maps import load_prob
from filament.segmentation.dataset import tubed_skeleton

argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

p = get_data_paths()
coco = json.load(open(p.annotations))
oof = Path("outputs/unet_convnext_t_skel050/oof")
stems = {f.stem for f in oof.glob("*.png")}
by_img = {}
for im in coco["images"]:
    stem = Path(im["file_name"]).stem
    if stem in stems:
        by_img.setdefault(im["id"], (stem, im["height"], im["width"]))
# pick the largest filament on a fold-0 image
best = max((a for a in coco["annotations"] if a["image_id"] in by_img), key=lambda a: a["area"])
stem, h, w = by_img[best["image_id"]]
mask = np.zeros((h, w), np.uint8)
for a in coco["annotations"]:
    if a["image_id"] == best["image_id"]:
        mask |= mu.decode(mu.merge(mu.frPyObjects(a["segmentation"], h, w)))
x, y, bw, bh = map(int, best["bbox"])
cx, cy, s = x + bw // 2, y + bh // 2, max(bw, bh) // 2 + 40
sl = (slice(max(cy - s, 0), cy + s), slice(max(cx - s, 0), cx + s))
img = read_image(p.train_images / f"{stem}.jpeg")[sl]
m = mask[sl]
sk = tubed_skeleton(m, 1)
prob = load_prob(oof, stem)[sl]
rec = (prob * sk).sum() / sk.sum()

fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
ax[0].imshow(img, cmap="gray"); ax[0].set_title(f"H-alpha crop ({stem})")
ax[1].imshow(img, cmap="gray"); ax[1].imshow(np.ma.masked_equal(m, 0), cmap="autumn", alpha=0.45)
ax[1].set_title("annotator mask (BCE + Dice target)")
ax[2].imshow(m, cmap="gray", vmax=3); ax[2].imshow(np.ma.masked_equal(sk, 0), cmap="cool", vmin=0, vmax=1)
ax[2].set_title("tubed skeleton (Skeleton Recall target)")
ax[3].imshow(prob, cmap="magma", vmin=0, vmax=1); ax[3].contour(sk, levels=[0.5], colors="cyan", linewidths=0.6)
ax[3].set_title(f"model probability; recall on skeleton = {rec:.2f}")
for a in ax:
    a.axis("off")
plt.tight_layout()
out = Path("outputs/figures/skeleton_recall_example.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=110)
print(out, stem, f"recall={rec:.3f}", m.sum(), sk.sum())

