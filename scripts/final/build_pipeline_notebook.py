"""Generate notebooks/solar_filament_pipeline.ipynb, the end-to-end walk-through of the submitted pipeline.

Edit the notebook here, then run it from the repo root so its outputs are saved:
    python scripts/final/build_pipeline_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/solar_filament_pipeline.ipynb \
        --ExecutePreprocessor.timeout=-1
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

CELLS = [
    ("md", """# Solar filament segmentation: the submitted pipeline

This notebook goes from raw GONG H-alpha images to the submitted CSV. It needs nothing beyond this repository and the
competition's images: the 15 checkpoints are downloaded from the repository's GitHub release.

| Section | What it does | Time |
|---|---|---|
| 1. Setup | finds or clones the repository | 1 min (5 with `INSTALL`) |
| 2. Data | the images, the annotations, why several annotators matter | 1 min |
| 3. Validation | folds and the official metric | instant |
| 4. Training | the three model configs and the training commands | off by default, ~40 GPU hours |
| 5. Weights | downloads and hash-checks the checkpoints | ~5 min |
| 6. One image | every stage of the pipeline on one test image | 1 min |
| 7. Test set | writes the submission CSV | ~77 min on a GPU |
| 8. Check | compares the result with the file submitted to Kaggle | 1 min |
| 9. Results | the out-of-fold scores and timing | instant |

**Environment:** Python 3.12 with `requirements.txt`, on a GPU with at least 4 GiB. On Kaggle, attach the competition
data, enable a GPU and set `INSTALL = True`. Elsewhere, set `FILAMENT_DATA` to the `MAGFiLO_1.0_Kaggle_2026` folder, or
extract the data into `filament-segmentation-2026/` in the repo root."""),

    ("md", "## 1. Setup"),
    ("code", """import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LucaT2/solar-filament-segmentation.git"
INSTALL = False        # True: install the pinned requirements (restart the kernel afterwards if torch changed)
RUN_TRAINING = False   # True: retrain all 15 models (~40 GPU hours) instead of downloading them
RUN_TEST_SET = True    # False: skip section 7 if its CSV does not exist yet

root = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src/filament/final.py").exists()), None)
if root is None:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
    root = Path.cwd() / "solar-filament-segmentation"
if INSTALL:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(root / "requirements.txt")], check=True)
sys.path.insert(0, str(root / "src"))


def run(script, *args):
    \"\"\"Run a repository script, with the library on its path.\"\"\"
    import os
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    subprocess.run([sys.executable, "-W", "ignore", str(root / script), *map(str, args)], check=True, cwd=root, env=env)


import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from filament.data.annotations import annotation_table, load_coco
from filament.data.folds import load_folds
from filament.data.images import read_image
from filament.data.rle import counts_to_mask, rle_to_mask
from filament.data.submission import read_submission, validate_submission
from filament.evaluation.metric import pq_score
from filament.final import DETECTOR_WEIGHT, UNET_WEIGHT, FINAL_CONFIG_DIR, FinalPipeline
from filament.paths import get_data_paths, get_output_dir
from filament.postprocess.pipeline import load_postprocess
from filament.postprocess.rescoring import component_table
from filament.segmentation.dataset import image_info

paths = get_data_paths()
print("repo:", root)
print("data:", paths.root)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (very slow)")"""),

    ("md", """## 2. Data

707 training and 180 test images of 2048×2048 pixels from the six GONG sites. Each annotation is one filament.

296 training images were annotated by two or three people independently, and the official metric scores each
annotator's version separately against the same prediction. Annotators often disagree on outlines and on whether a
filament is one object or several, so the score has a ceiling well below 1: **one annotator scored against the others
gets PQ 0.34**."""),
    ("code", """ann = annotation_table(load_coco(paths.annotations))
per_stem = ann.groupby("stem").annotator_image.nunique()
print(f"{len(ann)} filaments on {ann.stem.nunique()} images; {(per_stem > 1).sum()} images have several annotators")


def outline(ax, mask, color):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        ax.plot(*np.append(c[:, 0], c[:1, 0], axis=0).T, color=color, lw=0.8)


stem = per_stem[per_stem == 3].index[0]
img = read_image(paths.train_images / f"{stem}.jpeg")
versions = list(ann[ann.stem == stem].groupby("annotator_image"))
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
for ax, (name, rows), color in zip(axes, versions, ("tab:orange", "tab:cyan", "tab:green")):
    ax.imshow(img, cmap="gray")
    for r in rows.rle:
        outline(ax, rle_to_mask(r), color)
    ax.set_title(f"annotator {name.split('-')[0]}: {len(rows)} filaments", fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.show()"""),

    ("md", """## 3. Validation

The folds are grouped by image, so the same image never appears in training and validation under different
annotators, and stratified by site and number of annotators. Every model and setting was chosen on out-of-fold (OOF)
predictions, scored with a reimplementation of the official metric that the tests check against the organizers' code.
Two models are compared on the same images with a paired bootstrap, which resolves differences of about 0.01 PQ on one
fold. The public leaderboard shows 2 decimals on about 90 images, so it cannot."""),
    ("code", """folds = load_folds()
print(folds.groupby("fold").size().rename("images").to_frame().T.to_string())"""),

    ("md", """## 4. Training

Three model families, five folds each, so every training image has one out-of-fold prediction from a model that did
not train on it. Only the settings that differ between them are shown; the configs are complete."""),
    ("code", """CONFIGS = {
    "U-Net ResNet34": "configs/segmentation/unet_r34_skel050.yaml",
    "U-Net ConvNeXt-T": "configs/segmentation/unet_convnext_t_skel050.yaml",
    "Mask R-CNN R50-FPN v2": "configs/detection/maskrcnn/maskrcnn_r50_v2_m56_ftall.yaml",
}
KEYS = ["arch", "encoder", "crop", "batch_size", "epochs", "lr", "bce_weight", "dice_weight", "skel_weight",
        "mask_size", "trainable_layers", "samples_per_epoch"]
loaded = {name: yaml.safe_load((root / path).read_text()) for name, path in CONFIGS.items()}
pd.DataFrame({name: {k: c.get(k, "-") for k in KEYS} for name, c in loaded.items()})"""),
    ("code", """TRAINING = [
    ("scripts/segmentation/train.py", "--config", CONFIGS["U-Net ResNet34"], "--fold", 0, 1, 2, 3, 4),
    ("scripts/segmentation/train.py", "--config", CONFIGS["U-Net ConvNeXt-T"], "--fold", 0, 1, 2, 3, 4),
    ("scripts/detection/train.py", "--config", CONFIGS["Mask R-CNN R50-FPN v2"], "--fold", 0, 1, 2, 3, 4,
     "--set", "name=maskrcnn_ftall_5f"),
]
for command in TRAINING:
    print("python", *command)
    if RUN_TRAINING:
        run(*command)"""),

    ("md", """The README's *Retraining from scratch* section lists the remaining steps after training: OOF maps, map average,
tuning and the classifier. Their result is committed in `configs/final/`:"""),
    ("code", """cfg = load_postprocess(FINAL_CONFIG_DIR / "postprocess.json")
print(cfg)
print(pd.read_json(FINAL_CONFIG_DIR / "rescorer.json", typ="series").to_string())"""),

    ("md", "## 5. Download the checkpoints"),
    ("code", """if not RUN_TRAINING:
    run("scripts/final/download_weights.py")"""),

    ("md", """## 6. One image, stage by stage

The two ensembles make different mistakes: the U-Nets outline well, Mask R-CNN knows better what is one object. Their
weighted average is thresholded into connected components, and the classifier then removes the ones unlikely to match
an annotated filament."""),
    ("code", """pipeline = FinalPipeline.load()
print(f"{len(pipeline.unets)} U-Nets, {len(pipeline.detectors)} Mask R-CNNs on {pipeline.device}")

path = paths.test_images / "20140420195854Bh.jpeg"
info = image_info(path)
unet, detector = pipeline.maps(path, info)
prob = UNET_WEIGHT * unet + DETECTOR_WEIGHT * detector
img = read_image(path)
comps, table = component_table(path.stem, prob, img, pipeline.cfg, info.disk)
keep = set(pipeline.rescorer.keep_ids(table).tolist())

fig, axes = plt.subplots(2, 2, figsize=(11, 11))
for ax, (title, data) in zip(axes.flat, [("U-Net ensemble", unet), ("Mask R-CNN ensemble", detector),
                                         (f"average {UNET_WEIGHT} / {DETECTOR_WEIGHT}", prob)]):
    ax.imshow(data, cmap="magma", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
ax = axes[1, 1]
ax.imshow(img, cmap="gray")
ax.axis("off")
for row in table.itertuples():
    color = "lime" if row.comp in keep else "red" if row.kept else "yellow"
    outline(ax, comps.labels == row.comp, color)
n_small, n_removed = int((~table.kept).sum()), int(table.kept.sum()) - len(keep)
ax.set_title(f"green: {len(keep)} kept; red: {n_removed} removed by the classifier; yellow: {n_small} too small",
             fontsize=10)
plt.tight_layout()
plt.show()"""),

    ("md", "What the classifier saw on this image: its probability of matching an annotated filament, per component."),
    ("code", """shown = table[table.kept].assign(p_match=lambda d: pipeline.rescorer.probabilities(d).round(3)).round(2)
shown[["comp", "area", "mean_prob", "contrast", "length", "elongation", "r_norm", "site", "p_match"]].reset_index(drop=True)"""),

    ("md", "## 7. The test set"),
    ("code", """del pipeline
torch.cuda.empty_cache()
regenerated = get_output_dir() / "submission_regenerated.csv"
if RUN_TEST_SET and not regenerated.exists():
    run("scripts/final/predict.py", "--out", regenerated)
df = read_submission(regenerated)
print(f"{len(df)} filaments, {len(df) / 180:.2f} per image; validation problems: {validate_submission(df)}")"""),

    ("md", """## 8. Does it match what was submitted?

`submissions/submission_final_w40.csv` is the file Kaggle scored (public LB 0.39). The agreement below is PQ with the
submitted file standing in for the ground truth, so 1.0 would mean the same filaments with the same masks. Masks are
not bit-identical: the submitted run saved each stage's map as an 8-bit PNG, while this pipeline keeps floats."""),
    ("code", """submitted = read_submission(root / "submissions/submission_final_w40.csv")
reference = submitted.assign(filament_id="x-" + submitted.filament_id)
agreement = pq_score(reference, df)
print(f"submitted {len(submitted)} filaments, regenerated {len(df)}")
print(f"agreement: PQ {agreement.pq:.4f}, SQ {agreement.sq:.4f}, RQ {agreement.rq:.4f} "
      f"(matched {agreement.tp}, only regenerated {agreement.fp}, only submitted {agreement.fn})")"""),

    ("md", """## 9. Results

Out-of-fold on all 707 training images, each scored as it would be submitted, from
`scripts/evaluation/rubric_metrics.py`. `final_w40` is this pipeline; `det5f_tta` is the Mask R-CNN ensemble alone."""),
    ("code", """summary = pd.read_csv(root / "notes/results/rubric_summary.csv").set_index("experiment")
names = {"final_w40": "submitted", "det5f_tta": "Mask R-CNN alone", "human": "human agreement"}
columns = {"pq": "PQ", "sq": "SQ", "rq": "RQ", "precision": "precision", "recall": "recall",
           "overlap_median_dice": "median pair Dice", "pixel_dice_median": "median pixel Dice",
           "gt_missed": "GT missed", "gt_split_1ton": "1:n splits", "pred_merge_nto1": "n:1 merges"}
summary.loc[list(names), list(columns)].rename(index=names, columns=columns).round(3).T"""),
    ("code", """timing = pd.read_csv(root / "notes/results/inference_timing.csv")
timing[["pipeline", "device", "total_s_per_image", "test_set_min", "peak_gpu_gib"]].round(2)"""),
]


def main() -> None:
    nb = nbf.v4.new_notebook(metadata={"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"}})
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in CELLS]
    out = Path(__file__).resolve().parents[2] / "notebooks" / "solar_filament_pipeline.ipynb"
    nbf.write(nb, out)
    print(out)


if __name__ == "__main__":
    main()
