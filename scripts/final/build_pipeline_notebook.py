"""Generate notebooks/solar_filament_pipeline.ipynb, the end-to-end walk-through of the submitted pipeline, and
notebooks/kaggle/kaggle_check_pipeline.ipynb, which runs that walk-through on Kaggle from a fresh install.

Running this overwrites both notebooks without outputs. Edit the notebooks here, then run the walk-through from the
repo root so its outputs are saved:
    python scripts/final/build_pipeline_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/solar_filament_pipeline.ipynb \
        --ExecutePreprocessor.timeout=-1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

CELLS = [
    ("md", """# Solar filament segmentation: the submitted pipeline

This notebook goes from raw GONG H-alpha images to the submitted CSV. It needs nothing beyond this repository and the
competition's images: the checkpoint is downloaded from the repository's GitHub release. Built with DINOv3.

The method is one Mask R-CNN with a ConvNeXt-T backbone (DINOv3 weights), trained on all 707 training images, applied
to 8 views of each image.

| Section | What it does | Time |
|---|---|---|
| 1. Setup | finds or clones the repository | 1 min (5 with `INSTALL`) |
| 2. Data | the images, the annotations, why several annotators matter | 1 min |
| 3. Validation | folds and the official metric | instant |
| 4. Training | the model config and the training command | off by default, ~5 h on a Kaggle T4 |
| 5. Weights | downloads and hash-checks the checkpoint | 1 min |
| 6. One image | every stage of the pipeline on one test image | 1 min |
| 7. Test set | writes the submission CSV | ~7 min on a GPU |
| 8. Check | compares the result with the file submitted to Kaggle | 1 min |
| 9. Results | the held-out scores and timing | instant |

**Environment:** Python 3.12 with `requirements.txt`, on a GPU with at least 2 GiB. On Kaggle, attach the competition
data, enable a GPU and set `INSTALL = True`. Elsewhere, set `FILAMENT_DATA` to the `MAGFiLO_1.0_Kaggle_2026` folder, or
extract the data into `filament-segmentation-2026/` in the repo root."""),

    ("md", "## 1. Setup"),
    ("code", """import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/LucaT2/solar-filament-segmentation.git"
INSTALL = False        # True: install the pinned requirements (restart the kernel afterwards if torch changed)
RUN_TRAINING = False   # True: retrain the model (~5 h on a Kaggle T4) instead of downloading it
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
from filament.data.rle import rle_to_mask
from filament.data.submission import read_submission, validate_submission
from filament.data.images import disk_mask
from filament.detection import tta
from filament.detection.dataset import full_image_input
from filament.detection.instances import to_components, to_prob_map
from filament.detection.model import predict_instances
from filament.evaluation.metric import pq_score
from filament.final import FINAL_CONFIG_DIR, VIEWS, FinalPipeline
from filament.paths import get_data_paths, get_output_dir
from filament.postprocess.pipeline import load_postprocess
from filament.segmentation.dataset import image_info

paths = get_data_paths()
print("repo:", root)
print("data:", paths.root)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (slow)")"""),

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
annotators, and stratified by site and number of annotators. Every model and setting was chosen on held-out
predictions, scored with a reimplementation of the official metric that the tests check against the organizers' code.
Two models are compared on the same images with a paired bootstrap, which resolves differences of about 0.01 PQ on one
fold. The public leaderboard shows 2 decimals on about 90 images, so it cannot.

The final model trains on all images, so it has no held-out score of its own. Its architecture, its schedule and its
post-processing settings were all chosen with a model trained on folds 1–4 and scored on fold 0."""),
    ("code", """folds = load_folds()
print(folds.groupby("fold").size().rename("images").to_frame().T.to_string())"""),

    ("md", """## 4. Training

The backbone starts from Meta's DINOv3 ConvNeXt-T weights (timm `convnext_tiny.dinov3_lvd1689m`), the RPN and box and
mask heads from COCO Mask R-CNN weights. Only the FPN's input convolutions start from scratch. All backbone layers are
trained, at a quarter of the learning rate."""),
    ("code", """CONFIG = "configs/detection/maskrcnn/maskrcnn_dinov3_convnext_t_m56.yaml"
OVERRIDES = {"epochs": 29, "fit_epochs": False, "max_hours": 6.5}   # as the submitted run: fixed schedule, seed 42

config = {**yaml.safe_load((root / CONFIG).read_text()), **OVERRIDES}
KEYS = ["backbone", "pretrained", "trainable_layers", "backbone_lr_mult", "mask_size", "crop", "batch_size", "epochs",
        "samples_per_epoch", "lr", "weight_decay", "warmup_epochs", "augment", "seed"]
print(pd.Series({k: config[k] for k in KEYS}).to_string())

command = ("scripts/detection/train.py", "--config", CONFIG, "--fold", -1, "--set",
           *(f"{k}={str(v).lower()}" for k, v in OVERRIDES.items()))
print("\\npython", *command)
if RUN_TRAINING:
    run(*command)"""),

    ("md", """`--fold -1` trains on all images and saves `full_s42.pt` at the last epoch. The settings below were tuned on
fold 0 with `scripts/postprocess/tune_postprocess.py`; the README lists the commands."""),
    ("code", """cfg = load_postprocess(FINAL_CONFIG_DIR / "postprocess.json")
print(cfg)"""),

    ("md", "## 5. Download the checkpoint"),
    ("code", """if not RUN_TRAINING:
    run("scripts/final/download_weights.py")"""),

    ("md", """## 6. One image, stage by stage

The model is a torchvision Mask R-CNN whose ResNet50 is replaced by a timm ConvNeXt-T with DINOv3 weights. Images go
in at full resolution (2048×2048, not resized), because filaments are only about 8 px wide."""),
    ("code", """pipeline = FinalPipeline.load()
model, device = pipeline.model, pipeline.device


def millions(*modules):
    return sum(p.numel() for m in modules for p in m.parameters()) / 1e6


heads = model.roi_heads
print(f"backbone  ConvNeXt-T, DINOv3 weights, features at strides 4-32  {millions(model.backbone.body):5.1f} M parameters")
print(f"FPN       256 channels, 4 levels + 1 pooled level                {millions(model.backbone.fpn):5.1f} M")
print(f"RPN       anchors -> object proposals                            {millions(model.rpn):5.1f} M")
print(f"box head  filament vs background, box refinement                 {millions(heads.box_head, heads.box_predictor):5.1f} M")
print(f"mask head one 56 x 56 soft mask per detection                    {millions(heads.mask_head, heads.mask_predictor):5.1f} M")"""),

    ("md", """This is what `FinalPipeline.segment` does for one image, written out. Detections are merged across 8 views, and a
filament's score is the sum of its scores divided by the number of views, so one seen in only a few views scores low.
Each soft mask is pasted at the mask threshold, highest score first, so instances never overlap. Instances under the
score cut-off are dropped."""),
    ("code", """path = paths.test_images / "20140420195854Bh.jpeg"
info = image_info(path)        # finds the solar disk; mean and spread of the on-disk brightness
x = full_image_input(info)     # (3, 2048, 2048): the normalised image repeated to 3 channels
img = read_image(path)

per_view = []
for view in tta.VIEWS[VIEWS]:  # (transpose, flip_x, flip_y)
    found = predict_instances(model, tta.view_image(x, view), device, torch.float16)
    per_view.append(tta.unview_instances(found, view, x.shape[-2:]))   # back to the original orientation
detections = tta.merge_views(per_view)

region = disk_mask(img.shape, info.disk, cfg.disk_scale)               # only the solar disk
comps = to_components(detections, cfg.threshold, region)               # pasted masks, no overlaps
keep = comps.keep(cfg.min_area, cfg.min_score)
masks = [comps.labels == i for i in keep]

print("detections per view:", [len(v) for v in per_view])
print(f"{len(detections)} after merging, {comps.n} instances on the disk, {len(keep)} above score {cfg.min_score}")
same = [np.array_equal(a, b) for a, b in zip(masks, pipeline.segment(path))]
print(f"identical to FinalPipeline.segment: {len(same) == len(masks) and all(same)}")"""),
    ("code", """kept = set(keep.tolist())
fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("input", fontsize=10)
axes[1].imshow(to_prob_map(detections), cmap="magma", vmin=0, vmax=1)
axes[1].set_title(f"{VIEWS} views merged: score-weighted soft masks", fontsize=10)
axes[2].imshow(img, cmap="gray")
for i in range(1, comps.n + 1):
    outline(axes[2], comps.labels == i, "lime" if i in kept else "red")
axes[2].set_title(f"green: {len(kept)} kept; red: {comps.n - len(kept)} below score {cfg.min_score}", fontsize=10)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()"""),
    ("code", """pd.DataFrame({"score": comps.mean_prob[1:], "area_px": comps.area[1:],
              "kept": [i in kept for i in range(1, comps.n + 1)]}).sort_values("score", ascending=False).round(3)"""),

    ("md", "## 7. The test set"),
    ("code", """del pipeline
torch.cuda.empty_cache()
regenerated = get_output_dir() / "submission_regenerated.csv"
if RUN_TEST_SET and not regenerated.exists():
    run("scripts/final/predict.py", "--out", regenerated)
df = read_submission(regenerated)
print(f"{len(df)} filaments, {len(df) / 180:.2f} per image; validation problems: {validate_submission(df)}")"""),

    ("md", """## 8. Does it match what was submitted?

`submissions/submission_mrcnn_dino_full_s42.csv` is the file Kaggle scored (public LB 0.39). On the machine that
produced it, the regenerated file is byte-identical. A different GPU can change a few pixels through floating-point
rounding, so the agreement is also given as PQ with the submitted file standing in for the ground truth."""),
    ("code", """submitted = read_submission(root / "submissions/submission_mrcnn_dino_full_s42.csv")
same = len(set(submitted.segmentation_rle) & set(df.segmentation_rle))
reference = submitted.assign(filament_id="x-" + submitted.filament_id)
agreement = pq_score(reference, df)
print(f"submitted {len(submitted)} filaments, regenerated {len(df)}, identical masks {same}")
print(f"agreement: PQ {agreement.pq:.4f}, SQ {agreement.sq:.4f}, RQ {agreement.rq:.4f}")"""),

    ("md", """## 9. Results

The fold-0 model (same recipe, trained on folds 1–4) on the 142 images of fold 0, from
`scripts/evaluation/rubric_metrics.py`. The final model trained on 25% more images, which cannot be measured here.
The human reference is on all multi-annotated images."""),
    ("code", """model = pd.read_csv(root / "notes/results/dino_fold0/summary.csv").set_index("experiment")
human = pd.read_csv(root / "notes/results/rubric_summary.csv").set_index("experiment").loc[["human"]]
columns = {"pq": "PQ", "sq": "SQ", "rq": "RQ", "precision": "precision", "recall": "recall",
           "overlap_median_dice": "median pair Dice", "pixel_dice_median": "median pixel Dice",
           "gt_missed": "GT missed", "gt_split_1ton": "1:n splits", "pred_merge_nto1": "n:1 merges"}
table = pd.concat([model, human]).rename(index={"mrcnn_dino_f0_tta8": "fold-0 model", "human": "human agreement"})
table[list(columns)].rename(columns=columns).round(3).T"""),
    ("code", """timing = pd.read_csv(root / "notes/results/inference_timing.csv")
timing[["pipeline", "device", "total_s_per_image", "test_set_min", "peak_gpu_gib"]].round(2)"""),
]


CHECK_CELLS = [
    ("md", """# Clean-environment check of the submitted pipeline

Runs `notebooks/solar_filament_pipeline.ipynb` the way a reviewer would: a fresh clone, `pip install -r
requirements.txt`, then the notebook in a new kernel. The executed copy, the regenerated CSV and `check.json` are written
to the output.

While the repository is private, the clone and the checkpoint download use a GitHub token. Once it is public, the
walk-through needs neither: its first cell clones the repository and section 5 downloads the checkpoint.

### Settings
* Accelerator: **GPU T4** (one is enough). Internet: **on**.
* *Add Input*: competition **filament-segmentation-2026**.
* *Add-ons → Secrets*: the GitHub token (label `GITHUB_SECRET`). Run with **Save Version → Save & Run All** (~25 min)."""),
    ("code", """GITHUB_SECRET = "Github_key"
REPO = "LucaT2/solar-filament-segmentation"
BRANCH = "main"
WEIGHTS_TAG = "v2.0-weights"

import hashlib, json, platform, shutil, subprocess, sys, time
from pathlib import Path

import requests
from kaggle_secrets import UserSecretsClient

WORK = Path("/kaggle/working")
ROOT = WORK / "solar-filament-segmentation"
T0 = time.time()
report = {"python": platform.python_version()}


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def sh(cmd, what, timeout=3600):
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    log(f"{what}: exit {res.returncode}")
    return res


token = UserSecretsClient().get_secret(GITHUB_SECRET)
gpu = sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], "nvidia-smi")
report["gpu"] = gpu.stdout.strip()
log(f"Python {report['python']}; GPU {report['gpu']}")"""),
    ("code", """res = sh(["git", "clone", "--depth", "1", "-b", BRANCH, f"https://x-access-token:{token}@github.com/{REPO}.git",
          str(ROOT)], "git clone", 300)
if res.returncode:
    raise RuntimeError("git clone failed: " + res.stderr.replace(token, "***"))
report["commit"] = subprocess.run(["git", "-C", str(ROOT), "log", "--oneline", "-1"], capture_output=True,
                                  text=True).stdout.strip()
log(f"commit {report['commit']}")

# the checkpoint from the (private) release, where scripts/final/download_weights.py would put it
headers = {"Authorization": f"Bearer {token}"}
release = requests.get(f"https://api.github.com/repos/{REPO}/releases/tags/{WEIGHTS_TAG}", headers=headers,
                       timeout=60).json()
manifest = json.loads((ROOT / "configs/final/weights.json").read_text())
for checkpoint, expected in manifest["files"].items():
    asset = next(a for a in release["assets"] if a["name"] == checkpoint.replace("/", "-"))
    target = WORK / checkpoint
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(asset["url"], headers={**headers, "Accept": "application/octet-stream"}, stream=True,
                      timeout=600) as r, open(target, "wb") as f:
        r.raise_for_status()
        shutil.copyfileobj(r.raw, f, 1 << 24)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    report["checkpoint_hash_ok"] = digest == expected
    log(f"{checkpoint}: {target.stat().st_size / 2**20:.0f} MiB, hash ok {digest == expected}")"""),
    ("code", """before = {p: subprocess.run([sys.executable, "-c", f"import {p}; print({p}.__version__)"], capture_output=True,
                           text=True).stdout.strip() for p in ("torch", "numpy")}
log(f"preinstalled: {before}")
res = sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")], "pip install -r requirements.txt")
report["pip_requirements_ok"] = res.returncode == 0
print(res.stdout[-3000:], res.stderr[-5000:])
res = sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(ROOT)], "pip install -e .")
probe = ("import torch, numpy, timm, cv2, sklearn; print(torch.__version__, torch.version.cuda, "
         "torch.cuda.is_available(), numpy.__version__, timm.__version__, cv2.__version__, sklearn.__version__)")
res = sh([sys.executable, "-c", probe], "import check")
report["versions"] = res.stdout.strip() or res.stderr[-1500:]
log(f"torch, CUDA, GPU usable, numpy, timm, opencv, sklearn: {report['versions']}")"""),
    ("code", """executed = WORK / "solar_filament_pipeline_executed.ipynb"
start = time.time()
res = sh([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
          str(ROOT / "notebooks/solar_filament_pipeline.ipynb"), "--output", str(executed),
          "--ExecutePreprocessor.timeout=-1"], "notebook", 3 * 3600)
report["notebook_minutes"] = round((time.time() - start) / 60, 1)
report["notebook_ok"] = res.returncode == 0
if res.returncode:
    print(res.stderr[-6000:])
if executed.exists():
    for cell in json.loads(executed.read_text())["cells"]:
        for out in cell.get("outputs", []):
            text = "".join(out.get("text", ""))
            if out.get("output_type") == "error":
                report.setdefault("errors", []).append(f"{out['ename']}: {out['evalue'][:500]}")
            elif any(k in text for k in ("device:", "identical", "agreement", "validation problems")):
                print(text)
                report.setdefault("key_outputs", []).append(text.strip())
(WORK / "check.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))"""),
]


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    nb = nbf.v4.new_notebook(metadata={"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"}})
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in CELLS]
    notebooks = Path(__file__).resolve().parents[2] / "notebooks"
    nbf.write(nb, notebooks / "solar_filament_pipeline.ipynb")
    check = nbf.v4.new_notebook(metadata={**nb.metadata, "kaggle": {"accelerator": "nvidiaTeslaT4",
                                                                    "isInternetEnabled": True}})
    check.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
                   for kind, src in CHECK_CELLS]
    nbf.write(check, notebooks / "kaggle" / "kaggle_check_pipeline.ipynb")
    print("written", notebooks / "solar_filament_pipeline.ipynb", "and", notebooks / "kaggle" / "kaggle_check_pipeline.ipynb")


if __name__ == "__main__":
    main()
