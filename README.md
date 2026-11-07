# Solar Filament Segmentation

Instance segmentation of solar filaments in GONG H-alpha full-disk images, for the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(Kaggle, IEEE BigData 2026 BigDataCup), scored with Panoptic Quality (PQ). Built with DINOv3.

**The whole pipeline is walked through in
[notebooks/solar_filament_pipeline.ipynb](notebooks/solar_filament_pipeline.ipynb).** It needs only this repository
and the competition's images: the checkpoint is downloaded automatically.

## The submitted pipeline

One Mask R-CNN, trained on all 707 training images, applied to 8 views of each image:

```
image ─ 8 views (flips, transposes) ─ Mask R-CNN ─ merge detections per filament ─ paste masks ─ score ≥ 0.8 ─ submission
```

- **Model.** ConvNeXt-T backbone with DINOv3 weights, an FPN, and COCO-pretrained RPN and heads, with a 56×56 mask head
  to keep the outlines of thin filaments. All backbone layers are trained.
- **Views.** Detections of the same filament are merged across views. A filament's score is the sum of its scores
  divided by the number of views, so one seen in only a few views scores low.
- **Masks.** Soft masks are pasted at threshold 0.45, highest score first, so instances never overlap.

The architecture, schedule and settings were chosen on held-out images with a paired bootstrap, not on the public
leaderboard. The code for this path is [src/filament/final.py](src/filament/final.py).

## Results

The final model trains on all images, so it has no held-out score of its own. The table gives the same recipe trained
on folds 1–4 and scored on the 142 images of fold 0 with the official metric. Human agreement is one annotator scored
against the others, on all multi-annotated images.

| | PQ | SQ | RQ | Median pair Dice | Missed GT | 1:n splits | n:1 merges | Public LB |
|---|---|---|---|---|---|---|---|---|
| **Fold-0 model** | **0.452** | 0.685 | 0.660 | 0.790 | 19.4% | 3.9% | 2.9% | 0.39 |
| **Full-data model (submitted)** | not measurable | | | | | | | **0.39** |
| Human agreement | 0.340 | 0.633 | 0.537 | 0.718 | 17.6% | 3.2% | 3.0% | – |

Fold 0 is slightly easier than the average fold, so read 0.452 as an upper estimate for the fold-0 model.

**Inference:** 2.3 s per 2048×2048 image on an RTX 5070 Laptop GPU, 7 minutes for the 180 test images, 1.45 GiB peak GPU
memory. Running `scripts/final/predict.py` on that machine reproduces the submitted CSV byte for byte. On another GPU
the half-precision numerics differ slightly: from a fresh install on a Kaggle T4, the regenerated file agrees with the
submitted one at PQ 0.996 (1,307 vs 1,308 filaments), and the whole walk-through runs in 12.5 minutes.

**Why one model.** An ensemble of 15 models (U-Nets + Mask R-CNN, with a component classifier) reached the same public
LB of 0.39 at 25.7 s per image, 11× slower. On held-out images its PQ advantage was too small to resolve:
ΔPQ −0.006 [−0.019, +0.006] for the fold-0 DINOv3 model against it. Its file is
`submissions/submission_final_w40.csv`.

[PLAN.md](PLAN.md) is the full experiment log, including everything that did not help (a component classifier on this
model, consensus targets, pseudo-labels, neighbouring frames, copy-paste, Cascade R-CNN, Mask2Former, and more).

## Quick start

Requires Python 3.12 or newer (tested on 3.12 and 3.13) and, for reasonable speed, a GPU.

```bash
git clone https://github.com/LucaT2/solar-filament-segmentation.git
cd solar-filament-segmentation
python -m venv .venv
source .venv/bin/activate                     # Windows: .venv\Scripts\activate
pip install -r requirements.txt               # a specific CUDA build: see the comment in requirements.txt
pip install -e .

kaggle competitions download -c filament-segmentation-2026      # after accepting the competition rules
# extract into filament-segmentation-2026/, or set FILAMENT_DATA to the MAGFiLO_1.0_Kaggle_2026 folder

python scripts/final/download_weights.py      # the checkpoint (200 MB) from the GitHub release, hash-checked
python scripts/final/predict.py --out outputs/submission_final.csv
```

The submitted file is `submissions/submission_mrcnn_dino_full_s42.csv`.

## Retraining from scratch

The final training takes about 5 hours on a Kaggle T4. The settings need one more run of the same length, on folds 1–4.

```bash
CONFIG=configs/detection/maskrcnn/maskrcnn_dinov3_convnext_t_m56.yaml
EXP=maskrcnn_dinov3_convnext_t_m56

# 1. tune the settings on fold 0
python scripts/detection/train.py --config $CONFIG --fold 0
python scripts/detection/predict.py --exp $EXP --folds 0 --tta 8 --split oof
python scripts/postprocess/tune_postprocess.py --exp $EXP --min-scores 0 0.3 0.5 0.6 0.7 0.8 0.9 0.95
cp outputs/$EXP/postprocess.json configs/final/

# 2. train on all images (--fold -1 writes outputs/<exp>/full_s42.pt) and predict
python scripts/detection/train.py --config $CONFIG --fold -1 --set epochs=29 fit_epochs=false max_hours=6.5
python scripts/final/predict.py
```

To compare two experiments on the same images, use `scripts/evaluation/compare.py` (paired ΔPQ with a bootstrap
confidence interval). `scripts/evaluation/rubric_metrics.py` computes the table above, `scripts/figures/report_figures.py` the report's figures and 95% intervals, and
`scripts/evaluation/time_inference.py` the timing. Results are in [notes/results/](notes/results/).

Runs are logged to Weights & Biases when credentials are available; otherwise nothing is logged.

## Repository layout

The library (`src/filament/`) and the scripts (`scripts/`) follow the same order:
**data → model → predictions → post-processing → submission**, with evaluation alongside.

```
src/filament/
  final.py                the submitted pipeline: image -> masks
  data/                   annotations (COCO -> tables), images + solar disk, RLE, folds, submission CSVs
  detection/              Mask R-CNN and variants, dataset, training, TTA, saved detections
  segmentation/           U-Net family, losses, dataset + augmentation, training loop, classical baseline
  postprocess/            maps or instances -> components, component classifier, submission
  evaluation/             official PQ metric, fast OOF scoring + grid search, error analysis
  analysis/               exploratory analyses behind notebooks/analysis/01_eda
scripts/
  final/                  download_weights, predict, build_pipeline_notebook: the submitted pipeline
  setup/ data/ segmentation/ detection/ postprocess/ evaluation/ semi/ kaggle/ figures/
configs/
  final/                  tuned settings and checkpoint hash of the submitted pipeline
  folds.csv               the fixed 5-fold split
  segmentation/ detection/  one YAML per experiment; archive/ holds screened-out experiments
notebooks/                the pipeline walk-through; analysis/ (EDA, error analysis); kaggle/ (training, clean-install check)
notes/                    data findings, error analysis, literature, results, figures
submissions/              the submitted CSV files
tests/                    metric equivalence, RLE/submission round-trip, folds, post-processing, training parts
```

Every script prints its usage with `--help`. Experiments are referred to by name (`outputs/<name>`) or by folder path.
On Kaggle, the notebooks in `notebooks/kaggle/` run the same steps; they are generated by
`scripts/kaggle/build_notebooks.py`.

Tests: `pip install -r requirements-dev.txt`, then `python -m pytest -W ignore::DeprecationWarning`. The metric tests
compare against the organizers' self-evaluation notebook when it is in the repo root (it is not redistributed here).

## Data and rules

Only the images and annotations provided by the competition were used. The public MAGFiLO release was deliberately
not used, because it contains the test labels, and the spine annotations were not used. The submitted model uses no
neighbouring frames. Ground truth comes from the MAGFiLO dataset
([Ahmadzadeh et al., 2024](https://doi.org/10.1038/s41597-024-03876-y)).

## Licenses

The code is under the [MIT license](LICENSE). **The released checkpoint is not:** it is a derivative of Meta's DINOv3
weights and is distributed under the [DINOv3 License](LICENSES/DINOv3_LICENSE.md), which also applies to anything you
build from it. Other pretrained weights used as starting points belong to their authors: COCO Mask R-CNN (through
`torchvision`) and, for the U-Net experiments, ImageNet ResNet34 and ConvNeXt-T (through `timm`).
