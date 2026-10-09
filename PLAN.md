# Solar Filament Segmentation Challenge 2026 — Plan

Competition: https://www.kaggle.com/competitions/filament-segmentation-2026
Deadline: **2026-11-15 06:00 UTC** (Kaggle submissions + report + public repo via Google form)

## Guiding principle

The leaderboard metric is **Panoptic Quality (PQ)**. A predicted filament only counts as a true positive if its IoU with a
ground-truth filament is **> 0.5**. Fragmenting a filament, merging two filaments, and leaving small false blobs are all
penalized (see [notes/competition_overview.md](notes/competition_overview.md)).

Stage 1 ([notes/findings.md](notes/findings.md)) showed that true filaments almost never touch, so **instances =
connected components of a good binary mask**. The real PQ levers are:
1. **boundary precision** (±1 px of outline width changes PQ a lot; filaments are about 8 px wide),
2. **connectivity** (predictions must not break a filament into pieces),
3. **the confidence/size cut-off** for small, faint blobs.

The human-vs-human score is **PQ 0.34**; a model predicting the annotators' consensus can beat it.

**Model selection uses cross-validated PQ, not the public leaderboard.** The leaderboard covers only ~50% of the 180 test
images, and the organizers have said some leaderboard scores are gamed.

## Hardware

- Laptop: RTX 5070 Laptop GPU, 8 GB. Used for analysis, debugging and small-crop runs.
- Kaggle: 16 GB GPUs. Used for full-resolution training, cross-validation folds and ensembles.

---

## Stage 0 — Setup
- [x] Python 3.12 venv. PyTorch with CUDA ≥ 12.8 (needed for Blackwell), `segmentation_models_pytorch`, `timm`,
      `albumentations`, `pycocotools`, `opencv-python`, `scikit-image`. Pinned `requirements.txt`.
      *(torch 2.11.0+cu128 on RTX 5070 sm_120 verified; `scripts/check_env.py` passes.)*
- [x] Git repo + GitHub remote (private now, public at the deadline).
- [x] Code layout:
  - `src/filament/`: data, preprocess, rle, metric, models, postprocess, train, infer
  - `configs/*.yaml`: one config per experiment
  - `scripts/`: CLI entry points
  - `notebooks/`: EDA + Kaggle train/submit notebook

## Stage 1 — Understand the problem (data analysis + literature)
- [x] **Annotator agreement ceiling:** PQ between annotators on the 296 multi-annotated training images.
      → **human vs human PQ = 0.340** (SQ 0.63, RQ 0.54).
- [x] **Instance definition:** do GT instances touch? → almost never (5 overlapping pairs out of 8,199 filaments). Connected
      components on the union mask lose nothing (PQ 0.3399 → 0.3391). Outline width must be accurate to about 1 px
      (−1 px costs 18% of PQ).
- [x] **Image properties:** residual limb darkening beyond about 0.5 R, disk radius and center vary per site, Mauna Loa
      is blurry and low-contrast, filaments are only about 12% darker than their surroundings.
- [x] **Leakage / splits:** test images are as close in time to train as train images are to each other → folds
      grouped by image stem match the test conditions.
- [x] **Literature review** → [notes/literature.md](notes/literature.md).
- [x] Deliverables: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) + [notes/findings.md](notes/findings.md).

## Stage 2 — Validation harness (before any model)
- [x] Exact reimplementation of the official PQ, plus Dice/IoU distributions and 1:n / n:1 counts
      (`src/filament/metric.py`). `tests/test_metric.py` checks it against the organizers' own functions, loaded from
      their notebook, on deliberately damaged predictions (wider/thinner, shifted, dropped, split, merged, duplicated,
      false blobs, empty images, unknown images): **identical PQ and IoU matrices**.
- [x] RLE round-trip tests; GT written through the submission writer and read back scores PQ = 1.0
      (`tests/test_rle_submission.py`). Submission writer and validator in `src/filament/submission.py`.
- [x] 5-fold cross-validation grouped by image stem, stratified by site × annotator count →
      [configs/folds.csv](configs/folds.csv) (`scripts/make_folds.py`, guarded by `tests/test_folds.py`).
- [x] Multi-annotator ground truth scored per **annotator-image**, exactly like the official metric.
- [x] OOF evaluator: `python scripts/evaluate.py --pred <csv> [--fold k]` → PQ/SQ/RQ, overlap and split/merge rates,
      breakdowns by site, annotator count and fold.
- **Noise level:** human-vs-human PQ ranges from 0.316 to 0.385 across folds. Compare models on 5-fold OOF PQ, not on
      single folds or the about 90-image public leaderboard.

## Stage 3 — Baselines
- [x] **Classical:** local darkness vs a 101 px median background → threshold → connected components → size and
      confidence filters. **OOF PQ 0.183** (all 707 images). Mostly false positives (49% of blobs unmatched: fibrils,
      sunspots), 31% of filaments missed, 13% split.
- [ ] **Deep baseline:**
  - U-Net (EfficientNet/ResNet34 encoder) for binary filament segmentation, **1 input channel** (grayscale).
  - Per-image normalization; predictions outside the solar disk are masked (disk estimated per image).
  - Every annotator-image is used as a separate training sample, as the organizers allow.
  - Trained on full-resolution 512–1024 crops; inference with an overlapping sliding window.
  - Connected components (no linking of nearby pieces) + minimum-area filter → instances → RLE.
  - Threshold and minimum area tuned for PQ on out-of-fold predictions.
  - [x] Pipeline built: `scripts/train.py` / `predict.py` / `tune_postprocess.py` / `make_submission.py`,
        config `configs/unet_r34.yaml`, Kaggle notebook `notebooks/kaggle_train_submit.ipynb`.
  - [x] Fold 0 trained locally (RTX 5070 laptop, ~40 min): **PQ 0.391** after tuning. On fold 0's multi-annotated
        images: model ≈ 0.38 vs human 0.328.
  - [x] Folds 1–4 trained locally (`scripts/run_pipeline.ps1`, ~2 h 40 min) → **5-fold OOF PQ 0.403** (folds
        0.391–0.412), **0.425 with the rescorer** (nested); 5-model ensemble + flip TTA for test.
- [ ] First Kaggle submission (CSV upload). *(Ready: `outputs/unet_r34/submission.csv` from the fold-0 model + flip
      TTA.)*

## Stage 4 — Error analysis
*Done on fold 0 only (142 images): [notes/error_analysis.md](notes/error_analysis.md),
[notebooks/02_error_analysis.ipynb](notebooks/02_error_analysis.ipynb). Re-run with all folds once they exist.*
- [x] Error buckets vs human: model match 66% vs 54%, better in every bucket **except splits** (5.8% vs 2.9%;
      19% for filaments > 3,200 px).
- [x] Breakdowns: small (35% missed, same as humans), unidentifiable (20% missed), near the limb (35% missed, n = 34);
      no site stands out.
- [x] Outline width: no bias (±1 px both hurt). Linking pieces: ≤ +0.004, so splits need a model-side fix.
- [x] False positives: on multi-annotated images only 32% of kept components match all annotators. Pure FPs look like
      unlabeled faint filaments near the limb.
- [x] **Component rescoring classifier: PQ 0.391 → 0.419** (AUC 0.78 vs 0.66 for size alone).

## Stage 5 — Roadmap (agreed 2026-10-09)
Each step builds on the winner of the previous one; every idea is screened on fold 0 against the current baseline
with a paired comparison (ΔPQ with a confidence interval), and only winners get a 5-fold run (Kaggle T4×2).
Cut-off for new architectures: ~Oct 29; then final ensemble, report, repo, form (deadline Nov 15).

- [ ] **Step 0 — tooling:** `scripts/compare.py` (paired ΔPQ + bootstrap CI on the same images); tiled prediction for
      models that don't fit 2048² in memory; notebook option to run a different config per GPU.
- [ ] **Step 1 — splits of large filaments:** 1024 px crops + Skeleton Recall loss + longer training (40–50 epochs);
      ablate if it wins.
- [ ] **Step 2 — consensus targets:** average of all annotators' masks as a soft target on multi-annotated images.
- [ ] **Step 3 — encoders, then decoders:** U-Net with ResNet34 / ConvNeXt-T / EfficientNet-B4 / MiT-B2 → best 1–2
      encoders with U-Net / UNet++ / UPerNet or FPN. Per-encoder learning rate (lower for the encoder).
- [ ] **Step 4 — other architectures** (time-boxed): Mask2Former if splits/grouping still dominate, HRNet or DSCNet if
      boundaries do.
- [ ] Cheap extras along the way: classifier features (fragment, probability spread, texture) + lower size filter,
      `pos_fraction` 0.5, disk-distance input channel, 8-way TTA, multi-scale TTA.

## Stage 5 — Main approach: precise, connected, consensus segmentation
*Priorities from Stage 4: (1) component rescoring on all-fold OOF, (2) large and chained filaments: bigger
crops, Skeleton Recall, longer training, (3) 5-fold ensemble. Outline width and linking need no work.*
*Revised after Stage 1: GT instances almost never touch, so connected components replace the planned instance
head and watershed. The PQ levers are boundary precision (±1 px), connectivity, and the confidence/size cut-off.*
- [ ] **Stronger full-resolution model:** U-Net++ / UPerNet with a ConvNeXt or EfficientNetV2 encoder. No more than 2×
      downsampling.
- [ ] **Consensus training:** pick a random annotator per sample each epoch vs. averaged soft targets on
      multi-annotated images. The model should learn the annotators' consensus, which can beat the human-vs-human 0.34.
- [ ] **Losses:** BCE + Dice + **Skeleton Recall** (cheap connectivity loss, so filaments don't break at thin necks).
- [ ] **Input:** limb flattening or a radial channel, disk mask, blur and contrast augmentation (Mauna Loa).
- [ ] **Post-processing tuned on CV PQ:** probability threshold (≈ outline width), minimum area, per-component
      confidence cut-off, optional gap-closing only if Stage 4 shows predicted fragmentation.
- [ ] **Object-level scoring:** compare (a) a small classifier that rescores each U-Net connected component vs.
      (b) Mask R-CNN (torchvision, full resolution) as a separate model or ensemble partner. Mask R-CNN's 28×28 mask
      head and NMS cost almost nothing on this data (measured); its learned per-object confidence may improve RQ
      (missed faint filaments, false blobs).
- [ ] Optional: semi-supervised training on extra unlabeled GONG Hα images (external data allowed; no extra labels).
- [ ] **Open question:** may filament **spines** in the annotation JSON be used as an extra training target? The rules forbid
      "other ground-truth metadata for training". Ask on the competition forum before using them.

## Stage 6 — Polish and final deliverables
- [ ] Combine fold models + flip/rotation averaging at test time (TTA). Re-tune post-processing on out-of-fold predictions.
- [ ] Watch inference time (computational efficiency is judged).
- [ ] Repo cleanup: docs, modular code, pinned `requirements.txt`, reproducible end-to-end notebook.
- [ ] 4-page technical report (Overleaf template) → PDF.
- [ ] Google form submission before Nov 15; repo public until winners are announced (Nov 30).

---

## Kaggle notebook — `notebooks/kaggle_train_submit.ipynb`
Kaggle itself only needs an uploaded CSV; this isn't a code competition. The notebook gives us free 16 GB GPUs and a
single training → inference → submission run. It also becomes the basis for the **required** repo notebook that
illustrates and reproduces the whole pipeline for the final judging.

1. **Setup:** pip install + clone this repo from GitHub (or attach it as a Kaggle Dataset).
2. **Data:** attach the competition data.
3. **Train:** pick a config/fold, mixed-precision training on the 16 GB GPU, save weights to the notebook output.
4. **Infer:** sliding window + TTA + post-processing on `test/test_images` → `submission.csv`.
5. **Validate** the CSV by decoding it back, then submit (download, or `kaggle competitions submit`).

All logic lives in `src/`, so the notebook stays short and matches local runs.

## Experiment log

| Date | Experiment | CV PQ | Public LB | Notes |
|------|------------|-------|-----------|-------|
| 2026-10-04 | Reference: one annotator vs the others (multi-annotated images only) | 0.340 | – | SQ 0.633, RQ 0.537; per fold 0.316–0.385; worst sites U 0.304, M 0.314 |
| 2026-10-04 | Classical darkness detector (thr 0.2, min area 500, min mean 0.4) | 0.183 | 0.16 | all 707 images; FP-dominated (RQ 0.28) |
| 2026-10-04 | U-Net ResNet34, 512 crops, BCE+Dice, 30 ep × 4000 crops, **fold 0 only** (thr 0.5, min area 300) | 0.391 (f0) | 0.34 | untuned val 0.359; SQ 0.676, RQ 0.578; FP 1043 vs FN 552; best epoch = last (still improving); threshold surface flat 0.5–0.9; the LB upload used flip TTA on test |
| 2026-10-04 | + component rescoring classifier (P(match) ≥ 0.3), fold 0, CV within fold | 0.419 (f0) | – | analysis estimate on non-TTA OOF maps; AUC 0.78 |
| 2026-10-04 | U-Net fold 0, OOF maps re-predicted **with flip TTA** (same as test) | 0.402 (f0) | (0.34) | TTA alone +0.011 on OOF |
| 2026-10-04 | + rescorer in the pipeline (`fit_rescorer.py`, min_p 0.3, trained on fold-0 components) | 0.427 (f0); nested 0.424 | 0.36 | `submission_rescored.csv`; keeps 80% of components on test vs 79% on OOF; nested = cut-off chosen without the held-out images |
| 2026-10-04 | **5 folds**, TTA OOF, tuned on 707 images (thr 0.6, min area 400) | **0.403** | – | folds 0.391–0.412; sites L 0.386 / M 0.388 lowest, U 0.422 / T 0.416 highest |
| 2026-10-04 | + rescorer on 9,142 OOF components (min_p 0.4); test = 5-model avg + TTA | **0.425** nested | **0.37** | `submission_5fold.csv`; averaging lowers mean_prob → classifier keeps 71% on test vs 78% on OOF (≈ −0.004) |
| 2026-10-05 | same, cut-off calibrated to the OOF keep rate (min_p 0.297 on test) | 0.425 | **0.37** | `submission_5fold_calibrated.csv`; no visible difference from the uncalibrated one |
| 2026-10-09 | 2 full-data models (all 707 images, seeds 42/43, last epoch), trained on Kaggle T4×2 | – (no validation) | **0.37** | `outputs/kaggle_runs/solar-fillament_v1/submission_full.csv`; 1,168 filaments; ~70 min per model on a T4 |
| 2026-10-09 | 5 folds + 2 full-data models (7-model average) | – | **0.37** | `submission_folds_full.csv`; Kaggle's "folds" variant reproduced the local 5-fold submission exactly (same 1,073 filaments, same classifier) |

*Full-data models: no measurable gain on the public LB (all three at 0.37). Decision for the final recipe: optional; add 1–2 only if GPU time is left over at the end.*

*Public LB ≈ 0.87 × local OOF PQ for both models so far (public test subset somewhat harder, plus noise on about 90 images).*
