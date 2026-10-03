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
- [ ] Exact reimplementation of the official PQ (from `self-evaluation-notebook.ipynb`), plus Dice/IoU distributions and
      1:n / n:1 counts. *(Fast RLE-based version already in `src/filament/metric.py`; still to do: an equivalence
      test against the official dense implementation.)*
- [ ] RLE encode/decode round-trip tests. GT written as a submission must score PQ = 1.0.
- [ ] 5-fold cross-validation grouped by image stem, stratified by site and annotator count (Stage 1 showed this
      matches the train→test time structure).
- [ ] Multi-annotator ground truth: replicate the official scoring. GT is grouped per **annotator-image**
      (`<prefix>-<stem>`), and each annotator's set is scored separately against the single prediction for that
      image, so multi-annotated images count 2–3×. Folds are grouped by image stem so one image never appears in
      both train and validation under different annotators.

## Stage 3 — Baselines
- [ ] **Classical:** limb-darkening correction → adaptive threshold → morphology → connected components. A floor score and
      a comparison point for the report.
- [ ] **Deep baseline:**
  - U-Net (EfficientNet/ResNet34 encoder) for binary filament segmentation, **1 input channel** (grayscale).
  - Per-image normalization; predictions outside the solar disk are masked (disk estimated per image).
  - Every annotator-image is used as a separate training sample, as the organizers allow.
  - Trained on full-resolution 512–1024 crops; inference with an overlapping sliding window.
  - Connected components (no linking of nearby pieces) + minimum-area filter → instances → RLE.
  - Threshold and minimum area tuned for PQ on out-of-fold predictions.
- [ ] First Kaggle submission (CSV upload).

## Stage 4 — Error analysis
- [ ] Sort out-of-fold errors into buckets: false-positive blobs / missed filaments / fragmented / merged / IoU just below 0.5.
      Use the same categories as the human-vs-human analysis so the model can be compared with human disagreement.
- [ ] Break down by site (especially Mauna Loa), filament size, category and distance from the disk center.
- [ ] Check whether predicted masks are systematically too thin or too wide (outline width is the most sensitive knob).
- [ ] The largest bucket decides the Stage 5 priorities.

## Stage 5 — Main approach: precise, connected, consensus segmentation
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
|      |            |       |           |       |
