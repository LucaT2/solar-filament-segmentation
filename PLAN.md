# Solar Filament Segmentation Challenge 2026 — Plan

Competition: https://www.kaggle.com/competitions/filament-segmentation-2026
Deadline: **2026-11-15 06:00 UTC** (Kaggle submissions + report + public repo via Google form)

## Guiding principle

The leaderboard metric is **Panoptic Quality (PQ)**. A predicted filament only counts as a true positive if its IoU with a
ground-truth filament is **> 0.5**. Fragmenting a filament, merging two filaments, and leaving small false blobs are all
penalized. Correct **instance grouping** matters as much as pixel accuracy.

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
- [ ] **Annotator agreement ceiling:** PQ between annotators on the 296 multi-annotated training images. This is the
      realistic upper bound.
- [ ] **Instance definition:** do GT instances touch? Do annotators group nearby fragments into one outline? How tight are
      the outlines relative to the dark pixels?
- [ ] **Image properties:** limb darkening, per-site (B/C/L/M/T/U) differences, noise and clouds; filament location, size
      and shape statistics.
- [ ] **Leakage / splits:** train vs test date overlap. Folds grouped by date.
- [ ] **Literature review**, with a short note on what each source implies:
  - MAGFiLO paper (Ahmadzadeh et al. 2024, *Sci. Data*, doi 10.1038/s41597-024-03876-y)
  - Prior filament segmentation (Mask R-CNN, U-Net, semi-supervised H-alpha detection)
  - Thin/elongated structure segmentation (vessels, roads, clDice)
- [ ] Deliverables: `notebooks/01_eda.ipynb` + `notes/findings.md`.

## Stage 2 — Validation harness (before any model)
- [ ] Exact reimplementation of the official PQ (from `self-evaluation-notebook.ipynb`), plus Dice/IoU distributions and
      1:n / n:1 counts.
- [ ] RLE encode/decode round-trip tests. GT written as a submission must score PQ = 1.0.
- [ ] 5-fold grouped cross-validation (by date).
- [ ] Policy for multi-annotator ground truth in validation (one annotator chosen at random vs averaged over annotators),
      decided after Stage 1.

## Stage 3 — Baselines
- [ ] **Classical:** limb-darkening correction → adaptive threshold → morphology → connected components. A floor score and
      a comparison point for the report.
- [ ] **Deep baseline:**
  - U-Net (EfficientNet/ResNet34 encoder) for binary filament segmentation.
  - Trained on full-resolution 512–1024 crops; inference with an overlapping sliding window.
  - Connected components + minimum-area filter → instances → RLE.
  - Threshold and minimum area tuned for PQ on out-of-fold predictions.
- [ ] First Kaggle submission through the notebook.

## Stage 4 — Error analysis
- [ ] Sort out-of-fold errors into buckets: false-positive blobs / missed filaments / fragmented / merged / IoU just below 0.5.
- [ ] The largest bucket decides the Stage 5 priorities.

## Stage 5 — Main approach: segmentation + instance-boundary learning
- [ ] **Stronger high-resolution model:** U-Net++ / UPerNet with a ConvNeXt or EfficientNetV2 encoder.
- [ ] **Multi-task outputs:** filament mask + boundary/contact map + distance-transform map, combined by marker-based
      watershed to split touching filaments and keep each filament's pieces together.
- [ ] **Annotator-aware training:** pick a random annotator per sample each epoch, or use averaged soft targets.
- [ ] **Losses:** BCE + Dice, optionally clDice for continuity.
- [ ] **Learned fragment-merging step:** a classifier on neighbouring predicted pieces (gap, orientation, confidence).
- [ ] Alternative, only if analysis shows many touching instances: an instance segmentation model (Mask2Former / Mask R-CNN).
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
