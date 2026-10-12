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
      *(torch 2.11.0+cu128 on RTX 5070 sm_120 verified; `scripts/setup/check_env.py` passes.)*
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
- [x] Deliverables: [notebooks/analysis/01_eda.ipynb](notebooks/analysis/01_eda.ipynb) + [notes/findings.md](notes/findings.md).

## Stage 2 — Validation harness (before any model)
- [x] Exact reimplementation of the official PQ, plus Dice/IoU distributions and 1:n / n:1 counts
      (`src/filament/evaluation/metric.py`). `tests/test_metric.py` checks it against the organizers' own functions, loaded from
      their notebook, on deliberately damaged predictions (wider/thinner, shifted, dropped, split, merged, duplicated,
      false blobs, empty images, unknown images): **identical PQ and IoU matrices**.
- [x] RLE round-trip tests; GT written through the submission writer and read back scores PQ = 1.0
      (`tests/test_rle_submission.py`). Submission writer and validator in `src/filament/data/submission.py`.
- [x] 5-fold cross-validation grouped by image stem, stratified by site × annotator count →
      [configs/folds.csv](configs/folds.csv) (`scripts/setup/make_folds.py`, guarded by `tests/test_folds.py`).
- [x] Multi-annotator ground truth scored per **annotator-image**, exactly like the official metric.
- [x] OOF evaluator: `python scripts/evaluation/evaluate.py --pred <csv> [--fold k]` → PQ/SQ/RQ, overlap and split/merge rates,
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
  - [x] Pipeline built: `scripts/segmentation/train.py` / `predict.py` / `tune_postprocess.py` / `make_submission.py`,
        config `configs/segmentation/unet_r34.yaml`, Kaggle notebook `notebooks/kaggle/kaggle_train_submit.ipynb`.
  - [x] Fold 0 trained locally (RTX 5070 laptop, ~40 min): **PQ 0.391** after tuning. On fold 0's multi-annotated
        images: model ≈ 0.38 vs human 0.328.
  - [x] Folds 1–4 trained locally (`scripts/run_pipeline.ps1`, ~2 h 40 min) → **5-fold OOF PQ 0.403** (folds
        0.391–0.412), **0.425 with the rescorer** (nested); 5-model ensemble + flip TTA for test.
- [ ] First Kaggle submission (CSV upload). *(Ready: `outputs/unet_r34/submission.csv` from the fold-0 model + flip
      TTA.)*

## Stage 4 — Error analysis
*Done on fold 0 only (142 images): [notes/error_analysis.md](notes/error_analysis.md),
[notebooks/analysis/02_error_analysis.ipynb](notebooks/analysis/02_error_analysis.ipynb). Re-run with all folds once they exist.*
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

- [x] **Step 0 — tooling:** `scripts/evaluation/compare.py` (paired ΔPQ + bootstrap CI on the same images; resolves ~0.01 on one
      fold: TTA vs no TTA on fold 0 = −0.0106, CI [−0.018, −0.002]); tiled prediction (`infer_tile`); the Kaggle
      notebook runs a list of EXPERIMENTS spread over the GPUs and compares each with the baseline.
- [ ] **Step 1 — splits of large filaments:** 1024 px crops + Skeleton Recall loss + longer training (40–50 epochs);
      ablate if it wins. *Fold-0 screen (2026-10-09/10):* 1024 crops + 45 epochs alone: no effect (−0.004, splits not
      reduced) → keep 512 crops. Skeleton Recall (weight 1): splits of large filaments 19% → 12%, missed 16.5% →
      8.5%, but masks too wide (merges 3% → 7.6%); needs a stricter threshold (0.85). With it + classifier: honest
      fold-0 0.420 (baseline 0.421), **LB 0.37 as a single model** (baseline single model 0.36).
- [x] **Step 1b — skeleton weight at 512 crops / 30 epochs:** fold-0 screen (2026-10-10).
      **`unet_r34_skel050` wins:** ΔPQ +0.010 [−0.001, +0.021], P = 0.96 (threshold 0.9); with classifier 0.435 vs
      0.421. `unet_r34_skel100`: ±0.000. **5-fold confirmed (2026-10-10):** OOF PQ 0.412 vs 0.403,
      ΔPQ **+0.010 [+0.005, +0.014]**, P = 1.00 (−430 FP); with classifier 0.436 vs 0.425. Threshold plateau 0.90–0.95
      on single-model OOF maps; the 5-model test average is ~6% thinner at 0.95, so 0.90 is used for the ensemble.
      **skel050 is the new baseline.**
- [x] **Step 2 — consensus targets:** average of all annotators' masks as a soft target on multi-annotated images
      (`target_mode: consensus`). Fold-0 screen vs skel050 (2026-10-10): **no gain**, ΔPQ −0.001 [−0.010, +0.008];
      more conservative (−154 FP, −78 TP), with classifier 0.434 vs 0.435. Kept as a possible ensemble partner.
      Weight sweep: `unet_r34_skel025` ΔPQ −0.003 [−0.011, +0.006] (classifier 0.437) → 0.25–0.5 is a plateau;
      **recipe stays skel050**.
- [x] **Step 3a result (2026-10-10, fold 0 vs skel050/ResNet34):** **ConvNeXt-Tiny** ΔPQ **+0.008** [−0.004, +0.018],
      P = 0.92, better outlines (SQ +0.005), −119 FP; with classifier 0.439 vs 0.435. EfficientNet-B4 ΔPQ −0.007
      [−0.021, +0.006]; with classifier 0.419. (Note: both also use batch 16 / lr 7e-4.) Decision (user, 2026-10-10):
      **no 5-fold runs for now**; keep screening architectures on fold 0 (ConvNeXt-T + U-Net is the fold-0 reference).
- [x] **Decoder / transformer screen (2026-10-11, fold 0 vs ConvNeXt-T + U-Net):** **UPerNet** ΔPQ −0.015
      [−0.024, −0.006] (outlines worse, SQ −0.0125: its 1/4-resolution output blurs 8 px filaments) → U-Net stays.
      **MAnet** diverged (fp16 overflow → NaN from epoch 6): no valid result. **MiT-B2 / SegFormer** crashed with out of
      memory in the first full-image validation (4 GiB attention block on top of training's cached memory).
      Fixes: `decoder_fp32` (MAnet), 1024 px tiled validation/prediction (MiT), `empty_cache()` before validation,
      expandable CUDA segments, non-finite batches skipped and runs aborted once an epoch is mostly non-finite.
      Re-run of MiT-B2 + U-Net and MAnet (fp32 decoder) in `kaggle_screen.ipynb`; SegFormer dropped (1/4-res head).
- [x] **Cheap tests (2026-10-11, fold 0, local):**
      *Ensemble* ResNet34 skel050 + ConvNeXt-T (equal-weight map average, `scripts/postprocess/ensemble_maps.py`): ΔPQ +0.002 vs
      ConvNeXt [−0.007, +0.011], +0.012 vs ResNet34 [+0.002, +0.022]; with classifier 0.445 vs 0.439 / 0.435.
      *8-way TTA* (`--tta-mode d8`: 4 rotations × mirror): ConvNeXt +0.004 [−0.001, +0.009], ResNet34 +0.003
      [−0.002, +0.010] (better outlines); with classifier 0.446 / 0.441. *Both:* ensemble of 8-way maps + classifier
      **0.449** (threshold 0.8: averaged maps are softer). → Final recipe: multi-architecture ensemble + 8-way TTA.
- [x] **Step 3c — transformer encoder:** MiT-B2 (SegFormer's encoder) + U-Net (`unet_mitb2_skel050`); encoder lr ×0.1,
      weight decay 0.01 (SegFormer fine-tuning defaults), tiled 1024 px validation. *Result (2026-10-11):* ties
      ConvNeXt-T, ΔPQ −0.001 [−0.013, +0.010]; LB 0.36. As a third ensemble member: +0.004 before the classifier,
      −0.004 after (0.441 vs 0.445) → not added. SegFormer head dropped (1/4-res output).
- [x] **Step 3b — decoders with ConvNeXt-Tiny:** UNet++ is incompatible (needs 1/2-res features); **MAnet**
      (U-Net + attention, full-res) and **UPerNet** (multi-scale context, 1/4 res) on fold 0 vs ConvNeXt + U-Net.
      *Result:* UPerNet −0.015, MAnet −0.006 [−0.015, +0.004] → **U-Net stays.** Benchmarks at batch 8: U-Net
      4.0 GiB / MAnet 4.4 / UPerNet 5.2 / FPN 4.0 / SegFormer 4.4.
- [x] **U-Net phase concluded (2026-10-11):** ConvNeXt-T ≈ MiT-B2 > ResNet34 > EfficientNet-B4; U-Net decoder;
      Skeleton Recall 0.5; ensemble + 8-way TTA + classifier = best (fold-0 nested 0.449, LB 0.38). Recent screens all
      fall within ±0.01. The forum (2026-10-11) suggests our gap is detection (RQ), not outlines (SQ) → next: Mask R-CNN
      (Step 4) and the cheap classifier/size-filter extras.
- [x] **Step 3a — encoder screen (2026-10-10):** skel050 recipe with `tu-efficientnet_b4`
      (`configs/segmentation/archive/unet_effb4_skel050.yaml`) and `tu-convnext_tiny` (`configs/segmentation/unet_convnext_t_skel050.yaml`), fold 0, one per
      GPU in `notebooks/kaggle/kaggle_screen.ipynb`, compared against skel050 (ResNet34). Both: batch 16, lr 7e-4, encoder lr
      ×0.25 (`encoder_lr_mult`), 2 warm-up epochs. Laptop benchmark at batch 8: EfficientNet-B4 5.8 GiB / ConvNeXt-T
      4.0 GiB / ResNet34 2.2 GiB; full 2048² inference ~2 GiB for all (no tiling needed).
- [x] **Step 3 — encoders, then decoders:** U-Net with ResNet34 / ConvNeXt-T / EfficientNet-B4 / MiT-B2 → best 1–2
      encoders with U-Net / UNet++ / UPerNet or FPN. Per-encoder learning rate (lower for the encoder).
- [ ] **Step 4 — other architectures** (time-boxed): Mask2Former if splits/grouping still dominate, HRNet or DSCNet if
      boundaries do. *Revised 2026-10-11:* first **Mask R-CNN** (torchvision ResNet50-FPN v2, COCO weights; package
      `filament.detection` prepared), alone and combined with the U-Net (detection score as a classifier feature;
      recovering missed filaments). Cascade Mask R-CNN / Mask2Former only if this shows the instance route helps.
      *Implemented (2026-10-11):* `filament.detection` (dataset, model, train, instances); raw detections saved as
      `<stem>.npz` and loaded through `filament.postprocess.predictions`, so tuning / compare / submissions are shared
      with the U-Net. Local smoke test (RTX 5070, 200 crops): 0.34 s/step at batch 2, full-image validation of 142
      images in 23 s, 5 GiB peak at batch 4 with a 56 px mask head; PQ 0.24 after 100 steps.
      *Revised screen:* default anchors already cover filament shapes (best anchor IoU < 0.5 for only 4.5% of GT boxes;
      elongated ratios 0.25–4 would bring it to 1.3%), so instead of 28 vs 56 px masks the second run tests a
      **guided** Mask R-CNN: 2nd input channel = `unet_r34_skel050`'s 5-fold OOF probability map (test: its fold-0
      test maps, the same kind of single-model map). Smoke test after 100 steps: PQ 0.34 guided vs 0.24 plain.
      Guide maps uploaded as private dataset `tasadanluca/solar-filament-unet-r34-skel050-maps`. Note: training-image
      guides come from fold models that saw fold 0's labels (stacking; second-order, validation guides are clean).
      Kaggle fold-0 screen in `kaggle_screen.ipynb`: `maskrcnn_r50_v2_m56` and `maskrcnn_r50_v2_m56_guided`.
      *From the next run on:* batch 8 with lr 2e-4 (linear scaling; 2,000 crops per epoch -> 250 steps), measured
      ~1 GiB per image -> ~8.4 GiB at batch 8. Runs started before this change used batch 4, lr 1e-4.
      *Next screen (2026-10-12):* 32 epochs (its own log gives epochs 16/24/32), and `maskrcnn_r50_v2_m56_ftall`:
      whole ResNet50 trained (`trainable_layers` 5 instead of 3, which froze conv1 + layer1 = 0.23 M of 23.5 M
      weights) with the backbone at lr × 0.25 (`backbone_lr_mult`); ~12.1 GiB at batch 8 (1.5 GiB per image).
- [x] **Repo reorganised (2026-10-11):** library and scripts grouped by pipeline stage (`data`, `segmentation`,
      `detection`, `postprocess`, `evaluation`, `analysis`); configs split into current and `archive/`; Kaggle notebooks
      generated by `scripts/kaggle/build_notebooks.py`. Old `rescorer.pkl` files still load; rebuilt submissions are
      byte-identical.
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

## Kaggle notebook — `notebooks/kaggle/kaggle_train_submit.ipynb`
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

| 2026-10-10 | fold-0 screen: 1024 crops + 45 epochs (`unet_r34_c1024`), single fold-0 model + classifier | 0.401 (f0); nested 0.411 | 0.36 | paired ΔPQ vs baseline −0.004 [−0.014, +0.006]; splits not reduced |
| 2026-10-10 | fold-0 screen: + Skeleton Recall w=1 (`unet_r34_c1024_skel`), thr 0.85, single model + classifier | 0.401 (f0, thr 0.85); nested 0.420 | **0.37** | at thr 0.6: ΔPQ −0.021 [−0.036, −0.007] (masks too wide); splits of large filaments 19% → 12%, missed 16.5% → 8.5%, merges 3.0% → 7.6% |

| 2026-10-10 | fold-0 screen: Skeleton Recall **w=0.5**, 512 crops, 30 ep (`unet_r34_skel050`), thr 0.9 | **0.415 (f0)**; nested **0.435** | **0.38** (single fold-0 model; best so far) | ΔPQ +0.010 [−0.001, +0.021], P(better) 0.96; +61 TP / −61 FN; Mauna Loa +0.028 |
| 2026-10-10 | fold-0 screen: Skeleton Recall w=1.0, 512 crops, 30 ep (`unet_r34_skel100`), thr 0.9 | 0.405 (f0); nested 0.426 | ? | ΔPQ −0.000 [−0.013, +0.011] |

| 2026-10-10 | fold-0 screen: skel050 + consensus targets (`unet_r34_skel050_cons`) | 0.414 (f0); nested 0.434 | 0.37 | vs skel050: ΔPQ −0.001 [−0.010, +0.008]; −154 FP, −78 TP |
| 2026-10-10 | fold-0 screen: Skeleton Recall w=0.25 (`unet_r34_skel025`) | 0.412 (f0); nested 0.437 | 0.38 | vs skel050: ΔPQ −0.003 [−0.011, +0.006] |

| 2026-10-10 | **skel050, 5 folds** (OOF on 707 images), 5-model avg + TTA + classifier | **0.412**; nested **0.436** | **0.38** (thr 0.90) / 0.37 (thr 0.95) | vs unet_r34: ΔPQ +0.010 [+0.005, +0.014]; `submission_skel050_5fold_thr095.csv` (Kaggle, thr 0.95) and `_thr090.csv` (thr 0.90, matched to the ensemble's thinner masks) |

| 2026-10-10 | fold-0 screen: ConvNeXt-Tiny encoder (`unet_convnext_t_skel050`), batch 16 | **0.422 (f0)**; nested **0.439** | 0.38 | vs skel050 (r34): ΔPQ +0.008 [−0.004, +0.018], P 0.92; SQ 0.679 |
| 2026-10-10 | fold-0 screen: EfficientNet-B4 encoder (`unet_effb4_skel050`), batch 16 | 0.408 (f0); nested 0.419 | ? | vs skel050: ΔPQ −0.007 [−0.021, +0.006] |

| 2026-10-11 | fold-0 screen: ConvNeXt-T + **UPerNet** (`upernet_convnext_t_skel050`), thr 0.8 | 0.407 (f0); nested 0.431 | 0.37 | vs ConvNeXt-T + U-Net: ΔPQ −0.015 [−0.024, −0.006]; SQ −0.0125 |
| 2026-10-11 | ConvNeXt-T + MAnet / MiT-B2 + U-Net / SegFormer | – | – | failed (NaN / out of memory); fixed and re-queued (MAnet, MiT-B2 + U-Net) |
| 2026-10-11 | local: ensemble ResNet34 skel050 + ConvNeXt-T (fold-0 maps, 4 flips) | 0.424 (f0); nested 0.445 | – | vs ConvNeXt +0.002 [−0.007, +0.011] |
| 2026-10-11 | local: 8-way TTA, ConvNeXt-T / ResNet34 skel050 | 0.426 / 0.415 (f0); nested 0.446 / 0.441 | – | +0.004 / +0.003 vs 4 flips |
| 2026-10-11 | local: ensemble of both with 8-way TTA, thr 0.8 | 0.424 (f0); nested **0.449** | 0.38 | best fold-0 result so far; `outputs/submission_ens_d8_r34_convnext_fold0.csv` (fold-0 models only; test maps as wide as OOF maps; classifier keeps 75.7% on test vs 74.8% OOF) |
| 2026-10-11 | local: ensemble ResNet34 skel050 **5 folds** + ConvNeXt-T fold 0, both 8-way TTA; settings of the fold-0 ensemble (thr 0.8/300, classifier 0.4) | – (no OOF for this mix) | 0.38 | test maps as wide as the tuned-on OOF maps (filament px 0.00390 vs 0.00381; classifier keeps 74.0% vs 74.8%); more ResNet34 folds: no visible LB change |
| 2026-10-12 | fold-0 screen: **Mask R-CNN** R50-FPN v2, 56 px masks, 16 epochs, batch 4 (`maskrcnn_r50_v2_m56`); tuned: mask thr 0.5, score ≥ 0.8, min area 200, no classifier | **0.442 (f0)** | 0.38 | vs ConvNeXt-T + U-Net: ΔPQ **+0.019 [+0.008, +0.033]**, P 1.00; SQ −0.003, RQ +0.032, FP 756 → 575 at equal TP; best epoch = last (still improving). Finds 25% of the filaments the U-Net misses → complementary |
| 2026-10-12 | fold-0 screen: Mask R-CNN **guided** by the U-Net OOF map (`maskrcnn_r50_v2_m56_guided`) | 0.438 (f0) | 0.38 | vs plain: −0.004 [−0.015, +0.007]; inherits the guide's misses (rescues 18% of U-Net misses vs 25% for plain) → dropped |
| 2026-10-12 | local: **combining** Mask R-CNN with the U-Net ensemble (`ens_d8_r34_convnext_f0`), fold 0, all nested | see note | – | U-Net ens + classifier 0.445 (0.451 in fuse.py); Mask R-CNN score only 0.442, + classifier 0.421–0.434 (the score already ranks duplicates better); map average (detections → score-weighted soft maps, `instances_to_maps.py`) 50/50 + classifier 0.450, 67/33 0.447; one classifier over both candidate pools (`fuse.py`) 0.443 / 0.458 / 0.451 for detection pools score ≥ 0.05 / 0.3 / 0.5 → unstable, ≤ +0.005 robustly. Everything converges to ~0.45 → scale up the single Mask R-CNN instead |
| 2026-10-11 | re-run fold-0 screen: **MiT-B2 + U-Net** (`unet_mitb2_skel050`, tiled validation), thr 0.7 | 0.421 (f0); nested 0.434 | 0.36 | vs ConvNeXt-T + U-Net: ΔPQ −0.001 [−0.013, +0.010], P 0.44: a tie with a transformer encoder |
| 2026-10-11 | re-run fold-0 screen: ConvNeXt-T + **MAnet**, fp32 decoder, thr 0.85 | 0.417 (f0); nested 0.441 | ? | vs ConvNeXt-T + U-Net: ΔPQ −0.006 [−0.015, +0.004], P 0.10; trained cleanly (no skipped batches) |
| 2026-10-11 | local: ensemble ResNet34 + ConvNeXt-T + MiT-B2 (fold-0 maps, 4 flips), thr 0.85 | 0.428 (f0); nested 0.441 | – | vs 2-model ensemble: +0.004 [−0.006, +0.015], P 0.78 before the classifier; after the classifier, below it (0.441 vs 0.445) |

*Full-data models: no measurable gain on the public LB (all three at 0.37). Decision for the final recipe: optional; add 1–2 only if GPU time is left over at the end.*

*Public LB ≈ 0.87 × local OOF PQ for both models so far (public test subset somewhat harder, plus noise on about 90 images).*
