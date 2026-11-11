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
      *Next screen (2026-10-12):* two instance models with identical settings except the box
      heads: whole ResNet50 trained (`trainable_layers` 5 instead of 3, which froze conv1 + layer1 = 0.23 M of 23.5 M
      weights) at lr × 0.25 (`backbone_lr_mult`), batch 6 / lr 1.5e-4, 2 warm-up epochs, 32 epochs, 56 px masks:
      `maskrcnn/maskrcnn_r50_v2_m56_ftall` (~9.1 GiB) and **Cascade Mask R-CNN**
      `cascade/cascade_r50_v2_m56_ftall` (~11.3 GiB; batch 8 would need ~15 GiB). Cascade implemented in torchvision
      (`filament.detection.cascade`): 3 box stages at IoU 0.5/0.6/0.7, regression weights 10/20/30, stage loss
      weights 1/0.5/0.25, detached refinement, mask head on stage-0 positives (training) / final boxes (inference),
      score = mean over stages; all stage heads start from the COCO box head. Mask2Former considered and not chosen
      (1/4-resolution masks vs 8 px filaments, slow convergence, ~2 days of integration).
- [x] **Repo reorganised (2026-10-11):** library and scripts grouped by pipeline stage (`data`, `segmentation`,
      `detection`, `postprocess`, `evaluation`, `analysis`); configs split into current and `archive/`; Kaggle notebooks
      generated by `scripts/kaggle/build_notebooks.py`. Old `rescorer.pkl` files still load; rebuilt submissions are
      byte-identical.
- [ ] Cheap extras along the way: classifier features (fragment, probability spread, texture) + lower size filter,
      `pos_fraction` 0.5, disk-distance input channel, 8-way TTA, multi-scale TTA.

## Stage 5b — Roadmap after the Mask R-CNN screens (agreed 2026-10-16)
*Where we stand: every approach converges to ~0.45 honest fold-0 PQ (U-Net ensemble + classifier 0.445–0.451,
Mask R-CNN 0.442 → 0.446 with the whole backbone; combinations ≤ +0.005). Annotators agree at only ~0.34 (SQ 0.634),
so more gain must come from new information (data) or from aligning decisions with the metric, not from mixing
models trained on the same 707 images. Kaggle quota is the bottleneck (30 h/week); the public LB (2 decimals,
~90 images) cannot show gains this size — local paired comparisons decide.*

**Status (2026-10-18)** — tried so far, all fold 0, honest PQ; references: whole-backbone Mask R-CNN 0.446,
ConvNeXt U-Net 0.422 (0.439 with classifier), U-Net ensemble + classifier 0.449:

| Step | What | Result | Verdict |
|---|---|---|---|
| A1 | Neighbouring GONG frames as training data | U-Net −0.008 (significant), Mask R-CNN −0.001; LB 0.38 both | tried — no gain; **parked** |
| B1 | Cascade Mask R-CNN (19 epochs, time budget) | 0.444 (−0.002); ahead at equal epochs; LB 0.37 alone | tried — tie |
| B3 | Dice term in the mask loss (20 epochs) | 0.446 (−0.001); SQ +0.002 | tried — tie |
| D3 | Map average U-Net ensemble + Mask R-CNN + classifier | **0.461**, LB **0.39** (with cascade 0.460, LB 0.38) | tried — **best; final recipe** |
| – | Whole backbone (trainable_layers 5, lr × 0.25), guided Mask R-CNN, 3-detector average, fusion classifier | 0.446 / 0.438 / 0.450 / ≤ +0.005 | tried (see experiment log) |
| Bet 2 | Mask R-CNN at 1.5× / 2× input resolution; 1× detections + 2× outlines | 0.446 / 0.431; hybrid 0.451 (LB 0.38), in the combination 0.451 (LB 0.39) | tried — outlines better, detection worse |
| A2 (bet 1) | Semi-supervised: students on 3,251 pseudo-labelled frames (cut-off 0.6 / 0.5) | 0.447 / 0.448 (LB 0.38 both); finer grid, nested: 0.451 / 0.444 vs reference 0.444. With the U-Net + classifier (nested): 0.6 student 0.456, 0.5 student 0.451, both students 0.459, reference + 0.6 0.461 (`outputs/submission_ens_unet_det2.csv`, 1,264 filaments: **LB 0.39**), reference + both 0.453, 0.6 with detections scored ≥0.5 / ≥0.75 0.452 / 0.455 (without the classifier 0.454, the best so far) — best combination stays 0.461 | tried — detector alone +0.007, gone in the combination; parked |
| D1 | Mask R-CNN TTA (`filament.detection.tta`, `predict.py --tta 8`): flips + transposes, detections matched across views (box IoU ≥ 0.55), mean box / mask, score = sum / views; on the 0.6 student | alone, nested: 1 / 4 / 8 views 0.451 / 0.455 / **0.460**; paired 8 vs 1 view +0.011 [+0.002, +0.019], P 0.99 (RQ +0.013, better at 5 of 6 sites). U-Net + (reference + student TTA) + classifier **0.465** (vs 0.461); U-Net + student TTA 0.460 (0.3 / 0.7 weights 0.459). `outputs/submission_ens_unet_det2_tta.csv`, 1,310 filaments: **LB 0.39** | tried — **gain**; use in the final (TTA the reference too: its checkpoint is not local) |
| E0 | Fine structures / barbs (organiser hint, 2026-10-20): analysis on fold 0. Thin = GT pixels removed by an opening with a 7 px disk | 7.8% of GT pixels; covered by the combination 54% (U-Net 56%, detector 48%) vs 74% of thick pixels; another annotator covers only 50% of them. Oracle, perfect thin parts PQ 0.436 → 0.489 (per-annotator upper bound), perfect thick parts → 0.713. Hysteresis growth into fainter pixels (low 0.3–0.5, ≤ 3–20 px) hurts: −0.003 to −0.032 | analysed — secondary for PQ, but matters for the qualitative 30%; training-side options open (E1–E3) |
| E1 / E2 | ConvNeXt U-Net fold 0 + boundary head (channel 1 = 5x5 boundary band, BCE + Dice, weight 0.5; `_bnd`) / + thin mask pixels (7x7 opening residue) weighted 5x (`_thin`); vs the ConvNeXt reference 0.422 | boundary head: 0.419 (−0.004 [−0.013, +0.006]; SQ +0.001, FP +68), with classifier 0.433 (ref 0.439); thin pixels covered 53.0% (ref 53.7%). Thin weighting: 0.412 (**−0.010** [−0.021, +0.001], P 0.03; SQ −0.008, FP +198), with classifier 0.435; thin pixels covered **63.0%** (thick 75.0% vs 73.2%), tuned threshold 0.95 | tried 2026-10-20 — no gain; boundary head changes nothing, thin weighting draws more barbs but costs PQ (barbs annotated by only ~half the annotators). Kept as an option for the morphology part of the report |
| D2 | Final stage: whole-backbone Mask R-CNN folds 1–4 (2026-10-23, two sessions × 2 GPUs, 32 epochs, 4.8–5.1 h each; `outputs/maskrcnn_ftall_5f`) | val PQ (score 0.7): folds 0–4 0.436 / 0.434 / 0.432 / 0.430 / 0.424 | trained. 8-way TTA confirmed on folds 1–4 (565 images, detector alone, nested): 0.438 → **0.451** (+0.012); test predictions with TTA per fold in `test_tta8_fold<k>`. Fold-0 checkpoint recovered from the Oct 16 ERROR version (browser download): fold 0 0.444 → 0.450 with TTA. All 707 OOF images and 5 test folds now have TTA predictions. Fold-0 combinations (nested, U-Net + classifier): reference TTA 0.457, reference TTA + student TTA 0.451 (vs 0.465 for reference no-TTA + student TTA) — differences of this size are noise on 142 images. Interim submission `outputs/submission_ens_unet_det5f_tta.csv` (U-Net R34 5f + ConvNeXt f0, d8; detector 5 folds × TTA 8, fold maps averaged; tuned on fold 0; 1,215 filaments): **LB 0.39**. Next: ConvNeXt folds 1–4 (main notebook), the 707-image combination |
| Final (5 folds) | ConvNeXt U-Net folds 1–4 trained (2026-10-24; val PQ 0.369 / 0.362 / 0.357 / 0.348 vs fold 0 0.362). All 707 OOF images, everything with 8-way TTA: U-Net = ResNet34 5f + ConvNeXt 5f (`unet5f_d8`), detector = whole-backbone Mask R-CNN 5f (`ref_tta_maps`; test = mean of the 5 fold maps). Tuned + classifier on 707 images (8,543 components), nested: U-Net alone 0.445; U-Net / detector 50/50 0.452, 40/60 0.452, **60/40 0.455** (thr 0.65, area 400, classifier 0.4); detector alone (no classifier) ~0.451. Note: the earlier LB 0.39 submissions used ResNet34 + ConvNeXt **fold 0 only** on the U-Net side. Submission `outputs/submission_final_w60.csv` (1,176 filaments; test calibration matches OOF: px 0.00376 vs 0.00378, classifier keeps 78.7% vs 80.0%) Detector-based alternatives: (a) detector alone, 5 folds × TTA merged per filament (same rule as the TTA views; score = sum / 5), tuned on 707 images: thr 0.45, area 100, score 0.8 → 0.452 in-sample (~0.451 nested); test calibration matches (6.36 vs 6.30 instances/img); `outputs/submission_det5f_tta.csv`, 1,144 filaments: **LB 0.39**. (b) combination 40/60, `outputs/submission_final_w40.csv`, 1,161 filaments: **LB 0.39**. (The 60/40 submission: **LB 0.38**.) | built — the honest 707-image estimate is lower than the fold-0 numbers (0.461–0.465 were optimistic: fold 0 easier + noise on 142 images) |
| B5 | **Mask2Former** (`filament.detection.mask2former`, Hugging Face `transformers`; COCO instance weights, 100 queries, 1 class): same `Instances` output as Mask R-CNN (query score = class prob × mean mask prob; 1/4-resolution masks cut out around their box at 128 × 128), so TTA / tuning / fusion / submissions are unchanged. AdamW 1e-4, backbone × 0.1, wd 0.05, clip 0.01, 1024 crops, batch 3, pixel decoder recomputed (6.0 → 3.6 GiB per crop); ~0.9 s per crop on a T4 (est.) → ~30 min per epoch; `fit_epochs` shortens the cosine schedule after epoch 1 to what fits in 5.6 h. Fold-0 screen (2026-10-24, screen notebook): `mask2former_swin_t` (GPU 0) and `mask2former_swin_s` (GPU 1), vs the whole-backbone Mask R-CNN fold 0 (0.444 without TTA) | **Result (2026-10-25):** real T4 speed 1,430 / 1,540 s per epoch (Swin-T / S) → 12 / 11 epochs in 5.6 h; no diverged batches. Tuned, vs Mask R-CNN fold 0 without TTA (0.446 with the same grid; reference `outputs/mrcnn_ftall_f0_notta`): Swin-T **0.416** (−0.030 [−0.046, −0.015]; SQ −0.004, RQ −0.041, FP +128, FN +40, worse at all 6 sites), Swin-S 0.397 (−0.049 [−0.068, −0.033]; far fewer detections: TP −255). Outlines are nearly as good despite the 1/4-resolution masks; detection is the problem. Learns more slowly: val PQ 0.40 only at the end of its 12-epoch schedule, Mask R-CNN (fold 1) 0.40 at epoch 4 and 0.42 at epoch 12. Merged per filament with Mask R-CNN (as TTA views; masks resized to 56): 0.400 (+74 TP, +514 FP). Score cut-off at the grid edge (0.8 / 0.9). Outputs: `outputs/kaggle_runs/solar-filament-v2_m2f`. Submission Swin-T fold 0, no TTA (`outputs/submission_m2f_swin_t_f0.csv`, 1,391 filaments, 7.7 per image vs 7.4 on fold 0): **LB 0.37** | tried — **clearly worse; dropped**: matching Mask R-CNN would need ≥ 3× the epochs (~15 h per fold), 5 folds do not fit the remaining quota |
| D5 | **Per-filament fusion** instead of map averaging (2026-10-25, local, 707 OOF images, U-Net `unet5f_d8` thr 0.8, detector `det5f_tta` thr 0.45 / area 100). Motivation (complementarity analysis, 8,199 GT filaments, IoU > 0.5): found by both 56.1%, U-Net only 10.5%, detector only 6.3%; outlines equal when both find it (mean IoU 0.688 each, best-of-two oracle 0.705); detector FP 2,174 (84% also predicted by the U-Net), U-Net FP 3,763 (55%). (1) detector filaments (score ≥ 0.8) + U-Net components not covered by them (< 50% covered, ≥ 100 free px) added when a classifier trusts them (rescoring features + coverage / nearby detection score; AUC 0.785 vs 0.782 for the U-Net probability alone). (2) detections with score ≥ 0.3 kept by a classifier with U-Net agreement features (U-Net mean probability inside, share > 0.5 / 0.8; AUC 0.853 vs 0.846 for the score). (1) + (2) with both cut-offs chosen jointly | detector alone 0.452 (nested 0.4517). (1) **0.456** nested (cut-off ~0.6: +156 TP, +144 FP); (2) 0.4536 nested; (1) + (2) 0.4537 nested (gains do not stack: the extra detections cover what the U-Net would add) | tried — **no gain over the map average** (60/40 + classifier 0.455): both fusions converge to ~0.455; not used. The one-off scripts are not in the repository |
| A3 / E3 / A2b | **Last screens (agreed 2026-10-25)**, fold 0 vs the whole-backbone Mask R-CNN (0.446), two sessions × 2 GPUs. (a) **Copy-paste** (`filament.detection.copy_paste`, `_cp`): half of the annotated crops get 1–3 training filaments of the same site pasted in, as darkness relative to their own inpainted background (sources ≥ 5% darker; 0.8% skipped), on the disk at a similar distance from the centre (±0.15 R), away from every annotator's boxes (+10 px); +~50 ms CPU per crop. (b) **112 px mask head** (`m112`): 2.44 vs 1.83 GiB and 0.225 vs 0.135 s per image (laptop) → batch 4 / lr 1e-4, 24 epochs. (c) (a) + (b). (d) **Pseudo-label student with full real exposure** (`_pseudo_full`): 2,000 real + 1,000 pseudo crops per epoch (first students: 1,000 + 1,000), same fold-0-clean labels, cut-off 0.6. Screen: (a) + (b) in `kaggle_screen.ipynb`, (d) + (c) in `kaggle_train_submit.ipynb`. First attempt: both 112 px runs out of memory at batch 4 (~step 220–250; up to 128 positive RoIs per image × ~19 MB at 112 px) → batch 2 / lr 5e-5, sessions restarted | **Result (2026-10-26)**, tuned, paired vs the reference (0.446, same grid; outputs `outputs/kaggle_runs/solar-fillament_last`, `solar-filament-v2_last`): (d) pseudo full **0.449** (+0.003 [−0.007, +0.012], P 0.71; FP −35, TP −9; best val 0.443 vs 0.436); merged per filament with the reference 0.448 (+0.002 [−0.006, +0.009]). (b) 112 px 0.441 (−0.005 [−0.018, +0.006]; **SQ −0.001**: finer masks do not give better outlines; TP −54, FP −78). (a) copy-paste 0.439 (−0.007 [−0.016, +0.002], P 0.06; SQ −0.003, RQ −0.008). (c) 0.437 (−0.009 [−0.020, +0.001]). Epoch times on a T4: 550 / 950 / 1,000 / 735 s for (a) / (b) / (c) / (d). Submission (d), fold 0, no TTA (`outputs/submission_det_pseudo_full_f0.csv`, 1,298 filaments on 177 images, 7.3 per image as on fold 0; the pipeline reproduces 0.4491 on fold 0): **LB 0.37** (earlier students 0.38) | tried — **no gain; architecture phase closed**. The pseudo student ties again (as at half exposure); the outline limit is the annotations, not the mask resolution; pasted filaments slightly hurt (likely a context shortcut) |
| Rubric | `scripts/evaluation/rubric_metrics.py` (707 OOF images, classifier out of fold; outputs in `outputs/rubric`): PQ / matched Dice / pair Dice / pixel Dice / 1:n / n:1 / missed. w40 0.453 / 0.810 / 0.735 / 0.665 / 3.4% / 2.2% / 23.2%; w60 0.455 / 0.809 / 0.732 / 0.671 / 3.5% / 2.4% / 22.1%; detector 0.452 / 0.810 / 0.731 / 0.647 / 2.9% / 3.3% / 22.9%; human vs human 0.340 / 0.773 / 0.660 / 0.613 / 3.2% / 3.0% / 17.6%. Found rate small / medium / large GT: w60 46% / 72% / 69%, detector 52% / 67% / 67%, human 41% / 58% / 63% | done 2026-10-25 — the three candidates tie on every criterion; the map average has better pixel Dice and fewer merges, the detector finds more small filaments and splits less |
| C2b | **DINOv3 pretraining (agreed 2026-10-26)**: same ConvNeXt architectures, weights distilled from Meta's self-supervised DINOv3 (timm `*.dinov3_lvd1689m`, not gated, DINOv3 license; external pretrained weights are allowed like ImageNet / COCO). Literature: fine-tuned DINOv3 ConvNeXt-T beat ImageNet ConvNeXt-T by ~10 mIoU on industrial segmentation (arXiv 2605.23472; one study). (a) U-Net ConvNeXt-T DINOv3 vs `unet_convnext_t_skel050` (0.422 / 0.439 with classifier), config otherwise identical; (b) U-Net ConvNeXt-S DINOv3; (c) Mask R-CNN with a DINOv3 ConvNeXt-T backbone (`TimmFPNBackbone`: new FPN input convs, COCO FPN output convs / RPN / heads kept; laptop: 2.17 vs 1.79 GiB per image, 0.38 vs 0.28 s per step → batch 4 / lr 1e-4, schedule fitted to 5.6 h) vs 0.446. ConvNeXt-S detector: 1.34 s per step on the laptop → not screened. Screen notebook: (a) then (b) on GPU 0, (c) on GPU 1 (~6 h of the 11 h left this week) | **Result (2026-10-26)**, fold 0, paired (`compare.py`): (a) U-Net T DINOv3 0.415 vs 0.422 (−0.008 [−0.016, +0.000]; 0.436 vs 0.439 with classifier); (b) U-Net S DINOv3 0.418 (−0.005 [−0.014, +0.005]; 0.438 with classifier); (c) detector 0.444 vs 0.446 (−0.003 [−0.016, +0.009]; same settings −0.003): better outlines (SQ +0.004) but fewer detections (TP −130, FP −230; best score cut-off 0.9). As a second model merged per filament with the ResNet50 detector (`merge_views`, tuned on fold 0): 0.451 (+0.0045 [−0.004, +0.013]) vs 0.448 for ResNet50 + the pseudo-label ResNet50, so the backbone diversity adds +0.002 [−0.006, +0.010] over a second ResNet50. Outputs: `outputs/kaggle_runs/solar-filament-v2_dino`, `outputs/merge_{dino,pseudo}_f0`. Fold-0 submissions (2026-10-27): U-Net S + classifier (no TTA; 0.438), U-Net S d8 TTA + classifier (`outputs/unet_s_dinov3_f0_d8`; re-tuned and refit on TTA maps: 0.410 without / 0.433 nested with classifier, TTA *hurts* here), detector TTA8 (`outputs/mrcnn_dino_f0_tta8`: 0.452, +0.009 vs 1 view), detector TTA8 → maps + classifier (`outputs/mrcnn_dino_f0_tta8_maps`: 0.429 without, **0.455 nested** with). LB: 0.37 / 0.38 / **0.39** / 0.38 (in that order); LB/local 0.83-0.88 as before, no new best (5-fold detector TTA and w40 are also 0.39). **Full-data models (2026-11-03)**, `solar-filament-v2_full`: DINOv3 detector on all 707 images, seeds 42 / 43, 29 epochs (last epoch kept; fold-0 val PQ flat from epoch 18), ~630 s per epoch, no skipped batches. Test with TTA8 and the fold-0 TTA settings (thr 0.45, area 0, score 0.8; `outputs/mrcnn_dino_full_tta8`): 7.27 / 7.40 filaments per image vs 6.88 for the fold-0 model, same median area (~1,280 px); agreement (PQ of one against the other) s42 vs s43 0.87, full vs fold 0 0.84, vs det5f 0.80. The score cut-off sits on a plateau on fold 0 (0.7 / 0.8 / 0.9: 0.449 / 0.452 / 0.443), so the +6% detections are within it. Submissions `submission_mrcnn_dino_full_{s42,s43,both}.csv` (both = seeds merged per filament) and `_both_s083.csv` (cut-off 0.83: count matched to the fold-0 model): LB **0.39** all three. Final picks: `submission_final_w40.csv` (best measured, both families) + `submission_mrcnn_dino_full_s42.csv` (efficient: one model, ~2.2 s per image). **U-Net + DINOv3 detector (2026-11-03)**, fold 0, map average with the U-Net ensemble (`unet5f_d8`) + classifier, nested: DINOv3 w40 **0.461** / w60 0.459 vs ResNet50 (fold-0 OOF of the 5-fold TTA maps) w40 0.461 / w60 0.455, i.e. identical; without classifier 0.441 / 0.435 vs 0.436 / 0.434. Test: U-Net ensemble 40 + full-data s42 detector maps 60 with the fold-0 settings (thr 0.55, area 300, classifier 0.4): `outputs/cmb_dino_w40_f0/submission_unet5f_dino_full_w40.csv`, 1,294 filaments (7.19 per image vs 6.50 on fold-0 OOF; the full-data detector already finds +6%); agreement with w40 0.83: **LB 0.39**; without classifier (`_noclf.csv`, 1,686 filaments; fold 0 0.441): **LB 0.37** (as w40: 0.39 / 0.37). Final picks unchanged | tried — **no gain; dropped**: the U-Nets are slightly worse, the detector ties; +0.002 from a two-backbone ensemble is noise-level and would cost ~18 h of quota for 5 folds |
| A4, B2, B4, C1, C3 | — | — | not tried yet |

**A. Data (new information)**
- [x] **A1 — Neighbouring GONG frames as training data** *(tried 2026-10-18: no gain; parked)*. ~4 frames per training image (±2, ±5 min,
      same site) from gong2.nso.edu, aligned onto the training image's disk, trained with the same masks; inference
      stays single-image. Host ruling: allowed for training, never at inference (see forum). Feasibility checked
      2026-10-12: same image product on the disk (corr 0.994, fit 0.999·x + 0.6, residual 2.65 grey levels = JPEG),
      disk centre/radius within 1 px, a frame every minute; off-disk background differs (23.6 vs 17.3) → use the
      training image's off-disk area. Exclude frames within 30 min of any test image.
      *Built 2026-10-17/18:* `filament.data.neighbours`, `scripts/data/download_neighbours.py` + `align_neighbours.py`,
      `NeighbourSampler` in both datasets (`neighbours` / `neighbour_p` config options; validation and OOF stay on the
      originals). 2,628 frames downloaded (1.6 GB, 0 failures); annotated images are often a site's first frame of
      the day → later fallbacks (+8, +10 min). Lessons: bilinear sub-pixel alignment erased ~70% of the fine texture;
      disk-edge fitting added 3–5 px errors, while the archive frames are already co-registered by NSO (~1 px) → frames
      are used as they are, moved only for a clear content offset (≥ 3 px, 35 frames); quality check on smoothed
      images (σ 3 px), since raw correlation is dominated by seeing noise. 2,396 kept (91%) for 663 images; random
      sample of 30: masks fit within median 1 px (90% ≤ 2 px) relative to the training image itself; filament
      contrast 0.93× the original. Private Kaggle dataset `tasadanluca/solar-filament-neighbours`. Screen prepared:
      ConvNeXt and ResNet34 U-Net + neighbours (GPU 0), whole-backbone Mask R-CNN + neighbours, 32 epochs (GPU 1).
      **Result (2026-10-18, fold 0, neighbour_p 0.5, 1,912 frames for 531 of 565 training images):** ConvNeXt U-Net
      0.422 → 0.415, ΔPQ −0.008 [−0.015, −0.000] (SQ −0.005, FP +70; worse at B, L, M, T); with classifier 0.439 →
      0.435. Mask R-CNN whole backbone 0.446 → 0.445, −0.001 [−0.010, +0.007]. → no gain in this form. Likely causes:
      ~1–2 px residual misalignment acts as outline label noise for the pixel-wise U-Net; neighbours are slightly
      softer (contrast 0.93×) than the annotated frames; little variation the augmentations do not already cover.
      **Parked, not dropped (user, 2026-10-18).** Follow-ups if revisited: neighbours for the detector only (no harm
      there); lower `neighbour_p` (0.2–0.3); only neighbours with sharpness ≥ the original; per-frame mask registration
      instead of image registration; neighbours as an extra input for semi-supervised pseudo-labels (A2).
      (The ResNet34 + neighbours run silently used the reference config — CRLF line endings broke the config edit —
      and retrained plain ResNet34; fixed, guarded by a test, not re-run.)
- [ ] A2 — Semi-supervised: pseudo-labels on unlabelled GONG images (allowed per the quoted clarification); after A1.
- [ ] A3 — Copy-paste augmentation of real filaments (Ghiasi et al. 2021).
- [ ] A4 — Consensus-aware detector targets (share of annotators who drew a filament); low odds (U-Net consensus
      targets gave no gain).
- [x] Spines: named on the rules page as "additional annotations" next to "may not use any other ground-truth
      metadata for training" → treated as forbidden; masks-derived skeletons are fine.

**B. Heads and losses on Mask R-CNN**
- [x] **B1 — Cascade Mask R-CNN** (`filament.detection.cascade`) *(tried 2026-10-16: tie)*. Batch 4 after an OOM at batch
      6; stopped by the time budget after epoch 19: 0.444 vs 0.446 (−0.002 [−0.011, +0.006]), but ahead at equal epochs
      (val 0.444 vs 0.427 at epoch 18); fewer FP (415 vs 568), more FN; LB 0.37 alone, 0.38 with the U-Net ensemble.
- [ ] B2 — Mask Scoring head (score × predicted mask IoU): ranks detections by mask quality, as PQ rewards.
- [x] B3 — Dice / boundary-weighted loss in the mask head *(tried 2026-10-16, Dice only: tie)* (outlines; limited: our
      SQ 0.676 > human-vs-human 0.634). Result: 0.446 at 20 epochs (−0.001 [−0.010, +0.008]), SQ +0.002.
      *Implemented 2026-10-16* (`filament.detection.losses`, `mask_dice_weight`); queued with B1: both 20 epochs,
      `max_hours` 3.8, fold 0, vs the whole-backbone Mask R-CNN (0.446 at 32 epochs; its epoch-20 validation 0.428).
- [ ] B4 — Boundary branch (BMask R-CNN) / PointRend: outlines and fine structures; small expected gain.

**Risky bets (agreed 2026-10-18; the 5-fold final waits until the end)**
- [ ] **Bet 2 — higher input resolution for Mask R-CNN** (`scale` option): training crops of 1024/scale native px
      upscaled to 1024 (cubic; masks nearest), full images predicted at 2048 × scale (boxes mapped back). Memory at
      inference: 1.7 / 3.6 / 6.2 GiB at 1× / 1.5× / 2×. Screen: `maskrcnn_r50_v2_m56_ftall_s2` (GPU 0) and `_s15`
      (GPU 1), 32 epochs, validation every 4 epochs, vs 0.446 at 1×.
      **Result (2026-10-18, fold 0):** 1.5× 0.446 (−0.000 [−0.011, +0.011]); 2× 0.431 (**−0.015** [−0.028, −0.002]).
      Outlines improve steadily with resolution (SQ +0.006 / +0.009), detection gets worse (RQ −0.006 / −0.030; at
      2× FN 553 → 714, small filaments < 800 px found 58% → 43%): each crop covers ¼ of the area and the receptive
      field half the solar distance, so faint and small filaments are recognised worse. → no gain as a whole model.
      Follow-up (local): detections from the 1× model, outlines from the 2× model where they match.
      **Follow-up result (`scripts/detection/swap_outlines.py`, box IoU ≥ 0.5, ~79% of detections swapped):**
      1× detections + 2× outlines **0.451** (+0.005 [−0.001, +0.010], P 0.95; SQ +0.002, RQ +0.005) — best single
      detector; + 1.5× outlines 0.450 (+0.003). In the U-Net combination it does not help (nested 0.451 vs 0.461;
      without classifier 0.432 vs 0.433): the U-Net already supplies the outlines. Submission
      `outputs/submission_det1x_outline2x.csv`: **LB 0.38**. U-Net ensemble + hybrid submission
      `outputs/submission_ens_unet_hybrid.csv` (1,254 filaments; test maps match OOF: filament px 0.00373 vs 0.00364):
      **LB 0.39** (same as with the plain 1× Mask R-CNN).
- [ ] **Bet 1 — semi-supervised (A2):** unlabelled frames from days ≥ 2 days from every training/test image (one
      middle frame per competition site and day; `scripts/data/download_unlabelled.py`, CPU notebook
      `kaggle_download.ipynb`, ~3,000 frames); teacher = fold-0-clean combination (ResNet34 + ConvNeXt + Mask R-CNN,
      fold 0 only); confident instances as targets, crops with unsure instances skipped; student Mask R-CNN on
      50% real + 50% pseudo crops vs 0.446.
      *Built 2026-10-18/19:* 3,252 frames from 600 days (3,251 pass; 448–581 per site; Kaggle CPU download timed out →
      local). Teacher `teacher_f0` (map average of ResNet34 + ConvNeXt U-Nets, 4-flip TTA, and the first Mask R-CNN,
      all fold 0; nested fold-0 PQ 0.449), ~1.7 s per frame. 14,528 candidates: 8,628 with P ≥ 0.6 (more than the
      8,199 annotated filaments), 2,686 unsure (0.3–0.6), 3,214 background; median 1 confident per frame. Archive
      frames carry text/logo outside the disk → blanked for the student. Private dataset
      `tasadanluca/solar-filament-pseudo`. Screen: students with cut-off 0.6 (GPU 0) and 0.5 (GPU 1).
      **Result (2026-10-19, fold 0, 32 epochs, 50% pseudo crops):** cut-off 0.6 0.447 (+0.001 [−0.008, +0.010]),
      cut-off 0.5 0.448 (+0.0015 [−0.008, +0.010]) vs 0.446; FP/FN and found rate per size unchanged; best validation
      PQ slightly higher (0.444 vs 0.436). The students saw half as many real crops and still matched the reference.
      In the combination: U-Net + student 0.456, U-Net + mean(reference, student) 0.461 (= best). → tie; parked.
      LB 0.38 for both students (= reference).
      Possible follow-up: keep the full real exposure (4,000 crops per epoch, half pseudo; ~2× training time).

**C. New architectures (parked)**
- C1 Mask2Former: 1/4-resolution masks vs 8 px filaments, slow convergence, ~2 days → parked.
- C2 Bigger detector backbones (ResNeXt-101 / Swin): need detectron2 / MMDetection for COCO weights.
- C3 U-Net variants (Lovász, disk-distance channel, multi-scale TTA): saturated, ±0.005.

**D. Inference / combination (for the final submission)**
- [ ] D1 — TTA for Mask R-CNN (flips/rotations, instance-level merge).
- [ ] D2 — Several fold models of the best detector, merged.
- [x] D3 — Map averaging with the U-Net ensemble + classifier *(tried: best result)*: first Mask R-CNN ≤ +0.005;
      whole-backbone Mask R-CNN **nested 0.461, LB 0.39** (best so far) → the final recipe.
- [x] D4 — Explicit pixel-overlap check in `validate_submission` *(done 2026-10-26)*: pairwise RLE IoU finds candidate
      pairs, only those are decoded; reports pairs / images / shared pixels; tested; run by `make_submission.py` and the
      notebooks. The final candidates have 0 overlapping pixels (the pipeline is overlap-free by construction).

**Order (updated 2026-10-18):** 5 folds of the final recipe (whole-backbone Mask R-CNN + ConvNeXt U-Net, folds 1–4)
on Kaggle, while D4, D1 and B2 are built locally → B2 screen if ready → A2 if there is headroom → from ~Oct 29:
final combination tuned on all 707 OOF images, D1–D2, report (deadline Nov 15).

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
- [x] Watch inference time (computational efficiency is judged). *Measured 2026-10-26*
      (`scripts/evaluation/time_inference.py`, RTX 5070 Laptop, 20 test images after 2 warm-up, `outputs/inference_timing.csv`),
      seconds per 2048² image: detector 1 model, 1 view **0.45** (peak 1.7 GiB); detector 5 folds × 8 views **15.8**
      (2.4 GiB; 40 passes at 0.39 s vs 0.24 s for a single pass); U-Net ConvNeXt 1 model **0.68**; w40 + classifier **25.7**
      (U-Net 10 models × 8 views 10.4 s, detector 14.8 s, post-processing + classifier + RLE 0.25 s; 3.6 GiB). Whole
      test set (180 images): 1.3 / 47 / 2.0 / 77 min. Quality per pipeline (707 OOF images): one detector without TTA
      ~0.438–0.444 vs 0.452 with 5 folds × TTA → 35× faster for −0.01 PQ. CPU stages (JPEG, disk fit, components,
      classifier, encoding) are < 0.4 s.
- [x] **Final pick (2026-11-05): one selectable submission,
      `submission_mrcnn_dino_full_s42.csv`** (full-data DINOv3 Mask R-CNN, seed 42, 8 views, thr 0.45, score 0.8, no
      classifier; LB 0.39; measured 2.32 s per image, 7 min for the test set, 1.45 GiB peak, RTX 5070 Laptop, 40 images).
      Against `final_w40` (LB 0.39, 25.7 s per image, 15 models, classifier): the rubric's 70% lists mean Dice, PQ and the
      Dice / IoU / 1:n / n:1 distributions, while efficiency appears only in the Overview. Paired on fold 0, each scored as
      submitted: fold-0 DINOv3 detector −0.006 [−0.019, +0.006] vs final_w40, so the PQ gap is not resolved and the
      full-data model's extra 25% of images cannot be measured. Chosen for speed, simplicity and a real end-to-end
      reproduction. Rubric metrics of the fold-0 model on 142 images: `notes/results/dino_fold0/` (PQ 0.452, SQ 0.685,
      RQ 0.660, 1:n 3.9%, n:1 2.9%). A pixel-Dice comparison of the two was not run.
- [ ] Repo cleanup: docs, modular code, pinned `requirements.txt`, reproducible end-to-end notebook.
      Rule (Final Submission tab): "The committee should not require any additional file to reproduce the submitted
      predictions", so the checkpoint is downloaded by `scripts/final/download_weights.py` from the GitHub release
      `v2.0-weights`; `filament.final` + `scripts/final/predict.py` go from images to CSV and reproduce the submitted file
      byte for byte on the local machine (1,308 of 1,308 masks). The checkpoint is a derivative of DINOv3 weights, so it is
      distributed under the DINOv3 License (`LICENSES/DINOv3_LICENSE.md`, attached to the release), with "Built with
      DINOv3" in the README; the report must acknowledge DINOv3. Release `v1.0-weights` (the 15 `final_w40` checkpoints)
      was deleted on 2026-11-07, as nothing used it. Still to test once the repo is public: `download_weights.py` from a clean folder.
      Clean-install check on Kaggle (2026-11-07, `notebooks/kaggle/kaggle_check_pipeline.ipynb`, commit d72171a, T4,
      Python 3.13.16): `pip install -r requirements.txt` OK (torch 2.11.0+cu130 sees the GPU; the only pip warnings are
      conflicts with unrelated preinstalled Kaggle packages), checkpoint hash OK, the walk-through ran without errors in
      12.5 min. On the T4 the regenerated CSV is not byte-identical (1,307 vs 1,308 filaments, 854 masks identical,
      agreement with the submitted file PQ 0.996): fp16 numerics differ between GPUs, and a few detections sit at the
      score and mask thresholds.
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
| 2026-10-12 | fold-0 screen: Mask R-CNN, **whole backbone** at lr × 0.25, batch 6, 2 warm-up, 32 epochs (`maskrcnn_r50_v2_m56_ftall`); tuned 0.5 / 200 / score ≥ 0.8 | **0.446 (f0)** | 0.38 | vs first Mask R-CNN: +0.005 [−0.006, +0.015], P 0.80 (TP +13, FP −7); val PQ rises to ~epoch 22–28, then flat (best epoch 26); LB 0.38 |
| 2026-10-16 | fold-0 screen: **Cascade** Mask R-CNN (`cascade_r50_v2_m56_ftall`), batch 4 / lr 1e-4, whole backbone, stopped by the time budget after epoch 19 (best 18) | 0.444 (f0) | – | vs whole-backbone Mask R-CNN (32 ep): −0.002 [−0.011, +0.006]; very different operating point: FP 568 → 415, FN 553 → 636 (precise, cautious). **LB 0.37** alone. At equal epochs ahead: val PQ 0.444 vs 0.427 at epoch 18. ~650 s per epoch on a T4 |
| 2026-10-16 | fold-0 screen: Mask R-CNN + **Dice** in the mask loss (`maskrcnn_r50_v2_m56_ftall_dice`), 20 epochs | 0.446 (f0) | – | vs whole-backbone Mask R-CNN (32 ep): −0.001 [−0.010, +0.008]; SQ +0.002 (outlines, as intended), RQ −0.003. At epoch 18: val 0.437 vs 0.427. ~527 s per epoch |
| 2026-10-16 | local: U-Net ensemble + **cascade** / + Dice model / + average of the 3 detectors (map average 50/50 + classifier) | nested **0.460** / 0.449 / 0.450 (no classifier 0.439 / 0.437 / 0.436) | 0.38 (U-Net + cascade) | none beats U-Net + whole-backbone Mask R-CNN (0.461); cascade ties it with 19 vs 32 epochs and ~650 s vs 569 s per epoch → cheapest detector for the 5 folds. Differences of ±0.01 between these variants are within fold-0 noise (142 images) → decide on more folds |
| 2026-10-16 | local: **U-Net ensemble + whole-backbone Mask R-CNN**, map average 50/50 (`ens_unet_mrcnnft`; detections → score-weighted soft maps), thr 0.65 / 300, classifier min_p 0.3; test = 5-fold ResNet34 + ConvNeXt (8-way TTA) + Mask R-CNN fold 0 | nested **0.461** (best so far; without classifier 0.433) | **0.39** (first above 0.38) | without classifier vs U-Net ensemble: +0.009 [+0.003, +0.016]; vs Mask R-CNN alone −0.013 (averaging blurs its score filtering; the classifier restores it). Test maps match OOF (filament px 0.00375 vs 0.00366; classifier keeps 82% vs 78%). `outputs/submission_ens_unet_mrcnnft.csv`, 1,344 filaments, 0 overlaps |
| 2026-10-12 | fold-0 screen: **Cascade** Mask R-CNN, same settings, batch 6 | – | – | **out of GPU memory** in epoch 2 (> 14.5 GiB; the 11.3 GiB estimate from short runs missed crops with many filaments); epoch 1 took 632 s. Re-run queued at batch 4 / lr 1e-4 together with the whole-backbone Mask R-CNN on fold 1 (start of its 5-fold set); with 6 h 17 m of Kaggle quota left: 20 / 22 epochs and a `max_hours` 3.8 budget per job (~4.2 h session; T4: Mask R-CNN 569 s per epoch + 47 s per validation) |
| 2026-10-12 | fold-0 screen: Mask R-CNN **guided** by the U-Net OOF map (`maskrcnn_r50_v2_m56_guided`) | 0.438 (f0) | 0.38 | vs plain: −0.004 [−0.015, +0.007]; inherits the guide's misses (rescues 18% of U-Net misses vs 25% for plain) → dropped |
| 2026-10-12 | local: **combining** Mask R-CNN with the U-Net ensemble (`ens_d8_r34_convnext_f0`), fold 0, all nested | see note | – | U-Net ens + classifier 0.445 (0.451 in fuse.py); Mask R-CNN score only 0.442, + classifier 0.421–0.434 (the score already ranks duplicates better); map average (detections → score-weighted soft maps, `instances_to_maps.py`) 50/50 + classifier 0.450, 67/33 0.447; one classifier over both candidate pools (`fuse.py`) 0.443 / 0.458 / 0.451 for detection pools score ≥ 0.05 / 0.3 / 0.5 → unstable, ≤ +0.005 robustly. Everything converges to ~0.45 → scale up the single Mask R-CNN instead |
| 2026-10-11 | re-run fold-0 screen: **MiT-B2 + U-Net** (`unet_mitb2_skel050`, tiled validation), thr 0.7 | 0.421 (f0); nested 0.434 | 0.36 | vs ConvNeXt-T + U-Net: ΔPQ −0.001 [−0.013, +0.010], P 0.44: a tie with a transformer encoder |
| 2026-10-11 | re-run fold-0 screen: ConvNeXt-T + **MAnet**, fp32 decoder, thr 0.85 | 0.417 (f0); nested 0.441 | ? | vs ConvNeXt-T + U-Net: ΔPQ −0.006 [−0.015, +0.004], P 0.10; trained cleanly (no skipped batches) |
| 2026-10-11 | local: ensemble ResNet34 + ConvNeXt-T + MiT-B2 (fold-0 maps, 4 flips), thr 0.85 | 0.428 (f0); nested 0.441 | – | vs 2-model ensemble: +0.004 [−0.006, +0.015], P 0.78 before the classifier; after the classifier, below it (0.441 vs 0.445) |

| 2026-10-26 | **ablation submissions** (5-fold models, tuned on 707 OOF; `outputs/submission_*_noclf.csv`, `submission_unet5f_clf.csv`) | U-Net 5f no classifier 0.425 / + classifier 0.444 (nested) / w60 no classifier 0.431 / w40 no classifier 0.433 | **0.36 / 0.37 / 0.36 / 0.37** | the classifier is worth +0.01 on the LB (+0.02 locally), the detector +0.01 without and +0.02 with the classifier (w40 + classifier 0.39, detector alone 0.39). LB / local ratio 0.83–0.87 for every file |

*Full-data models: no measurable gain on the public LB (all three at 0.37). Decision for the final recipe: optional; add 1–2 only if GPU time is left over at the end.*

*Public LB ≈ 0.87 × local OOF PQ for both models so far (public test subset somewhat harder, plus noise on about 90 images).*
