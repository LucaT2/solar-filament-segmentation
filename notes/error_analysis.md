# Stage 4: Error analysis of the U-Net baseline

Source: [notebooks/analysis/02_error_analysis.ipynb](../notebooks/analysis/02_error_analysis.ipynb) (code in `src/filament/evaluation/error_analysis.py`).
Model: `unet_r34` (U-Net ResNet34, 30 epochs), **fold 0 only** (142 images, 235 annotator-images, 1,643 true
filaments). Post-processing: threshold 0.5, minimum area 300 px. **PQ 0.391** (SQ 0.676, RQ 0.578; TP 1,091,
FP 1,043, FN 552). One fold is a small sample (±0.03 noise), so trends matter more than exact values.

## 1. The model already beats human-vs-human disagreement, except on splits

What happens to each true filament, compared with one annotator scored against another:

| | match | IoU ≤ 0.5 | missing | split | merged |
|---|---|---|---|---|---|
| model | **66.4%** | 12.4% | 11.8% | **5.8%** | 3.5% |
| human vs human | 53.6% | 19.5% | 17.8% | 2.9% | 6.3% |

Splitting is the only bucket where the model is worse than a human, and it grows with size: **19% of the largest
filaments (> 3,200 px) are split**.

## 2. Where the errors are

- **Small filaments** (< 633 px): 35% missed (humans: 37%). This is mostly inherent ambiguity.
- **Unidentifiable filaments** (faint, no barbs): 20% missed vs 7% for Left/Right.
- **Near the limb** (r > 0.85 R): 35% missed, but only 34 filaments.
- **Sites:** no site stands out on this fold. Mauna Loa has the most near-misses (16%), El Teide the most misses (18%).

## 3. Missed filaments and near-misses

- Of 194 missed filaments, **110 were seen** (model probability > 0.5 inside them); only 77 were truly invisible to
  the model. The seen ones were removed by the 300 px size filter or covered by a non-matching piece.
- Of 204 near-misses (overlap but IoU ≤ 0.5), 59% have IoU between 0.4 and 0.5.

## 4. Predicted components: many "false positives" are label disagreement

Kept components (after the size filter):

| kind | count | meaning |
|---|---|---|
| match_all | 582 | true positive for every annotator |
| match_some | 183 | true positive for some annotators only |
| overlap_no_match | 261 | touches a filament but IoU ≤ 0.5 |
| pure_fp | 261 | touches no annotator's filament |

- On multi-annotated images only 32% of kept components match *all* annotators, while 33% match *some*: the
  annotators disagree, not the model.
- **Pure false positives** are smaller (median 636 vs 1,364 px), further out (median r = 0.70 R vs 0.56 R) and
  slightly less confident. **Visually most look like real faint filaments near the limb that nobody labelled.**
- **The size filter is efficient:** it drops 624 components, of which only 35 would have matched any annotator.

## 5. Outline width is already right

Grow or shrink every predicted mask:

| change | −2 px | −1 px | 0 | +1 px | +2 px |
|---|---|---|---|---|---|
| PQ | 0.253 | 0.376 | **0.391** | 0.363 | 0.296 |

No systematic bias. The threshold surface is flat for the same reason: the model's probabilities are sharp.

## 6. Fragmentation can't be fixed by linking pieces

Joining components closer than about 2 × `link_px`: best +0.004 (link 4 px: 0.3908 → 0.3945), within noise.
From 10 px on it merges separate filaments and hurts. The gallery shows why: split filaments are mostly chains of
dark patches that some annotators covered with **one long smooth band** including the fainter gaps. The model outlines
only the dark patches, and the gaps are far wider than a few pixels.

## 7. Rescoring components: the biggest cheap gain (+0.03)

A gradient-boosted classifier predicts whether a component matches at least one annotator from: log area, mean
probability, local contrast, elongation, length, r / R and site. It is evaluated out-of-fold within fold 0, grouped by
image.

- AUC 0.78 on kept components vs 0.66 for area alone (0.88 on all components).
- Most important features: **mean probability**, then area, r / R, contrast.
- Dropping kept components with P(match) < 0.3: **PQ 0.391 → 0.419** (FP 1,043 → 622, TP 1,091 → 1,014). The gain is
  stable for cut-offs from 0.2 to 0.5. Replacing the size filter entirely gives the same (0.420).
- The cut-off was chosen on the same fold, so the true gain is probably slightly smaller.

## Priorities for Stage 5

1. **Component rescoring** (+0.03 on fold 0, cheap): train the classifier on the OOF components of all 5 folds, tune
   its cut-off on OOF PQ, and apply it to test components. Needs folds 1–4.
2. **Large and chained filaments** (the model's only bucket worse than humans):
   - larger context: 1024 px crops, or a deeper or attention-based encoder;
   - **Skeleton Recall loss** (rewards covering the whole centreline, including faint connecting parts);
   - longer training: fold 0 was still improving at epoch 30.
3. **5-fold ensemble** with flip TTA (averaging usually adds +0.01–0.02 and stabilizes the components).
4. **Leave alone:** outline width (already optimal), linking (≤ +0.004), small filaments (mostly label ambiguity, at
   human level).
