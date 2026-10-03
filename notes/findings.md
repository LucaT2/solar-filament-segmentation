# Stage 1 findings

Source: [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb) (analysis code in `src/filament/eda.py`). Literature:
[literature.md](literature.md).

## 1. Humans agree with each other at only PQ ≈ 0.34

Each annotator was scored against every other annotator of the same image, exactly the way the official metric
scores a submission (296 images with 2–3 annotations, 1,196 ordered pairs).

| | PQ | SQ (mean IoU of matches) | RQ |
|---|---|---|---|
| Annotator vs annotator | **0.340** | 0.634 | 0.536 |

- PQ per pair: 10th percentile 0.13, median 0.34, 90th percentile 0.55.
- Agreement differs between annotator groups (about 0.28 for group `0104` vs 0.42 for group `0501`).
- Pixel-level Dice between annotators' union masks: median 0.64.
- Even *matched* filaments overlap only moderately (median IoU 0.63).

**What happens to a filament when another annotator draws it:**

| outcome | share | notes |
|---|---|---|
| match (IoU > 0.5) | 53.6% | |
| overlap but IoU ≤ 0.5 | 19.5% | boundary disagreement |
| missing (no overlap) | 17.8% | **37%** for the smallest 20% of filaments (< 633 px), **30%** for "unidentifiable" |
| merged with a neighbour | 6.3% | |
| split into pieces | 2.9% | mostly the largest filaments |

**Consequences**
- This explains the organizers' remark that "PQ > 0.35 is of great value": it is human-level.
- A model can beat a single human. On a multi-annotated test image the same prediction is scored against 2–3
  annotators, and a prediction close to their *consensus* scores better than any one person's drawing (like averaging
  in an ensemble). Training on all annotators pushes the model toward that consensus.
- Small and faint filaments are inherently ambiguous. FP and FN cost the same (0.5 each), so whether to output a
  borderline blob is a confidence and size threshold question to **tune on cross-validation**, not to guess.

## 2. Filaments almost never touch → semantic segmentation + connected components is enough

- Among 36,064 filament pairs within the same annotator-image, only **5 overlap** and 46 come within 5 px of each other.
  1.1% of filaments have a neighbour within 5 px.
- **What-if test:** take one annotator's masks, discard instance IDs, re-split the union into connected components, and
  score against the other annotators. PQ goes from 0.3399 to **0.3391**, practically no loss. Linking nearby pieces only
  hurts (link 6 px: 0.337, 20 px: 0.326).
- 0.4% of masks have more than one piece (the MAGFiLO cleanup merged or split them).

**Consequence:** the planned instance head (boundary/distance maps + watershed) is **not needed**. Instances =
connected components of a good binary mask. The remaining instance risk is on the *prediction* side: a model breaking
one filament into pieces at thin necks, which is a connectivity problem (Skeleton Recall loss and post-processing), not
a touching-instances problem.

## 3. Outline width must be accurate to about 1 px

Same what-if setup: grow or shrink annotator B's masks, then score against annotator A.

| change | −2 px | −1 px | 0 | +1 px | +2 px | +3 px |
|---|---|---|---|---|---|---|
| PQ | 0.146 | 0.280 | **0.340** | 0.338 | 0.289 | 0.234 |

- Filaments are thin: typical width about 8 px (5th–95th percentile 4.7–18 px), length about 100 px (40–350 px).
  Shifting the boundary by one pixel changes the IoU of a thin shape enough to push many matches below 0.5.
- Shrinking hurts much more than growing: −1 px costs 18% of PQ, +1 px about 0.5%.

**Consequences**
- Predict at **full resolution** (or no more than 2× downsampled with careful upsampling). 4× downsampling makes thin
  filaments 2 px wide and boundaries impossible to place accurately.
- The probability threshold (equivalently, the mask width) is one of the most important hyperparameters. Tune it on
  cross-validated PQ, and when unsure prefer slightly wider masks over narrower ones.
- Losses and augmentations must not systematically shrink masks (e.g. aggressive downscaling of the targets).

## 4. Filaments are faint, thin, and on a non-uniform disk

- Filaments are only about **12% darker** than a 15 px ring around them (median inside 119 vs ring 136).
  Unidentifiable ones are the faintest (10%).
- Outlines are fairly tight: about 91% of mask pixels are darker than the surrounding ring, and only 2–4% of the ring is
  as dark as the filament core.
- Unidentifiable filaments are smaller and smoother (solidity 0.74 vs 0.50, median length 74 vs 125–130 px), as the
  annotation rules required.
- No filament centre lies off the disk (r > R), so predictions outside the disk can be masked away.

## 5. Image properties

- Decoded JPEGs are single-channel grayscale, 2048×2048, with no saturation.
- **Disk geometry varies:** radius 904–1033 px (site-dependent: Mauna Loa about 916, El Teide about 1012), and the center
  is up to about 75 px from the image center. A disk mask needs a per-image estimate (simple thresholding works).
- **Residual limb darkening:** the median radial profile is flat to about 0.4 R, then falls (139 → 125 at 0.74 R → 102 at
  0.89 R → 59 at 0.96 R). Filaments near the limb sit on a darker background. Options: a radius channel or flattening the
  radial profile as input preprocessing (cheap to test).
- **Site differences:** Mauna Loa (M) images are much blurrier (Laplacian variance about 69 vs about 280 elsewhere) and
  lower in contrast (disk std 13.5 vs about 22). Use augmentation with blur and contrast changes, and check CV per site.
- Train and test have the same distributions of site, brightness, sharpness and radius.

## 6. Time structure and cross-validation design

- Train and test cover 2011–2022 in similar proportions, and every site appears in both.
- The time from a test image to its nearest training image (median 37 h; 35% within 24 h) matches the time between
  training images themselves (median 36 h; 31% within 24 h). The organizers split at random over images.

**Consequence:** cross-validation with folds **grouped by image stem** (all annotators of an image in the same fold)
reproduces the train→test situation faithfully. Grouping by date would be more pessimistic than the real test. Folds
are stratified by site and annotator count.

## 7. Annotator styles differ

- Some annotators draw one smooth band over a chain of dark fragments; others outline each piece tightly (see the
  notebook's overlay of `20111218085014Th`).
- Training on all annotations (the organizers say to treat each annotator-image as a separate image) gives the model an
  averaged style. Averaging annotators into soft targets is a second option to compare.

## Changes to the plan

1. **Drop** the multi-task instance head and watershed. Use semantic segmentation → connected components (no linking).
2. **Focus** on (a) accurate boundaries at full resolution, (b) connectivity so filaments don't fragment (Skeleton Recall
   loss), (c) tuning the threshold and minimum area for PQ on CV, (d) small and faint filament recall versus false
   positives.
3. Cross-validation: 5 folds grouped by stem, stratified by site and annotator count. Scoring reproduces the official
   per-annotator-image metric.
4. Input: grayscale with a single channel, per-image normalization, disk mask, an optional radial channel or limb
   flattening, and blur and contrast augmentation.
5. Realistic target: beat the human-vs-human 0.34 on CV. The headroom comes from predicting the annotators' consensus.
