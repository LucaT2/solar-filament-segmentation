# Literature notes

Short summaries of prior work and what each implies for this competition. Each entry links to its source.

## The dataset itself

**MAGFiLO: Ahmadzadeh et al. 2024, *Scientific Data* 11:1031** ([paper](https://doi.org/10.1038/s41597-024-03876-y), [site](https://www.mlecofi.net/magfilo))
- 10,244 filaments on 958 unique GONG Hα images (2011–2022). Images were annotated by 24 groups of 3 annotators, each
  working on its own copy of a 60-image batch, double-blind, with two review rounds.
- Masks were painted with brush and eraser tools (often started from a clustering-based "auto-annotate" draft that
  ignores barbs), then stored as polygons.
- Rules: "unidentifiable" filaments (37%) are drawn **smooth, without barbs**; left- and right-chiral filaments must show
  barbs clearly enough that the chirality can be read from the mask alone. Disconnected pieces of one filament were
  either joined manually or split into separate filaments; islands under 5% of the area were dropped.
- The only published agreement number is κ = 0.47 / 0.66 on *chirality labels*. The authors say the segmentation task is
  partly subjective, and they deliberately kept disagreements, including "one filament vs. several".
- The JPEGs are NSO's own conversion: exposure normalized to the disk center, disk size normalized, and **limb darkening
  removed by a Fourier "sharpening" filter**.
- *Implication:* the labels are noisy by design. Mask detail depends on the class, which the model has to infer
  implicitly. Heavy brightness correction is probably unnecessary (we check this in the EDA).

## Prior filament segmentation work

**Ahmadzadeh et al. 2019, "Toward Filament Segmentation Using Deep Neural Networks", IEEE BigData** ([arXiv 1912.02743](https://arxiv.org/abs/1912.02743))
- Mask R-CNN on 2048² BBSO images, trained on labels from a classical detection algorithm. This is the organizers'
  own earlier work.
- *Implication:* the organizers know the instance-detection route. I first assumed its low-resolution 28×28 mask head
  would be a poor fit for thin filaments, but **measurement disagrees**:
  - Ground-truth masks sent through a 28×28 head keep PQ 0.94 (filaments are small; median box 96 px).
  - Annotator-vs-annotator PQ only drops from 0.340 to 0.338 when one side passes through the head.
  - Box-overlap suppression (NMS) is no issue: only 1 filament pair has box IoU > 0.5.

  Mask R-CNN remains a legitimate candidate (see "Why U-Net first" below).

**Zhu et al. 2019, "Solar Filament Recognition Based on Deep Learning", *Solar Physics*** ([arXiv 1909.06580](https://arxiv.org/abs/1909.06580))
- U-Net with dropout on BBSO images, trained on automatically generated, hand-corrected masks.

**Guo et al. 2022, "Solar-Filament Detection and Classification Based on Deep Learning", *Solar Physics* 297:104**
- Cited by EdgeAttNet as prior detection and classification work; not read in detail yet.

**Diercke et al. 2024, A&A 686:A213: "A universal method for solar filament detection from Hα observations using semi-supervised deep learning"** ([paper](https://www.aanda.org/articles/aa/full_html/2024/06/aa48314-23/aa48314-23.html), [code](https://github.com/adiercke/DeepFilamentSegmentation))
- YOLOv5 trained on ChroTel → boxes on the whole GONG archive → U-Net trained semi-supervised on pseudo-masks;
  92% accuracy.
- *Implication:* the GONG archive offers far more *unlabeled* Hα images. Pseudo-labeling or self-training on extra GONG
  images is allowed (external data is fine, other ground truth is not) and is a possible late-stage boost.

**Zhu et al. 2025, "Flat U-Net", ApJ 980:176** ([paper](https://doi.org/10.3847/1538-4357/adadff))
- An ultra-lightweight U-Net with channel-attention blocks; Dice 0.82, recall 0.74.
- *Implication:* small models can do well, which helps with the efficiency part of the judging. Recall (missed faint
  filaments) is the typical weak spot.

**Solomon et al. 2025, "EdgeAttNet: Towards Barb-Aware Filament Segmentation"** ([arXiv 2509.02964](https://arxiv.org/abs/2509.02964), [code](https://github.com/dasjar/EdgeAttNet))
- Trained **on MAGFiLO**: a U-Net with edge-guided self-attention at the bottleneck. mIoU 0.645 vs 0.572 for a plain
  U-Net, trained from scratch **without pretrained weights or augmentation**, 50 epochs.
- Their split is random over annotation entries, so the same image under different annotators can land in both train
  and test, which probably inflates their numbers.
- *Implication:* the most directly relevant baseline. Global context (attention) and edge cues help with barbs. A
  pretrained encoder with augmentation should comfortably beat their plain U-Net.

**UMSL student report, "Pixel-Perfect Segmentation of Solar Filaments"** ([pdf](https://irl.umsl.edu/cgi/viewcontent.cgi?article=1358&context=urs))
- U-Net on MAGFiLO, IoU 0.52, precision 0.75, recall 0.63.
- Failure modes they list: **filaments split by small gaps, very small filaments missed, several small filaments grouped
  as one, thin sections dropped.** These are exactly PQ's fragmentation and merging penalties.

**"New dataset and framework for filament detection, classification, and segmentation using deep learning"** ([index](https://exa.ai/library/publication/x3cr115gsbn); authors and venue not checked)
- GONG Hα: DETR for detection, then a U-Net for segmentation and classification. Another two-stage pipeline.

**"Data-Driven Segmentation of Solar Filaments Based on Ground-Based Instrument Data", *Solar Physics* 2026** ([paper](https://link.springer.com/article/10.1007/s11207-026-02741-y))
- Compact U-Nets on Kanzelhöhe Hα. Studies **leakage-aware multi-year sampling** and fixed vs. native input normalization.
- *Implication:* validation splits must avoid temporal leakage, and the choice of input normalization matters for
  small models.

**Hu et al. 2026, "A Modern ConvNet for Solar Filament Detection", RAA (accepted)** ([arXiv 2607.24525](https://arxiv.org/abs/2607.24525))
- MORDEN (Multiscale ORiented DENdritic), a semantic segmentation ConvNet built for multiscale features. It is trained
  on a small hand-annotated Hα set (MHAS) and then used to generate a large set (AHAS).
- Post-processing: **DenseCRF** sharpens mask edges; **DBSCAN** groups pixels into filament instances. The paper reports
  that MORDEN beats the open-access filament models it was compared against.
- *Implication:* the newest filament work still does semantic segmentation followed by instance post-processing, like
  our pipeline. DenseCRF is a cheap post-processing step we could test; DBSCAN could merge nearby pieces of one
  filament, the job we tested with `link_px`.

**"Chirality classification of solar filaments on MAGFiLO", 2025** ([arXiv 2509.18214](https://arxiv.org/abs/2509.18214))
- ConvNeXt-Base classifies filament crops by chirality; per-class accuracy about 0.69 / 0.73.
- *Implication:* not segmentation, but it shows ConvNeXt features transfer well to MAGFiLO crops (consistent with our
  ConvNeXt-T encoder being our best).

## Thin-structure and connectivity methods (vessels, roads, cracks)

**clDice: Shit et al., CVPR 2021** ([paper](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html))
- A Dice-style loss on soft skeletons that rewards connected predictions. Costs about +88% training time.

**Skeleton Recall Loss: Kirchhoff et al., ECCV 2024** ([paper](https://arxiv.org/abs/2404.03010), [code](https://github.com/MIC-DKFZ/Skeleton-Recall))
- Skeletonize the *ground truth* on the CPU, thicken it slightly, and add a soft-recall loss on that tube. It beats clDice
  on 5 datasets for about +8% training time.
- *Implication:* the cheapest way to fight **fragmentation** (filaments broken at thin necks), our main PQ risk. A
  first-choice addition to BCE+Dice in Stage 5.

**DenseCRF: Krähenbühl & Koltun, NeurIPS 2011** ([paper](https://arxiv.org/abs/1210.5644)); used as post-processing in DeepLab v1/v2
- A fully connected conditional random field refines a probability map: pixels that are close together *and* similar
  in intensity are pushed to the same label, so mask edges snap to image edges. Mean-field inference makes it fast.
- *Implication:* a training-free step between the probability map and the threshold. It can help when masks are
  blurry at their edges; it can also erase thin, faint parts. Must be tested with paired CV before use.

## Instance segmentation scored with Panoptic Quality

**Panoptic Segmentation: Kirillov et al., CVPR 2019** ([paper](https://doi.org/10.1109/CVPR.2019.00963))
- Defines PQ = SQ × RQ. With IoU > 0.5, matching is unique when instances don't overlap.
- Note: the competition's metric skips one-to-one assignment, but that only matters for overlapping predictions,
  which we won't produce.

**HoVer-Net: Graham et al., *Medical Image Analysis* 2019** ([arXiv 1812.06499](https://arxiv.org/abs/1812.06499))
- Nuclear instance segmentation evaluated with PQ. Predicts foreground plus horizontal and vertical distance maps, and
  separates touching instances with marker-controlled watershed.
- *Implication:* a proven recipe when instances touch. Whether we need it depends on the EDA finding about how often
  filaments touch or merge.

**StarDist (Schmidt et al. 2018) / Cellpose (Stringer et al. 2021)**
- Learned instance representations (star-convex polygons / gradient flows) for touching objects. StarDist's
  star-convexity fits elongated, curved filaments poorly. Cellpose flows handle arbitrary shapes but are more complex
  than we likely need.

## Why U-Net first, and when Mask R-CNN

Measured on our data (2026-10-03): Mask R-CNN's known weaknesses barely matter here (28×28 mask head: annotator PQ
0.340 → 0.338; NMS affects 0.02% of filaments). Its main strength, separating touching objects, isn't needed either:
filaments almost never touch, and connected components lose nothing (0.3399 → 0.3391).

U-Net (semantic segmentation + connected components) comes first for practical reasons:
- It works on full-resolution 2048² images by simple sliding-window tiling with stitched probabilities. Detectors
  usually resize to about 1333 px or need instance merging across tile borders.
- Dense per-pixel supervision gets the most out of 707 images, and strong pretrained encoders plug in directly
  (`segmentation_models_pytorch`).
- The probability threshold directly controls outline width, the most sensitive PQ knob.
- It's simpler and faster, which helps the efficiency and code-quality parts of the judging.

Mask R-CNN's potential advantage is **object-level reasoning**: a learned confidence per filament, and deciding from
context what counts as one filament. That targets the weakest part of PQ, recognition (human RQ 0.54: missing faint
filaments, merges, splits). It is therefore worth testing in Stage 5, either as a model of its own or as an ensemble
partner. The cheaper hybrid is a U-Net plus a small classifier that rescores each connected component.

## Summary for our approach

1. A full-resolution U-Net-style model with a pretrained encoder is the established baseline. Published MAGFiLO numbers
   are modest (IoU ≈ 0.52–0.65), so there is room to improve.
2. The known failure modes (gaps, thin necks, merged small filaments, missed small ones) map directly onto PQ's
   penalties. The levers are connectivity-aware losses (Skeleton Recall), fragment-merging post-processing, and
   small-object recall.
3. Labels are subjective by design. Train on all annotators, and don't over-trust single-annotator boundaries.
4. Unlabeled GONG images allow semi-supervised boosts later; the MAGFiLO labels must not be used beyond what the
   competition provides.
