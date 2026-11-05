# Solar Filament Segmentation

Instance segmentation of solar filaments in GONG H-alpha full-disk images, for the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(Kaggle, IEEE BigData 2026 BigDataCup), scored with Panoptic Quality (PQ).

**The whole pipeline is walked through in
[notebooks/solar_filament_pipeline.ipynb](notebooks/solar_filament_pipeline.ipynb).** It needs only this repository
and the competition's images: the checkpoints are downloaded automatically.

## The submitted pipeline

```
image ─┬─ U-Net ensemble: ResNet34 + ConvNeXt-T, 5 folds each, 8 views ──── × 0.4 ─┐
       │                                                                           ├─ average map ─ threshold 0.6
       └─ Mask R-CNN ensemble: ResNet50-FPN v2, 5 folds, 8 views ───────── × 0.6 ─┘     │
                                                                                         ▼
                 submission ◄─ classifier P(match) ≥ 0.4 ◄─ size ≥ 400 px ◄─ connected components
```

- **The U-Nets** give precise outlines at full resolution. They are trained with BCE + Dice + Skeleton Recall, which
  keeps thin filaments connected.
- **Mask R-CNN** gives object-level confidence and keeps neighbouring filaments apart. It is trained with a 56×56 mask
  head and all backbone layers trainable, and its detections are drawn as score-weighted soft masks.
- **The component classifier** is a gradient-boosted model on each component's area, shape, contrast, position,
  observing site and mean probability. It removes candidates unlikely to match an annotated filament, which is worth
  +0.02 PQ.

Every setting was chosen on out-of-fold (OOF) predictions with a paired bootstrap, not on the public leaderboard. The
code for this path is [src/filament/final.py](src/filament/final.py).

## Results

Out-of-fold on all 707 training images, scored with the official metric. Human agreement is one annotator scored
against the others, on the images with several annotators.

| | PQ | SQ | RQ | Median pair Dice | Missed GT | 1:n splits | n:1 merges | Public LB |
|---|---|---|---|---|---|---|---|---|
| **Submitted pipeline** | **0.453** (nested 0.452) | 0.685 | 0.661 | 0.791 | 23.2% | 3.4% | 2.2% | **0.39** |
| Human agreement | 0.340 | 0.633 | 0.537 | 0.718 | 17.6% | 3.2% | 3.0% | – |

Inference on an RTX 5070 Laptop GPU: 25.7 s per 2048×2048 image (77 min for the 180 test images, peak 3.6 GiB).

**Efficient alternative.** One Mask R-CNN with a DINOv3 ConvNeXt-T backbone, trained on all images, with 8 views and
no classifier, scores the same public LB (0.39) at about 2.2 s per image. It is the better choice when speed matters:
`submissions/submission_mrcnn_dino_full_s42.csv`, config
`configs/detection/maskrcnn/maskrcnn_dinov3_convnext_t_m56.yaml`.

[PLAN.md](PLAN.md) is the full experiment log, including everything that did not help (consensus targets, pseudo-labels,
neighbouring frames, copy-paste, Cascade R-CNN, Mask2Former, and more).

## Quick start

Requires Python 3.12 and, in practice, a GPU.

```bash
git clone https://github.com/LucaT2/solar-filament-segmentation.git
cd solar-filament-segmentation
python -m venv .venv
source .venv/bin/activate                     # Windows: .venv\Scripts\activate
pip install -r requirements.txt               # a specific CUDA build: see the comment in requirements.txt
pip install -e .

kaggle competitions download -c filament-segmentation-2026      # after accepting the competition rules
# extract into filament-segmentation-2026/, or set FILAMENT_DATA to the MAGFiLO_1.0_Kaggle_2026 folder

python scripts/final/download_weights.py      # the 15 checkpoints (2 GB) from the GitHub release, hash-checked
python scripts/final/predict.py --out outputs/submission_final.csv
```

The submitted files are in [submissions/](submissions/) for comparison.

## Retraining from scratch

Training all 15 models takes about 40 GPU hours on a Kaggle T4: about 1.5 h per U-Net and 5 h per Mask R-CNN. Each
command below can be run alone.

```bash
# 1. train the 15 models (each also writes its out-of-fold predictions)
python scripts/segmentation/train.py --config configs/segmentation/unet_r34_skel050.yaml --fold 0 1 2 3 4
python scripts/segmentation/train.py --config configs/segmentation/unet_convnext_t_skel050.yaml --fold 0 1 2 3 4
python scripts/detection/train.py --config configs/detection/maskrcnn/maskrcnn_r50_v2_m56_ftall.yaml --fold 0 1 2 3 4 \
    --set name=maskrcnn_ftall_5f

# 2. out-of-fold maps with 8 views
python scripts/segmentation/predict.py --exp unet_r34_skel050 --tta-mode d8 --split oof
python scripts/segmentation/predict.py --exp unet_convnext_t_skel050 --tta-mode d8 --split oof
python scripts/detection/predict.py --exp maskrcnn_ftall_5f --tta 8 --split oof --out oof_tta8
python scripts/detection/instances_to_maps.py --exp maskrcnn_ftall_5f --split oof_tta8 --out ref_tta_maps --out-split oof

# 3. average the maps, tune the post-processing, fit the classifier
python scripts/postprocess/ensemble_maps.py --exps unet_r34_skel050 unet_convnext_t_skel050 --out unet5f_d8 --split oof
python scripts/postprocess/ensemble_maps.py --exps unet5f_d8 ref_tta_maps --weights 0.4 0.6 --out final_w40 --split oof
python scripts/postprocess/tune_postprocess.py --exp final_w40 --thresholds 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 \
    --min-areas 100 200 300 400 500
python scripts/postprocess/fit_rescorer.py --exp final_w40

# 4. use the new settings
cp outputs/final_w40/postprocess.json outputs/final_w40/rescorer.pkl configs/final/
python scripts/final/predict.py
```

To compare two experiments on the same images, use `scripts/evaluation/compare.py` (paired ΔPQ with a bootstrap
confidence interval). `scripts/evaluation/rubric_metrics.py` computes the table above, and
`scripts/evaluation/time_inference.py` the timing.

Runs are logged to Weights & Biases when credentials are available; otherwise nothing is logged.

## Repository layout

The library (`src/filament/`) and the scripts (`scripts/`) follow the same order:
**data → model → probability maps → post-processing → submission**, with evaluation alongside.

```
src/filament/
  final.py                the submitted pipeline: image -> masks
  data/                   annotations (COCO -> tables), images + solar disk, RLE, folds, submission CSVs
  segmentation/           U-Net family, losses, dataset + augmentation, training loop, classical baseline
  detection/              Mask R-CNN and variants, dataset, training, TTA, saved detections
  postprocess/            maps or instances -> components, component classifier, submission
  evaluation/             official PQ metric, fast OOF scoring + grid search, error analysis
  analysis/               exploratory analyses behind notebooks/analysis/01_eda
scripts/
  final/                  download_weights, predict: the submitted pipeline
  setup/ data/ segmentation/ detection/ postprocess/ evaluation/ semi/ kaggle/ figures/
configs/
  final/                  tuned settings, classifier, checkpoint hashes of the submitted pipeline
  folds.csv               the fixed 5-fold split
  segmentation/ detection/  one YAML per experiment; archive/ holds screened-out experiments
notebooks/                the pipeline walk-through; analysis/ (EDA, error analysis); kaggle/ (training notebooks)
notes/                    data findings, error analysis, literature, figures
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
not used, because it contains the test labels, and the spine annotations were not used. The submitted models use no
neighbouring frames. Ground truth comes from the MAGFiLO dataset
([Ahmadzadeh et al., 2024](https://doi.org/10.1038/s41597-024-03876-y)).

## Licenses

The code is under the [MIT license](LICENSE). Pretrained weights used as starting points belong to their authors:
ResNet34 and ConvNeXt-T (ImageNet, through `timm` / `segmentation-models-pytorch`), Mask R-CNN ResNet50-FPN v2 (COCO,
through `torchvision`). The efficient alternative uses DINOv3 weights (Meta, DINOv3 license).
