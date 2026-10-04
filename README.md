# Solar Filament Segmentation

Instance segmentation of solar filaments in GONG H-alpha full-disk images, for the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(Kaggle, IEEE BigData 2026 BigDataCup). Ground truth comes from the MAGFiLO dataset
([Ahmadzadeh et al., 2024](https://doi.org/10.1038/s41597-024-03876-y)). Submissions are scored with Panoptic Quality.

See [PLAN.md](PLAN.md) for the roadmap and experiment log.

## Layout

```
src/filament/     library code (paths, data, preprocessing, RLE, metric, models, post-processing, train, infer)
configs/          one YAML config per experiment
scripts/          command-line entry points
notebooks/        data analysis + the Kaggle train/submit notebook
tests/            unit tests (RLE/submission round-trip, metric equivalence, folds, post-processing)
notes/            competition overview, data findings, literature notes
outputs/          checkpoints, predictions, submissions (git-ignored)
```

## Setup (local, Windows/Linux)

Requires Python 3.12. PyTorch is installed separately so the CUDA build matches the GPU
(CUDA 12.8+ is required for RTX 50-series / Blackwell).

```bash
py -3.12 -m venv .venv                      # Linux: python3.12 -m venv .venv
# activate: .venv\Scripts\activate (Windows) | source .venv/bin/activate (Linux)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-dev.txt         # or requirements.txt without Jupyter/pytest
pip install -e .
python scripts/check_env.py
```

On Windows with Smart App Control enabled, use the official python.org build of Python, which is signed. Unsigned
builds (e.g. the Python that `uv` downloads) and some very recent package releases get their DLLs blocked. See
the comment in `requirements.txt`.

## Data

Download with the Kaggle CLI (after accepting the competition rules):

```bash
kaggle competitions download -c filament-segmentation-2026
```

Extract it into `filament-segmentation-2026/` in the repo root, or set `FILAMENT_DATA` to the
`MAGFiLO_1.0_Kaggle_2026` folder. On Kaggle the data is found automatically under `/kaggle/input`.

## Pipeline

Every model writes one probability map per image (`outputs/<exp>/oof/` for out-of-fold training images,
`outputs/<exp>/test/` for test images). Post-processing (threshold → connected components → size and confidence
filters) is tuned separately on the out-of-fold maps against the exact competition metric.

```bash
python scripts/make_folds.py                                    # once: configs/folds.csv (already committed)

# U-Net: train all folds (writes OOF maps), predict test with the fold ensemble
python scripts/train.py --config configs/unet_r34.yaml --fold 0 1 2 3 4
python scripts/predict.py --exp unet_r34 --tta

# classical baseline instead of the U-Net: python scripts/run_classical.py   (exp name: classical)

python scripts/tune_postprocess.py --exp unet_r34               # -> outputs/unet_r34/postprocess.json
python scripts/make_submission.py --exp unet_r34                # -> outputs/unet_r34/submission.csv
python scripts/make_submission.py --exp unet_r34 --split oof    # OOF CSV, then:
python scripts/evaluate.py --pred outputs/unet_r34/oof_submission.csv
```

**Tracking:** training and tuning runs are logged to Weights & Biases (project `solar-filament-segmentation`, one run
per fold, grouped by experiment) when credentials are available (`wandb login`, or `WANDB_API_KEY`). Without them
nothing is logged. Force with `--set wandb=on|offline|off` (training) or `--wandb ...` (tuning).

On Kaggle, [notebooks/kaggle_train_submit.ipynb](notebooks/kaggle_train_submit.ipynb) runs the same steps, training
two folds in parallel on a T4 ×2 machine.

Tests: `python -m pytest -W ignore::DeprecationWarning`. The metric tests compare against the organizers'
self-evaluation notebook when it is present in the repo root (it is not redistributed here).
