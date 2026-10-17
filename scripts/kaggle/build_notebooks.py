"""Build the two Kaggle notebooks from one source, so their code cells never drift apart.

* notebooks/kaggle/kaggle_train_submit.ipynb  main session (multi-fold runs, submissions)
* notebooks/kaggle/kaggle_screen.ipynb        parallel session (fold-0 screens)

They differ only in the title and the configuration cell (EXPERIMENTS, inputs, baseline, submissions): edit those
here, re-run, commit, push, then re-import the notebook on Kaggle.

Usage:
    python scripts/kaggle/build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[2]

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Solar Filament Segmentation: experiments on Kaggle

Runs one or more **experiments** (a config + the folds / full-data models to train) on Kaggle GPUs, then for each:
out-of-fold maps → tuned post-processing → component classifier → **paired comparison against the baseline** →
optional submissions. All logic lives in the repository
([LucaT2/solar-filament-segmentation](https://github.com/LucaT2/solar-filament-segmentation)); this notebook only
orchestrates it.

### Settings (*Notebook options* in the right-hand panel)
* Accelerator: **GPU T4 ×2**. Internet: **on**.
* *Add Input*: competition **filament-segmentation-2026**, plus the datasets listed in `CHECKPOINT_DATASETS` (earlier
  results, e.g. the baseline **solar-filament-unet-r34**, created with `scripts/kaggle/prepare_kaggle_dataset.py`).
* *Add-ons → Secrets*: the GitHub token (label `GITHUB_SECRET`) and, optionally, the W&B key (`WANDB_SECRET`). Tick both.
* Run with **Save Version → Save & Run All** so it keeps running in the background.
""")

code("""
# ---- configuration ---------------------------------------------------------------------------------------
# Each experiment: a config from the repo, the folds to train (cross-validation, with validation) and/or seeds of
# full-data models (all 707 images, no validation). Jobs of all experiments are spread over the GPUs.
EXPERIMENTS = [   # fold-0 screen of the transformer encoder MiT-B2, one per GPU
    {"config": "configs/segmentation/unet_mitb2_skel050.yaml",      "folds": [0], "full_seeds": [], "overrides": []},  # + U-Net
    {"config": "configs/segmentation/archive/segformer_mitb2_skel050.yaml", "folds": [0], "full_seeds": [], "overrides": []},  # SegFormer
]
# attached inputs with earlier experiment results, matched by "<notebook>/<experiment>" in their path
# (Add Input -> Your Work -> Notebooks): the screen notebook's output, holding ConvNeXt-Tiny + U-Net fold 0
CHECKPOINT_DATASETS = ["solar-filament-v2/unet_convnext_t_skel050"]
BASELINE = "unet_convnext_t_skel050"   # current best fold-0 model
TTA = True                             # average 4 flips per model when predicting
SUBMISSIONS = {                        # output name -> (experiment, [model names / glob patterns])
    "unet_mitb2_fold0": ("unet_mitb2_skel050", ["fold0"]),
    "segformer_mitb2_fold0": ("segformer_mitb2_skel050", ["fold0"]),
}
GITHUB_SECRET = "Github_key"           # label of the Kaggle secret holding the GitHub token
WANDB_SECRET = "WANDB_API_KEY"         # label of the (optional) Kaggle secret holding the W&B API key
REPO_URL = "github.com/LucaT2/solar-filament-segmentation.git"
REPO_BRANCH = "main"
REPO_DATASET = None                    # alternative to the GitHub token: name of a dataset containing the repo
""")

md("## 1. Setup")
code("""
import os, sys, shutil, subprocess, glob, json, time
from pathlib import Path

T0 = time.time()
def log(msg):   # timestamped progress, so a stalled step is easy to spot in the log
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s] {msg}", flush=True)

log("session started")
# no step may wait for input: git never asks for a password (a rejected token fails instead of hanging)
os.environ["GIT_TERMINAL_PROMPT"] = "0"

def run(cmd, timeout, what):
    \"\"\"Run a setup command; fail loudly (instead of hanging) after ``timeout`` seconds.\"\"\"
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{what} did not finish within {timeout} s (network problem?)") from None
    return res

log("reading secrets")
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
# less GPU-memory fragmentation between training and full-image validation (inherited by all subprocesses)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
try:   # optional W&B tracking; training subprocesses inherit the environment variable
    os.environ["WANDB_API_KEY"] = secrets.get_secret(WANDB_SECRET)
    log("W&B tracking enabled")
except Exception:
    os.environ["WANDB_MODE"] = "disabled"
    log(f"no '{WANDB_SECRET}' secret: tracking disabled")

def github_token():
    try:
        return secrets.get_secret(GITHUB_SECRET)
    except Exception as exc:
        raise RuntimeError(
            f"Kaggle secret '{GITHUB_SECRET}' is not attached to this notebook. In the editor: Add-ons -> Secrets -> "
            f"make sure a secret with exactly this label exists and its checkbox is ticked, or change GITHUB_SECRET "
            f"in the configuration cell. Then Save Version -> Save & Run All again."
        ) from exc

def find_input(name, marker):
    \"\"\"Locate an attached dataset by name (Kaggle's mount layout varies), via a file it must contain.\"\"\"
    hits = [Path(p).parent for p in glob.glob(f"/kaggle/input/**/{marker}", recursive=True) if name in p]
    if not hits:
        raise FileNotFoundError(f"attached dataset '{name}' with {marker} not found under /kaggle/input")
    return min(hits, key=lambda p: len(str(p)))

REPO = Path("/kaggle/working/repo")
if REPO_DATASET:
    shutil.copytree(find_input(REPO_DATASET, "pyproject.toml"), REPO, dirs_exist_ok=True)
elif not REPO.exists():
    token = github_token()
    log("GitHub token read; cloning the repository")
    # token as the password (required for fine-grained tokens; also works for classic ones)
    res = run(["git", "clone", "--depth", "1", "-b", REPO_BRANCH,
               f"https://x-access-token:{token}@{REPO_URL}", str(REPO)], 120, "git clone")
    if res.returncode != 0:   # token never printed
        raise RuntimeError(
            "git clone failed: " + res.stderr.replace(token, "***") +
            "\\nCheck the GitHub token: not expired; 'Repository access' includes solar-filament-segmentation; "
            "'Repository permissions -> Contents: Read-only'.")
log("repo: " + subprocess.run(["git", "-C", str(REPO), "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip())
""")
code("""
# Kaggle images ship torch, numpy, pandas, opencv, scikit-image, scikit-learn, timm and wandb. Install only what is
# missing, pinned to the versions tested locally (requirements.txt), without touching the preinstalled stack.
log("installing packages")
for cmd in ([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "segmentation-models-pytorch==0.5.0",
             "albumentations==2.0.8", "albucore==0.0.24", "pycocotools"],
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO)]):
    res = run(cmd, 300, "pip install")
    if res.returncode != 0:
        raise RuntimeError("pip install failed:\\n" + res.stderr[-2000:])
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import torch, yaml
log(f"packages installed; torch {torch.__version__}, "
    f"GPUs {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
!python scripts/setup/check_env.py
log("environment checked")
""")
code("""
WORK = Path("/kaggle/working")

def load_checkpoint_dataset(name):
    \"\"\"Copy an attached experiment dataset into /kaggle/working/<experiment name>, unpacking zipped folders.\"\"\"
    src = find_input(name, "postprocess.json")
    exp = yaml.safe_load((src / "config.yaml").read_text())["name"] if (src / "config.yaml").exists() else name
    dst = WORK / exp
    shutil.copytree(src, dst, dirs_exist_ok=True)
    # `kaggle datasets create --dir-mode zip` uploads sub-folders as zips; Kaggle may or may not extract them
    for archive in dst.glob("*.zip"):
        if not (dst / archive.stem).is_dir():
            shutil.unpack_archive(archive, dst / archive.stem)
        archive.unlink()
    for sub in [d for d in dst.iterdir() if d.is_dir()]:
        nested = sub / sub.name   # some zips contain the folder itself
        if nested.is_dir():
            for f in nested.iterdir():
                f.rename(sub / f.name)
            nested.rmdir()
    folders = {d.name: len(list(d.iterdir())) for d in sorted(dst.iterdir()) if d.is_dir()}
    log(f"{name} -> {dst}: {sorted(p.name for p in dst.glob('*.pt'))} | prediction folders: {folders}")
    return exp

loaded = []
for n in CHECKPOINT_DATASETS:   # a missing input only costs its comparison, not the whole run
    try:
        loaded.append(load_checkpoint_dataset(n))
    except FileNotFoundError as exc:
        log(f"WARNING: {exc}; continuing without it")
for e in EXPERIMENTS:
    e["name"] = yaml.safe_load((REPO / e["config"]).read_text())["name"]
    # configs/detection/... -> instance model (Mask R-CNN), configs/segmentation/... -> probability-map model
    e["kind"] = "detection" if "/detection/" in e["config"] else "segmentation"
    print(f"{e['kind']} experiment {e['name']}: folds {e['folds']}, full-data seeds {e['full_seeds']}, "
          f"overrides {e['overrides']}")

def kind_of(exp):
    \"\"\"'detection' or 'segmentation' for an experiment folder in /kaggle/working.\"\"\"
    cfg = WORK / exp / "config.yaml"
    return yaml.safe_load(cfg.read_text()).get("kind", "segmentation") if cfg.exists() else "segmentation"

if any(e["kind"] == "detection" for e in EXPERIMENTS):   # download the COCO weights once, before parallel jobs
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
    MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT.get_state_dict(progress=False)
    log("Mask R-CNN COCO weights downloaded")
log("setup done")
""")

md("""
## 2. Train

All jobs (every fold and full-data model of every experiment) are dealt out to the GPUs in turn; each GPU works
through its list in one background process. Logs: `/kaggle/working/train_gpu<k>.log`.
""")
code("""
n_gpu = max(torch.cuda.device_count(), 1)
jobs = []
for e in EXPERIMENTS:
    over = list(e["overrides"])
    # two training processes share Kaggle's 4 CPU cores: 2 data-loading workers each unless overridden
    if n_gpu > 1 and not any(o.startswith("num_workers") for o in over):
        over.append("num_workers=2")
    train = f"scripts/{e['kind']}/train.py"
    for k in e["folds"]:
        jobs.append([train, "--config", e["config"], "--fold", str(k)] + (["--set", *over] if over else []))
    for s in e["full_seeds"]:
        jobs.append([train, "--config", e["config"], "--fold", "-1", "--set", *over, f"seed={s}"])

procs = []
for gpu in range(n_gpu):
    my_jobs = jobs[gpu::n_gpu]
    if not my_jobs:
        continue
    script = " && ".join(subprocess.list2cmdline(["python", "-W", "ignore", *j]) for j in my_jobs)
    log = open(WORK / f"train_gpu{gpu}.log", "w")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
    procs.append((subprocess.Popen(["bash", "-c", script], env=env, stdout=log, stderr=subprocess.STDOUT), log, gpu))
    print(f"GPU {gpu}: {[' '.join(j[2:5]) for j in my_jobs]}")

# Watch the jobs: every few minutes print each GPU's latest log line (visible live in the Kaggle viewer), and warn
# when a log has not changed for a long time.
import time
POLL_S, STALL_S = 120, 1200
last_change = {gpu: (time.time(), 0) for _, _, gpu in procs}
first = True
while any(p.poll() is None for p, _, _ in procs):
    time.sleep(30 if first else POLL_S)   # a first sign of life soon, then every POLL_S
    first = False
    for p, _, gpu in procs:
        path = WORK / f"train_gpu{gpu}.log"
        size = path.stat().st_size
        lines = [l for l in path.read_text(errors="replace").splitlines() if l.strip() and "wandb:" not in l]
        if size != last_change[gpu][1]:
            last_change[gpu] = (time.time(), size)
        quiet = time.time() - last_change[gpu][0]
        state = "running" if p.poll() is None else f"done (exit {p.returncode})"
        warn = f"  !! no new output for {quiet / 60:.0f} min" if quiet > STALL_S and p.poll() is None else ""
        print(f"{time.strftime('%H:%M:%S')} GPU {gpu} {state}: {lines[-1][:160] if lines else '(no output yet)'}{warn}",
              flush=True)
for p, log, gpu in procs:
    log.close()
    print(f"GPU {gpu} finished with exit code {p.returncode}")
    print("".join(open(WORK / f"train_gpu{gpu}.log").readlines()[-4:]))
""")
code("""
# Training curves (fold models: validation PQ per threshold; full-data models: loss only)
import pandas as pd
for e in EXPERIMENTS:
    for f in sorted((WORK / e["name"]).glob("log_*.csv")):
        log = pd.read_csv(f)
        pq_cols = [c for c in log.columns if c.startswith("pq@")]
        best = f"best val PQ {log[pq_cols].max(axis=1).max():.4f}" if pq_cols else "no validation"
        print(f"{e['name']}/{f.stem}: {len(log)} epochs, {best}, last loss {log.loss.iloc[-1]:.4f}")
""")

md("""
## 3. Out-of-fold maps, post-processing, classifier, and comparison with the baseline

For every experiment with fold models trained here: predict their OOF maps with the same TTA as the test set, tune
post-processing on them, fit the component classifier (honest nested estimate in `rescorer.json`), and compare with
`BASELINE` on the same images (ΔPQ with a 95% bootstrap interval, without the classifier).
Detection experiments already wrote their OOF instances during training; their tuning searches the mask threshold
and the detection-score cut-off, and they have no component classifier (yet).
""")
code("""
tta_flag = "--tta" if TTA else ""
for e in EXPERIMENTS:
    if not e["folds"]:
        continue
    name = e["name"]
    new_folds = " ".join(map(str, e["folds"]))          # trained in this session -> need OOF maps
    all_folds = " ".join(sorted(p.stem[4:] for p in (WORK / name).glob("fold*.pt")))  # incl. loaded ones
    print(f"\\n==================== {name}: new folds [{new_folds}], all folds [{all_folds}] ====================")
    if e["kind"] == "segmentation":
        !python -W ignore scripts/segmentation/predict.py --exp {name} {tta_flag} --split oof --folds {new_folds}
        !python -W ignore scripts/postprocess/tune_postprocess.py --exp {name} --workers 4 --fold {all_folds} \\
            --thresholds 0.4 0.5 0.6 0.7 0.8 0.85 0.9 0.95 --min-areas 100 200 300 400 500 --min-scores 0
        !python -W ignore scripts/postprocess/fit_rescorer.py --exp {name}
    else:   # thresholds = mask threshold, min-scores = detection-score cut-off
        !python -W ignore scripts/postprocess/tune_postprocess.py --exp {name} --workers 4 --fold {all_folds} \\
            --thresholds 0.3 0.4 0.5 0.6 0.7 --min-areas 0 100 200 300 400 \\
            --min-scores 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    if (WORK / BASELINE / "oof").is_dir():
        print("--- vs baseline, each with its own tuned post-processing:")
        !python -W ignore scripts/evaluation/compare.py --a {WORK / BASELINE} --b {WORK / name} --folds {all_folds}
        if e["kind"] == kind_of(BASELINE):   # settings only carry over between models of the same kind
            print("--- vs baseline, both with the baseline's post-processing:")
            !python -W ignore scripts/evaluation/compare.py --a {WORK / BASELINE} --b {WORK / name} --folds {all_folds} --post a
    if (WORK / name / "rescorer.json").exists():
        print(json.dumps(json.loads((WORK / name / "rescorer.json").read_text()), indent=2))
""")

md("""
## 4. Submissions (optional)

For every entry in `SUBMISSIONS`: average the selected models (× 4 flips) on the test images, post-process, apply the
classifier, and write `/kaggle/working/submission_<name>.csv`. Fold-0 screens usually skip this.
""")
code("""
from filament.data.submission import read_submission, validate_submission
from filament.paths import get_data_paths
test_stems = [p.stem for p in get_data_paths().test_images.glob("*.jpeg")]
for sub_name, (exp, models) in SUBMISSIONS.items():
    print(f"\\n===== {sub_name}: {exp} {models} =====")
    model_args = " ".join(f'"{m}"' for m in models)
    if kind_of(exp) == "segmentation":
        !python -W ignore scripts/segmentation/predict.py --exp {exp} {tta_flag} --models {model_args} --out test_{sub_name}
    else:
        !python -W ignore scripts/detection/predict.py --exp {exp} --models {model_args} --out test_{sub_name}
    !python -W ignore scripts/postprocess/make_submission.py --exp {exp} --maps test_{sub_name} --out submission_{sub_name}.csv
    shutil.copy(WORK / exp / f"submission_{sub_name}.csv", WORK / f"submission_{sub_name}.csv")
    sub = read_submission(WORK / f"submission_{sub_name}.csv")
    print(f"{len(sub)} filaments on {sub.filament_id.str.rsplit('_', n=1).str[0].nunique()}/{len(test_stems)} images,",
          "validation:", validate_submission(sub, expected_stems=test_stems) or "OK")
""")

md("""
## 5. Keep the results

Everything is in `/kaggle/working/<experiment>/` (checkpoints, OOF maps, logs, `postprocess.json`, `rescorer.json`).
Download what you need from the **Output** tab, e.g. with the Kaggle CLI:
`kaggle kernels output <user>/<notebook> -p outputs/kaggle_runs/<run> --file-pattern "<regex>"`.
To reuse an experiment in a later session, add this notebook's output (or a dataset made from it) as an input and
list it in `CHECKPOINT_DATASETS`.
""")

META = {
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True},
}
nbf.write(nbf.v4.new_notebook(cells=cells, metadata=META), REPO_ROOT / "notebooks/kaggle/kaggle_train_submit.ipynb")

# Second notebook for a parallel session (Kaggle allows 2 GPU sessions at once): same code, different experiments.
SCREEN_CONFIG = '''
# ---- configuration ---------------------------------------------------------------------------------------
# Bet 2: whole-backbone Mask R-CNN at higher input resolution, fold 0, one per GPU (reference at 1x: 0.446):
# * GPU 0: 2x (training crops of 512 native px upscaled to 1024; full images predicted at 4096 px, ~6.2 GiB)
# * GPU 1: 1.5x (683 px crops; full images at 3072 px)
# Same settings as the reference otherwise (32 epochs; validation every 4 epochs, since full-image prediction is
# slower). max_hours stops a job cleanly after the last epoch that still fits. The 1x reference is compared locally.
EXPERIMENTS = [
    {"config": "configs/detection/maskrcnn/maskrcnn_r50_v2_m56_ftall_s2.yaml",  "folds": [0], "full_seeds": [],
     "overrides": ["max_hours=6.8"]},
    {"config": "configs/detection/maskrcnn/maskrcnn_r50_v2_m56_ftall_s15.yaml", "folds": [0], "full_seeds": [],
     "overrides": ["max_hours=6.8"]},
]
# attached inputs, matched by "<notebook>/<experiment>" in their path: this notebook's previous output
# (Add Input -> Your Work -> Notebooks -> this notebook), holding ConvNeXt-Tiny + U-Net fold 0 and its OOF maps
CHECKPOINT_DATASETS = ["solar-filament-v2/unet_convnext_t_skel050"]
BASELINE = "unet_convnext_t_skel050"   # in-notebook reference; vs the 1x Mask R-CNN (0.446): locally
TTA = True                             # U-Net only: average 4 flips per model when predicting
SUBMISSIONS = {                        # output name -> (experiment, [model names / glob patterns])
    "maskrcnn_ftall_s2_fold0": ("maskrcnn_r50_v2_m56_ftall_s2", ["fold0"]),
    "maskrcnn_ftall_s15_fold0": ("maskrcnn_r50_v2_m56_ftall_s15", ["fold0"]),
}
GITHUB_SECRET = "Github_key"           # label of the Kaggle secret holding the GitHub token
WANDB_SECRET = "WANDB_API_KEY"         # label of the (optional) Kaggle secret holding the W&B API key
REPO_URL = "github.com/LucaT2/solar-filament-segmentation.git"
REPO_BRANCH = "main"
REPO_DATASET = None                    # alternative to the GitHub token: name of a dataset containing the repo
'''
screen_cells = list(cells)
screen_cells[1] = nbf.v4.new_code_cell(SCREEN_CONFIG.strip())
screen_cells[0] = nbf.v4.new_markdown_cell(cells[0].source.replace(
    "# Solar Filament Segmentation: experiments on Kaggle",
    "# Solar Filament Segmentation: fold-0 screen (parallel session)"))
nbf.write(nbf.v4.new_notebook(cells=screen_cells, metadata=META), REPO_ROOT / "notebooks/kaggle/kaggle_screen.ipynb")

# Third notebook, CPU only (no GPU quota): downloads unlabelled GONG frames from new days into its output, to be
# attached as an input of the GPU notebooks (pseudo-labelling, semi-supervised training).
DOWNLOAD_TITLE = """
# Solar Filament Segmentation: download unlabelled GONG frames (CPU)

Downloads unlabelled H-alpha frames from days at least 2 days away from every training and test image (one frame per
competition site and day) into `/kaggle/working/unlabelled/` (`frames/*.jpg`, `manifest.csv`), with
`scripts/data/download_unlabelled.py`. The output is then attached to the GPU notebooks for pseudo-labelling.

### Settings
* Accelerator: **None** (CPU only: no GPU quota). Internet: **on**.
* *Add Input*: competition **filament-segmentation-2026** (to know the training and test dates).
* *Add-ons → Secrets*: the GitHub token (label `GITHUB_SECRET`). Run with **Save Version → Save & Run All**.
"""
DOWNLOAD_CONFIG = '''
# ---- configuration ---------------------------------------------------------------------------------------
DAYS = 600                             # days sampled evenly over 2011-2022 (~5 frames per day: ~3,000 frames)
MARGIN_DAYS = 2                        # minimum distance from every training and test image
GITHUB_SECRET = "Github_key"           # label of the Kaggle secret holding the GitHub token
WANDB_SECRET = "WANDB_API_KEY"         # (not needed here)
REPO_URL = "github.com/LucaT2/solar-filament-segmentation.git"
REPO_BRANCH = "main"
REPO_DATASET = None
'''
DOWNLOAD_RUN = """
# ~40 min of archive listings, then ~2 h of downloads (one polite request per ~2 s); resumable
!python -W ignore scripts/data/download_unlabelled.py --days {DAYS} --margin-days {MARGIN_DAYS} --out /kaggle/working/unlabelled
import pandas as pd
m = pd.read_csv("/kaggle/working/unlabelled/manifest.csv")
log(f"{len(m)} frames, {int(m.ok.sum())} pass the quality check; per site: {m[m.ok].site.value_counts().to_dict()}")
"""
download_cells = [nbf.v4.new_markdown_cell(DOWNLOAD_TITLE.strip()), nbf.v4.new_code_cell(DOWNLOAD_CONFIG.strip()),
                  cells[2], cells[3], cells[4], nbf.v4.new_markdown_cell("## 2. Download"),
                  nbf.v4.new_code_cell(DOWNLOAD_RUN.strip())]
CPU_META = {**META, "kaggle": {"accelerator": "none", "isInternetEnabled": True}}
nbf.write(nbf.v4.new_notebook(cells=download_cells, metadata=CPU_META),
          REPO_ROOT / "notebooks/kaggle/kaggle_download.ipynb")
print("written", len(cells), "cells x 2 notebooks + the download notebook")
