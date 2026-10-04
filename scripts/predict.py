"""Predict probability maps for the test images with the trained fold models (averaged), optionally with flip TTA.

Writes outputs/<exp>/test/<stem>.png; then run scripts/make_submission.py --exp <exp>.

Usage:
    python scripts/predict.py --exp unet_r34                 # all fold*.pt found in outputs/unet_r34
    python scripts/predict.py --exp unet_r34 --folds 0 1 --tta
"""

from __future__ import annotations

import argparse
import time

import torch
from tqdm import tqdm

from filament.dataset import image_info, normalize_with
from filament.inference import save_prob
from filament.model import build_model, predict_full
from filament.paths import get_data_paths, get_output_dir
from filament.preprocess import read_image
from filament.train import amp_dtype


def load_models(exp_dir, folds, device):
    ckpts = sorted(exp_dir.glob("fold*.pt")) + sorted(exp_dir.glob("full.pt"))
    if folds is not None:
        ckpts = [c for c in ckpts if c.stem in {f"fold{k}" for k in folds}]
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints in {exp_dir}")
    models, cfg = [], None
    for path in ckpts:
        state = torch.load(path, map_location=device)
        cfg = state["config"]
        model = build_model(cfg["arch"], cfg["encoder"], pretrained=False)
        model.load_state_dict(state["model"])
        models.append(model.to(device).eval())
        print(f"loaded {path.name} (epoch {state.get('epoch')}, val PQ {state.get('val_pq', float('nan')):.4f})")
    return models, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--folds", type=int, nargs="*")
    parser.add_argument("--tta", action="store_true", help="average over 4 flips")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = get_data_paths()
    exp_dir = get_output_dir(args.exp)
    models, cfg = load_models(exp_dir, args.folds, device)
    dtype = amp_dtype(cfg.get("amp", "fp16"))

    t0 = time.time()
    images = sorted(paths.test_images.glob("*.jpeg"))
    for path in tqdm(images, desc="test"):
        info = image_info(path)
        prob = predict_full(models, normalize_with(read_image(path), info), device, tta=args.tta, amp_dtype=dtype)
        save_prob(exp_dir / "test", path.stem, prob)
    print(f"{len(images)} test maps in {time.time() - t0:.0f}s -> {exp_dir / 'test'}")


if __name__ == "__main__":
    main()
