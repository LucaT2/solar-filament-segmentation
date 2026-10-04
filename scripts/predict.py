"""Predict probability maps with the trained fold models, optionally with flip TTA.

--split test (default): every test image, averaged over all selected fold models -> outputs/<exp>/test/
--split oof:            each fold model predicts its own validation images        -> outputs/<exp>/oof/
                        (use the same --tta setting as for test, so OOF-tuned post-processing and the
                        component rescorer see identically produced maps)

Usage:
    python scripts/predict.py --exp unet_r34 --tta                 # test, all fold*.pt in outputs/unet_r34
    python scripts/predict.py --exp unet_r34 --tta --split oof
    python scripts/predict.py --exp unet_r34 --folds 0 1 --tta
"""

from __future__ import annotations

import argparse
import time

import torch
from tqdm import tqdm

from filament.dataset import image_info, normalize_with
from filament.folds import load_folds
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
    parser.add_argument("--split", choices=["test", "oof"], default="test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = get_data_paths()
    exp_dir = get_output_dir(args.exp)

    if args.split == "test":
        models, cfg = load_models(exp_dir, args.folds, device)
        jobs = [(models, sorted(paths.test_images.glob("*.jpeg")), cfg)]
    else:
        folds = load_folds()
        fold_ids = args.folds if args.folds is not None else sorted(
            int(p.stem[4:]) for p in exp_dir.glob("fold*.pt"))
        jobs = []
        for k in fold_ids:
            models, cfg = load_models(exp_dir, [k], device)
            jobs.append((models, [paths.train_images / f"{s}.jpeg" for s in folds.stem[folds.fold == k]], cfg))

    t0, n = time.time(), 0
    for models, images, cfg in jobs:
        dtype = amp_dtype(cfg.get("amp", "fp16"))
        for path in tqdm(images, desc=args.split):
            info = image_info(path)
            prob = predict_full(models, normalize_with(read_image(path), info), device, tta=args.tta, amp_dtype=dtype)
            save_prob(exp_dir / args.split, path.stem, prob)
        n += len(images)
    print(f"{n} {args.split} maps (tta={args.tta}) in {time.time() - t0:.0f}s -> {exp_dir / args.split}")


if __name__ == "__main__":
    main()
