"""Predict probability maps with trained models, optionally with flip TTA.

--split test (default): every test image, averaged over the selected models -> outputs/<exp>/<out or "test">/
--split oof:            each fold model predicts its own validation images  -> outputs/<exp>/<out or "oof">/
                        (use the same --tta setting as for test, so OOF-tuned post-processing and the
                        component rescorer see identically produced maps)

Models are checkpoint names in outputs/<exp>/: fold0 ... fold4 (cross-validation folds) and full_s<seed>
(trained on all images). --models accepts names or glob patterns; default for test: all fold models.

Usage:
    python scripts/predict.py --exp unet_r34 --tta                                   # 5 fold models -> test/
    python scripts/predict.py --exp unet_r34 --tta --split oof
    python scripts/predict.py --exp unet_r34 --tta --models "full_s*" --out test_full
    python scripts/predict.py --exp unet_r34 --tta --models "fold*" "full_s*" --out test_folds_full
"""

from __future__ import annotations

import argparse
import fnmatch
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


def select_checkpoints(exp_dir, patterns: list[str]) -> list:
    available = sorted(exp_dir.glob("*.pt"))
    chosen = [c for c in available if any(fnmatch.fnmatch(c.stem, p) for p in patterns)]
    if not chosen:
        raise FileNotFoundError(f"no checkpoints matching {patterns} in {exp_dir} "
                                f"(available: {[c.stem for c in available]})")
    return chosen


def load_models(ckpts, device):
    models, cfg = [], None
    for path in ckpts:
        state = torch.load(path, map_location=device)
        cfg = state["config"]
        model = build_model(cfg["arch"], cfg["encoder"], pretrained=False, decoder_fp32=cfg.get("decoder_fp32", False))
        model.load_state_dict(state["model"])
        models.append(model.to(device).eval())
        print(f"loaded {path.name} (epoch {state.get('epoch')}, val PQ {state.get('val_pq', float('nan')):.4f})")
    return models, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--models", nargs="+", default=["fold*"], help="checkpoint names or glob patterns")
    parser.add_argument("--folds", type=int, nargs="*", help="shorthand for --models fold<k> ...")
    parser.add_argument("--tta", action="store_true", help="average over 4 flips")
    parser.add_argument("--split", choices=["test", "oof"], default="test")
    parser.add_argument("--out", help="output sub-folder of outputs/<exp>/ (default: the split name)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = get_data_paths()
    exp_dir = get_output_dir(args.exp)
    patterns = [f"fold{k}" for k in args.folds] if args.folds is not None else args.models
    out_dir = exp_dir / (args.out or args.split)

    if args.split == "test":
        models, cfg = load_models(select_checkpoints(exp_dir, patterns), device)
        jobs = [(models, sorted(paths.test_images.glob("*.jpeg")), cfg)]
    else:  # OOF: only fold models, each on its own validation images
        folds = load_folds()
        jobs = []
        for ckpt in select_checkpoints(exp_dir, patterns):
            if not ckpt.stem.startswith("fold"):
                continue
            k = int(ckpt.stem[4:])
            models, cfg = load_models([ckpt], device)
            jobs.append((models, [paths.train_images / f"{s}.jpeg" for s in folds.stem[folds.fold == k]], cfg))

    t0, n = time.time(), 0
    for models, images, cfg in jobs:
        dtype = amp_dtype(cfg.get("amp", "fp16"))
        tiling = {"tile": cfg.get("infer_tile"), "overlap": cfg.get("infer_overlap", 128)}
        for path in tqdm(images, desc=args.split):
            info = image_info(path)
            prob = predict_full(models, normalize_with(read_image(path), info), device, tta=args.tta, amp_dtype=dtype,
                                **tiling)
            save_prob(out_dir, path.stem, prob)
        n += len(images)
    print(f"{n} {args.split} maps (tta={args.tta}) in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
