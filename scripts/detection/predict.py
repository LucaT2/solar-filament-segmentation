"""Predict instances (boxes, scores, soft masks) with trained Mask R-CNN models.

--split test (default): every test image -> outputs/<exp>/<out or "test">/<stem>.npz
--split oof:            each fold model predicts its own validation images -> outputs/<exp>/<out or "oof">/
                        (training already writes these; re-run after changing the prediction code)

With several models (e.g. 5 folds on the test set), their detections are pooled into one list per image; the
overlap rule in post-processing then keeps the highest-scoring copy of each filament.

Guided models (config ``guide``) read the guide experiment's maps: its oof/ maps for --split oof, and for the test set
the folder named by the config's ``guide_test_maps`` (default test/; override with --guide-maps).

The saved predictions are post-processed like probability maps: scripts/postprocess/tune_postprocess.py,
scripts/evaluation/compare.py and scripts/postprocess/make_submission.py work on them unchanged.

Usage:
    python scripts/detection/predict.py --exp maskrcnn_r50_v2 --folds 0              # fold-0 model -> test/
    python scripts/detection/predict.py --exp maskrcnn_r50_v2 --split oof
    python scripts/detection/predict.py --exp maskrcnn_r50_v2 --split oof --tta 8 --out oof_tta8
"""

from __future__ import annotations

import argparse
import fnmatch
import time

import numpy as np
import torch
from tqdm import tqdm

from filament.data.folds import load_folds
from filament.detection.dataset import full_image_input
from filament.detection.instances import Instances, save_instances
from filament.detection.model import predict_instances
from filament.detection.tta import predict_tta
from filament.detection.train import build_from_config, guide_dir
from filament.paths import get_data_paths, resolve_exp_dir
from filament.segmentation.dataset import image_info
from filament.segmentation.train import amp_dtype


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
        model = build_from_config(cfg, pretrained=False)
        model.load_state_dict(state["model"])
        models.append(model.to(device).eval())
        print(f"loaded {path.name} (epoch {state.get('epoch')}, val PQ {state.get('val_pq', float('nan')):.4f})")
    return models, cfg


def pooled(insts: list[Instances]) -> Instances:
    if len(insts) == 1:
        return insts[0]
    return Instances(np.concatenate([i.boxes for i in insts]), np.concatenate([i.scores for i in insts]),
                     np.concatenate([i.masks for i in insts]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--models", nargs="+", default=["fold*"], help="checkpoint names or glob patterns")
    parser.add_argument("--folds", type=int, nargs="*", help="shorthand for --models fold<k> ...")
    parser.add_argument("--split", choices=["test", "oof"], default="test")
    parser.add_argument("--out", help="output sub-folder of outputs/<exp>/ (default: the split name)")
    parser.add_argument("--guide-maps", help="for guided models: sub-folder of the guide experiment with the maps "
                                             "(default: oof/, or the config's guide_test_maps for the test set)")
    parser.add_argument("--tta", type=int, choices=[1, 2, 4, 8], default=1,
                        help="views per image: flips (4) and transposed flips (8), merged per filament")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = get_data_paths()
    exp_dir = resolve_exp_dir(args.exp)
    patterns = [f"fold{k}" for k in args.folds] if args.folds is not None else args.models
    out_dir = exp_dir / (args.out or args.split)

    if args.split == "test":
        models, cfg = load_models(select_checkpoints(exp_dir, patterns), device)
        jobs = [(models, sorted(paths.test_images.glob("*.jpeg")), cfg)]
    else:
        folds = load_folds()
        jobs = []
        for ckpt in select_checkpoints(exp_dir, patterns):
            if ckpt.stem.startswith("fold"):
                models, cfg = load_models([ckpt], device)
                k = int(ckpt.stem[4:])
                jobs.append((models, [paths.train_images / f"{s}.jpeg" for s in folds.stem[folds.fold == k]], cfg))

    t0, n = time.time(), 0
    for models, images, cfg in jobs:
        dtype = amp_dtype(cfg.get("amp", "fp16"))
        guides = guide_dir(cfg, args.split)
        if guides is not None and args.guide_maps:
            guides = guides.parent / args.guide_maps
        if guides is not None:
            print(f"guide maps: {guides}")
        for path in tqdm(images, desc=args.split):
            x = full_image_input(image_info(path), guides)
            save_instances(out_dir, path.stem, pooled([
                predict_tta(lambda v: predict_instances(m, v, device, dtype, cfg.get("scale", 1.0)), x, args.tta)
                for m in models]))
        n += len(images)
    print(f"{n} {args.split} predictions in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
