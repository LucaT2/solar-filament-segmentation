"""Inference time and GPU memory of the submission pipelines, per image and per stage, on real test images.

Pipelines (the same functions as the submissions; checkpoints and tuned settings from outputs/):
* ``detector_5f_tta8`` (submission_det5f_tta): Mask R-CNN, 5 fold models x 8 views, merged per filament, score cut-off.
* ``w40_classifier`` (submission_final_w40): U-Net ResNet34 + ConvNeXt (5 folds each, 8 views) and the detector maps
  (5 folds x 8 views, score-weighted soft masks), averaged 0.4 / 0.6, components + classifier.
* ``dino_full_tta8`` (the submitted pipeline, filament.final): one full-data Mask R-CNN, 8 views, score cut-off.
* light variants: ``detector_1model`` (fold 0, one view) and ``unet_1model`` (ConvNeXt fold 0, one view, no classifier).

Stages: load (JPEG + disk + normalisation, CPU), unet, detector, merge (views / folds -> one list or map), post
(components, classifier, RLE encoding, CPU). GPU stages are synchronised before the clock stops; peak GPU memory is
measured per pipeline.

Usage:
    python scripts/evaluation/time_inference.py --images 20 --out outputs/inference_timing.csv
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from filament.data.images import disk_mask, read_image
from filament.data.submission import instances_to_rows
from filament.detection.dataset import full_image_input
from filament.detection.instances import to_components, to_prob_map
from filament.detection.model import predict_instances
from filament.detection.tta import merge_views, predict_tta
from filament.detection.train import build_from_config
from filament.final import FINAL_CONFIG_DIR, FinalPipeline
from filament.paths import get_data_paths, resolve_exp_dir
from filament.postprocess.pipeline import load_postprocess
from filament.postprocess.rescoring import Rescorer, component_table
from filament.segmentation.dataset import image_info, normalize_with
from filament.segmentation.model import build_model, n_outputs, predict_full

UNETS = ["unet_r34_skel050", "unet_convnext_t_skel050"]
DETECTOR = "maskrcnn_ftall_5f"


def load_unets(names, folds, device):
    out = []
    for name in names:
        for k in folds:
            state = torch.load(resolve_exp_dir(name) / f"fold{k}.pt", map_location=device)
            cfg = state["config"]
            m = build_model(cfg["arch"], cfg["encoder"], pretrained=False, decoder_fp32=cfg.get("decoder_fp32", False),
                            classes=n_outputs(cfg))
            m.load_state_dict(state["model"])
            out.append(m.to(device).eval())
    return out


def load_detectors(folds, device):
    out = []
    for k in folds:
        state = torch.load(resolve_exp_dir(DETECTOR) / f"fold{k}.pt", map_location=device)
        m = build_from_config(state["config"], pretrained=False)
        m.load_state_dict(state["model"])
        out.append(m.to(device).eval())
    return out


class Clock:
    def __init__(self, device):
        self.device, self.times = device, defaultdict(float)

    def __call__(self, stage, fn, *args, **kwargs):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn(*args, **kwargs)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.times[stage] += time.perf_counter() - t
        return out


def detections(models, x, device, views):
    """Per model: ``views`` views merged per filament (one Instances per model)."""
    return [predict_tta(lambda v, m=m: predict_instances(m, v, device, torch.float16), x, views) for m in models]


def encode(stem, labels, ids):
    return instances_to_rows(stem, [labels == i for i in ids])


def run_detector(clock, path, models, views, cfg, device):
    info = clock("load", image_info, path)
    x = clock("load", full_image_input, info)
    per_model = clock("detector", detections, models, x, device, views)
    inst = clock("merge", merge_views, per_model)
    region = disk_mask((2048, 2048), info.disk, cfg.disk_scale)
    comps = clock("post", to_components, inst, cfg.threshold, region)
    return clock("post", encode, path.stem, comps.labels, comps.keep(cfg.min_area, cfg.min_score))


def run_unet(clock, path, unets, tta, cfg, device):
    info = clock("load", image_info, path)
    img = clock("load", read_image, path)
    prob = clock("unet", predict_full, unets, normalize_with(img, info), device, tta)
    comps, table = clock("post", component_table, path.stem, prob, img, cfg, info.disk)
    return clock("post", encode, path.stem, comps.labels, table.comp[table.kept].to_numpy())


def run_w40(clock, path, unets, detectors, cfg, rescorer, device):
    info = clock("load", image_info, path)
    img = clock("load", read_image, path)
    u = clock("unet", predict_full, unets, normalize_with(img, info), device, "d8")
    x = clock("load", full_image_input, info)
    per_model = clock("detector", detections, detectors, x, device, 8)
    d = clock("merge", lambda insts: np.mean([to_prob_map(i) for i in insts], axis=0), per_model)
    prob = 0.4 * u + 0.6 * d
    comps, table = clock("post", component_table, path.stem, prob, img, cfg, info.disk)
    return clock("post", encode, path.stem, comps.labels, clock("post", rescorer.keep_ids, table))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=int, default=20, help="test images timed (after 2 warm-up images)")
    parser.add_argument("--out", default="outputs/inference_timing.csv")
    parser.add_argument("--only", nargs="*", help="time only these pipelines")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(get_data_paths().test_images.glob("*.jpeg"))
    warm, timed = paths[:2], paths[2:2 + args.images]
    w40 = resolve_exp_dir("final_w40")
    cfg_w40, rescorer = load_postprocess(w40 / "postprocess.json"), Rescorer.load(w40 / "rescorer.pkl")
    cfg_det = load_postprocess(resolve_exp_dir("det5f_tta") / "postprocess.json")
    cfg_unet = load_postprocess(resolve_exp_dir("unet5f_d8") / "postprocess.json")
    cfg_final = load_postprocess(FINAL_CONFIG_DIR / "postprocess.json")

    pipelines = {
        "dino_full_tta8": lambda: (lambda c, p, m=[FinalPipeline.load(device).model]: run_detector(
            c, p, m, 8, cfg_final, device)),
        "detector_1model": lambda: (lambda c, p, m=load_detectors([0], device): run_detector(c, p, m, 1, cfg_det,
                                                                                               device)),
        "detector_5f_tta8": lambda: (lambda c, p, m=load_detectors(range(5), device): run_detector(c, p, m, 8, cfg_det,
                                                                                                    device)),
        "unet_1model": lambda: (lambda c, p, m=load_unets(UNETS[1:], [0], device): run_unet(c, p, m, "none", cfg_unet,
                                                                                            device)),
        "w40_classifier": lambda: (lambda c, p, u=load_unets(UNETS, range(5), device),
                                   d=load_detectors(range(5), device): run_w40(c, p, u, d, cfg_w40, rescorer, device)),
    }
    rows = []
    for name, build in pipelines.items():
        if args.only and name not in args.only:
            continue
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        run = build()
        load_s = time.perf_counter() - t0
        for p in warm:
            run(Clock(device), p)
        clock, n_filaments = Clock(device), 0
        for p in timed:
            n_filaments += len(run(clock, p))
        per = {k: v / len(timed) for k, v in clock.times.items()}
        row = {"pipeline": name, "images": len(timed), "model_load_s": load_s, **{f"{k}_s": v for k, v in per.items()},
               "total_s_per_image": sum(per.values()), "test_set_min": sum(per.values()) * 180 / 60,
               "filaments_per_image": n_filaments / len(timed),
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0}
        rows.append(row)
        print("  ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        del run
    table = pd.DataFrame(rows)
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    table.insert(1, "device", gpu)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"written {args.out} ({gpu})")


if __name__ == "__main__":
    main()
