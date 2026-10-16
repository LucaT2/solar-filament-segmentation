"""Train one fold of a segmentation model and write its out-of-fold probability maps.

Usage (via scripts/segmentation/train.py):
    python scripts/segmentation/train.py --config configs/segmentation/unet_r34.yaml --fold 0
    python scripts/segmentation/train.py --config configs/segmentation/unet_r34.yaml --fold 0 --set epochs=2 samples_per_epoch=200   # smoke test
    python scripts/segmentation/train.py --config configs/segmentation/unet_r34.yaml --fold -1     # train on all images (no validation)

Outputs (outputs/<name>/): fold<k>.pt (best checkpoint by validation PQ), log_fold<k>.csv, oof/<stem>.png.
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from filament.data.annotations import annotation_table, load_coco
from filament.data.folds import load_folds
from filament.data.images import disk_mask, read_image
from filament.evaluation.metric import score_images
from filament.evaluation.scoring import GroundTruth, score_components
from filament.paths import get_data_paths, get_output_dir
from filament.postprocess.components import find_components
from filament.postprocess.maps import save_prob
from filament.segmentation.dataset import CropDataset, build_augment, image_info, normalize_with
from filament.segmentation.model import SegLoss, build_model, predict_full
from filament.tracking import start_run


@dataclass
class TrainConfig:
    name: str = "unet_r34"
    arch: str = "Unet"
    encoder: str = "resnet34"
    pretrained: bool = True
    crop: int = 512
    batch_size: int = 8
    epochs: int = 30
    samples_per_epoch: int = 4000
    pos_fraction: float = 0.7
    augment: str = "base"
    lr: float = 5e-4
    encoder_lr_mult: float = 1.0  # encoder learning rate = lr * this (pretrained encoders often prefer < 1)
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    skel_weight: float = 0.0  # Skeleton Recall loss weight (0 = off)
    skel_radius: int = 2  # thickness of the tubed skeleton (diamond dilation radius, px)
    target_mode: str = "random"  # random: one annotator's mask per sample | consensus: average of all annotators
    neighbours: str | None = None  # aligned neighbouring GONG frames (filament.data.neighbours) used as training images
    neighbour_p: float = 0.5  # probability that a training crop comes from a neighbour instead of the image itself
    decoder_fp32: bool = False  # run the decoder in float32 under mixed precision (fp16 overflow, e.g. MAnet)
    grad_clip: float = 0.0  # max gradient norm (0 = off)
    infer_tile: int | None = None  # full-image prediction in tiles of this size (None = one pass)
    infer_overlap: int = 128
    amp: str = "fp16"  # fp16 | bf16 | none
    num_workers: int = 4
    val_every: int = 2
    val_thresholds: list = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6])
    val_min_area: int = 50
    seed: int = 42
    wandb: str = "auto"  # auto | on | offline | off (see filament.tracking)
    wandb_project: str | None = None
    wandb_entity: str | None = None

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> "TrainConfig":
        data = yaml.safe_load(Path(path).read_text()) if path else {}
        types = {f.name: f.type for f in fields(cls)}
        for item in overrides or []:
            key, value = item.split("=", 1)
            if key not in types:
                raise KeyError(f"unknown config key {key!r}")
            data[key] = yaml.safe_load(value)
        return cls(**data)


def load_training_neighbours(name: str | None, train_stems) -> dict | None:
    """Aligned neighbour frames of the training images of this fold (never of its validation images)."""
    if not name:
        return None
    from filament.data.neighbours import find_neighbour_dir, load_neighbours

    folder = find_neighbour_dir(name)
    keep = set(train_stems)
    nb = {s: paths for s, paths in load_neighbours(folder).items() if s in keep}
    print(f"neighbour frames: {sum(map(len, nb.values()))} for {len(nb)} of {len(keep)} training images ({folder})",
          flush=True)
    return nb


def amp_dtype(name: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "none": None}[name]


def lr_lambda(cfg: TrainConfig, steps_per_epoch: int):
    warmup, total = cfg.warmup_epochs * steps_per_epoch, cfg.epochs * steps_per_epoch

    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return f


def validate(model, cfg, infos, gt: GroundTruth, device, out_dir: Path | None = None) -> dict:
    """Full-image predictions for the validation images -> PQ at a few thresholds (min_area fixed)."""
    model.eval()
    per_thr: dict[float, list] = {t: [] for t in cfg.val_thresholds}
    pool = ThreadPoolExecutor(4)

    def score(stem, prob):
        region = disk_mask(prob.shape, infos[stem].disk, 1.0)
        res = {}
        for t in cfg.val_thresholds:
            comps = find_components(prob, t, region)
            res[t] = score_components(stem, comps, comps.keep(cfg.val_min_area), gt)
        if out_dir is not None:
            save_prob(out_dir, stem, prob)
        return res

    futures = []
    for stem in gt.stems:
        x = normalize_with(read_image(infos[stem].path), infos[stem])
        prob = predict_full([model], x, device, amp_dtype=amp_dtype(cfg.amp), tile=cfg.infer_tile,
                            overlap=cfg.infer_overlap)
        futures.append(pool.submit(score, stem, prob))
    for fut in futures:
        for t, s in fut.result().items():
            per_thr[t] += s
    pool.shutdown()
    model.train()
    return {t: score_images(s).pq for t, s in per_thr.items()}


def run(cfg: TrainConfig, fold: int) -> dict:
    torch.manual_seed(cfg.seed + fold)
    np.random.seed(cfg.seed + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = get_data_paths()
    out = get_output_dir(cfg.name)
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg)))

    ann = annotation_table(load_coco(paths.annotations))
    folds = load_folds()
    train_stems = folds.stem[folds.fold != fold].tolist()
    val_stems = folds.stem[folds.fold == fold].tolist() if fold >= 0 else []

    t0 = time.time()
    with ThreadPoolExecutor(8) as pool:
        infos = {i.stem: i for i in pool.map(image_info, [paths.train_images / f"{s}.jpeg" for s in folds.stem])}
    print(f"image stats for {len(infos)} images in {time.time() - t0:.0f}s")

    ds = CropDataset(ann, infos, train_stems, cfg.crop, cfg.samples_per_epoch, cfg.pos_fraction,
                     build_augment(cfg.augment), seed=cfg.seed + fold, skeleton=cfg.skel_weight > 0,
                     skel_radius=cfg.skel_radius, target_mode=cfg.target_mode,
                     neighbours=load_training_neighbours(cfg.neighbours, train_stems), neighbour_p=cfg.neighbour_p)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                        pin_memory=True, drop_last=True, persistent_workers=cfg.num_workers > 0)
    gt = GroundTruth(ann, val_stems) if val_stems else None

    model = build_model(cfg.arch, cfg.encoder, cfg.pretrained, decoder_fp32=cfg.decoder_fp32).to(device)
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    param_groups = [
        {"params": [p for p in model.parameters() if id(p) in encoder_ids], "lr": cfg.lr * cfg.encoder_lr_mult},
        {"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr": cfg.lr},
    ]
    opt = torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg, len(loader)))
    scaler = torch.amp.GradScaler(enabled=cfg.amp == "fp16" and device.type == "cuda")
    loss_fn = SegLoss(cfg.bce_weight, cfg.dice_weight, cfg.skel_weight)
    dtype = amp_dtype(cfg.amp)

    # fold models: fold<k>.pt; full-data models (fold -1, no validation, last epoch): full_s<seed>.pt
    tag = f"fold{fold}" if fold >= 0 else f"full_s{cfg.seed}"
    ckpt_path = out / f"{tag}.pt"
    tracker = start_run(
        cfg.wandb,
        name=f"{cfg.name}-{tag}",
        group=cfg.name,
        job_type="train",
        config={**asdict(cfg), "fold": fold, "n_train_images": len(train_stems), "n_val_images": len(val_stems),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
        tags=[cfg.arch, cfg.encoder],
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
    )
    best, log = -1.0, []
    for epoch in range(1, cfg.epochs + 1):
        t0, losses, skipped = time.time(), [], 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None and device.type == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):  # e.g. fp16 overflow: skip the batch instead of corrupting training
                skipped += 1
                sched.step()
                continue
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        lrs = sched.get_last_lr()  # [encoder, decoder]
        row = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else float("nan"), "lr": lrs[-1],
               "lr_encoder": lrs[0], "train_s": time.time() - t0, "skipped_batches": skipped}
        if skipped > 0.5 * len(loader):  # training has diverged: stop instead of wasting the remaining GPU time
            log.append(row)
            pd.DataFrame(log).to_csv(out / f"log_{tag}.csv", index=False)
            tracker.finish()
            raise RuntimeError(f"{cfg.name} {tag}: {skipped}/{len(loader)} batches with non-finite loss in epoch "
                               f"{epoch}; training diverged (try decoder_fp32, grad_clip or a lower lr)")

        if gt is not None and (epoch % cfg.val_every == 0 or epoch == cfg.epochs):
            if device.type == "cuda":  # hand training's cached-but-unused memory back before full-image validation
                torch.cuda.empty_cache()
            t1 = time.time()
            pqs = validate(model, cfg, infos, gt, device)
            row.update({f"pq@{t}": v for t, v in pqs.items()}, val_s=time.time() - t1)
            score = max(pqs.values())
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch, "val_pq": score}, ckpt_path)
                row["saved"] = True
        elif gt is None and epoch == cfg.epochs:
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch}, ckpt_path)
        log.append(row)
        pd.DataFrame(log).to_csv(out / f"log_{tag}.csv", index=False)
        tracker.log({**{k.replace("pq@", "val/pq@"): v for k, v in row.items()},
                     "val/pq_best_thr": max((v for k, v in row.items() if k.startswith("pq@")), default=None)},
                    step=epoch)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    result = {"fold": fold, "best_val_pq": best}
    if gt is not None:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        final = validate(model, cfg, infos, gt, device, out_dir=out / "oof")  # writes OOF maps of the best model
        result.update({"best_epoch": state["epoch"], "final_pq": final})
    (out / f"result_{tag}.json").write_text(json.dumps(result, indent=2, default=str))
    tracker.summary({"best_val_pq": best, "best_epoch": result.get("best_epoch"),
                     **{f"final/pq@{t}": v for t, v in result.get("final_pq", {}).items()}})
    tracker.finish()
    print(result)
    return result
