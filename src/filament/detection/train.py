"""Train one fold of Mask R-CNN and write its out-of-fold instance predictions.

Usage (via scripts/detection/train.py):
    python scripts/detection/train.py --config configs/detection/maskrcnn_r50_v2.yaml --fold 0
    python scripts/detection/train.py --config configs/detection/maskrcnn_r50_v2.yaml --fold 0 \
        --set epochs=1 samples_per_epoch=40 batch_size=2                                     # smoke test

Outputs (outputs/<name>/): fold<k>.pt (best checkpoint by validation PQ), log_fold<k>.csv, result_fold<k>.json and
oof/<stem>.npz (raw detections of the best model on the fold's images; tune them with
scripts/postprocess/tune_postprocess.py like probability maps).
"""

from __future__ import annotations

import json
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
from filament.data.images import disk_mask
from filament.detection.dataset import InstanceCropDataset, collate, full_image_input
from filament.detection.instances import save_instances, to_components
from filament.detection.model import build_maskrcnn, predict_instances, set_input_size
from filament.evaluation.metric import score_images
from filament.evaluation.scoring import GroundTruth, score_components
from filament.paths import get_data_paths, get_output_dir, resolve_exp_dir
from filament.segmentation.dataset import build_augment, image_info
from filament.segmentation.train import amp_dtype, lr_lambda
from filament.tracking import start_run


@dataclass
class DetTrainConfig:
    name: str = "maskrcnn_r50_v2"
    pretrained: bool = True  # COCO weights
    trainable_layers: int = 3  # ResNet stages fine-tuned (0-5); the rest stay frozen
    mask_size: int = 28  # mask head output (28 = torchvision default; 56 = finer, more memory)
    guide: str | None = None  # segmentation experiment whose probability maps become the 2nd input channel
    # (its oof/ maps for training images; OOF, so no image's guide comes from a model that saw its labels)
    guide_test_maps: str = "test"  # the guide's folder with test maps. Training sees single-model OOF maps, so
    # single-model test maps (e.g. test_fold0) match them better than a softer multi-fold average
    detections_per_img: int = 100
    crop: int = 1024
    batch_size: int = 4
    epochs: int = 24
    samples_per_epoch: int = 2000
    pos_fraction: float = 0.8
    min_instance_area: int = 30  # drop filament pieces smaller than this after cropping (px)
    augment: str = "base"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    grad_clip: float = 10.0  # max gradient norm (0 = off)
    amp: str = "fp16"  # fp16 | bf16 | none
    num_workers: int = 4
    val_every: int = 2
    val_mask_threshold: float = 0.5
    val_score_thresholds: list = field(default_factory=lambda: [0.3, 0.5, 0.7])
    val_min_area: int = 100
    seed: int = 42
    wandb: str = "auto"  # auto | on | offline | off (see filament.tracking)
    wandb_project: str | None = None
    wandb_entity: str | None = None

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> "DetTrainConfig":
        data = yaml.safe_load(Path(path).read_text()) if path else {}
        names = {f.name for f in fields(cls)}
        for item in overrides or []:
            key, value = item.split("=", 1)
            if key not in names:
                raise KeyError(f"unknown config key {key!r}")
            data[key] = yaml.safe_load(value)
        return cls(**data)


def build_from_config(cfg: dict | DetTrainConfig, pretrained: bool | None = None):
    c = cfg if isinstance(cfg, dict) else asdict(cfg)
    return build_maskrcnn(c["pretrained"] if pretrained is None else pretrained, c["trainable_layers"],
                          c["detections_per_img"], mask_size=c["mask_size"], size=c["crop"])


def guide_dir(cfg: dict | DetTrainConfig, split: str) -> Path | None:
    """Folder with the guide experiment's probability maps for ``split`` ('oof', 'test' or a folder name), or None."""
    c = cfg if isinstance(cfg, dict) else asdict(cfg)
    if not c.get("guide"):
        return None
    folder = c.get("guide_test_maps", "test") if split == "test" else split
    return resolve_exp_dir(c["guide"]) / folder


def validate(model, cfg: DetTrainConfig, infos, gt: GroundTruth, device, out_dir: Path | None = None) -> dict:
    """Full-image detections for the validation images -> PQ at a few score cut-offs (mask threshold, min_area fixed)."""
    per_thr: dict[float, list] = {t: [] for t in cfg.val_score_thresholds}
    guides = guide_dir(cfg, "oof")
    pool = ThreadPoolExecutor(4)

    def score(stem, inst):
        comps = to_components(inst, cfg.val_mask_threshold, disk_mask((2048, 2048), infos[stem].disk, 1.0))
        res = {t: score_components(stem, comps, comps.keep(cfg.val_min_area, t), gt) for t in cfg.val_score_thresholds}
        if out_dir is not None:
            save_instances(out_dir, stem, inst)
        return res

    futures = []
    for stem in gt.stems:
        x = full_image_input(infos[stem], guides)
        futures.append(pool.submit(score, stem, predict_instances(model, x, device, amp_dtype(cfg.amp))))
    for fut in futures:
        for t, s in fut.result().items():
            per_thr[t] += s
    pool.shutdown()
    return {t: score_images(s).pq for t, s in per_thr.items()}


def run(cfg: DetTrainConfig, fold: int) -> dict:
    torch.manual_seed(cfg.seed + fold)
    np.random.seed(cfg.seed + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = get_data_paths()
    out = get_output_dir(cfg.name)
    (out / "config.yaml").write_text(yaml.safe_dump({**asdict(cfg), "kind": "detection"}))

    ann = annotation_table(load_coco(paths.annotations))
    folds = load_folds()
    train_stems = folds.stem[folds.fold != fold].tolist()
    val_stems = folds.stem[folds.fold == fold].tolist() if fold >= 0 else []

    t0 = time.time()
    with ThreadPoolExecutor(8) as pool:
        infos = {i.stem: i for i in pool.map(image_info, [paths.train_images / f"{s}.jpeg" for s in folds.stem])}
    print(f"image stats for {len(infos)} images in {time.time() - t0:.0f}s")

    ds = InstanceCropDataset(ann, infos, train_stems, cfg.crop, cfg.samples_per_epoch, cfg.pos_fraction,
                             build_augment(cfg.augment), cfg.min_instance_area, guide_dir(cfg, "oof"))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True,
                        drop_last=True, persistent_workers=cfg.num_workers > 0, collate_fn=collate)
    gt = GroundTruth(ann, val_stems) if val_stems else None

    model = build_from_config(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg, len(loader)))
    scaler = torch.amp.GradScaler(enabled=cfg.amp == "fp16" and device.type == "cuda")
    dtype = amp_dtype(cfg.amp)

    tag = f"fold{fold}" if fold >= 0 else f"full_s{cfg.seed}"
    ckpt_path = out / f"{tag}.pt"
    tracker = start_run(
        cfg.wandb, name=f"{cfg.name}-{tag}", group=cfg.name, job_type="train",
        config={**asdict(cfg), "fold": fold, "n_train_images": len(train_stems), "n_val_images": len(val_stems),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
        tags=["maskrcnn"], project=cfg.wandb_project, entity=cfg.wandb_entity,
    )
    best, log = -1.0, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        set_input_size(model, cfg.crop)
        t0, losses, parts, skipped = time.time(), [], {}, 0
        for images, targets in loader:
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None and device.type == "cuda"):
                loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):  # skip the batch instead of corrupting training
                skipped += 1
                sched.step()
                continue
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            for k, v in loss_dict.items():
                parts.setdefault(k, []).append(v.item())
        row = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else float("nan"),
               **{k: float(np.mean(v)) for k, v in parts.items()},
               "lr": sched.get_last_lr()[0], "train_s": time.time() - t0, "skipped_batches": skipped}
        if skipped > 0.5 * len(loader):
            log.append(row)
            pd.DataFrame(log).to_csv(out / f"log_{tag}.csv", index=False)
            tracker.finish()
            raise RuntimeError(f"{cfg.name} {tag}: {skipped}/{len(loader)} batches with non-finite loss in epoch "
                               f"{epoch}; training diverged (try amp=bf16/none, grad_clip or a lower lr)")

        if gt is not None and (epoch % cfg.val_every == 0 or epoch == cfg.epochs):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            t1 = time.time()
            pqs = validate(model, cfg, infos, gt, device)
            row.update({f"pq@{t}": v for t, v in pqs.items()}, val_s=time.time() - t1)
            score = max(pqs.values())
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch, "val_pq": score},
                           ckpt_path)
                row["saved"] = True
        elif gt is None and epoch == cfg.epochs:
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch}, ckpt_path)
        log.append(row)
        pd.DataFrame(log).to_csv(out / f"log_{tag}.csv", index=False)
        tracker.log({k.replace("pq@", "val/pq@score"): v for k, v in row.items()}, step=epoch)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    result = {"fold": fold, "best_val_pq": best}
    if gt is not None:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        final = validate(model, cfg, infos, gt, device, out_dir=out / "oof")  # OOF detections of the best model
        result.update({"best_epoch": state["epoch"], "final_pq": final})
    (out / f"result_{tag}.json").write_text(json.dumps(result, indent=2, default=str))
    tracker.summary({"best_val_pq": best, "best_epoch": result.get("best_epoch"),
                     **{f"final/pq@score{t}": v for t, v in result.get("final_pq", {}).items()}})
    tracker.finish()
    print(result)
    return result
