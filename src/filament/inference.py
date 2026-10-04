"""Shared inference plumbing: cached probability maps -> instances -> submission table."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pandas as pd

from filament.evaluation import load_prob, region_for
from filament.postprocess import PostConfig, instances, prob_to_uint8
from filament.submission import build_submission


def save_prob(prob_dir: Path, stem: str, prob) -> None:
    prob_dir = Path(prob_dir)
    prob_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(prob_dir / f"{stem}.png"), prob_to_uint8(prob))


def probs_to_submission(prob_dir, stems, cfg: PostConfig, image_dir) -> pd.DataFrame:
    """Post-process cached probability maps of ``stems`` into a submission DataFrame."""
    preds, disks = {}, {}
    for stem in stems:
        prob = load_prob(prob_dir, stem)
        preds[stem] = instances(prob, cfg, region_for(stem, image_dir, cfg.disk_scale, disks))
    return build_submission(preds)


def save_postprocess(path: Path, cfg: PostConfig, extra: dict | None = None) -> None:
    Path(path).write_text(json.dumps({**cfg.to_dict(), **(extra or {})}, indent=2))


def load_postprocess(path: Path) -> PostConfig:
    data = json.loads(Path(path).read_text())
    return PostConfig(**{k: data[k] for k in PostConfig.__dataclass_fields__ if k in data})
