"""Cached probability maps -> instances -> submission table, plus the saved post-processing settings."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from filament.data.submission import build_submission
from filament.postprocess.components import PostConfig, instances, region_for
from filament.postprocess.maps import load_prob


def probs_to_submission(prob_dir, stems, cfg: PostConfig, image_dir, rescorer=None) -> pd.DataFrame:
    """Post-process cached probability maps of ``stems`` into a submission DataFrame.

    With a ``filament.postprocess.rescoring.Rescorer``, components must also pass its P(match) cut-off.
    """
    from filament.data.images import read_image
    from filament.postprocess.rescoring import component_table

    preds, disks = {}, {}
    for stem in stems:
        prob = load_prob(prob_dir, stem)
        if rescorer is None:
            preds[stem] = instances(prob, cfg, region_for(stem, image_dir, cfg.disk_scale, disks))
        else:
            comps, table = component_table(stem, prob, read_image(Path(image_dir) / f"{stem}.jpeg"), cfg)
            preds[stem] = [comps.labels == i for i in rescorer.keep_ids(table)]
    return build_submission(preds)


def save_postprocess(path: Path, cfg: PostConfig, extra: dict | None = None) -> None:
    Path(path).write_text(json.dumps({**cfg.to_dict(), **(extra or {})}, indent=2))


def load_postprocess(path: Path) -> PostConfig:
    data = json.loads(Path(path).read_text())
    return PostConfig(**{k: data[k] for k in PostConfig.__dataclass_fields__ if k in data})
