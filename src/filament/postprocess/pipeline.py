"""Saved predictions -> instances -> submission table, plus the saved post-processing settings."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from filament.data.submission import build_submission
from filament.postprocess.components import PostConfig, region_for
from filament.postprocess.maps import load_prob
from filament.postprocess.predictions import load_components, prediction_kind


def predictions_to_submission(pred_dir, stems, cfg: PostConfig, image_dir, rescorer=None) -> pd.DataFrame:
    """Post-process the saved predictions (probability maps or instances) of ``stems`` into a submission DataFrame.

    With a ``filament.postprocess.rescoring.Rescorer``, components must also pass its P(match) cut-off.
    """
    from filament.data.images import read_image
    from filament.postprocess.rescoring import component_table

    if rescorer is not None and prediction_kind(pred_dir) != "maps":
        raise NotImplementedError("the component classifier is only fitted on probability maps so far")
    preds, disks = {}, {}
    for stem in stems:
        if rescorer is None:
            comps = load_components(pred_dir, stem, cfg.threshold, region_for(stem, image_dir, cfg.disk_scale, disks),
                                    cfg.link_px)
            preds[stem] = [comps.labels == i for i in comps.keep(cfg.min_area, cfg.min_score)]
        else:
            prob = load_prob(pred_dir, stem)
            comps, table = component_table(stem, prob, read_image(Path(image_dir) / f"{stem}.jpeg"), cfg)
            preds[stem] = [comps.labels == i for i in rescorer.keep_ids(table)]
    return build_submission(preds)


def save_postprocess(path: Path, cfg: PostConfig, extra: dict | None = None) -> None:
    Path(path).write_text(json.dumps({**cfg.to_dict(), **(extra or {})}, indent=2))


def load_postprocess(path: Path) -> PostConfig:
    data = json.loads(Path(path).read_text())
    return PostConfig(**{k: data[k] for k in PostConfig.__dataclass_fields__ if k in data})
