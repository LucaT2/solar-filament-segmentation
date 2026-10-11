"""A model's saved predictions for one image, whatever the model type, as components.

* segmentation models save a probability map (``<dir>/<stem>.png``): threshold + connected components
* detection models save scored instances (``<dir>/<stem>.npz``): pasted masks with overlaps resolved

Both become ``Components`` (label map, areas, a per-component score), so tuning, comparison, error analysis and
submissions work the same way for both. The ``PostConfig`` fields keep their meaning where it carries over:

=============  ==========================================  ==========================================
field          probability maps                            instances
=============  ==========================================  ==========================================
threshold      probability threshold                       mask threshold (inside each detection)
min_area       minimum component area (px)                 minimum instance area (px, after overlaps)
min_score      minimum mean probability of a component     minimum detection score
link_px        join fragments closer than ~2 * link_px     not used
=============  ==========================================  ==========================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from filament.postprocess.components import Components, find_components
from filament.postprocess.maps import load_prob

KINDS = {".png": "maps", ".npz": "instances"}


def prediction_kind(pred_dir: Path) -> str:
    """'maps' or 'instances', from the files in ``pred_dir``."""
    for suffix, kind in KINDS.items():
        if next(Path(pred_dir).glob(f"*{suffix}"), None) is not None:
            return kind
    raise FileNotFoundError(f"no predictions (*.png or *.npz) in {pred_dir}")


def available_stems(pred_dir: Path) -> set[str]:
    return {p.stem for p in Path(pred_dir).iterdir() if p.suffix in KINDS} if Path(pred_dir).is_dir() else set()


def component_builder(pred_dir: Path, stem: str, region: np.ndarray | None = None) -> Callable[..., Components]:
    """Read the image's predictions once; return ``f(threshold, link_px=0) -> Components`` (for threshold grids)."""
    pred_dir = Path(pred_dir)
    if (pred_dir / f"{stem}.npz").exists():
        from filament.detection.instances import load_instances, to_components

        inst = load_instances(pred_dir, stem)
        return lambda threshold, link_px=0: to_components(inst, threshold, region)
    prob = load_prob(pred_dir, stem)
    return lambda threshold, link_px=0: find_components(prob, threshold, region, link_px)


def load_components(pred_dir: Path, stem: str, threshold: float, region: np.ndarray | None = None,
                    link_px: int = 0) -> Components:
    return component_builder(pred_dir, stem, region)(threshold, link_px)
