"""Build, write, read and validate submission files.

Format (one row per predicted filament)::

    filament_id,segmentation_rle
    20150125172714Mh_1,<COCO RLE counts string>

``filament_id`` = image stem + ``_`` + unique suffix; the RLE size is implicitly 2048 x 2048.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from filament.data.rle import IMAGE_SIZE, counts_to_mask, counts_to_rle, mask_to_counts, rle_area, rle_iou

COLUMNS = ["filament_id", "segmentation_rle"]
STEM_PATTERN = re.compile(r"^\d{14}[A-Z]h$")


def instances_to_rows(stem: str, instances: Iterable[np.ndarray | str]) -> list[dict]:
    """Rows for one image. ``instances`` are binary masks or already-encoded RLE counts strings."""
    rows = []
    for k, inst in enumerate(instances, start=1):
        counts = inst if isinstance(inst, str) else mask_to_counts(inst)
        rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": counts})
    return rows


def build_submission(predictions: Mapping[str, Iterable[np.ndarray | str]]) -> pd.DataFrame:
    """``{stem: [mask_or_counts, ...]}`` -> submission DataFrame. Images without predictions get no rows."""
    rows = []
    for stem, instances in predictions.items():
        rows.extend(instances_to_rows(stem, instances))
    return pd.DataFrame(rows, columns=COLUMNS)


def write_submission(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # QUOTE_NONE: the counts must be written without quotation marks (RLE counts never contain commas).
    df[COLUMNS].to_csv(path, index=False, quoting=3)
    return path


def read_submission(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, quoting=3)


def validate_submission(df: pd.DataFrame, expected_stems: Iterable[str] | None = None) -> list[str]:
    """Return a list of problems (empty list = valid).

    Checks columns, id format, uniqueness, that every RLE decodes to a non-empty 2048 x 2048 mask, that no two
    filaments of an image overlap (a rule of the competition) and (optionally) that stems belong to the expected
    image set.
    """
    problems: list[str] = []
    if list(df.columns) != COLUMNS:
        return [f"columns must be {COLUMNS}, got {list(df.columns)}"]
    if df["filament_id"].duplicated().any():
        problems.append(f"{int(df['filament_id'].duplicated().sum())} duplicated filament_id values")

    stems = df["filament_id"].str.rsplit("_", n=1).str[0]
    bad_ids = df.loc[~df["filament_id"].str.contains("_") | ~stems.str.match(STEM_PATTERN), "filament_id"]
    if len(bad_ids):
        problems.append(f"{len(bad_ids)} malformed filament_id values, e.g. {bad_ids.iloc[0]!r}")
    if expected_stems is not None:
        unknown = set(stems) - set(expected_stems)
        if unknown:
            problems.append(f"{len(unknown)} stems not in the expected image set, e.g. {sorted(unknown)[0]!r}")

    quoted = df["segmentation_rle"].str.contains(r"[\"']", regex=True)
    if quoted.any():
        problems.append(f"{int(quoted.sum())} RLE strings contain quotes")
    try:
        areas = rle_area([counts_to_rle(c) for c in df["segmentation_rle"]])
        if (areas == 0).any():
            problems.append(f"{int((areas == 0).sum())} empty masks")
        if (areas > IMAGE_SIZE * IMAGE_SIZE).any():
            problems.append("RLE decodes to more pixels than a 2048 x 2048 image")
    except Exception as exc:  # malformed counts strings raise inside pycocotools
        problems.append(f"RLE decoding failed: {exc}")
    if not problems:  # overlaps are only checked on decodable, well-formed files
        problems += overlap_problems(df)
    return problems


def overlapping_pairs(df: pd.DataFrame) -> list[tuple[str, str, int]]:
    """Pairs of filaments of the same image that share pixels: ``(filament_id, filament_id, shared pixels)``.

    The pairwise IoU on the RLEs finds the candidates; only those pairs are decoded to count the shared pixels.
    """
    stems = df["filament_id"].str.rsplit("_", n=1).str[0]
    pairs = []
    for _, group in df.groupby(stems, sort=False):
        if len(group) < 2:
            continue
        rles = [counts_to_rle(c) for c in group["segmentation_rle"]]
        first, second = np.nonzero(np.triu(rle_iou(rles, rles) > 0, k=1))
        for i, j in zip(first, second):
            shared = int((counts_to_mask(group["segmentation_rle"].iloc[i])
                          & counts_to_mask(group["segmentation_rle"].iloc[j])).sum())
            pairs.append((group["filament_id"].iloc[i], group["filament_id"].iloc[j], shared))
    return pairs


def overlap_problems(df: pd.DataFrame) -> list[str]:
    """The rules forbid overlapping predictions within an image; one message if any filaments share pixels."""
    pairs = overlapping_pairs(df)
    if not pairs:
        return []
    a, b, shared = max(pairs, key=lambda p: p[2])
    return [f"{len(pairs)} overlapping filament pairs in {len({p[0].rsplit('_', 1)[0] for p in pairs})} images "
            f"({sum(p[2] for p in pairs)} shared pixels), e.g. {a} and {b} share {shared} px"]
