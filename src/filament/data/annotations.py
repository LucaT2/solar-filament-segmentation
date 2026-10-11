"""Load the MAGFiLO COCO annotations into tidy tables.

Terminology used across the project:
    stem             image file name without extension, e.g. ``20111116063134Lh``
    annotator_image  one annotator's annotation set for one image, e.g. ``050103-20111116063134Lh``
                     (the COCO ``image.id``). The official metric scores each annotator-image separately.
    annotator        the 6-digit prefix of ``annotator_image``, e.g. ``050103``
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from filament.data.rle import polygons_to_rle

SITE_NAMES = {
    "B": "Big Bear",
    "C": "Cerro Tololo",
    "L": "Learmonth",
    "M": "Mauna Loa",
    "T": "El Teide",
    "U": "Udaipur",
}

CATEGORY_NAMES = {1: "Left", 2: "Right", 3: "Unidentifiable", 4: "Ambiguous"}


def parse_stem(stem: str) -> tuple[datetime, str]:
    """``YYYYMMDDHHMMSS`` + site letter + ``h`` -> (capture time, site letter)."""
    return datetime.strptime(stem[:14], "%Y%m%d%H%M%S"), stem[14]


def load_coco(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def image_table(file_names: list[str]) -> pd.DataFrame:
    """One row per image file, with capture time and site parsed from the name."""
    rows = []
    for name in sorted(file_names):
        stem = Path(name).stem
        time, site = parse_stem(stem)
        rows.append({"stem": stem, "file_name": name, "time": time, "date": time.date(), "site": site})
    return pd.DataFrame(rows)


def annotator_image_table(coco: dict) -> pd.DataFrame:
    """One row per annotator-image (COCO image entry)."""
    rows = []
    for img in coco["images"]:
        annotator, stem = img["id"].split("-", 1)
        rows.append(
            {
                "annotator_image": img["id"],
                "annotator": annotator,
                "stem": stem,
                "file_name": img["file_name"],
                "height": img["height"],
                "width": img["width"],
            }
        )
    df = pd.DataFrame(rows)
    df["n_annotators"] = df.groupby("stem")["annotator_image"].transform("count")
    return df


def annotation_table(coco: dict, with_rle: bool = True) -> pd.DataFrame:
    """One row per filament annotation, optionally with its mask as a compressed RLE dict."""
    sizes = {img["id"]: (img["height"], img["width"]) for img in coco["images"]}
    rows = []
    for ann in coco["annotations"]:
        annotator, stem = ann["image_id"].split("-", 1)
        x, y, w, h = ann["bbox"]
        row = {
            "ann_id": ann["id"],
            "annotator_image": ann["image_id"],
            "annotator": annotator,
            "stem": stem,
            "category": CATEGORY_NAMES.get(ann["category_id"], str(ann["category_id"])),
            "area": ann["area"],
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "n_polygons": len(ann["segmentation"]),
            "spine": ann.get("spine"),
        }
        if with_rle:
            height, width = sizes[ann["image_id"]]
            row["rle"] = polygons_to_rle(ann["segmentation"], height, width)
        rows.append(row)
    return pd.DataFrame(rows)


def ground_truth_table(annotations: pd.DataFrame, stems=None) -> pd.DataFrame:
    """Ground truth in the hidden-GT format (``<annotator>-<stem>_<k>``), optionally restricted to some stems."""
    if stems is not None:
        annotations = annotations[annotations["stem"].isin(set(stems))]
    return to_submission_format(annotations, key="annotator_image")


def to_submission_format(annotations: pd.DataFrame, key: str = "annotator_image") -> pd.DataFrame:
    """Convert an annotation table into the ``filament_id, segmentation_rle`` format.

    ``key="annotator_image"`` reproduces the hidden ground-truth format (``<annotator>-<stem>_<k>``);
    ``key="stem"`` produces a prediction-style table (``<stem>_<k>``).
    """
    df = annotations[[key, "rle"]].copy()
    df["k"] = df.groupby(key).cumcount() + 1
    return pd.DataFrame(
        {
            "filament_id": df[key] + "_" + df["k"].astype(str),
            "segmentation_rle": df["rle"].map(lambda r: r["counts"]),
        }
    ).reset_index(drop=True)
