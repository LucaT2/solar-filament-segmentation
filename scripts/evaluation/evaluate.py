"""Score a predictions CSV (submission format, keyed by image stem) against the training ground truth.

Used for out-of-fold (OOF) validation. Images are scored only if they appear in the selected folds; every
annotator-image is scored separately, exactly like the official metric.

Usage:
    python scripts/evaluation/evaluate.py --pred outputs/oof/unet_baseline.csv              # all folds
    python scripts/evaluation/evaluate.py --pred outputs/oof/unet_baseline.csv --fold 0     # one fold
    python scripts/evaluation/evaluate.py --reference human                                 # human-vs-human reference
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from filament.data.annotations import annotation_table, annotator_image_table, ground_truth_table, load_coco
from filament.data.folds import load_folds
from filament.data.submission import build_submission, read_submission, validate_submission
from filament.evaluation.metric import pq_by_group, pq_score, summarize
from filament.paths import get_data_paths


def human_reference(ann: pd.DataFrame, ai: pd.DataFrame, stems: list[str]) -> pd.DataFrame:
    """'Predictions' = the first annotator of each image. On multi-annotated images this scores a human against
    the other humans; single-annotated images trivially score 1 and are excluded."""
    multi = ai[(ai.n_annotators > 1) & ai.stem.isin(stems)]
    first = multi.drop_duplicates("stem").annotator_image
    preds = {stem: [r["counts"] for r in g.rle] for stem, g in ann[ann.annotator_image.isin(first)].groupby("stem")}
    return build_submission(preds), multi, set(first)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", help="predictions CSV in submission format")
    parser.add_argument("--fold", type=int, nargs="*", help="restrict to these folds (default: all)")
    parser.add_argument("--reference", choices=["human"], help="score a built-in reference instead of --pred")
    parser.add_argument("--json", help="also write the summary to this JSON file")
    args = parser.parse_args()
    if not args.pred and not args.reference:
        parser.error("give --pred or --reference")

    coco = load_coco(get_data_paths().annotations)
    ann, ai = annotation_table(coco), annotator_image_table(coco)
    folds = load_folds()
    if args.fold:
        folds = folds[folds.fold.isin(args.fold)]
    stems = folds.stem.tolist()

    if args.reference == "human":
        pred, multi, used = human_reference(ann, ai, stems)
        gt = ground_truth_table(ann[~ann.annotator_image.isin(used)], multi.stem)
    else:
        pred = read_submission(args.pred)
        problems = validate_submission(pred)
        if problems:
            print("WARNING, submission problems:", *problems, sep="\n  ")
        pred_stems = set(pred.filament_id.str.rsplit("_", n=1).str[0])
        missing = set(stems) - pred_stems
        if missing:
            print(f"note: {len(missing)} of {len(stems)} images have no predictions (count as all-FN)")
        gt = ground_truth_table(ann, stems)

    result = pq_score(gt, pred)
    summary = summarize(result)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()}, indent=2))

    site = lambda key: key.split("-", 1)[1][14]  # noqa: E731
    n_ann = ai.set_index("annotator_image").n_annotators.to_dict()
    fold_of = folds.set_index("stem").fold.to_dict()
    for name, group_of in [
        ("site", site),
        ("annotators per image", lambda k: n_ann[k]),
        ("fold", lambda k: fold_of[k.split("-", 1)[1]]),
    ]:
        print(f"\nby {name}:")
        print(pq_by_group(result, group_of).round(4).to_string(index=False))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
