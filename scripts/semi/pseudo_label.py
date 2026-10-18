"""Pseudo-label unlabelled GONG frames with the fold-0-clean teacher (filament.semi.pseudo).

For every frame in --frames: a uint16 label map of the teacher's size-filtered filament candidates
(<out>/labels/<stem>.png), one row per candidate with the classifier's P(match) (<out>/candidates.csv), and one row per
frame with its disk and brightness statistics (<out>/frames.csv). Frames whose disk or brightness looks wrong are
recorded and skipped. Resumable: frames with an existing label map are skipped.

Usage:
    python scripts/semi/pseudo_label.py --frames outputs/unlabelled/frames --out outputs/pseudo
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import pandas as pd
import torch
from tqdm import tqdm

from filament.detection.train import build_from_config
from filament.paths import resolve_exp_dir
from filament.postprocess.pipeline import load_postprocess
from filament.postprocess.rescoring import Rescorer
from filament.segmentation.model import build_model
from filament.semi.pseudo import Teacher

UNETS = ["unet_r34_skel050/fold0.pt", "outputs/kaggle_runs/solar-filament-v2_rerun/unet_convnext_t_skel050/fold0.pt"]
DETECTOR = "outputs/kaggle_runs/solar-filament-v2_maskrcnn/maskrcnn_r50_v2_m56/fold0.pt"


def load_unet(path: Path, device):
    state = torch.load(path, map_location=device)
    c = state["config"]
    model = build_model(c["arch"], c["encoder"], pretrained=False, decoder_fp32=c.get("decoder_fp32", False))
    model.load_state_dict(state["model"])
    return model.to(device).eval(), c


def load_detector(path: Path, device):
    state = torch.load(path, map_location=device)
    model = build_from_config(state["config"], pretrained=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state["config"]


def resolve_ckpt(spec: str) -> Path:
    p = Path(spec)
    if p.exists():
        return p
    exp, name = spec.split("/", 1)
    return resolve_exp_dir(exp) / name


def append_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", default="outputs/unlabelled/frames")
    parser.add_argument("--out", default="outputs/pseudo")
    parser.add_argument("--teacher", default="teacher_f0", help="experiment with the teacher's postprocess.json / rescorer.pkl")
    parser.add_argument("--unets", nargs="+", default=UNETS)
    parser.add_argument("--detector", default=DETECTOR)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tdir = resolve_exp_dir(args.teacher)
    unets = [load_unet(resolve_ckpt(u), device) for u in args.unets]
    det, det_cfg = load_detector(resolve_ckpt(args.detector), device)
    teacher = Teacher([m for m, _ in unets], [c for _, c in unets], det, det_cfg,
                      load_postprocess(tdir / "postprocess.json"), Rescorer.load(tdir / "rescorer.pkl"), device)
    print(f"teacher: {[c['name'] for _, c in unets]} + {det_cfg['name']}, {teacher.cfg}, min_p {teacher.rescorer.min_p}")

    out = Path(args.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    frames = sorted(Path(args.frames).glob("*.jpg"))[: args.limit]
    todo = [f for f in frames if not (out / "labels" / f"{f.stem}.png").exists()]
    print(f"{len(frames)} frames, {len(todo)} to label", flush=True)
    t0 = time.time()
    for f in tqdm(todo, desc="pseudo-label"):
        try:
            labels, rows, info = teacher.label(f)
        except Exception as exc:  # unreadable frame or no disk found
            append_csv(out / "frames.csv", [{"stem": f.stem, "ok": False, "reason": str(exc)[:80]}])
            cv2.imwrite(str(out / "labels" / f"{f.stem}.png"), (0 * cv2.imread(str(f), 0)).astype("uint16"))
            continue
        ok = 850 <= info.disk.radius <= 1050 and 60 <= info.mean <= 200 and info.std > 5
        cv2.imwrite(str(out / "labels" / f"{f.stem}.png"), labels)
        append_csv(out / "candidates.csv", rows if ok else [])
        append_csv(out / "frames.csv", [{"stem": f.stem, "ok": ok, "reason": "" if ok else "disk/brightness",
                                         "cx": info.disk.cx, "cy": info.disk.cy, "radius": info.disk.radius,
                                         "mean": info.mean, "std": info.std, "n_candidates": len(rows)}])
    print(f"labelled {len(todo)} frames in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
