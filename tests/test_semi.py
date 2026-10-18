"""Student side of semi-supervised training: sampling crops from pseudo-labelled frames (synthetic folder)."""

import cv2
import numpy as np
import pandas as pd

from filament.semi.data import OFF_DISK_VALUE, PseudoFrames


def _pseudo_folder(tmp_path, unsure_p: float):
    (tmp_path / "frames").mkdir()
    (tmp_path / "labels").mkdir()
    img = np.full((256, 256), 120, np.uint8)
    img[:20, :60] = 250  # "text" in a corner, outside the disk
    cv2.imwrite(str(tmp_path / "frames" / "f1.jpg"), img)
    labels = np.zeros((256, 256), np.uint16)
    labels[100:110, 60:120] = 1  # confident filament
    labels[100:110, 150:200] = 2  # second candidate (confident or unsure, depending on unsure_p)
    cv2.imwrite(str(tmp_path / "labels" / "f1.png"), labels)
    pd.DataFrame([{"stem": "f1", "ok": True, "cx": 128.0, "cy": 128.0, "radius": 110.0, "mean": 120.0,
                   "std": 10.0}]).to_csv(tmp_path / "frames.csv", index=False)
    pd.DataFrame([{"stem": "f1", "id": 1, "area": 600, "p": 0.9, "cy": 105.0, "cx": 90.0},
                  {"stem": "f1", "id": 2, "area": 500, "p": unsure_p, "cy": 105.0, "cx": 175.0}]).to_csv(
        tmp_path / "candidates.csv", index=False)
    return PseudoFrames(tmp_path, conf=0.6, unsure=0.3)


def test_confident_candidates_become_instances_and_text_is_blanked(tmp_path):
    pf = _pseudo_folder(tmp_path, unsure_p=0.8)  # both confident
    rng = np.random.default_rng(0)
    img, layers, info = pf.sample(rng, 256, pos_fraction=1.0)
    assert len(layers) == 2 and all(m.dtype == np.uint8 for m in layers)
    assert info.mean == 120.0 and info.disk.radius == 110.0
    assert abs(int(img[5, 5]) - OFF_DISK_VALUE) <= 2  # corner text replaced
    assert abs(int(img[128, 128]) - 120) <= 3  # the disk is untouched


def test_crops_containing_an_unsure_candidate_are_rejected(tmp_path):
    pf = _pseudo_folder(tmp_path, unsure_p=0.45)  # candidate 2 is unsure
    assert pf.sample(np.random.default_rng(0), 256, pos_fraction=1.0) is None  # every full-frame crop contains it
    rng = np.random.default_rng(1)
    got = [pf.sample(rng, 64, pos_fraction=1.0) for _ in range(30)]
    ok = [g for g in got if g is not None]
    assert ok and all(len(layers) <= 1 for _, layers, _ in ok)  # only the confident filament, never candidate 2


def test_background_candidates_are_ignored(tmp_path):
    pf = _pseudo_folder(tmp_path, unsure_p=0.1)  # candidate 2 is very likely false: plain background
    img, layers, _ = pf.sample(np.random.default_rng(0), 256, pos_fraction=1.0)
    assert len(layers) == 1 and layers[0][100:110, 60:120].all()
