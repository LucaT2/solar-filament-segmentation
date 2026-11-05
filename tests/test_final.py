"""The files committed for the submitted pipeline are complete and consistent with filament.final."""

import json

from filament.final import DETECTOR, FINAL_CONFIG_DIR, FOLDS, UNETS, DETECTOR_WEIGHT, UNET_WEIGHT
from filament.postprocess.pipeline import load_postprocess
from filament.postprocess.rescoring import Rescorer


def test_manifest_lists_every_checkpoint():
    manifest = json.loads((FINAL_CONFIG_DIR / "weights.json").read_text())
    expected = {f"{exp}/fold{k}.pt" for exp in (*UNETS, DETECTOR) for k in FOLDS}
    assert set(manifest["files"]) == expected
    assert all(len(h) == 64 for h in manifest["files"].values())


def test_map_weights_sum_to_one():
    assert UNET_WEIGHT + DETECTOR_WEIGHT == 1.0


def test_tuned_settings_and_classifier_load():
    cfg = load_postprocess(FINAL_CONFIG_DIR / "postprocess.json")
    rescorer = Rescorer.load(FINAL_CONFIG_DIR / "rescorer.pkl")
    assert (cfg.threshold, cfg.min_area, rescorer.min_p) == (0.6, 400, 0.4)
