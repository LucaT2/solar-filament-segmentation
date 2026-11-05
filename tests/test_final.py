"""The files committed for the submitted pipeline are complete and consistent with filament.final."""

import json

from filament.final import CHECKPOINT, EXPERIMENT, FINAL_CONFIG_DIR
from filament.postprocess.pipeline import load_postprocess


def test_manifest_lists_the_checkpoint():
    manifest = json.loads((FINAL_CONFIG_DIR / "weights.json").read_text())
    assert set(manifest["files"]) == {f"{EXPERIMENT}/{CHECKPOINT}"}
    assert all(len(h) == 64 for h in manifest["files"].values())


def test_tuned_settings_load():
    cfg = load_postprocess(FINAL_CONFIG_DIR / "postprocess.json")
    assert (cfg.threshold, cfg.min_area, cfg.min_score) == (0.45, 0, 0.8)
