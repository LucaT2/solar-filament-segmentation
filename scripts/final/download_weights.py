"""Download the checkpoint of the submitted pipeline from the GitHub release into outputs/, checking its sha256.

A file already present with the right hash is skipped, so it is safe to run twice. Needs no account or token.

Usage:
    python scripts/final/download_weights.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from filament.final import FINAL_CONFIG_DIR
from filament.paths import get_output_dir


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def asset_name(checkpoint: str) -> str:
    """Release assets are flat: ``<experiment>/full_s42.pt`` is published as ``<experiment>-full_s42.pt``."""
    return checkpoint.replace("/", "-")


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    manifest = json.loads((FINAL_CONFIG_DIR / "weights.json").read_text())
    for checkpoint, expected in manifest["files"].items():
        target = get_output_dir() / checkpoint
        if target.exists() and sha256(target) == expected:
            print(f"ok          {checkpoint}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {checkpoint}", flush=True)
        partial = target.with_suffix(".part")
        with urllib.request.urlopen(f"{manifest['release']}/{asset_name(checkpoint)}") as response, \
                open(partial, "wb") as f:
            shutil.copyfileobj(response, f, 1 << 24)
        if sha256(partial) != expected:
            partial.unlink()
            raise SystemExit(f"sha256 mismatch for {checkpoint}")
        partial.replace(target)
    print(f"{len(manifest['files'])} checkpoint(s) in {get_output_dir()}")


if __name__ == "__main__":
    main()
