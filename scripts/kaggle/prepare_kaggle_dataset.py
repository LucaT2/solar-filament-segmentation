"""Package an experiment's trained artifacts as a (private) Kaggle Dataset, for reuse in the Kaggle notebook.

Copies fold checkpoints, out-of-fold maps, post-processing settings and logs from outputs/<exp>/ into
outputs/kaggle_upload/<slug>/ with a dataset-metadata.json, then prints the Kaggle CLI command to upload it.

Usage:
    python scripts/kaggle/prepare_kaggle_dataset.py --exp unet_r34 --user tasadanluca
    py -3.14 -m kaggle datasets create -p outputs/kaggle_upload/solar-filament-unet-r34 --dir-mode zip   # first time
    py -3.14 -m kaggle datasets version -p outputs/kaggle_upload/solar-filament-unet-r34 -m "msg" --dir-mode zip  # update
"""

from __future__ import annotations

import argparse
import json
import shutil

from filament.paths import REPO_ROOT, get_output_dir, resolve_exp_dir

PATTERNS = ["fold*.pt", "full_s*.pt", "postprocess.json", "rescorer.json", "config.yaml", "log_*.csv", "result_*.json"]


def subtitle(exp: str) -> str:
    """Kaggle requires 20-80 characters."""
    text = f"Solar filament models + OOF maps: {exp}"
    return (text if len(text) <= 80 else text[:77] + "...").ljust(20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--user", required=True, help="Kaggle username")
    parser.add_argument("--slug", help="dataset slug (default: solar-filament-<exp with _ -> ->)")
    args = parser.parse_args()

    exp_dir = resolve_exp_dir(args.exp)  # name or folder path
    slug = args.slug or f"solar-filament-{args.exp.replace('_', '-')}"
    dst = REPO_ROOT / "outputs" / "kaggle_upload" / slug
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    n = 0
    for pattern in PATTERNS:
        for f in sorted(exp_dir.glob(pattern)):
            shutil.copy2(f, dst / f.name)
            n += 1
    if (exp_dir / "oof").is_dir():
        shutil.copytree(exp_dir / "oof", dst / "oof")
    (dst / "dataset-metadata.json").write_text(json.dumps({
        "title": slug,
        "id": f"{args.user}/{slug}",
        "licenses": [{"name": "other"}],
        "subtitle": subtitle(args.exp),
    }, indent=2))

    size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 2**20
    n_oof = len(list((dst / "oof").glob("*.png"))) if (dst / "oof").exists() else 0
    print(f"{dst}: {n} files + {n_oof} OOF maps, {size:.0f} MB")
    print("\nUpload (private by default):")
    print(f"  py -3.14 -m kaggle datasets create -p \"{dst}\" --dir-mode zip")
    print("Later updates:")
    print(f"  py -3.14 -m kaggle datasets version -p \"{dst}\" -m \"update\" --dir-mode zip")


if __name__ == "__main__":
    main()
