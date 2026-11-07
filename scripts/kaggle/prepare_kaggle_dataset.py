"""Package an experiment's trained artifacts as a (private) Kaggle Dataset, for reuse in the Kaggle notebook.

Copies fold checkpoints, prediction folders (default: oof/), post-processing settings and logs from outputs/<exp>/
into outputs/kaggle_upload/<slug>/ with a dataset-metadata.json, then prints the Kaggle CLI command to upload it.

Usage:
    python scripts/kaggle/prepare_kaggle_dataset.py --exp unet_r34 --user tasadanluca
    python scripts/kaggle/prepare_kaggle_dataset.py --exp unet_r34_skel050 --user tasadanluca \
        --maps oof test --no-checkpoints --slug solar-filament-unet-r34-skel050-maps     # maps only (guide input)
    py -3.14 -m kaggle datasets create -p outputs/kaggle_upload/solar-filament-unet-r34 --dir-mode zip   # first time
    py -3.14 -m kaggle datasets version -p outputs/kaggle_upload/solar-filament-unet-r34 -m "msg" --dir-mode zip  # update
"""

from __future__ import annotations

import argparse
import json
import shutil

from filament.paths import REPO_ROOT, resolve_exp_dir

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
    parser.add_argument("--maps", nargs="*", default=["oof"], help="prediction sub-folders to include")
    parser.add_argument("--no-checkpoints", action="store_true", help="leave out the model weights (*.pt)")
    args = parser.parse_args()

    exp_dir = resolve_exp_dir(args.exp)  # name or folder path
    slug = args.slug or f"solar-filament-{args.exp.replace('_', '-')}"
    dst = REPO_ROOT / "outputs" / "kaggle_upload" / slug
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    n = 0
    for pattern in PATTERNS:
        if args.no_checkpoints and pattern.endswith(".pt"):
            continue
        for f in sorted(exp_dir.glob(pattern)):
            shutil.copy2(f, dst / f.name)
            n += 1
    for sub in args.maps:
        if not (exp_dir / sub).is_dir():
            raise SystemExit(f"{exp_dir / sub} does not exist")
        shutil.copytree(exp_dir / sub, dst / sub)
    (dst / "dataset-metadata.json").write_text(json.dumps({
        "title": slug,
        "id": f"{args.user}/{slug}",
        "licenses": [{"name": "other"}],
        "subtitle": subtitle(args.exp),
    }, indent=2))

    size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 2**20
    maps = ", ".join(f"{len(list((dst / s).iterdir()))} in {s}/" for s in args.maps)
    print(f"{dst}: {n} files + predictions ({maps or 'none'}), {size:.0f} MB")
    print("\nUpload (private by default):")
    print(f"  py -3.14 -m kaggle datasets create -p \"{dst}\" --dir-mode zip")
    print("Later updates:")
    print(f"  py -3.14 -m kaggle datasets version -p \"{dst}\" -m \"update\" --dir-mode zip")


if __name__ == "__main__":
    main()
