"""Sanity-check the environment: packages, GPU, and competition data.

Usage:
    python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import platform
import sys

PACKAGES = [
    "torch",
    "torchvision",
    "segmentation_models_pytorch",
    "timm",
    "albumentations",
    "pycocotools",
    "cv2",
    "skimage",
    "scipy",
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "yaml",
    "tqdm",
    "wandb",
]


def check_packages() -> bool:
    ok = True
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            print(f"  {name:<28} {getattr(module, '__version__', '?')}")
        except ImportError as exc:
            print(f"  {name:<28} MISSING ({exc})")
            ok = False
    return ok


def check_gpu() -> bool:
    import torch

    if not torch.cuda.is_available():
        print("  CUDA not available (CPU only)")
        return False
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} {props.name}  {props.total_memory / 2**30:.1f} GiB  sm_{props.major}{props.minor}")
    # A real kernel launch catches wheels built without this GPU's architecture.
    x = torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()
    print(f"  matmul test ok: {bool(torch.isfinite(x @ x).all())}   CUDA runtime {torch.version.cuda}")
    return True


def check_data() -> bool:
    from filament.paths import get_data_paths

    try:
        paths = get_data_paths()
    except FileNotFoundError as exc:
        print(f"  {exc}")
        return False
    n_train = len(list(paths.train_images.glob("*.jpeg")))
    n_test = len(list(paths.test_images.glob("*.jpeg")))
    print(f"  root:        {paths.root}")
    print(f"  train imgs:  {n_train}")
    print(f"  test imgs:   {n_test}")
    print(f"  annotations: {paths.annotations.exists()}")
    return n_train > 0 and n_test > 0 and paths.annotations.exists()


def main() -> int:
    print(f"Python {sys.version.split()[0]} on {platform.platform()}")
    print("Packages:")
    pkgs = check_packages()
    print("GPU:")
    gpu = check_gpu() if pkgs else False
    print("Data:")
    data = check_data()
    print(f"\npackages={'ok' if pkgs else 'FAIL'}  gpu={'ok' if gpu else 'none'}  data={'ok' if data else 'FAIL'}")
    return 0 if pkgs and data else 1


if __name__ == "__main__":
    sys.exit(main())
