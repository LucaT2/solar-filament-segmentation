"""Train a segmentation model on one fold (or all folds / all data).

Usage:
    python scripts/segmentation/train.py --config configs/segmentation/unet_r34.yaml --fold 0
    python scripts/segmentation/train.py --config configs/segmentation/unet_r34.yaml --fold 0 1 2 3 4
    python scripts/segmentation/train.py --config configs/segmentation/unet_r34.yaml --fold 0 --set epochs=2 samples_per_epoch=200
"""

import argparse

from filament.segmentation.train import TrainConfig, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, nargs="+", default=[0], help="-1 = train on all images")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. epochs=5 lr=3e-4")
    args = parser.parse_args()
    cfg = TrainConfig.load(args.config, args.set)
    print(cfg)
    for fold in args.fold:
        run(cfg, fold)


if __name__ == "__main__":
    main()
