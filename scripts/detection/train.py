"""Train Mask R-CNN on one fold (or several folds / all data).

Usage:
    python scripts/detection/train.py --config configs/detection/maskrcnn/maskrcnn_r50_v2_m56.yaml --fold 0
    python scripts/detection/train.py --config configs/detection/maskrcnn/maskrcnn_r50_v2_m56.yaml --fold 0 \
        --set epochs=1 samples_per_epoch=40 batch_size=2 num_workers=0                          # smoke test
"""

import argparse

from filament.detection.train import DetTrainConfig, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, nargs="+", default=[0], help="-1 = train on all images")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. epochs=12 lr=2e-4")
    args = parser.parse_args()
    cfg = DetTrainConfig.load(args.config, args.set)
    print(cfg)
    for fold in args.fold:
        run(cfg, fold)


if __name__ == "__main__":
    main()
