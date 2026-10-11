"""Solar filament instance segmentation (Kaggle Solar Filament Segmentation Challenge 2026).

Packages, in pipeline order:

* ``data``          competition data: annotations, images and the solar disk, RLE, folds, submission files
* ``segmentation``  per-pixel models (U-Net family, classical baseline) -> probability maps
* ``detection``     instance models (Mask R-CNN) -> scored instance masks
* ``postprocess``   probability maps -> instances: components, component classifier, submission
* ``evaluation``    the official PQ metric, fast OOF scoring, error analysis
* ``analysis``      exploratory data analysis behind the notebooks

Plus ``paths`` (where data and outputs live) and ``tracking`` (Weights & Biases).
"""

__version__ = "0.1.0"
