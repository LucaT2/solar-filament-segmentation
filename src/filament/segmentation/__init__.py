"""Per-pixel (semantic) models: each produces one filament-probability map per image.

* ``dataset``    training crops, augmentation, Skeleton Recall targets
* ``model``      architectures (segmentation_models_pytorch), losses, tiled full-image prediction with TTA
* ``train``      training loop with fold validation (``TrainConfig``, configs in ``configs/segmentation/``)
* ``classical``  the non-learned darkness baseline

The maps are written with ``filament.postprocess.maps`` and turned into instances by ``filament.postprocess``.
"""
