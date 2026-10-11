"""Instance models (planned: Mask R-CNN): each produces scored instance masks directly.

Planned modules:

* ``dataset``  instance targets (boxes + masks of one annotator) on training crops
* ``model``    torchvision Mask R-CNN (ResNet50-FPN v2, COCO weights), set up for 1-channel 2048 px images
* ``train``    training loop with fold validation (configs in ``configs/detection/``)
* ``predict``  full-image inference -> instance masks + scores -> submission rows

Shared pieces it reuses: ``filament.data`` (annotations, images, RLE, folds, submissions) and
``filament.evaluation`` (PQ). Detections can also be combined with the U-Net components as a classifier feature.
"""
