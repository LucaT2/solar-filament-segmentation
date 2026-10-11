"""Instance models (Mask R-CNN): each predicts scored instance masks directly.

* ``instances``  per-image detections (boxes, scores, small soft masks) saved as ``<stem>.npz``; pasting into the
                 image and overlap resolution -> ``Components``, the same structure the probability maps produce
* ``dataset``    training crops with one annotator's filaments as instances (same sampling/augmentation as U-Net)
* ``model``      torchvision Mask R-CNN ResNet50-FPN v2 for 1-channel images at native resolution; raw inference
* ``train``      training loop with fold validation (``DetTrainConfig``, configs in ``configs/detection/``)

Because saved instances load through ``filament.postprocess.predictions`` like probability maps, tuning
(``scripts/postprocess/tune_postprocess.py``), comparison (``scripts/evaluation/compare.py``) and submissions
(``scripts/postprocess/make_submission.py``) work for detection experiments unchanged.
"""
