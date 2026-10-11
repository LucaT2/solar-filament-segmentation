"""Competition data and file formats, shared by every model.

* ``annotations``  COCO annotation file -> tables of filaments, annotator-images and images
* ``images``       reading GONG JPEGs, finding the solar disk
* ``rle``          COCO run-length encoding (masks, polygons, IoU, area)
* ``folds``        the fixed 5-fold split (``configs/folds.csv``)
* ``submission``   building, writing, reading and validating submission CSVs
"""
