"""Probability maps -> filament instances -> submission.

* ``maps``        probability maps on disk (8-bit PNGs, one per image)
* ``components``  threshold + connected components + size filter (``PostConfig``), disk region
* ``rescoring``   the component classifier that decides which components to keep
* ``pipeline``    maps -> submission table; saving and loading the tuned settings (``postprocess.json``)
"""
