import json
import types

import pytest

from filament.paths import REPO_ROOT

OFFICIAL_NOTEBOOK = REPO_ROOT / "self-evaluation-notebook.ipynb"


@pytest.fixture(scope="session")
def coco():
    from filament.data.annotations import load_coco
    from filament.paths import get_data_paths

    try:
        paths = get_data_paths()
    except FileNotFoundError:
        pytest.skip("competition data not available")
    return load_coco(paths.annotations)


@pytest.fixture(scope="session")
def ann(coco):
    from filament.data.annotations import annotation_table

    return annotation_table(coco)


@pytest.fixture(scope="session")
def ai(coco):
    from filament.data.annotations import annotator_image_table

    return annotator_image_table(coco)


@pytest.fixture(scope="session")
def official():
    """The organizers' metric functions, executed straight from their self-evaluation notebook.

    The notebook is not redistributed with this repo (see .gitignore); tests using it are skipped without it.
    """
    if not OFFICIAL_NOTEBOOK.exists():
        pytest.skip("official self-evaluation notebook not available locally")
    nb = json.loads(OFFICIAL_NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # only the import cell and the cells defining the scoring helpers (skip plotting and the TODO cells)
        if source.lstrip().startswith("import") or "def get_overlap" in source or "def get_pq_score" in source:
            exec(compile(source, str(OFFICIAL_NOTEBOOK), "exec"), namespace)
    return types.SimpleNamespace(**{k: namespace[k] for k in ("get_overlap_df", "get_pq_score")})
