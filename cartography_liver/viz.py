import spatialdata as sd
from napari_spatialdata import Interactive
import logging

logging.basicConfig(level=logging.INFO)


def viz(path):
    logging.info(f"Loading {path}")
    sdata = sd.SpatialData.read(str(path))
    Interactive(sdata)

if __name__ == "__main__":
    import os
    from pathlib import Path
    root = Path(os.environ.get("DATA_ROOT", "data/"))
    viz(root / "ResolveHuman.zarr")