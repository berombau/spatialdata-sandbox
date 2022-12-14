import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

root = Path(os.environ.get("DATA_ROOT", "data/"))

logging.info(f"root = {root}")
dataset = root / "ResolveHuman"
if dataset.exists():
    logging.info(f"Already downloaded")
else:
    logging.info(f"Downloading data to {dataset}")
    raise NotImplementedError