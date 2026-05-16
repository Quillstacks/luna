import json
import numpy as np
from pathlib import Path
import logging

log = logging.getLogger(__name__)


class LunaNormalizer:
    def __init__(self, stats_path: str | Path):
        self.stats_path = Path(stats_path)
        self.stats: dict = {}
        if self.stats_path.exists():
            with open(self.stats_path) as f:
                self.stats = json.load(f)
            log.info("Loaded NAC stats from %s", self.stats_path)
        else:
            log.warning("Stats file %s not found. Using local normalization.", self.stats_path)

    def normalize(self, arr: np.ndarray, file_path: str | Path) -> np.ndarray:
        arr    = arr.astype(np.float32)
        nac_id = Path(file_path).stem.split("_")[-1]

        if nac_id in self.stats:
            lo, hi = self.stats[nac_id]["min"], self.stats[nac_id]["max"]
        else:
            lo, hi = arr.min(), arr.max()
            log.debug("NAC %s unknown, using local min/max", nac_id)

        return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)