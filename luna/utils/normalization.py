import json
import numpy as np
from pathlib import Path
import logging

log = logging.getLogger(__name__)

class LunaNormalizer:
    def __init__(self, stats_path: str | Path):
        self.stats_path = Path(stats_path)
        self.stats = {}
        if self.stats_path.exists():
            with open(self.stats_path, "r") as f:
                self.stats = json.load(f)
            log.info(f"Loaded NAC stats from {self.stats_path}")
        else:
            log.warning(f"Stats file {self.stats_path} not found. Using local normalization.")

    def normalize(self, arr: np.ndarray, file_path: str | Path) -> np.ndarray:
        arr = arr.astype(np.float32)
        nac_id = Path(file_path).stem.split('_')[-1]
        
        if nac_id in self.stats:
            p2 = self.stats[nac_id]["min"]
            p98 = self.stats[nac_id]["max"]
            arr = (arr - p2) / (p98 - p2 + 1e-6)
        else:
            # Fallback: Robuste lokale Percentiles
            p2, p98 = np.percentile(arr, [2, 98])
            arr = (arr - p2) / (p98 - p2 + 1e-6)
            log.debug(f"NAC {nac_id} unknown, using local P2/P98")

        return np.clip(arr, 0, 1)