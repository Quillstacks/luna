"""
luna.latent_map.prober
~~~~~~~~~~~~~~~~~~~~~~
Parallel Pithos prober and sampler for scanning index files in data/_scratch/indices_old.
"""

from __future__ import annotations

import logging
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

from luna.screening.pithos import PithosMIDB

log = logging.getLogger(__name__)


# Global Pithos handle per worker process
_db_instance: PithosMIDB | None = None


def _init_worker() -> None:
    global _db_instance
    _db_instance = PithosMIDB()


def _sample_index_worker(args: Tuple[str, int]) -> List[Dict[str, Any]]:
    """
    Subprocess worker: reads a single index file set, samples max_per_index tiles,
    and returns metadata + raw float32 embeddings.
    """
    meta_path_str, max_per_index = args
    meta_path = Path(meta_path_str)
    index_prefix = str(meta_path)[:-9]  # strip '_meta.pkl'
    fp16_path = Path(f"{index_prefix}.bin_fp16.bin")

    if not meta_path.exists() or not fp16_path.exists():
        return []

    try:
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        raw_bytes = fp16_path.read_bytes()
        vecs_fp16 = np.frombuffer(raw_bytes, dtype=np.float16).reshape(-1, 384)

        n = len(metadata)
        if len(vecs_fp16) != n:
            return []

        if n <= max_per_index:
            indices = np.arange(n)
        else:
            indices = np.random.choice(n, size=max_per_index, replace=False)

        results = []
        for idx in indices:
            meta = metadata[idx]
            vec_f32 = vecs_fp16[idx].astype(np.float32)
            results.append({
                "product_id": meta.product_id,
                "x_offset": meta.x_offset,
                "y_offset": meta.y_offset,
                "width": meta.width,
                "height": meta.height,
                "lat": meta.lat,
                "lon": meta.lon,
                "vector": vec_f32,
            })
        return results
    except Exception as e:
        log.debug("Failed sampling index %s: %e", index_prefix, e)
        return []


class LatentSpaceProber:
    """
    Samples background tiles across indices_old and probes Hamming distance distributions.
    """

    def __init__(self, indices_dir: str | Path = "data/_scratch/indices_old") -> None:
        self.indices_dir = Path(indices_dir)

    def sample_background_tiles(
        self, target_sample_count: int = 10000, max_workers: int = 16
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Sample background tiles uniformly from all index files in indices_dir.

        Returns:
            metadata_list: list of tile metadata dicts
            vectors: (sample_count, 384) float32 numpy array
        """
        meta_files = sorted(self.indices_dir.glob("*_meta.pkl"))
        num_indices = len(meta_files)

        if num_indices == 0:
            raise FileNotFoundError(f"No *_meta.pkl files found in {self.indices_dir}")

        log.info("Found %d index files in %s. Sampling background tiles...", num_indices, self.indices_dir)

        max_per_index = max(1, target_sample_count // num_indices + 1)
        tasks = [(str(p), max_per_index) for p in meta_files]

        all_sampled_tiles: List[Dict[str, Any]] = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for tile_batch in executor.map(_sample_index_worker, tasks):
                all_sampled_tiles.extend(tile_batch)

        log.info("Sampled %d background tiles across the Moon.", len(all_sampled_tiles))

        if not all_sampled_tiles:
            raise ValueError("No background tiles could be sampled from indices_old.")

        # Subsample to exact target_sample_count if needed
        if len(all_sampled_tiles) > target_sample_count:
            selected_indices = np.random.choice(len(all_sampled_tiles), size=target_sample_count, replace=False)
            all_sampled_tiles = [all_sampled_tiles[i] for i in selected_indices]

        vectors = np.stack([t["vector"] for t in all_sampled_tiles]).astype(np.float32)
        
        # Remove vector key from metadata dict to save memory
        metadata_list = []
        for t in all_sampled_tiles:
            meta = {k: v for k, v in t.items() if k != "vector"}
            metadata_list.append(meta)

        return metadata_list, vectors
