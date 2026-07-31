"""
luna.latent_map.dataset_converter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Stream-converts float16/float32 tile embeddings across index directories into
compressed columnar Parquet dataset format with 2D Parametric UMAP coordinates.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from luna.latent_map.parametric_umap import ParametricUMAPEncoder
from luna.screening.pithos import PithosMIDB

log = logging.getLogger(__name__)


class ManifoldDatasetConverter:
    """
    Converts compiled index files into 2D Parquet manifold dataset.
    """

    def __init__(
        self,
        encoder: ParametricUMAPEncoder | None = None,
        indices_dir: str | Path = "data/_scratch/indices_old",
    ) -> None:
        self.indices_dir = Path(indices_dir)
        self.encoder = encoder or ParametricUMAPEncoder()

    def convert_to_parquet(
        self,
        pit_anchors_embeddings: np.ndarray | None = None,
        out_parquet_path: str | Path = "data/_scratch/luna_manifold.parquet",
        max_indices: int | None = None,
    ) -> str:
        """
        Process index files, run GPU forward-pass, and write compressed Parquet file.
        """
        out_path = Path(out_parquet_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        meta_files = sorted(self.indices_dir.glob("*_meta.pkl"))
        if max_indices:
            meta_files = meta_files[:max_indices]

        log.info("Processing %d index files for Parquet conversion...", len(meta_files))

        all_rows: List[Dict[str, Any]] = []
        all_vecs: List[np.ndarray] = []

        # Prepare reference bit packing for Hamming distance calculation if anchors provided
        bits_anchors = None
        if pit_anchors_embeddings is not None and len(pit_anchors_embeddings) > 0:
            binary_anchors = PithosMIDB.binarize(pit_anchors_embeddings)
            bits_anchors = np.unpackbits(binary_anchors.view(np.uint8), axis=1)[:, :384]

        for meta_file in meta_files:
            prefix = str(meta_file)[:-9]
            fp16_file = Path(f"{prefix}.bin_fp16.bin")
            if not fp16_file.exists():
                continue

            try:
                with open(meta_file, "rb") as f:
                    metadata = pickle.load(f)

                raw_bytes = fp16_file.read_bytes()
                vecs_fp16 = np.frombuffer(raw_bytes, dtype=np.float16).reshape(-1, 384)

                n = len(metadata)
                if len(vecs_fp16) != n:
                    continue

                for i in range(n):
                    meta = metadata[i]
                    vec_f32 = vecs_fp16[i].astype(np.float32)

                    hamming_dist = 0
                    if bits_anchors is not None:
                        bin_tile = PithosMIDB.binarize(vec_f32.reshape(1, 384))
                        bits_tile = np.unpackbits(bin_tile.view(np.uint8), axis=1)[:, :384]
                        hamming_dist = int(np.min(np.sum(bits_anchors != bits_tile, axis=1)))

                    all_rows.append({
                        "product_id": meta.product_id,
                        "x_offset": meta.x_offset,
                        "y_offset": meta.y_offset,
                        "width": meta.width,
                        "height": meta.height,
                        "lat": meta.lat,
                        "lon": meta.lon,
                        "hamming_dist": hamming_dist,
                        "category": "Regolith Background",
                    })
                    all_vecs.append(vec_f32)
            except Exception as e:
                log.debug("Skipping %s due to read error: %s", prefix, e)

        if not all_vecs:
            raise ValueError("No tile vectors could be loaded for dataset conversion.")

        log.info("Loaded %d tiles. Running GPU Parametric UMAP projection...", len(all_vecs))
        embeddings_matrix = np.stack(all_vecs).astype(np.float32)
        coords_2d = self.encoder.project_embeddings_gpu(embeddings_matrix)

        df = pd.DataFrame(all_rows)
        df["x_2d"] = coords_2d[:, 0]
        df["y_2d"] = coords_2d[:, 1]

        df.to_parquet(out_path, compression="snappy", index=False)
        log.info("Saved compressed Parquet dataset → %s (Rows: %d)", out_path, len(df))

        return str(out_path)
