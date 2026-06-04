"""
luna.storage.lcvk_store
~~~~~~~~~~~~~~~~~~~~~~~
Drop-in replacement for FaissLocalStore that writes native LCVK PLAN binary
indices instead of FAISS flat indices.

Implements the ``VectorStore`` protocol exactly — call sites in
``DataIngestor`` and ``LunaPipeline`` require no changes beyond swapping the
class name.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from luna.screening.lcvk import LcvkEngine
from luna.screening.protocols import TileMetadata

log = logging.getLogger("luna.storage.lcvk_store")


class LcvkLocalStore:
    """
    In-memory buffer that compiles to a LCVK PLAN binary index on disk.

    Vectors are received as float32 DINOv3 embeddings and buffered in RAM.
    On ``save_to_disk`` they are binarized with the PolarQuant-Hadamard
    transform and compiled into a memory-mapped off-heap ``.bin`` file via
    ``vdb_compile_index_file``.

    Record IDs are assigned sequentially starting from 0, so they map
    directly to metadata list indices — exactly like the old FAISS store.

    Parameters
    ----------
    planet_id : int
        Planet registry byte.  Use ``LcvkEngine.MOON_ID`` (= 1) for the Moon.
    planet_radius : int
        Mean radius in metres.  Use ``LcvkEngine.MOON_RADIUS`` (= 1 737 400).
    """

    def __init__(
        self,
        planet_id:     int = LcvkEngine.MOON_ID,
        planet_radius: int = LcvkEngine.MOON_RADIUS,
    ) -> None:
        self._planet_id     = planet_id
        self._planet_radius = planet_radius
        self._vectors:  list[np.ndarray]    = []
        self._metadata: list[TileMetadata]  = []

    # ------------------------------------------------------------------
    # VectorStore protocol
    # ------------------------------------------------------------------

    def upsert(self, vectors: np.ndarray, metadata: list[TileMetadata]) -> None:
        """Buffer a batch of float32 embeddings and their metadata."""
        if len(vectors) == 0:
            log.debug("upsert called with empty batch — skipping")
            return
        if len(vectors) != len(metadata):
            raise ValueError(
                f"vectors/metadata length mismatch: {len(vectors)} vs {len(metadata)}"
            )
        self._vectors.append(vectors.astype(np.float32))
        self._metadata.extend(metadata)

    def save_to_disk(self, prefix_path: str) -> None:
        """
        Binarize all buffered vectors and compile a PLAN binary index.

        Writes two files:
        * ``{prefix_path}.bin``      — LCVK off-heap index (PLAN format)
        * ``{prefix_path}_meta.pkl`` — list[TileMetadata] (same layout as
                                        the old FAISS store)
        """
        if not self._vectors:
            log.warning("No vectors buffered — nothing to save")
            return

        bin_path  = f"{prefix_path}.bin"
        meta_path = f"{prefix_path}_meta.pkl"

        # 1. Concatenate all buffered batches
        all_vectors = np.ascontiguousarray(
            np.concatenate(self._vectors, axis=0), dtype=np.float32
        )
        n = len(all_vectors)
        log.info(
            "Binarizing %d vectors with PolarQuant-Hadamard transform …", n
        )

        # 2. Binarize: float32 (N, 384) → int64 (N, 6)
        binary_vecs = LcvkEngine.binarize(all_vectors)

        # 3. Sequential IDs — index into metadata list directly
        ids = np.arange(n, dtype=np.int64)

        # 4. Compile native PLAN index
        Path(bin_path).parent.mkdir(parents=True, exist_ok=True)
        log.info("Compiling LCVK PLAN index → %s …", bin_path)
        with LcvkEngine() as engine:
            engine.build_index(
                file_path     = bin_path,
                planet_id     = self._planet_id,
                planet_radius = self._planet_radius,
                ids           = ids,
                vectors       = binary_vecs,
            )

        # 5. Write metadata (list[TileMetadata], index == record ID)
        with open(meta_path, "wb") as f:
            pickle.dump(self._metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        log.info(
            "Saved %d records → %s  /  %s", n, bin_path, meta_path
        )
