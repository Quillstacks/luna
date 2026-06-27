"""
luna.storage.pithos_store
~~~~~~~~~~~~~~~~~~~~~~~~~
Drop-in replacement for FaissLocalStore that writes native Pithos PLAN binary
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

from luna.screening.pithos import PithosMIDB, MOON_ID, MOON_RADIUS
from luna.screening.protocols import TileMetadata

log = logging.getLogger("luna.storage.pithos_store")


class PithosStore:
    """
    In-memory buffer that compiles to a Pithos PLAN binary index on disk.

    Vectors are received as float32 DINOv3 embeddings and buffered in RAM.
    On ``save_to_disk`` they are compiled into a memory-mapped off-heap ``.bin`` file.

    Record IDs are assigned sequentially starting from 0, so they map
    directly to metadata list indices — exactly like the old FAISS store.

    Parameters
    ----------
    planet_id : int
        Planet registry byte.  Use ``MOON_ID`` (= 1) for the Moon.
    planet_radius : int
        Mean radius in metres.  Use ``MOON_RADIUS`` (= 1 737 400).
    """

    def __init__(
        self,
        planet_id:     int = MOON_ID,
        planet_radius: int = MOON_RADIUS,
    ) -> None:

    def __init__(
        self,
        planet_id:     int = MOON_ID,
        planet_radius: int = MOON_RADIUS,
    ) -> None:
        self._planet_id     = planet_id
        self._planet_radius = planet_radius
        self._vectors:  list[np.ndarray]   = []
        self._metadata: list[TileMetadata] = []

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
        Compile all buffered vectors into a PLAN binary index.

        Writes two files:
        * ``{prefix_path}.bin``      — Pithos off-heap index (PLAN format)
        * ``{prefix_path}_meta.pkl`` — list[TileMetadata] (same layout as
                                        the old FAISS store)
        """"
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

        # 2. Sequential IDs — index into metadata list directly
        ids = np.arange(n, dtype=np.int64)

        # 3. Compile native PLAN index
        Path(bin_path).parent.mkdir(parents=True, exist_ok=True)
        log.info("Compiling Pithos PLAN index → %s …", bin_path)
        with PithosMIDB() as db:
            db.build_index(
                file_path     = bin_path,
                planet_id     = self._planet_id,
                planet_radius = self._planet_radius,
                ids           = ids,
                vectors       = all_vectors,
            )

        # 5. Write metadata (list[TileMetadata], index == record ID)
        with open(meta_path, "wb") as f:
            pickle.dump(self._metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        log.info(
            "Saved %d records → %s  /  %s", n, bin_path, meta_path
        )
