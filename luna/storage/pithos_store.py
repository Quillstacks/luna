"""
luna.storage.pithos_store
~~~~~~~~~~~~~~~~~~~~~~~~~
VectorStore implementation backed by the Pithos MIDB native library.

Vectors are buffered in RAM as raw float32 and compiled to an off-heap
``.bin`` index on ``save_to_disk``.  Record IDs are sequential so they
map directly to the metadata list.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from luna.screening.pithos import PithosMIDB, MOON_ID, MOON_RADIUS, MOON_TIERS
from luna.screening.protocols import TileMetadata

log = logging.getLogger("luna.storage.pithos_store")


class PithosStore:
    """
    In-memory buffer that compiles to a Pithos PLAN binary index on disk.

    Parameters
    ----------
    planet_id     : Planet registry byte.  Use ``MOON_ID`` (= 1) for the Moon.
    planet_radius : Mean radius in metres.  Use ``MOON_RADIUS`` (= 1 737 400).
    tiers         : Matryoshka cascade tier boundaries (int32 array).
    """

    def __init__(
        self,
        planet_id:     int         = MOON_ID,
        planet_radius: int         = MOON_RADIUS,
        tiers:         np.ndarray  = MOON_TIERS,
        use_fp16:      bool        = False,
        use_cuda:      bool        = False,
    ) -> None:
        self._planet_id     = planet_id
        self._planet_radius = planet_radius
        self._tiers         = tiers
        self._use_fp16      = use_fp16
        self._use_cuda      = use_cuda
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
        Compile all buffered vectors into a Pithos index file.

        Writes two files:
        * ``{prefix_path}.bin``      — Pithos off-heap index (PLAN format)
        * ``{prefix_path}_meta.pkl`` — list[TileMetadata] (index == record ID)
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
        log.info("Compiling Pithos index with %d float32 vectors …", n)

        # 2. Sequential IDs — map directly to metadata list positions
        ids = np.arange(n, dtype=np.int64)

        # 3. Compile native Pithos index (binarization happens inside the lib)
        Path(bin_path).parent.mkdir(parents=True, exist_ok=True)
        log.info("Compiling Pithos PLAN index → %s …", bin_path)
        db = PithosMIDB(use_cuda=self._use_cuda)
        db.build_index(
            file_path     = bin_path,
            ids           = ids,
            vectors       = all_vectors,
            planet_id     = self._planet_id,
            planet_radius = self._planet_radius,
            tiers         = self._tiers,
            use_fp16      = self._use_fp16,
        )

        # 4. Write metadata (list[TileMetadata], index == record ID)
        with open(meta_path, "wb") as f:
            pickle.dump(self._metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        log.info("Saved %d records → %s  /  %s", n, bin_path, meta_path)
