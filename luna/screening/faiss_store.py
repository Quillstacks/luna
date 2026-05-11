import logging
import pickle
import os

import numpy as np

from luna.screening.protocols import VectorStore, TileMetadata

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

log = logging.getLogger("luna.screening.faiss_store")


class FaissLocalStore(VectorStore):
    def __init__(self, vector_dim: int) -> None:
        self.dim = vector_dim
        self._vectors: list[np.ndarray] = []
        self._metadata: list[TileMetadata] = []

    def upsert(self, vectors: np.ndarray, metadata: list[TileMetadata]) -> None:
        if len(vectors) == 0:
            log.debug("upsert called with empty vector batch — skipping")
            return

        self._vectors.append(vectors.astype(np.float32))
        self._metadata.extend(metadata)

    def save_to_disk(self, prefix_path: str) -> None:
        """Builds the FAISS index from all buffered vectors and writes it to disk."""
        import faiss  # lazy import — optional heavy dependency

        if not self._vectors:
            log.warning("No vectors buffered — nothing to save")
            return

        log.info("Building FAISS index from %d batches...", len(self._vectors))
        all_vectors = np.ascontiguousarray(np.concatenate(self._vectors, axis=0))

        faiss.normalize_L2(all_vectors)
        index = faiss.IndexFlatIP(self.dim)
        index.add(all_vectors)

        index_path = f"{prefix_path}.index"
        meta_path = f"{prefix_path}_meta.pkl"

        faiss.write_index(index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self._metadata, f)

        log.info("Saved %d vectors → %s / %s", len(self._metadata), index_path, meta_path)