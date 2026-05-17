# tests/test_smoke.py

# luna/tests/test_smoke.py
def test_ci_pipeline_gateway():
    """Import-Smoke-Test — kein Model-Load, kein GPU nötig."""
    from luna import LunaPipeline, CandidateHit
    from luna.config import DINO_DIM, TILE_SIZE
    from luna.screening.protocols import TileMetadata
    from luna.storage.faiss_store import FaissLocalStore

    assert DINO_DIM == 384
    assert TILE_SIZE == 256

    store = FaissLocalStore(vector_dim=DINO_DIM)
    assert store is not None