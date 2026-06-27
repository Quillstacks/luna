# luna/tests/test_smoke.py

def test_ci_pipeline_gateway():
    """Import smoke test — no model load, no GPU required."""
    from luna import LunaPipeline, CandidateHit
    from luna.config import DINO_DIM, TILE_SIZE
    from luna.screening.protocols import TileMetadata
    from luna.storage.pithos_store import PithosStore
    from luna.screening.pithos import PithosMIDB, MOON_ID

    assert DINO_DIM == 384
    assert TILE_SIZE == 256

    store = PithosStore()
    assert store is not None
    assert store._planet_id == MOON_ID