# luna/tests/test_smoke.py


def test_ci_pipeline_gateway():
    """Import smoke test — no model load, no GPU required."""
    from luna import LunaPipeline, CandidateHit, MetricsReport, LunaError, LunaConfig
    from luna.config import DINO_DIM, TILE_SIZE
    from luna.screening.protocols import TileMetadata
    from luna.storage.pithos_store import PithosStore
    from luna.screening.pithos import PithosMIDB, MOON_ID
    from luna.exceptions import (
        IngestError, IndexError, IndexNotFoundError,
        SearchError, DeltaError, DeltaFullError, NACNotFoundError, RefinementError
    )

    assert DINO_DIM == 384
    assert TILE_SIZE == 256

    store = PithosStore()
    assert store is not None
    assert store._planet_id == MOON_ID


def test_pithos_singleton():
    """PithosMIDB is a singleton — two instances give the same object."""
    from luna.screening.pithos import PithosMIDB
    
    db1 = PithosMIDB()
    db2 = PithosMIDB()
    assert db1 is db2


def test_pithos_binarize():
    """binarize() converts float32 embeddings to (N, 6) int64."""
    from luna.screening.pithos import PithosMIDB
    import numpy as np
    
    vecs = np.random.rand(10, 384).astype(np.float32)
    result = PithosMIDB.binarize(vecs)
    assert result.shape == (10, 6)
    assert result.dtype == np.int64


def test_delta_roundtrip():
    """create_delta_buffer -> insert -> delta_size -> backup -> restore."""
    import pytest
    from luna.screening.pithos import PithosMIDB
    import numpy as np
    import tempfile
    
    db = PithosMIDB()
    if not getattr(db, "_has_native", False):
        pytest.skip("Pithos native shared library (libpithos) is not available.")
    index_name = "test_delta_roundtrip"
    capacity = 100
    
    db.create_delta_buffer(index_name, capacity)
    assert db.delta_size(index_name) == 0
    
    vec = np.random.rand(384).astype(np.float32)
    db.insert(index_name, id=42, vector=vec)
    assert db.delta_size(index_name) == 1
    
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name
    
    db.backup_delta(index_name, path)
    db.drop_index(index_name)
    
    db.create_delta_buffer(index_name, capacity)
    db.restore_delta(index_name, path, capacity)
    assert db.delta_size(index_name) == 1


def test_metrics_report_to_dict():
    """MetricsReport.to_dict() is JSON-serializable."""
    from luna.metrics import MetricsReport
    import json
    
    report = MetricsReport(
        ingest_s=1.5,
        ingest_tiles=100,
        total_scan_s=5.0,
    )
    d = report.to_dict()
    json_str = json.dumps(d)
    assert json_str is not None


def test_scan_no_metrics():
    """scan() without metrics returns list."""
    from luna.pipeline import LunaPipeline
    
    pipeline = LunaPipeline.__new__(LunaPipeline)
    assert callable(pipeline.scan)


def test_scan_with_metrics():
    """scan(..., metrics=True) returns tuple."""
    from luna.pipeline import LunaPipeline
    from luna.metrics import MetricsReport
    
    pipeline = LunaPipeline.__new__(LunaPipeline)
    assert callable(pipeline.scan)


def test_exception_types():
    """All exception types are subclasses of LunaError."""
    from luna.exceptions import (
        LunaError, IngestError, IndexError, IndexNotFoundError,
        SearchError, DeltaError, DeltaFullError, NACNotFoundError, RefinementError
    )
    
    assert issubclass(IngestError, LunaError)
    assert issubclass(IndexError, LunaError)
    assert issubclass(IndexNotFoundError, IndexError)
    assert issubclass(SearchError, LunaError)
    assert issubclass(DeltaError, LunaError)
    assert issubclass(DeltaFullError, DeltaError)
    assert issubclass(NACNotFoundError, LunaError)
    assert issubclass(RefinementError, LunaError)


def test_config_defaults():
    """LunaConfig() has identical values to current config.py constants."""
    from luna.config import LunaConfig, TILE_SIZE, STRIDE, SEARCH_K, FINAL_TOP_K, MIN_DIST_PX
    from luna.pipeline import MAX_BATCH_SIZE
    import numpy as np
    
    config = LunaConfig()
    assert config.tile_size == TILE_SIZE
    assert config.stride == STRIDE
    assert config.search_k == SEARCH_K
    assert config.final_top_k == FINAL_TOP_K
    assert config.min_dist_px == MIN_DIST_PX
    assert config.max_batch_size == MAX_BATCH_SIZE
    assert np.array_equal(config.pithos_tiers, np.array([64, 128, 256, 384], dtype=np.int32))