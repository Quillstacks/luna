"""Central configuration for the Luna pipeline.

All magic numbers live here. Import from this module instead of
hardcoding values at call sites.
"""

from __future__ import annotations
import logging
from pathlib import Path

log = logging.getLogger("luna.config")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
SCRATCH_DIR  = DATA_DIR / "_scratch"
SPICE_DIR    = DATA_DIR / "spice" / "lro"
STATS_FILE   = DATA_DIR / "nac_stats.json"
INDEX_DIR    = SCRATCH_DIR / "indices"
WEIGHTS_DIR  = DATA_DIR / "weights"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

HF_REPO_ID   = "F1nnSBK/lunar-dinov3-lora"
DINO_DIM     = 384

# ---------------------------------------------------------------------------
# Tiling  (must match Cython DEF constants in transformer.pyx)
# ---------------------------------------------------------------------------

TILE_SIZE = 256    # px — DEF TILE_SIZE in transformer.pyx
STRIDE    = 192    # px — DEF STRIDE    in transformer.pyx

# ---------------------------------------------------------------------------
# LROC / PDS
# ---------------------------------------------------------------------------

LROC_VALID_MIN = -32752   # DN — pixels at or below this are NULL/saturation
PDS3_OFFSET    = 5064     # bytes — standard PDS3 label size (1 record × 5064 B)

# ---------------------------------------------------------------------------
# Cython ring buffer
# ---------------------------------------------------------------------------

RING_SIZE      = 1024     # must be power of two

def get_dynamic_max_batch_size() -> int:
    """Recognise the active device and return dynamic batch size.

    CUDA with >40 GB VRAM -> 1024, >20 GB -> 512, >10 GB -> 256, otherwise -> 64.
    MPS (Apple Silicon) -> 64.
    CPU fallback -> 16.
    """
    try:
        import torch
        if torch.cuda.is_available():
            try:
                dev = torch.cuda.current_device()
                vram_gb = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
                if vram_gb > 40:
                    batch_size = 1024
                elif vram_gb > 20:
                    batch_size = 512
                elif vram_gb > 10:
                    batch_size = 256
                else:
                    batch_size = 64
                log.info("Resolved dynamic batch size %d for CUDA system (VRAM: %.2f GB)", batch_size, vram_gb)
                return batch_size
            except Exception as e:
                log.warning("Failed to query CUDA VRAM, falling back to 256. Error: %s", e)
                return 256
        elif torch.backends.mps.is_available():
            log.info("Resolved dynamic batch size 64 for MPS system")
            return 64
        else:
            log.info("Resolved dynamic batch size 16 for CPU system")
            return 16
    except ImportError:
        return 64

MAX_BATCH_SIZE = get_dynamic_max_batch_size()

# ---------------------------------------------------------------------------
# FAISS / retrieval
# ---------------------------------------------------------------------------

SEARCH_K    = 1000    # candidates per anchor query
FINAL_TOP_K = 100     # hits returned after NMS
MIN_DIST_PX = 512.0   # NMS suppression radius (px)
ZOOM_SIZE   = 256     # crop size for result visualisation (px)