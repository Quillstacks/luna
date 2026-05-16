"""Central configuration for the Luna pipeline.

All magic numbers live here. Import from this module instead of
hardcoding values at call sites.
"""

from __future__ import annotations
from pathlib import Path

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
DINO_INPUT   = 224       # ViT input resolution (px)

# DINOv3 ViT-S/16: 1 CLS token + 4 register tokens
N_SPECIAL_TOKENS = 5

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

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
MAX_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# FAISS / retrieval
# ---------------------------------------------------------------------------

SEARCH_K    = 1000    # candidates per anchor query
FINAL_TOP_K = 100     # hits returned after NMS
MIN_DIST_PX = 512.0   # NMS suppression radius (px)
ZOOM_SIZE   = 256     # crop size for result visualisation (px)