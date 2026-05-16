import logging
import os
from pathlib import Path
from time import perf_counter
import gc
import torch
import numpy as np

from luna.io.pds_fetch import fetch_nac
from luna.models.dinov3 import DINOEncoder
from luna.screening.candidate_gen import DataIngestor
from luna.screening.faiss_store import FaissLocalStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("luna.scripts.test_ingestor")

HF_REPO_ID = "F1nnSBK/lunar-dinov3-lora"
VECTOR_DIM = 384
TILE_SIZE  = 256
STRIDE     = 192
BATCH_SIZE = 256
MAX_BATCH  = 64

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_DIR  = PROJECT_ROOT / "data" / "_scratch"
STATS_FILE   = PROJECT_ROOT / "data" / "nac_stats.json"
INDEX_DIR    = SCRATCH_DIR / "indices"
QUERY_NPY    = SCRATCH_DIR / "pits" / "Aristarchus_6_M109548636LC.npy"

# Product IDs only — paths are resolved automatically
NAC_PRODUCT_IDS: list[str] = [
    "M1118880788RC",
    "M1210724899LC",
    # "M102285549RE",
]
# Or derive from a directory of already-downloaded files:
# NAC_PRODUCT_IDS = [p.stem for p in sorted(SCRATCH_DIR.glob("*.IMG"))]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_nac_paths(product_ids: list[str], dest_dir: Path) -> list[Path]:
    """Return local paths, downloading any missing NACs from PDS."""
    paths = []
    for pid in product_ids:
        local = dest_dir / f"{pid}.IMG"
        if local.exists():
            log.info("Found local: %s", local.name)
        else:
            log.info("Fetching from PDS: %s ...", pid)
            local = fetch_nac(pid, dest_dir=dest_dir)
        paths.append(local)
    return paths


def load_query_vector(encoder: DINOEncoder, query_path: Path) -> np.ndarray | None:
    import faiss
    if not query_path.exists():
        return None
    img   = np.load(query_path).astype(np.float32)
    valid = img[img > -32752]
    if valid.size == 0:
        return None
    f_min, f_max = valid.min(), valid.max()
    norm  = np.clip((img - f_min) / (f_max - f_min + 1e-6), 0, 1) if f_max > f_min else np.zeros_like(img)
    batch = (np.expand_dims(norm, 0) * 255).astype(np.uint8)
    vec   = np.ascontiguousarray(encoder.encode(batch), dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec


def run_sanity_check(index_path: str, q_vec: np.ndarray) -> None:
    import faiss
    index      = faiss.read_index(index_path)
    dists, ids = index.search(q_vec, 10)

    log.info("Top-10 hits for anchor query:")
    for rank, (d, idx) in enumerate(zip(dists[0], ids[0]), start=1):
        log.info("  Rank %02d | dist %.4f | meta-id %d", rank, d, idx)

    spread = float(dists[0][0] - dists[0][-1])
    if spread < 0.001:
        log.warning("Feature collapse detected — check Cython normalization or LoRA weights.")
    else:
        log.info("Healthy distance variance (spread=%.4f).", spread)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not NAC_PRODUCT_IDS:
        log.error("NAC_PRODUCT_IDS is empty.")
        return

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    try:
        nac_paths = resolve_nac_paths(NAC_PRODUCT_IDS, dest_dir=SCRATCH_DIR)
    except RuntimeError as e:
        log.error("Failed to resolve NAC files: %s", e)
        return

    encoder  = DINOEncoder(lora_dir=HF_REPO_ID, base_weights_path=HF_REPO_ID,
                           matryoshka_dim=VECTOR_DIM, device="mps")
    store    = FaissLocalStore(vector_dim=VECTOR_DIM)
    ingestor = DataIngestor(model=encoder, store=store,
                            stats_path=STATS_FILE, max_batch_size=MAX_BATCH)

    q_vec       = load_query_vector(encoder, QUERY_NPY)
    total_tiles = 0
    wall_start  = perf_counter()
    per_nac_stats: list[dict] = []

    for nac_path in nac_paths:
        log.info("── Ingesting %s ──", nac_path.name)

        store    = FaissLocalStore(vector_dim=VECTOR_DIM)
        ingestor = DataIngestor(model=encoder, store=store,
                                stats_path=STATS_FILE, max_batch_size=MAX_BATCH)

        t0    = perf_counter()
        tiles = ingestor.ingest_nac(path=nac_path, tile_size=TILE_SIZE,
                                    stride=STRIDE, batch_size=BATCH_SIZE)
        elapsed = perf_counter() - t0
        tps     = tiles / elapsed if elapsed > 0 else 0.0

        ingestor.screener.shutdown()
        del ingestor
        torch.mps.empty_cache()
        gc.collect()

        prefix = str(INDEX_DIR / f"faiss_{nac_path.stem}")
        store.save_to_disk(prefix)
        log.info("  Index saved → %s", prefix)
        del store
        gc.collect()

        per_nac_stats.append({"name": nac_path.name, "tiles": tiles,
                               "elapsed": elapsed, "tps": tps})
        total_tiles += tiles
        log.info("  %d tiles | %.2fs | %.0f tiles/s", tiles, elapsed, tps)

    total_elapsed = perf_counter() - wall_start

    log.info("═" * 60)
    log.info("INGESTION SUMMARY")
    log.info("  NACs processed : %d", len(nac_paths))
    for s in per_nac_stats:
        log.info("  %-30s %5d tiles  %6.2fs  %6.0f t/s",
                 s["name"], s["tiles"], s["elapsed"], s["tps"])
    log.info("  Total tiles    : %d", total_tiles)
    log.info("  Wall time      : %.2fs", total_elapsed)
    log.info("  Avg throughput : %.0f tiles/s", total_tiles / total_elapsed)
    log.info("═" * 60)

    last_index = str(INDEX_DIR / f"faiss_{nac_paths[-1].stem}.index")
    if q_vec is not None and os.path.exists(last_index):
        log.info("── Sanity check on %s ──", Path(last_index).name)
        run_sanity_check(last_index, q_vec)
    else:
        log.info("Sanity check skipped (index or query missing).")


if __name__ == "__main__":
    main()