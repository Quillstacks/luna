# scripts/test_ingestor.py
import gc
import logging
import os
import torch
from pathlib import Path
from time import perf_counter

import numpy as np

from luna.io.pds_fetch import fetch_nac
from luna.models.dinov3 import DINOEncoder
from luna.screening import DataIngestor
from luna.storage import FaissLocalStore
from luna.config import SCRATCH_DIR, INDEX_DIR, HF_REPO_ID, DINO_DIM, TILE_SIZE, STRIDE, MAX_BATCH_SIZE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("luna.scripts.test_ingestor")


QUERY_NPY    = SCRATCH_DIR / "pits" / "Aristarchus_6_M109548636LC.npy"

NAC_PRODUCT_IDS: list[str] = [
    "M1118880788RC",
    "M1210724899LC",
]


def resolve_nac_paths(product_ids: list[str], dest_dir: Path) -> list[Path]:
    paths = []
    for pid in product_ids:
        local = dest_dir / f"{pid}.IMG"
        if not local.exists():
            log.info("Fetching from PDS: %s ...", pid)
            local = fetch_nac(pid, dest_dir=dest_dir)
        else:
            log.info("Found local: %s", local.name)
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
    log.info("Top-10 hits:")
    for rank, (d, idx) in enumerate(zip(dists[0], ids[0]), start=1):
        log.info("  Rank %02d | dist %.4f | id %d", rank, d, idx)
    spread = float(dists[0][0] - dists[0][-1])
    if spread < 0.001:
        log.warning("Feature collapse detected.")
    else:
        log.info("Healthy variance (spread=%.4f).", spread)


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    nac_paths = resolve_nac_paths(NAC_PRODUCT_IDS, dest_dir=SCRATCH_DIR)
    encoder   = DINOEncoder(lora_dir=HF_REPO_ID, base_weights_path=HF_REPO_ID,
                            matryoshka_dim=DINO_DIM)
    q_vec     = load_query_vector(encoder, QUERY_NPY)

    total_tiles = 0
    wall_start  = perf_counter()
    stats: list[dict] = []

    for nac_path in nac_paths:
        log.info("── Ingesting %s ──", nac_path.name)

        # Fresh store + ingestor per NAC — prevents cumulative memory build-up
        store    = FaissLocalStore(vector_dim=DINO_DIM)
        ingestor = DataIngestor(model=encoder, store=store, max_batch_size=MAX_BATCH_SIZE)

        t0    = perf_counter()
        tiles = ingestor.ingest_nac(path=nac_path, tile_size=TILE_SIZE, stride=STRIDE)
        elapsed = perf_counter() - t0
        tps     = tiles / elapsed if elapsed > 0 else 0.0

        # Shut down Cython thread before FAISS index build
        ingestor.screener.shutdown()
        del ingestor
        if encoder.device.type == "mps":
            torch.mps.empty_cache()
        elif encoder.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


        prefix = str(INDEX_DIR / f"faiss_{nac_path.stem}")
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        store.save_to_disk(prefix)
        log.info("  Index saved → %s", prefix)
        del store
        gc.collect()

        stats.append({"name": nac_path.name, "tiles": tiles, "elapsed": elapsed, "tps": tps})
        total_tiles += tiles
        log.info("  %d tiles | %.2fs | %.0f tiles/s", tiles, elapsed, tps)

    total_elapsed = perf_counter() - wall_start
    log.info("═" * 60)
    log.info("SUMMARY")
    for s in stats:
        log.info("  %-30s %5d tiles  %6.2fs  %6.0f t/s",
                 s["name"], s["tiles"], s["elapsed"], s["tps"])
    log.info("  Total: %d tiles | %.2fs | %.0f t/s avg",
             total_tiles, total_elapsed, total_tiles / total_elapsed)
    log.info("═" * 60)

    last_index = str(INDEX_DIR / f"faiss_{nac_paths[-1].stem}.index")
    if q_vec is not None and os.path.exists(last_index):
        run_sanity_check(last_index, q_vec)


if __name__ == "__main__":
    main()