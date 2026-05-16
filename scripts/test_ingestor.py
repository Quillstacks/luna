import logging
import os
from pathlib import Path
from time import perf_counter

import numpy as np

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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_DIR  = PROJECT_ROOT / "data" / "_scratch"
STATS_FILE   = PROJECT_ROOT / "data" / "nac_stats.json"

TEST_NAC_IMG = SCRATCH_DIR / "M157906985RC.IMG"
QUERY_NPY    = SCRATCH_DIR / "pits" / "Aristarchus_6_M109548636LC.npy"


def main() -> None:
    if not TEST_NAC_IMG.exists():
        log.error(f"Please place a valid NAC .IMG file at {TEST_NAC_IMG}")
        return

    encoder = DINOEncoder(
        lora_dir=HF_REPO_ID,
        base_weights_path=HF_REPO_ID,
        matryoshka_dim=384,
        device="mps"
    )

    store = FaissLocalStore(vector_dim=384)

    ingestor = DataIngestor(
        model=encoder,
        store=store,
        stats_path=STATS_FILE,
        max_batch_size=16
    )

    log.info(f"Igniting Cython Disruptor Engine for {TEST_NAC_IMG.name}...")
    
    start_time = perf_counter()
    
    total_tiles = ingestor.ingest_nac(
        path=TEST_NAC_IMG,
        tile_size=256,
        stride=192,
        batch_size=128
    )
    
    elapsed = perf_counter() - start_time
    tiles_per_sec = total_tiles / elapsed

    log.info("Ingestion complete!")
    log.info(f"Processed {total_tiles} tiles in {elapsed:.2f}s ({tiles_per_sec:.0f} tiles/s)")

    prefix = str(SCRATCH_DIR / f"indices/faiss_{TEST_NAC_IMG.stem}")
    store.save_to_disk(prefix)
    log.info(f"FAISS index saved to {prefix}")

    index_file = f"{prefix}.index"
    if os.path.exists(index_file) and QUERY_NPY.exists():
        log.info("--- RUNNING FAISS SANITY CHECK ---")
        import faiss
        
        index = faiss.read_index(index_file)
        
        img_array = np.load(QUERY_NPY).astype(np.float32)
        valid = img_array[img_array > -32752]
        if valid.size > 0:
            f_min, f_max = valid.min(), valid.max()
            q_norm = np.clip((img_array - f_min) / (f_max - f_min + 1e-6), 0, 1) if f_max > f_min else np.zeros_like(img_array)
        else:
            q_norm = np.zeros_like(img_array)

        batch = q_norm.astype(np.float32)
        if batch.ndim == 2:
            batch = np.expand_dims(batch, axis=0)
        batch_uint8 = (batch * 255).astype(np.uint8)
        
        q_vec = encoder.encode(batch_uint8)
        
        q_vec = np.ascontiguousarray(q_vec, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q_vec)
        
        dists, ids = index.search(q_vec, 10)
        
        log.info("Top 10 Hits for Anchor:")
        for i, (d, idx) in enumerate(zip(dists[0], ids[0])):
            log.info(f"  Rank {i+1:02d} | Dist: {d:.4f} | Meta-ID: {idx}")

        spread = dists[0][0] - dists[0][-1]
        log.info(f"Distance Spread (Rank 1 to 10): {spread:.4f}")
        
        if spread < 0.001:
            log.warning("WARNING: Feature Collapse detected! All vectors look identical.")
            log.warning("Check your Cython local normalization or LoRA training weights.")
        else:
            log.info("SUCCESS: Healthy distance variance detected! Model is distinguishing features.")
    else:
        log.info(f"Sanity check skipped (Index {index_file} or QUERY_NPY missing).")


if __name__ == "__main__":
    main()