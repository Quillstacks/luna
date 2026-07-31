#!/usr/bin/env python3
"""Script to build Pithos indices for Aristarchus NAC products.

Optimized for local/laptop execution (MPS/CPU, low VRAM budget).
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('create_aristarchus_indices.log')
    ]
)
log = logging.getLogger(__name__)

# Performance tuning environment variables
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

def main():
    log.info("=" * 60)
    log.info("BUILDING ARISTARCHUS PITHOS INDICES")
    log.info("=" * 60)
    
    from luna.config import SCRATCH_DIR, INDEX_DIR
    from luna.io.pds_fetch import fetch_nac
    from luna.pipeline import LunaPipeline
    
    import torch
    device = (
        "mps" if torch.backends.mps.is_available() else 
        "cpu"
    )
    log.info(f"Target device: {device}")
    
    nac_ids = ["M109548636RC", "M109548636LC"]
    
    from luna.config import HF_REPO_ID
    try:
        pipeline = LunaPipeline.from_pretrained(
            HF_REPO_ID,
            device=device,
            config=None
        )
        log.info("Pipeline loaded successfully")
    except Exception as e:
        log.error(f"Error loading pipeline: {e}")
        return 1
    
    for nac_id in nac_ids:
        nac_path = SCRATCH_DIR / f"{nac_id}.IMG"
        
        if not nac_path.exists():
            log.info(f"{nac_id}: NAC raster not found at {nac_path}")
            continue
        
        log.info(f"\n{'='*60}")
        log.info(f"Processing {nac_id}...")
        log.info(f"{'='*60}")
        
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        start_time = time.time()
        
        try:
            log.info(f"  Step 1/2: Ingesting {nac_id}...")
            store, metadata = pipeline._ingest(nac_path)
            
            ingest_time = time.time() - start_time
            log.info(f"  Ingestion completed in {ingest_time:.1f}s")
            log.info(f"  Generated {len(metadata)} tile embeddings")
            
            log.info(f"  Step 2/2: Saving Pithos index...")
            index_path = pipeline._save_index(store, nac_path)
            
            save_time = time.time() - start_time
            log.info(f"  Index saved in {save_time:.1f}s")
            log.info(f"  Index path: {index_path}")
            
            del store
            del metadata
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            log.error(f"  Error processing {nac_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    log.info("\n" + "=" * 60)
    log.info("COMPLETE!")
    log.info("=" * 60)
    log.info("Generated indices:")
    for nac_id in nac_ids:
        index_files = list(INDEX_DIR.glob(f"pithos_{nac_id}*"))
        if index_files:
            log.info(f"  {nac_id}: {len(index_files)} files")
        else:
            log.info(f"  {nac_id}: No index files found")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
