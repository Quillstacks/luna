#!/usr/bin/env python3
"""
Global Lunar Sweep Orchestrator.
Targeted scan with hardcoded Product IDs and Auto-Download.
Dynamically creates a new Qdrant collection for each NAC and ingests tiles.
"""

import argparse
import logging
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from luna.screening.candidate_gen import DataIngestor
from luna.screening.wrappers import DinoV2Wrapper, QdrantWrapper
from luna.io import PDSIndex, fetch_nac

# --- HIER DEINE IDS DEFINIEREN ---
PIDS_TO_SCAN = [
    "M155607349RC"
]
# ---------------------------------

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Silence the HTTP loggers so they don't break the progress bar!
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    return logging.getLogger("LunarSweep")

def parse_args():
    parser = argparse.ArgumentParser(description="Sweep LROC NAC images for lunar pits.")
    parser.add_argument("--data-dir", type=Path, help="Local directory to scan.")
    parser.add_argument("--scratch-dir", type=Path, default=Path("data/_scratch"))
    parser.add_argument("--keep", action="store_true", help="Keep downloaded IMGs.")
    parser.add_argument("--adapter-file", type=str, 
                        default="luna/models/dinov2/adapter_model.safetensors")
    
    # Vector size for DinoV2 ViT-S is 384. Change this if you use a different model.
    parser.add_argument("--vector-size", type=int, default=384)
    return parser.parse_args()
    


def main():
    log = setup_logger()
    args = parse_args()
    
    # 1. Pipeline Initial
    try:
        log.info(f"Initializing DinoV2 with adapter: {args.adapter_file}")
        model = DinoV2Wrapper(model_name_or_path=args.adapter_file)
        idx = PDSIndex()
        
        # Initialize raw Qdrant client for collection management
        qdrant_admin = QdrantClient("http://localhost:6333")
    except Exception as e:
        log.error(f"Initialization failed: {e}")
        sys.exit(1)

    # 2. Target Acquisition
    targets = []
    if PIDS_TO_SCAN:
        targets = PIDS_TO_SCAN
        log.info(f"Using {len(targets)} hardcoded PIDs defined in script.")
    elif args.data_dir:
        targets = list(args.data_dir.glob("**/*.IMG"))
        log.info(f"Scanning {len(targets)} local images in {args.data_dir}")

    if not targets:
        log.error("No targets found! Define PIDS_TO_SCAN or provide --data-dir.")
        sys.exit(0)

    # 3. Sweep Loop
    for i, target in enumerate(targets, start=1):
        img_path = None
        is_downloaded = False
        
        try:
            if isinstance(target, str):
                log.info(f"[{i}/{len(targets)}] Fetching {target} from PDS...")
                img_path = fetch_nac(target, dest_dir=args.scratch_dir)
                is_downloaded = True
            else:
                img_path = target

            product_id = img_path.stem # e.g., M1193372314RC
            collection_name = f"lunar_global_{product_id}"

            log.info(f"[{i}/{len(targets)}] Creating Qdrant collection: {collection_name}")
            
            # --- CREATE DYNAMIC COLLECTION ---
            if qdrant_admin.collection_exists(collection_name):
                qdrant_admin.delete_collection(collection_name)
                
            qdrant_admin.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=args.vector_size, distance=Distance.COSINE),
            )

            # --- INITIALIZE WRAPPER AND INGESTOR FOR THIS IMAGE ---
            store = QdrantWrapper(collection_name=collection_name)
            ingestor = DataIngestor(model=model, store=store, use_multiprocessing=False)

            log.info(f"[{i}/{len(targets)}] Slicing and Ingesting {img_path.name}...")
            
            # Ingestion Pipeline
            total_ingested = ingestor.ingest_nac(
                path=img_path,
                tile_size=224,
                stride=112,
                batch_size=256
            )

            if total_ingested > 0:
                log.info(f"  -> SUCCESS: Ingested {total_ingested} tiles into '{collection_name}'!")
            else:
                log.info("  -> Results: No usable tiles found to ingest.")

        except Exception as e:
            log.error(f"  -> Failed to process {target}: {e}")
        
        finally:
            # Cleanup
            if is_downloaded and img_path and img_path.exists() and not args.keep:
                log.info(f"  -> Cleaning up {img_path.name}")
                img_path.unlink()
                xml_path = img_path.with_suffix(".xml")
                if xml_path.exists(): xml_path.unlink()

    log.info("Sweep complete.")

if __name__ == "__main__":
    main()