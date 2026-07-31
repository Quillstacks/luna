#!/usr/bin/env python3
"""Skript zum Erstellen von Pithos Indizes für Aristarchus NACs.

Optimiert für MacBook (MPS/CPU, kein CUDA, begrenzter RAM).
"""

import os
import sys
import time
import logging
from pathlib import Path

# Projekt-Root hinzufügen
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('create_aristarchus_indices.log')
    ]
)
log = logging.getLogger(__name__)

# Umgebungsvariablen für Performance
os.environ["OMP_NUM_THREADS"] = "2"  # Reduziert für Mac
os.environ["MKL_NUM_THREADS"] = "2"

def main():
    log.info("=" * 60)
    log.info("ARISTARCHUS INDIZES ERSTELLEN")
    log.info("=" * 60)
    
    from luna.config import SCRATCH_DIR, INDEX_DIR
    from luna.io.pds_fetch import fetch_nac
    from luna.pipeline import LunaPipeline
    
    # Device auswählen (MPS falls verfügbar, sonst CPU)
    import torch
    device = (
        "mps" if torch.backends.mps.is_available() else 
        "cpu"
    )
    log.info(f"Verwendetes Device: {device}")
    
    # NACs die wir indexieren wollen
    nac_ids = ["M109548636RC", "M109548636LC"]
    
    from luna.config import HF_REPO_ID
    try:
        pipeline = LunaPipeline.from_pretrained(
            HF_REPO_ID,
            device=device,
            config=None  # Standard Konfiguration
        )
        log.info("✅ Pipeline geladen")
    except Exception as e:
        log.error(f"Fehler beim Laden der Pipeline: {e}")
        return 1
    
    # Prozess jedes NAC
    for nac_id in nac_ids:
        nac_path = SCRATCH_DIR / f"{nac_id}.IMG"
        
        if not nac_path.exists():
            log.info(f"{nac_id}: NAC nicht gefunden unter {nac_path}")
            continue
        
        log.info(f"\n{'='*60}")
        log.info(f"Verarbeite {nac_id}...")
        log.info(f"{'='*60}")
        
        # Speicher vor der Verarbeitung bereinigen
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        start_time = time.time()
        
        try:
            # Schritt 1: NAC ingestieren (in Tiles aufteilen und embedden)
            log.info(f"  Schritt 1/2: Ingestiere {nac_id}...")
            store, metadata = pipeline._ingest(nac_path)
            
            ingest_time = time.time() - start_time
            log.info(f"  ✅ Ingestion abgeschlossen in {ingest_time:.1f}s")
            log.info(f"     Erstellte {len(metadata)} Tile-Embeddings")
            
            # Schritt 2: Index speichern
            log.info(f"  Schritt 2/2: Speichere Index...")
            index_path = pipeline._save_index(store, nac_path)
            
            save_time = time.time() - start_time
            log.info(f"  ✅ Index gespeichert in {save_time:.1f}s")
            log.info(f"     Index-Pfad: {index_path}")
            
            # Speicher nach jedem NAC bereinigen
            del store
            del metadata
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            log.error(f"  ❌ Fehler bei {nac_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    log.info("\n" + "=" * 60)
    log.info("FERTIG!")
    log.info("=" * 60)
    log.info("Erstellte Indizes:")
    for nac_id in nac_ids:
        index_files = list(INDEX_DIR.glob(f"pithos_{nac_id}*"))
        if index_files:
            log.info(f"  ✅ {nac_id}: {len(index_files)} Dateien")
        else:
            log.info(f"  ❌ {nac_id}: Keine Index-Dateien gefunden")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
