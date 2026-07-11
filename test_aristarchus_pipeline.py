#!/usr/bin/env python3
"""Komplettes Testskript für Aristarchus NACs: Stage-1 + Stage-2 mit Analyse.

Dieses Skript:
1. Führt Stage-1 Suche auf M109548636RC und M109548636LC durch
2. Findet Pit-Kandidaten
3. Führt Stage-2 auf den besten Kandidaten aus
4. Analysiert die Ergebnisse im Vergleich zu LPA-Katalog
"""

import os
import sys
import time
import logging
import csv
from pathlib import Path
from collections import defaultdict

# Projekt-Root hinzufügen
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_aristarchus_pipeline.log')
    ]
)
log = logging.getLogger(__name__)

# Umgebungsvariablen für Performance auf Mac
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

# Import nach Logging, um Import-Fehler zu loggen
import numpy as np
import torch

def load_lpa_catalog(catalog_path="Catalogs/lpa.csv"):
    """Lade LPA-Katalog und filtere Aristarchus-Pits."""
    catalog_pits = []
    try:
        with open(catalog_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["host"].strip() == "Aristarchus":
                    raw_lon = float(row["longitude"])
                    if raw_lon > 180.0:
                        raw_lon -= 360.0
                    catalog_pits.append({
                        "name": row["name"].strip(),
                        "lat": float(row["latitude"]),
                        "lon": raw_lon,
                        "depth_m": float(row["depth_m"]) if row["depth_m"] else 0.0,
                        "funnel_max_m": float(row["funnel_max_m"]) if row["funnel_max_m"] else 0.0,
                        "funnel_min_m": float(row["funnel_min_m"]) if row["funnel_min_m"] else 0.0,
                    })
        log.info(f"Geladen: {len(catalog_pits)} Aristarchus-Pits aus LPA-Katalog")
        for pit in catalog_pits:
            log.info(f"  - {pit['name']}: lat={pit['lat']:.4f}, lon={pit['lon']:.4f}, depth={pit['depth_m']}m")
    except Exception as e:
        log.error(f"Fehler beim Laden des LPA-Katalogs: {e}")
    return catalog_pits

def compute_lunar_distance(lat1, lon1, lat2, lon2):
    """Berechne Distanz auf dem Mond in Metern."""
    LUNAR_METERS_PER_DEGREE = 30323.35
    dlat = np.radians(lat1 - lat2)
    delta_lon = (lon1 % 360) - (lon2 % 360)
    if delta_lon > 180: delta_lon -= 360
    elif delta_lon < -180: delta_lon += 360
    dlon = np.radians(delta_lon)
    mean_lat = np.radians((lat1 + lat2) / 2.0)
    dy = LUNAR_METERS_PER_DEGREE * 57.2957795 * dlat
    dx = LUNAR_METERS_PER_DEGREE * 57.2957795 * dlon * np.cos(mean_lat)
    return np.sqrt(dx**2 + dy**2)

def run_stage1_search(pipeline, nac_ids, query_path, search_k=200):
    """Führe Stage-1 Suche auf den NACs durch."""
    from luna.config import INDEX_DIR, SCRATCH_DIR
    from luna.pipeline import apply_lunar_spatial_nms
    
    lpa_catalog = load_lpa_catalog()
    
    # Lade Query-Daten
    if not query_path.exists():
        log.error(f"Query-Datei nicht gefunden: {query_path}")
        return [], lpa_catalog
    
    log.info(f"Geladen: Query-Vektoren aus {query_path}")
    
    # Führe Suche mit der scan-Methode durch
    log.info(f"\n{'='*60}")
    log.info(f"Stage-1 Suche auf {nac_ids}...")
    log.info(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Suche durchführen mit scan-Methode
        hits = pipeline.scan(
            product_ids=nac_ids,
            query_dir=str(query_path.parent),  # Verwende das Verzeichnis
            search_k=search_k,
            top_k=search_k,
            force_reingest=False
        )
        
        search_time = time.time() - start_time
        log.info(f"  ✅ Suche abgeschlossen in {search_time:.1f}s")
        log.info(f"     Gesamt Kandidaten: {len(hits)}")
        
        # Analysiere Kandidaten
        if hits:
            scores = [h.score for h in hits]
            log.info(f"     Score-Bereich: {min(scores):.2f} - {max(scores):.2f}")
            log.info(f"     Durchschnittlicher Score: {np.mean(scores):.2f}")
        
    except Exception as e:
        log.error(f"  ❌ Fehler bei Suche: {e}")
        import traceback
        traceback.print_exc()
        return [], lpa_catalog
    
    log.info(f"\n{'='*60}")
    log.info(f"STAGE-1 ERGEBNIS")
    log.info(f"{'='*60}")
    log.info(f"Gesamt Kandidaten: {len(hits)}")
    
    return hits, lpa_catalog

def run_stage2_test(refiner, hits, lpa_catalog, out_dir, max_candidates=50):
    """Führe Stage-2 auf den besten Kandidaten durch."""
    from luna.config import SCRATCH_DIR
    
    # Sortiere nach Score (beste zuerst)
    hits_sorted = sorted(hits, key=lambda h: h.score, reverse=True)
    
    # Nimm nur die besten Kandidaten
    if len(hits_sorted) > max_candidates:
        log.info(f"Teste nur Top {max_candidates} von {len(hits_sorted)} Kandidaten")
        hits_to_test = hits_sorted[:max_candidates]
    else:
        hits_to_test = hits_sorted
    
    log.info(f"\n{'='*60}")
    log.info(f"STAGE-2 TEST (Top {len(hits_to_test)} Kandidaten)")
    log.info(f"{'='*60}")
    
    # Erstelle Output-Verzeichnis
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Führe Stage-2 aus
    start_time = time.time()
    refined_hits = refiner.refine(
        hits_to_test,
        out_dir=str(out_dir),
        score_thr=0.5,
        save_debug_plots=True
    )
    stage2_time = time.time() - start_time
    
    log.info(f"\nStage-2 abgeschlossen in {stage2_time:.1f}s")
    log.info(f"  Bestätigte Pits: {len(refined_hits)}")
    log.info(f"  Abgelehnt: {len(hits_to_test) - len(refined_hits)}")
    
    # Analysiere die Ergebnisse
    if refined_hits and lpa_catalog:
        log.info(f"\n{'='*60}")
        log.info(f"ANALYSE: Vergleich mit LPA-Katalog")
        log.info(f"{'='*60}")
        
        for refined in refined_hits:
            best_dist = float('inf')
            best_pit = None
            
            for pit in lpa_catalog:
                dist = compute_lunar_distance(
                    refined.essa_lat, refined.essa_lon,
                    pit["lat"], pit["lon"]
                )
                if dist < best_dist:
                    best_dist = dist
                    best_pit = pit
            
            # Auch Distanz zu Stage-1 Koordinaten
            stage1_dist = compute_lunar_distance(
                refined.lat, refined.lon,
                refined.essa_lat, refined.essa_lon
            )
            
            if best_pit:
                log.info(f"\n  ✅ {refined.product_id} Rank {refined.rank}:")
                log.info(f"     Stage-1: lat={refined.lat:.4f}, lon={refined.lon:.4f}")
                log.info(f"     Stage-2: lat={refined.essa_lat:.4f}, lon={refined.essa_lon:.4f}")
                log.info(f"     Stage-1 vs Stage-2 Distanz: {stage1_dist:.1f}m")
                log.info(f"     Nächster LPA-Pit: {best_pit['name']} (Distanz: {best_dist:.1f}m)")
                log.info(f"     Score: {refined.dino_score:.3f}, ESSA Score: {refined.essa_score:.3f}")
            else:
                log.info(f"\n  ⚠️  {refined.product_id} Rank {refined.rank}:")
                log.info(f"     Stage-1: lat={refined.lat:.4f}, lon={refined.lon:.4f}")
                log.info(f"     Stage-2: lat={refined.essa_lat:.4f}, lon={refined.essa_lon:.4f}")
                log.info(f"     Kein LPA-Pit in der Nähe gefunden")
                log.info(f"     Score: {refined.dino_score:.3f}, ESSA Score: {refined.essa_score:.3f}")
    
    return refined_hits

def main():
    log.info("=" * 60)
    log.info("ARISTARCHUS PIPELINE TEST: STAGE-1 + STAGE-2")
    log.info("=" * 60)
    
    from luna.config import SCRATCH_DIR
    from luna.pipeline import LunaPipeline
    from luna.models.stage2_decoder import Stage2Refiner, build_stage2_decoder
    from luna.models.dinov3 import DINOEncoder
    
    # Device auswählen
    device = (
        "mps" if torch.backends.mps.is_available() else 
        "cpu"
    )
    log.info(f"Verwendetes Device: {device}")
    
    # NACs die wir testen
    nac_ids = ["M109548636RC", "M109548636LC"]
    
    # Query-Datei (Pit-Patches)
    query_path = SCRATCH_DIR / "pits" / "dino_reference.npy"
    
    # Output-Verzeichnis
    out_dir = SCRATCH_DIR / "stage2_test_results"
    
    # ========================================================================
    # STAGE-1: Suche
    # ========================================================================
    log.info("\n" + "=" * 60)
    log.info("PHASE 1: STAGE-1 SUCHE")
    log.info("=" * 60)
    
    try:
        # Lade Pipeline
        log.info("Lade LunaPipeline...")
        pipeline = LunaPipeline.from_pretrained(
            "F1nnSBK/lunar-dinov3-lora",
            device=device,
            config=None
        )
        log.info("✅ Pipeline geladen")
        
        # Führe Stage-1 Suche durch
        hits, lpa_catalog = run_stage1_search(pipeline, nac_ids, query_path)
        
        if not hits:
            log.error("Keine Kandidaten gefunden! Beende.")
            return 1
        
    except Exception as e:
        log.error(f"Fehler in Stage-1: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================================================
    # STAGE-2: Verfeinerung
    # ========================================================================
    log.info("\n" + "=" * 60)
    log.info("PHASE 2: STAGE-2 VERFEINERUNG")
    log.info("=" * 60)
    
    try:
        # Speicher bereinigen
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # Lade DINO Encoder für Stage-2
        log.info("Lade DINO Encoder für Stage-2...")
        encoder = DINOEncoder(
            lora_dir="F1nnSBK/lunar-dinov3-lora",
            base_weights_path="F1nnSBK/lunar-dinov3-lora",
            device=device
        )
        log.info("✅ DINO Encoder geladen")
        
        # Lade Stage-2 Decoder mit den neuen Fixes
        log.info("Lade Stage-2 Decoder...")
        decoder = build_stage2_decoder(device=device)
        log.info("✅ Stage-2 Decoder geladen")
        
        # Erstelle Refiner
        refiner = Stage2Refiner(decoder=decoder, dino_encoder=encoder)
        log.info("✅ Refiner erstellt")
        
        # Führe Stage-2 Test durch
        refined_hits = run_stage2_test(refiner, hits, lpa_catalog, out_dir, max_candidates=50)
        
    except Exception as e:
        log.error(f"Fehler in Stage-2: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================================================
    # ZUSAMMENFASSUNG
    # ========================================================================
    log.info("\n" + "=" * 60)
    log.info("ZUSAMMENFASSUNG")
    log.info("=" * 60)
    log.info(f"Stage-1 Kandidaten: {len(hits)}")
    log.info(f"Stage-2 bestätigte Pits: {len(refined_hits)}")
    log.info(f"LPA-Katalog Pits: {len(lpa_catalog)}")
    log.info(f"Output-Verzeichnis: {out_dir}")
    
    # Berechne Match-Rate
    if refined_hits and lpa_catalog:
        matched = 0
        for refined in refined_hits:
            for pit in lpa_catalog:
                dist = compute_lunar_distance(
                    refined.essa_lat, refined.essa_lon,
                    pit["lat"], pit["lon"]
                )
                if dist < 300.0:  # Innerhalb von 300m gilt als Match
                    matched += 1
                    break
        
        match_rate = matched / len(lpa_catalog) * 100
        log.info(f"Match-Rate mit LPA-Katalog: {matched}/{len(lpa_catalog)} ({match_rate:.1f}%)")
    
    log.info("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
