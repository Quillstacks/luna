#!/usr/bin/env python3
"""Complete test script for Aristarchus NACs: Stage-1 + Stage-2 with analysis.

This script:
1. Executes Stage-1 candidate search across M109548636RC and M109548636LC
2. Surface pit candidate hits
3. Executes Stage-2 refinement on top candidates
4. Analyzes results against cataloged LPA reference pits
"""

import os
import sys
import time
import logging
import csv
from pathlib import Path
from collections import defaultdict

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_aristarchus_pipeline.log')
    ]
)
log = logging.getLogger(__name__)

# Performance tuning environment variables
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import torch

def load_lpa_catalog(catalog_path="Catalogs/lpa.csv"):
    """Load LPA catalog and filter Aristarchus region pits."""
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
        log.info(f"Loaded {len(catalog_pits)} Aristarchus pits from LPA catalog")
        for pit in catalog_pits:
            log.info(f"  - {pit['name']}: lat={pit['lat']:.4f}, lon={pit['lon']:.4f}, depth={pit['depth_m']}m")
    except Exception as e:
        log.error(f"Error loading LPA catalog: {e}")
    return catalog_pits

def compute_lunar_distance(lat1, lon1, lat2, lon2):
    """Compute surface distance on the Moon in meters."""
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
    """Run Stage-1 search across specified NAC products."""
    from luna.config import INDEX_DIR, SCRATCH_DIR
    from luna.pipeline import apply_lunar_spatial_nms
    
    lpa_catalog = load_lpa_catalog()
    
    if not query_path.exists():
        log.error(f"Query vector file not found at {query_path}")
        return [], lpa_catalog
    
    log.info(f"Loaded query vectors from {query_path}")
    
    log.info(f"\n{'='*60}")
    log.info(f"Running Stage-1 search on {nac_ids}...")
    log.info(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        hits = pipeline.scan(
            product_ids=nac_ids,
            query_dir=str(query_path.parent),
            search_k=search_k,
            top_k=search_k,
            force_reingest=False
        )
        
        search_time = time.time() - start_time
        log.info(f"  Search completed in {search_time:.1f}s")
        log.info(f"  Total candidate hits: {len(hits)}")
        
        if hits:
            scores = [h.score for h in hits]
            log.info(f"  Score range: {min(scores):.2f} - {max(scores):.2f}")
            log.info(f"  Mean score: {np.mean(scores):.2f}")
        
    except Exception as e:
        log.error(f"  Search error: {e}")
        import traceback
        traceback.print_exc()
        return [], lpa_catalog
    
    log.info(f"\n{'='*60}")
    log.info(f"STAGE-1 RESULTS SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"Total candidate hits: {len(hits)}")
    
    return hits, lpa_catalog

def run_stage2_test(refiner, hits, lpa_catalog, out_dir, max_candidates=50):
    """Run Stage-2 refinement on top candidate hits."""
    from luna.config import SCRATCH_DIR
    
    hits_sorted = sorted(hits, key=lambda h: h.score, reverse=True)
    
    if len(hits_sorted) > max_candidates:
        log.info(f"Testing top {max_candidates} of {len(hits_sorted)} candidates")
        hits_to_test = hits_sorted[:max_candidates]
    else:
        hits_to_test = hits_sorted
    
    log.info(f"\n{'='*60}")
    log.info(f"STAGE-2 REFINEMENT TEST (Top {len(hits_to_test)} candidates)")
    log.info(f"{'='*60}")
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    refined_hits = refiner.refine(
        hits_to_test,
        out_dir=str(out_dir),
        score_thr=0.5,
        save_debug_plots=True
    )
    stage2_time = time.time() - start_time
    
    log.info(f"\nStage-2 refinement completed in {stage2_time:.1f}s")
    log.info(f"  Confirmed pits: {len(refined_hits)}")
    log.info(f"  Rejected: {len(hits_to_test) - len(refined_hits)}")
    
    if refined_hits and lpa_catalog:
        log.info(f"\n{'='*60}")
        log.info(f"ANALYSIS: Comparison against LPA Reference Catalog")
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
            
            stage1_dist = compute_lunar_distance(
                refined.lat, refined.lon,
                refined.essa_lat, refined.essa_lon
            )
            
            if best_pit:
                log.info(f"\n  {refined.product_id} Rank {refined.rank}:")
                log.info(f"     Stage-1: lat={refined.lat:.4f}, lon={refined.lon:.4f}")
                log.info(f"     Stage-2: lat={refined.essa_lat:.4f}, lon={refined.essa_lon:.4f}")
                log.info(f"     Stage-1 vs Stage-2 offset: {stage1_dist:.1f}m")
                log.info(f"     Closest LPA pit: {best_pit['name']} (Distance: {best_dist:.1f}m)")
                log.info(f"     Score: {refined.dino_score:.3f}, ESSA Score: {refined.essa_score:.3f}")
            else:
                log.info(f"\n  {refined.product_id} Rank {refined.rank}:")
                log.info(f"     Stage-1: lat={refined.lat:.4f}, lon={refined.lon:.4f}")
                log.info(f"     Stage-2: lat={refined.essa_lat:.4f}, lon={refined.essa_lon:.4f}")
                log.info(f"     No nearby LPA pit found within range")
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
    
    device = (
        "mps" if torch.backends.mps.is_available() else 
        "cpu"
    )
    log.info(f"Target execution device: {device}")
    
    nac_ids = ["M109548636RC", "M109548636LC"]
    query_path = SCRATCH_DIR / "pits" / "dino_reference.npy"
    out_dir = SCRATCH_DIR / "stage2_test_results"
    
    # ========================================================================
    # STAGE-1: SEARCH
    # ========================================================================
    log.info("\n" + "=" * 60)
    log.info("PHASE 1: STAGE-1 SEARCH")
    log.info("=" * 60)
    
    try:
        from luna.config import HF_REPO_ID
        log.info("Loading LunaPipeline...")
        pipeline = LunaPipeline.from_pretrained(
            HF_REPO_ID,
            device=device,
            config=None
        )
        log.info("Pipeline loaded successfully")
        
        hits, lpa_catalog = run_stage1_search(pipeline, nac_ids, query_path)
        
        if not hits:
            log.error("No candidate hits found! Exiting.")
            return 1
        
    except Exception as e:
        log.error(f"Stage-1 error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================================================
    # STAGE-2: REFINEMENT
    # ========================================================================
    log.info("\n" + "=" * 60)
    log.info("PHASE 2: STAGE-2 REFINEMENT")
    log.info("=" * 60)
    
    try:
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        log.info("Loading DINO Encoder for Stage-2...")
        encoder = DINOEncoder(
            lora_dir=HF_REPO_ID,
            base_weights_path=HF_REPO_ID,
            device=device
        )
        log.info("DINO Encoder loaded successfully")
        
        log.info("Loading Stage-2 Decoder...")
        decoder = build_stage2_decoder(device=device)
        log.info("Stage-2 Decoder loaded successfully")
        
        refiner = Stage2Refiner(decoder=decoder, dino_encoder=encoder)
        log.info("Stage-2 Refiner constructed")
        
        refined_hits = run_stage2_test(refiner, hits, lpa_catalog, out_dir, max_candidates=50)
        
    except Exception as e:
        log.error(f"Stage-2 error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    log.info("\n" + "=" * 60)
    log.info("EXECUTION SUMMARY")
    log.info("=" * 60)
    log.info(f"Stage-1 Candidate Hits: {len(hits)}")
    log.info(f"Stage-2 Confirmed Pits: {len(refined_hits)}")
    log.info(f"LPA Catalog Reference Pits: {len(lpa_catalog)}")
    log.info(f"Output Directory: {out_dir}")
    
    if refined_hits and lpa_catalog:
        matched = 0
        for refined in refined_hits:
            for pit in lpa_catalog:
                dist = compute_lunar_distance(
                    refined.essa_lat, refined.essa_lon,
                    pit["lat"], pit["lon"]
                )
                if dist < 300.0:
                    matched += 1
                    break
        
        match_rate = matched / len(lpa_catalog) * 100
        log.info(f"Match rate with LPA catalog: {matched}/{len(lpa_catalog)} ({match_rate:.1f}%)")
    
    log.info("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
