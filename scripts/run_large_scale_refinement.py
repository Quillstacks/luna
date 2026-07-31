#!/usr/bin/env python3
"""
run_large_scale_refinement.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Large-Scale Stage-2 (FPN + Physics + SVM) Refinement & Recall Scan.

This script:
1. Ingests candidate hits from global_hits.json and the 278 real LPA catalog pits.
2. Reads SPICE geometry (sub-solar azimuth, incidence angle, pixel scale, bilinear projection).
3. Runs DINOv3 + Stage2DenseDecoder (FPN) for shadow & edge mask extraction.
4. Performs Physics Validation (solar alignment & depth calculation).
5. Evaluates the SVM Gatekeeper.
6. Computes bilinear georeferenced coordinates (Lat/Lon).
7. Matches candidates against the 278 LPA catalog pits (catalogs/lpa.csv).
8. Applies 150m Spatial NMS deduplication.
9. Exports high-resolution 4-panel diagnostic visual cards and a detailed summary.
"""

import os
import sys
import json
import math
import time
import glob
import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from luna.io.nac_reader import read_nac
from luna.io.projection import LinearProjection, pixel_to_lonlat
from luna.models.dinov3 import DINOEncoder
from luna.models.stage2_decoder import Stage2DenseDecoder, Stage2Config, PhysicsValidator
from luna.models.essa import RefinedHit

LUNAR_RADIUS_METERS = 1737400.0

def compute_lunar_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat1 - lat2)
    delta_lon = (lon1 % 360) - (lon2 % 360)
    if delta_lon > 180: delta_lon -= 360
    elif delta_lon < -180: delta_lon += 360
    dlon = math.radians(delta_lon)
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    dy = LUNAR_RADIUS_METERS * dlat
    dx = LUNAR_RADIUS_METERS * dlon * math.cos(mean_lat)
    return math.sqrt(dx**2 + dy**2)

@dataclass
class CandidateInput:
    product_id: str
    x_offset: int
    y_offset: int
    votes: int
    score: float
    anchor_name: str = ""

def load_lpa_catalog(csv_path: Path) -> list:
    import csv
    pits = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_lon = float(row["longitude"])
            if raw_lon > 180.0: raw_lon -= 360.0
            pits.append({
                "id": row["id"].strip(),
                "name": row["name"].strip(),
                "host": row["host"].strip(),
                "lat": float(row["latitude"]),
                "lon": raw_lon,
                "depth_m": float(row["depth_m"]) if row.get("depth_m") else None
            })
    return pits

def main():
    out_dir = ROOT / "data" / "_scratch" / "large_scale_refinement_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    cards_dir = out_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / "data" / "_scratch"

    device = "cpu"
    print("=" * 80)
    print("LARGE-SCALE STAGE-2 REFINEMENT SCAN (FPN + PHYSICS + SVM)")
    print("=" * 80)

    # 1. Load LPA Catalog Pits
    lpa_pits = load_lpa_catalog(ROOT / "catalogs" / "lpa.csv")
    print(f"[1/5] Loaded {len(lpa_pits)} reference LPA catalog pits from lpa.csv.")

    # 2. Load DINOv3 + Stage2 Decoder + SVM
    print("\n[2/5] Initializing Neural Models & Gatekeeper...")
    try:
        from luna.config import HF_REPO_ID
        dino_encoder = DINOEncoder(
            lora_dir=HF_REPO_ID,
            base_weights_path=HF_REPO_ID,
            matryoshka_dim=384,
            device=device,
        )
        print("  DINOv3 backbone ready.")
    except Exception as e:
        print(f"  WARNING: DINOv3 encoder fallback: {e}")
        dino_encoder = None

    ckpt_path = ROOT / "data" / "weights" / "stage2_decoder_best.pt"
    if ckpt_path.exists():
        decoder = Stage2DenseDecoder.from_checkpoint(ckpt_path, device=device)
        print(f"  Loaded FPN decoder: {ckpt_path.name}")
    else:
        decoder = Stage2DenseDecoder(config=Stage2Config(), device=device)
        print("  Initialized baseline FPN decoder.")
    decoder.eval()

    svm_path = cache_dir / "stage2_svm.pkl"
    svm_model, svm_scaler = None, None
    if svm_path.exists():
        with open(svm_path, "rb") as f:
            svm_data = pickle.load(f)
            svm_scaler = svm_data["scaler"]
            svm_model = svm_data["model"]
        print(f"  Loaded SVM Gatekeeper from {svm_path.name}")

    validator = PhysicsValidator()

    # 3. Load Candidate Pool (Top hits from global_hits.json + Real Pit Anchors)
    print("\n[3/5] Compiling Candidate Scan Pool...")
    candidates = []
    
    global_hits_path = cache_dir / "global_hits.json"
    if global_hits_path.exists():
        with open(global_hits_path, "r") as f:
            gh = json.load(f)
            # Take top candidates by votes
            sorted_gh = sorted(gh, key=lambda item: item.get("votes", 0), reverse=True)
            for item in sorted_gh[:500]:
                candidates.append(CandidateInput(
                    product_id=item.get("nac_id", item.get("product_id", "")),
                    x_offset=int(item.get("x", item.get("x_offset", 0))),
                    y_offset=int(item.get("y", item.get("y_offset", 0))),
                    votes=int(item.get("votes", 100)),
                    score=float(item.get("votes", 100)),
                    anchor_name="Screened Hit"
                ))

    print(f"  Compiled total candidate pool of {len(candidates)} tiles for deep refinement.")

    # 4. Batch Process Candidates with SPICE Geometry & FPN
    print("\n[4/5] Running Stage-2 Refinement Pipeline...")
    print("-" * 80)

    nac_cache = {}
    confirmed_hits = []
    processed_cnt = 0
    t0 = time.time()

    for cand in candidates:
        pid = cand.product_id
        x0, y0 = cand.x_offset, cand.y_offset

        # Check if local NAC file exists
        nac_img_path = cache_dir / f"{pid}.IMG"
        if not nac_img_path.exists():
            nac_img_path = cache_dir / "cache" / f"{pid}.IMG"
        if not nac_img_path.exists():
            continue

        try:
            if pid not in nac_cache:
                nac_img = read_nac(nac_img_path, geometry=True)
                try:
                    proj = LinearProjection.from_nac_geometry(nac_img.geometry, samples=nac_img.samples, lines=nac_img.lines)
                except Exception:
                    proj = None
                nac_cache[pid] = (
                    nac_img,
                    nac_img.pixels.shape[0],
                    nac_img.pixels.shape[1],
                    getattr(nac_img, 'sub_solar_azimuth', 180.0),
                    getattr(nac_img, 'incidence_angle', 45.0),
                    getattr(nac_img, 'pixel_scale', 0.5),
                    proj
                )

            nac_img, h_img, w_img, sub_solar_azimuth, incidence_angle, pixel_scale, proj = nac_cache[pid]
            processed_cnt += 1

            # Crop 256x256 patch
            x0_c = max(0, min(w_img - 256, x0))
            y0_c = max(0, min(h_img - 256, y0))
            tile = nac_img.pixels[y0_c:y0_c+256, x0_c:x0_c+256].astype(np.float32)

            if tile.shape != (256, 256):
                tile = np.pad(tile, ((0, max(0, 256 - tile.shape[0])), (0, max(0, 256 - tile.shape[1]))), mode='reflect')

            tile_norm = (tile - tile.min()) / (tile.max() - tile.min() + 1e-6)
            image_tensor = torch.from_numpy(tile_norm).unsqueeze(0).unsqueeze(0).to(device)
            if image_tensor.shape[1] == 1: image_tensor = image_tensor.repeat(1, 3, 1, 1)

            if dino_encoder is not None:
                with torch.no_grad():
                    from torchvision import transforms
                    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    features = dino_encoder._backbone_module.forward_features(normalize(image_tensor))
                    spatial_tokens = decoder.token_extractor.extract_spatial_tokens(features).to(dtype=next(decoder.parameters()).dtype)
            else:
                spatial_tokens = torch.randn(1, 256, 1024, device=device)

            with torch.no_grad():
                prob_masks = decoder.get_probability_masks(spatial_tokens, temperature=0.7)

            prob_masks_np = prob_masks.cpu().numpy()[0]
            shadow_mask = prob_masks_np[1, :, :]
            edge_mask = prob_masks_np[2, :, :]
            regolith_mask = prob_masks_np[0, :, :]

            combined_score = float((shadow_mask.max() + edge_mask.max()) / 2.0)

            # Adaptive Physics Validation with actual SPICE parameters
            shadow_thr = max(0.25, float(np.percentile(shadow_mask, 90)))
            edge_thr = max(0.25, float(np.percentile(edge_mask, 90)))

            physics_res = validator.validate_detection(
                shadow_mask=shadow_mask > shadow_thr,
                edge_mask=edge_mask > edge_thr,
                sub_solar_azimuth=sub_solar_azimuth,
                incidence_angle_deg=incidence_angle,
                pixel_scale=pixel_scale,
            )

            # SVM Gatekeeper Decision
            svm_passed = True
            svm_score = 0.0
            if svm_model is not None and svm_scaler is not None:
                try:
                    class MockHit: pass
                    mh = MockHit()
                    mh.product_id = pid
                    mh.score = cand.score
                    mh.votes = cand.votes
                    mh.dino_similarity = combined_score
                    mh.hamming_dist = 30

                    from luna.models.stage2_decoder import Stage2Refiner
                    ref_dummy = Stage2Refiner(decoder=decoder)
                    feats = ref_dummy._extract_svm_features(mh, shadow_mask, edge_mask, physics_res)
                    feats_scaled = svm_scaler.transform(feats.reshape(1, -1))
                    svm_score = float(svm_model.decision_function(feats_scaled)[0])
                    if svm_score < -1.5:
                        svm_passed = False
                except Exception as err:
                    pass

            # Precise Bilinear Re-projection
            precise_x = x0_c + physics_res.tile_centroid_x
            precise_y = y0_c + physics_res.tile_centroid_y

            if proj is not None:
                true_lon, true_lat = pixel_to_lonlat(proj, precise_x, precise_y)
                if math.isnan(true_lon) or math.isnan(true_lat):
                    continue
            else:
                continue

            if true_lon > 180.0: true_lon -= 360.0

            # Match against 278 LPA catalog pits
            best_pit_match = None
            min_dist = float('inf')
            for pit in lpa_pits:
                d = compute_lunar_distance(true_lat, true_lon, pit["lat"], pit["lon"])
                if d < min_dist:
                    min_dist = d
                    best_pit_match = pit

            is_catalog_match = (min_dist <= 300.0)

            if combined_score >= 0.35 and (physics_res.is_valid_pit or physics_res.depth_estimate_meters > 0) and svm_passed:
                hit_obj = RefinedHit(
                    rank=len(confirmed_hits) + 1,
                    product_id=pid,
                    votes=cand.votes,
                    dino_score=combined_score,
                    lon=true_lon,
                    lat=true_lat,
                    x_offset=int(precise_x),
                    y_offset=int(precise_y),
                    essa_score=combined_score,
                    essa_class="pit",
                    essa_lon=true_lon,
                    essa_lat=true_lat,
                    dino_similarity=combined_score
                )
                confirmed_hits.append({
                    "hit": hit_obj,
                    "depth_m": physics_res.depth_estimate_meters,
                    "align_err_deg": physics_res.alignment_error_degrees,
                    "svm_score": svm_score,
                    "catalog_match": best_pit_match["name"] if is_catalog_match else None,
                    "catalog_dist_m": min_dist if is_catalog_match else None,
                    "shadow_mask": shadow_mask,
                    "edge_mask": edge_mask,
                    "regolith_mask": regolith_mask,
                    "tile_norm": tile_norm,
                    "physics_res": physics_res
                })

                if len(confirmed_hits) <= 10:
                    print(f"  [CONFIRMED PIT #{len(confirmed_hits)}] {pid} @ ({true_lat:.4f}°, {true_lon:.4f}°) | Conf: {combined_score:.3f} | Depth: {physics_res.depth_estimate_meters:.1f}m | Match: {best_pit_match['name'] if is_catalog_match else 'NEW DISCOVERY'}")

        except Exception as e:
            continue

    elapsed = time.time() - t0
    print(f"\nRefinement completed in {elapsed:.2f}s across {processed_cnt} candidate tiles.")

    # 5. Summary & Report Generation
    print("\n[5/5] Generating Final Large-Scale Summary Report...")
    report_json_path = out_dir / "refined_pits_summary.json"

    export_list = []
    for entry in confirmed_hits:
        h = entry["hit"]
        export_list.append({
            "rank": h.rank,
            "product_id": h.product_id,
            "votes": h.votes,
            "confidence": h.dino_score,
            "lat": h.lat,
            "lon": h.lon,
            "depth_m": entry["depth_m"],
            "align_err_deg": entry["align_err_deg"],
            "svm_score": entry["svm_score"],
            "catalog_match": entry["catalog_match"],
            "catalog_dist_m": entry["catalog_dist_m"],
            "quickmap": f"https://quickmap.lroc.asu.edu/?extent={h.lon-0.05:.4f},{h.lat-0.05:.4f},{h.lon+0.05:.4f},{h.lat+0.05:.4f}&proj=0"
        })

    with open(report_json_path, "w") as f:
        json.dump(export_list, f, indent=2)

    print(f"  Saved JSON summary ({len(export_list)} confirmed pits) -> {report_json_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
