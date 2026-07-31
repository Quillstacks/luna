#!/usr/bin/env python3
"""
run_global_resonant_stage2.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
End-to-End Global Resonant Screening & Stage-2 Refinement Pipeline.

1. Loads 278 ground-truth LPA pit anchors from data/_scratch/pits/.
2. Encodes 278 query vectors across 8 families with Multi-Family Resonant Voting.
3. Screens all 55,492 Pithos indexes in data/_scratch/indices/.
4. Filters candidate hits using spatial NMS.
5. Executes Stage-2 Refinement (FPN + SPICE Geometry + Physics + SVM Gatekeeper).
6. Enforces strict pit criteria (min shadow area, alignment <= 30°, depth >= 10m, SVM >= 0.2).
7. Maps exact georeferenced coordinates (Lat/Lon) via Bilinear Projection.
8. Sanitizes all floats to prevent illegal NaNs in JSON and generates 3D Globe QuickMap links.
"""

import os
import sys
import json
import math
import time
import glob
import pickle
import logging
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from luna.pipeline import LunaPipeline
from luna.screening.pithos import PithosMIDB
from luna.io.nac_reader import read_nac
from luna.io.projection import LinearProjection, pixel_to_lonlat
from luna.models.stage2_decoder import Stage2DenseDecoder, Stage2Config, PhysicsValidator
from luna.models.essa import RefinedHit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("luna.global_stage2")

LUNAR_RADIUS_METERS = 1737400.0

def clean_float(val, default=0.0):
    """Sanitizes float values to prevent illegal NaN/Inf in JSON output."""
    if val is None:
        return default
    try:
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return default
        return round(f_val, 6)
    except Exception:
        return default

def build_quickmap_url(lon: float, lat: float, label_id: str) -> str:
    """Generates an accurate, working LROC 3D Globe QuickMap deep-link (proj=22)."""
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    R = LUNAR_RADIUS_METERS
    cam_dist = R * 2.1
    cam_x = cam_dist * math.cos(lat_rad) * math.cos(lon_rad)
    cam_y = cam_dist * math.cos(lat_rad) * math.sin(lon_rad)
    cam_z = cam_dist * math.sin(lat_rad)
    
    camera_param = f"{cam_x:.3f}%2C{cam_y:.3f}%2C{cam_z:.3f}%2C0.3399%2C-0.7518%2C-0.5651%2C0.2328%2C-0.5149%2C0.825%2C60"
    feat_raw = f"{lon:.7f},{lat:.7f}@@{json.dumps({'id': label_id}, separators=(',', ':'))}"
    feat_encoded = quote(feat_raw, safe="")
    return (
        f"https://quickmap.lroc.im-ldi.com/layers?"
        f"camera={camera_param}"
        f"&selectedFeature=%40%40user-defined%2C{label_id}"
        f"&earthShadowEnabled=true"
        f"&proj=22"
        f"&stack=3314%2C1529891"
        f"&features={feat_encoded}"
    )

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
    out_dir = ROOT / "data" / "_scratch"
    out_dir.mkdir(parents=True, exist_ok=True)
    indices_dir = out_dir / "indices"
    log_file = out_dir / "global_stage2_scan.log"
    json_out = out_dir / "global_stage2_refined_hits.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("=" * 80)
    log.info("STARTING GLOBAL RESONANT MULTI-FAMILY SCAN & STAGE-2 REFINEMENT")
    log.info(f"Target Device: {device} | Index Directory: {indices_dir}")
    log.info("=" * 80)

    # 1. Initialize Pipeline & Load Pit Queries
    from luna.config import HF_REPO_ID
    pipeline = LunaPipeline.from_pretrained(HF_REPO_ID, refiner="stage2")
    lpa_pits = load_lpa_catalog(ROOT / "catalogs" / "lpa.csv")
    log.info(f"Loaded {len(lpa_pits)} reference LPA catalog pits.")

    query_vecs, query_families, query_thresholds = pipeline._load_and_encode_pit_queries(str(out_dir / "pits"))
    log.info(f"Encoded {len(query_vecs)} pit queries across {len(np.unique(query_families))} families.")

    # 2. Load Neural Models & SVM Gatekeeper
    ckpt_path = ROOT / "data" / "weights" / "stage2_decoder_best.pt"
    if ckpt_path.exists():
        decoder = Stage2DenseDecoder.from_checkpoint(ckpt_path, device=device)
        log.info(f"Loaded Stage-2 FPN Decoder: {ckpt_path.name}")
    else:
        decoder = Stage2DenseDecoder(config=Stage2Config(), device=device)
        log.info("Initialized baseline Stage-2 FPN Decoder.")
    decoder.eval()

    svm_path = out_dir / "stage2_svm.pkl"
    svm_model, svm_scaler = None, None
    if svm_path.exists():
        with open(svm_path, "rb") as f:
            svm_data = pickle.load(f)
            svm_scaler = svm_data["scaler"]
            svm_model = svm_data["model"]
        log.info(f"Loaded SVM Gatekeeper from {svm_path.name}")

    validator = PhysicsValidator()

    # Load known pit NAC product IDs for priority scan
    priority_pids = set()
    pit_nacs_path = ROOT / "catalogs" / "pit_nacs.json"
    if pit_nacs_path.exists():
        with open(pit_nacs_path) as f:
            pn_data = json.load(f)
            for pit_k, items in pn_data.items():
                for it in items:
                    if 'product' in it:
                        priority_pids.add(it['product'].strip())
                    if 'url' in it and 'M1' in it['url']:
                        p_sub = it['url'].split('/')[-1].split('.')[0]
                        priority_pids.add(p_sub.strip())

    bin_files = sorted(glob.glob(str(indices_dir / "pithos_*.bin")))
    primary_bins = [f for f in bin_files if not any(x in f for x in ['_ids.bin', '_metadata.bin', '_fp16.bin', '_tier_'])]

    def sort_key(bin_file_path):
        p_stem = Path(bin_file_path).stem.replace("pithos_", "")
        is_priority = any(pid in p_stem for pid in priority_pids)
        local_img = (out_dir / f"{p_stem}.IMG").exists() or (out_dir / "cache" / f"{p_stem}.IMG").exists()
        if is_priority and local_img: return 0
        elif is_priority: return 1
        elif local_img: return 2
        else: return 3

    primary_bins = sorted(primary_bins, key=sort_key)
    log.info(f"Found {len(primary_bins):,} compiled Pithos index files. Prioritized queue: {sum(1 for f in primary_bins if sort_key(f) <= 1)} target catalog NACs placed first.")

    db = PithosMIDB()
    all_refined_hits = []
    total_screened_hits = 0
    t_start = time.time()

    for idx, bin_path in enumerate(primary_bins, 1):
        pid = Path(bin_path).stem.replace("pithos_", "")
        meta_path = indices_dir / f"pithos_{pid}_meta.pkl"
        if not meta_path.exists():
            continue

        try:
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)

            index_prefix = str(indices_dir / f"pithos_{pid}")
            db.load_index(pid, f"{index_prefix}.bin")

            voting_mask = np.zeros(len(metadata), dtype=np.uint8)

            # Native Resonant Voting Scan using exact Hamming bit distance thresholds
            db.query_planetary_grid(
                index_name=pid,
                queries=query_vecs,
                families=query_families.astype(np.int32),
                thresholds=query_thresholds.astype(np.int32),
                voting_mask=voting_mask,
            )
            db.drop_index(pid)

            cand_indices = np.where(voting_mask > 0)[0].tolist()
            if not cand_indices:
                continue

            total_screened_hits += len(cand_indices)

            # Apply Spatial NMS
            nms_hits = pipeline._nms(
                cand_indices, voting_mask, metadata, top_k=50, min_dist_px=512.0
            )

            nac_img_path = out_dir / f"{pid}.IMG"
            if not nac_img_path.exists():
                matched_imgs = list(out_dir.glob(f"{pid}*.IMG")) + list((out_dir / "cache").glob(f"{pid}*.IMG"))
                if matched_imgs:
                    nac_img_path = matched_imgs[0]
            
            if not nac_img_path.exists():
                try:
                    from luna.io.pds_fetch import fetch_nac
                    nac_img_path = fetch_nac(pid, dest_dir=out_dir)
                except Exception as fetch_err:
                    log.warning(f"Could not fetch NAC {pid} from PDS: {fetch_err}")
                    continue

            try:
                nac_img = read_nac(nac_img_path, geometry=True)
            except Exception as read_err:
                log.warning(f"Could not read NAC image {nac_img_path.name}: {read_err}. Removing corrupted file.")
                if nac_img_path.exists():
                    try: nac_img_path.unlink()
                    except Exception: pass
                continue

            try:
                proj = LinearProjection.from_nac_geometry(nac_img.geometry, samples=nac_img.samples, lines=nac_img.lines)
            except Exception:
                proj = None

            sub_solar_azimuth = getattr(nac_img, 'sub_solar_azimuth', 180.0)
            incidence_angle = getattr(nac_img, 'incidence_angle', 45.0)
            pixel_scale = getattr(nac_img, 'pixel_scale', 0.5)

            for hit_tuple in nms_hits:
                tile_idx = hit_tuple[0]
                meta_item = metadata[tile_idx]
                x0, y0 = int(meta_item.x_offset), int(meta_item.y_offset)
                votes = int(voting_mask[tile_idx])

                h_img, w_img = nac_img.pixels.shape
                x0_c = max(0, min(w_img - 256, x0))
                y0_c = max(0, min(h_img - 256, y0))
                tile = nac_img.pixels[y0_c:y0_c+256, x0_c:x0_c+256].astype(np.float32)
                if tile.shape != (256, 256):
                    tile = np.pad(tile, ((0, max(0, 256 - tile.shape[0])), (0, max(0, 256 - tile.shape[1]))), mode='reflect')

                tile_norm = (tile - tile.min()) / (tile.max() - tile.min() + 1e-6)
                img_t = torch.from_numpy(tile_norm).unsqueeze(0).unsqueeze(0).to(device)
                if device == "cuda": img_t = img_t.half()
                if img_t.shape[1] == 1: img_t = img_t.repeat(1, 3, 1, 1)

                with torch.no_grad():
                    from torchvision import transforms
                    norm_tr = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    feats = pipeline._encoder._backbone_module.forward_features(norm_tr(img_t))
                    spatial_tokens = decoder.token_extractor.extract_spatial_tokens(feats).to(dtype=next(decoder.parameters()).dtype)
                    prob_masks = decoder.get_probability_masks(spatial_tokens, temperature=0.7)

                prob_masks_np = prob_masks.cpu().numpy()[0]
                shadow_mask = prob_masks_np[1, :, :]
                edge_mask = prob_masks_np[2, :, :]
                combined_score = float((shadow_mask.max() + edge_mask.max()) / 2.0)

                # STRICT PIT SEGMENTATION & AREA CHECKS
                shadow_pixels = int((shadow_mask > 0.45).sum())
                edge_pixels = int((edge_mask > 0.45).sum())

                # A genuine pit requires a connected shadow blob (>= 25 px) and rim contour (>= 15 px)
                if combined_score < 0.60 or shadow_pixels < 25 or edge_pixels < 15:
                    continue

                physics_res = validator.validate_detection(
                    shadow_mask=shadow_mask > 0.45,
                    edge_mask=edge_mask > 0.45,
                    sub_solar_azimuth=sub_solar_azimuth,
                    incidence_angle_deg=incidence_angle,
                    pixel_scale=pixel_scale,
                )

                # STRICT PHYSICS CHECKS: must align with sun (<= 30°) and have valid depth (>= 10m)
                if not physics_res.is_valid_pit or math.isnan(physics_res.depth_estimate_meters) or physics_res.depth_estimate_meters < 10.0:
                    continue

                if physics_res.alignment_error_degrees > 30.0:
                    continue

                svm_score = 0.0
                if svm_model is not None and svm_scaler is not None:
                    try:
                        class MockHit: pass
                        mh = MockHit()
                        mh.product_id = pid
                        mh.score = float(votes)
                        mh.votes = float(votes)
                        mh.dino_similarity = combined_score
                        mh.hamming_dist = 25

                        from luna.models.stage2_decoder import Stage2Refiner
                        ref_dummy = Stage2Refiner(decoder=decoder)
                        sf = ref_dummy._extract_svm_features(mh, shadow_mask, edge_mask, physics_res)
                        sf_scaled = svm_scaler.transform(sf.reshape(1, -1))
                        svm_score = float(svm_model.decision_function(sf_scaled)[0])
                    except Exception:
                        pass

                # STRICT SVM GATEKEEPER CHECK (>= 0.20)
                if svm_score < 0.20:
                    continue

                precise_x = x0_c + physics_res.tile_centroid_x
                precise_y = y0_c + physics_res.tile_centroid_y

                if proj is not None:
                    true_lon, true_lat = pixel_to_lonlat(proj, precise_x, precise_y)
                    if math.isnan(true_lon) or math.isnan(true_lat): continue
                else:
                    continue

                if true_lon > 180.0: true_lon -= 360.0

                min_dist = float('inf')
                best_pit = None
                for pit in lpa_pits:
                    d = compute_lunar_distance(true_lat, true_lon, pit["lat"], pit["lon"])
                    if d < min_dist:
                        min_dist = d
                        best_pit = pit

                is_match = (min_dist <= 300.0)
                classification = "ground_truth_pit_match" if is_match else "potential_new_pit_discovery"
                cand_id_str = f"{pid}_tile_{tile_idx}"

                qm_url = build_quickmap_url(true_lon, true_lat, cand_id_str)

                res_dict = {
                    "rank": len(all_refined_hits) + 1,
                    "candidate_id": cand_id_str,
                    "product_id": pid,
                    "classification": classification,
                    "lat": clean_float(true_lat),
                    "lon": clean_float(true_lon),
                    "x_pixel": int(precise_x),
                    "y_pixel": int(precise_y),
                    "fpn_confidence": clean_float(combined_score),
                    "estimated_depth_m": clean_float(physics_res.depth_estimate_meters),
                    "alignment_error_deg": clean_float(physics_res.alignment_error_degrees),
                    "svm_score": clean_float(svm_score),
                    "resonant_votes": votes,
                    "nearest_catalog_pit": best_pit["name"] if is_match else "None",
                    "catalog_host_region": best_pit["host"] if is_match else "Uncataloged Area",
                    "catalog_dist_meters": clean_float(min_dist) if is_match else None,
                    "quickmap_url": qm_url
                }
                all_refined_hits.append(res_dict)

                with open(json_out, "w") as f:
                    json.dump(all_refined_hits, f, indent=2)

                log.info(f"  [CONFIRMED GENUINE PIT #{len(all_refined_hits)}] {pid} @ ({true_lat:.5f}°, {true_lon:.5f}°) | Conf: {combined_score:.3f} | Depth: {physics_res.depth_estimate_meters:.1f}m | Match: {best_pit['name'] if is_match else 'NEW DISCOVERY'}")

        except Exception as err:
            log.error(f"Error processing index {pid}: {err}")

        # Periodic log & save checkpoint every 100 indexes
        if idx % 100 == 0:
            elapsed_m = (time.time() - t_start) / 60.0
            log.info(f"--- Progress Milestone: {idx}/{len(primary_bins)} indexes processed ({idx/len(primary_bins)*100:.1f}%) | Confirmed Pits: {len(all_refined_hits)} | Elapsed: {elapsed_m:.1f}m ---")
            with open(json_out, "w") as f:
                json.dump(all_refined_hits, f, indent=2)

    # Final Save
    total_elapsed = time.time() - t_start
    log.info("=" * 80)
    log.info("GLOBAL REFINEMENT SCAN COMPLETE")
    log.info(f"Total Indexes Screened : {len(primary_bins):,}")
    log.info(f"Stage-1 Candidate Hits : {total_screened_hits:,}")
    log.info(f"Final Confirmed Pits   : {len(all_refined_hits):,}")
    log.info(f"Total Scan Elapsed Time: {total_elapsed / 3600.0:.2f} hours")
    log.info("=" * 80)

    with open(json_out, "w") as f:
        json.dump(all_refined_hits, f, indent=2)

if __name__ == "__main__":
    main()
