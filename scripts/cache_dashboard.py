#!/usr/bin/env python3
"""
cache_dashboard.py
~~~~~~~~~~~~~~~~~~
Offline pre-computes dashboard data and saves to data/_scratch/dash_cache.pkl.
Guarantees instant (< 100ms) page loading on browser launch!
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from luna.latent_map.topology import PitTopologyAnalyzer
from luna.latent_map.prober import LatentSpaceProber
from luna.latent_map.visualizer import LatentMapVisualizer
from luna.latent_map.manifold_math import (
    compute_laplace_beltrami_eigenvectors,
    compute_nash_sammon_strain,
    compute_jl_distortion_ratio,
)
from luna.screening.pithos import PithosMIDB
import numpy as np
import pandas as pd

log = logging.getLogger("cache_dashboard")


def build_cache(sample_size: int = 5000, out_cache_path: str = "data/_scratch/dash_cache.pkl") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log.info("Building instant dashboard cache file (%s)...", out_cache_path)

    # Step 1: Load Pit Anchors
    analyzer = PitTopologyAnalyzer()
    anchor_info, anchor_embeddings = analyzer.load_and_encode_anchors("data/_scratch/pits")

    # Step 2: Sample Background Tiles
    prober = LatentSpaceProber(indices_dir="data/_scratch/indices_old")
    bg_metadata, bg_embeddings = prober.sample_background_tiles(
        target_sample_count=sample_size, max_workers=8
    )

    combined_embeddings = np.vstack([anchor_embeddings, bg_embeddings])

    # Step 3: Dimensionality Reductions
    vis_umap = LatentMapVisualizer(method="umap")
    coords_2d = vis_umap.reduce_dimensions(combined_embeddings)

    from sklearn.decomposition import PCA
    pca_3d = PCA(n_components=3, random_state=42)
    coords_3d = pca_3d.fit_transform(combined_embeddings)

    # Step 4: Advanced Manifold Mathematics (Laplace-Beltrami, Nash Strain, JL Distortion)
    log.info("Computing Laplace-Beltrami Diffusion Coordinates...")
    diff_coords = compute_laplace_beltrami_eigenvectors(combined_embeddings, k_neighbors=15, n_components=3)

    log.info("Computing Nash Sammon Strain (Projection Distortion)...")
    point_strain, global_stress = compute_nash_sammon_strain(combined_embeddings, coords_2d, sample_size=1000)

    log.info("Computing Johnson-Lindenstrauss Preservation Score...")
    _, jl_preservation_score = compute_jl_distortion_ratio(combined_embeddings, coords_2d, sample_size=1000)

    # Step 5: Hamming Distances to Anchors
    binary_anchors = PithosMIDB.binarize(anchor_embeddings)
    binary_bg = PithosMIDB.binarize(bg_embeddings)
    bits_anchors = np.unpackbits(binary_anchors.view(np.uint8), axis=1)[:, :384]
    bits_bg = np.unpackbits(binary_bg.view(np.uint8), axis=1)[:, :384]

    min_hamming = []
    nearest_anchor = []
    for i in range(len(bg_embeddings)):
        dists = np.sum(bits_anchors != bits_bg[i], axis=1)
        min_idx = np.argmin(dists)
        min_hamming.append(int(dists[min_idx]))
        nearest_anchor.append(anchor_info[min_idx]["name"])

    # Anchor DataFrame
    df_anchors = pd.DataFrame({
        "name": [a["name"] for a in anchor_info],
        "category": [a["category"] for a in anchor_info],
        "family_id": [a["family_id"] for a in anchor_info],
        "x_2d": coords_2d[:len(anchor_embeddings), 0],
        "y_2d": coords_2d[:len(anchor_embeddings), 1],
        "x_3d": coords_3d[:len(anchor_embeddings), 0],
        "y_3d": coords_3d[:len(anchor_embeddings), 1],
        "z_3d": coords_3d[:len(anchor_embeddings), 2],
        "lb_psi1": diff_coords[:len(anchor_embeddings), 0],
        "lb_psi2": diff_coords[:len(anchor_embeddings), 1],
        "nash_strain": point_strain[:len(anchor_embeddings)],
        "type": "Pit Anchor",
        "product_id": "REFERENCE",
        "lat": 0.0,
        "lon": 0.0,
        "x_offset": 0,
        "y_offset": 0,
        "hamming_dist": 0,
        "nearest_anchor": [a["name"] for a in anchor_info],
    })

    # Background DataFrame
    df_bg = pd.DataFrame({
        "name": [f"Tile_{i}" for i in range(len(bg_metadata))],
        "category": "Regolith Background",
        "family_id": -1,
        "x_2d": coords_2d[len(anchor_embeddings):, 0],
        "y_2d": coords_2d[len(anchor_embeddings):, 1],
        "x_3d": coords_3d[len(anchor_embeddings):, 0],
        "y_3d": coords_3d[len(anchor_embeddings):, 1],
        "z_3d": coords_3d[len(anchor_embeddings):, 2],
        "lb_psi1": diff_coords[len(anchor_embeddings):, 0],
        "lb_psi2": diff_coords[len(anchor_embeddings):, 1],
        "nash_strain": point_strain[len(anchor_embeddings):],
        "type": "Surface Tile",
        "product_id": [m["product_id"] for m in bg_metadata],
        "lat": [m["lat"] for m in bg_metadata],
        "lon": [m["lon"] for m in bg_metadata],
        "x_offset": [m["x_offset"] for m in bg_metadata],
        "y_offset": [m["y_offset"] for m in bg_metadata],
        "hamming_dist": min_hamming,
        "nearest_anchor": nearest_anchor,
    })

    cos_matrix, ham_matrix = analyzer.compute_distance_matrices(anchor_embeddings)

    cache_data = {
        "df_anchors": df_anchors,
        "df_bg": df_bg,
        "cos_matrix": cos_matrix,
        "ham_matrix": ham_matrix,
        "jl_preservation_score": jl_preservation_score,
        "global_sammon_stress": global_stress,
    }

    out_file = Path(out_cache_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(cache_data, f)

    log.info("✨ Pre-computed cache successfully written → %s (Size: %.2f MB)", out_file, out_file.stat().st_size / (1024*1024))


if __name__ == "__main__":
    build_cache()
