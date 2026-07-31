#!/usr/bin/env python3
"""
train_parametric_umap.py
~~~~~~~~~~~~~~~~~~~~~~~~
Trains the PyTorch Parametric UMAP Neural Network encoder on landmark pit anchors
and background tiles with low CPU/GPU resource usage (protecting the active scan).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from luna.latent_map.topology import PitTopologyAnalyzer
from luna.latent_map.prober import LatentSpaceProber
from luna.latent_map.visualizer import LatentMapVisualizer
from luna.latent_map.parametric_umap import ParametricUMAPEncoder

log = logging.getLogger("train_parametric_umap")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Parametric UMAP Encoder")
    parser.add_argument("--landmark-samples", type=int, default=5000, help="Number of background tiles for landmark training")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--out-weights", type=str, default="data/_scratch/weights/parametric_umap_luna.pt", help="Weights destination")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    log.info("Loading landmark dataset (pit anchors + background tiles)...")
    analyzer = PitTopologyAnalyzer()
    anchor_info, anchor_embeddings = analyzer.load_and_encode_anchors("data/_scratch/pits")

    prober = LatentSpaceProber(indices_dir="data/_scratch/indices_old")
    bg_metadata, bg_embeddings = prober.sample_background_tiles(
        target_sample_count=args.landmark_samples, max_workers=8
    )

    combined_embeddings = np.vstack([anchor_embeddings, bg_embeddings])
    log.info("Total landmark embeddings: %d", len(combined_embeddings))

    # Generate teacher 2D coordinates via standard Cosine UMAP
    log.info("Generating UMAP teacher coordinates...")
    vis = LatentMapVisualizer(method="umap")
    teacher_2d_coords = vis.reduce_dimensions(combined_embeddings)

    # Train PyTorch Parametric UMAP Model
    log.info("Fitting PyTorch Parametric UMAP Encoder...")
    encoder = ParametricUMAPEncoder(model_path=args.out_weights)
    loss = encoder.train_on_landmarks(
        landmark_embeddings=combined_embeddings,
        target_2d_coords=teacher_2d_coords,
        epochs=args.epochs,
        batch_size=512,
    )

    log.info("✨ Parametric UMAP Model Training Complete! (Final Loss: %.6f)", loss)


if __name__ == "__main__":
    main()
