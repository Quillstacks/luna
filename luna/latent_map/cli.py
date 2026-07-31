"""
luna.latent_map.cli
~~~~~~~~~~~~~~~~~~~
CLI module for executing the latent space mapping pipeline.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


from luna.latent_map.topology import PitTopologyAnalyzer
from luna.latent_map.prober import LatentSpaceProber
from luna.latent_map.visualizer import LatentMapVisualizer

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="LUNA Latent Space Explorer & Visualizer")
    parser.add_argument("--pits-dir", type=str, default="data/_scratch/pits/", help="Directory containing pit query .npy patches")
    parser.add_argument("--indices-dir", type=str, default="data/_scratch/indices_old/", help="Directory containing Pithos index files")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of background tiles to sample")
    parser.add_argument("--method", type=str, default="umap", choices=["umap", "tsne", "pca"], help="Dimensionality reduction method")
    parser.add_argument("--workers", type=int, default=16, help="Parallel worker process count")
    parser.add_argument("--out", type=str, default="data/_scratch/luna_latent_map.html", help="Destination HTML dashboard path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    log.info("Starting Latent Space Exploration...")

    # Step 1: Topology analysis of reference pit anchors
    analyzer = PitTopologyAnalyzer()
    anchor_info, anchor_embeddings = analyzer.load_and_encode_anchors(args.pits_dir)
    log.info("Encoded %d reference pit anchors.", len(anchor_info))

    # Step 2: Sample background tiles from indices_old
    prober = LatentSpaceProber(indices_dir=args.indices_dir)
    bg_metadata, bg_vectors = prober.sample_background_tiles(
        target_sample_count=args.sample_size, max_workers=args.workers
    )

    # Step 3: Dimensionality reduction & Plotly HTML dashboard
    visualizer = LatentMapVisualizer(method=args.method)
    html_path = visualizer.build_dashboard(
        anchor_info=anchor_info,
        anchor_embeddings=anchor_embeddings,
        background_metadata=bg_metadata,
        background_embeddings=bg_vectors,
        out_html_path=args.out,
    )

    print("\n=========================================================================")
    print(f"✨ Latent Space Dashboard generated successfully!")
    print(f"📍 Dashboard HTML Path : {html_path}")
    print(f"📊 Pit Anchors Mapped  : {len(anchor_info)}")
    print(f"🌑 Background Samples  : {len(bg_metadata)}")
    print("=========================================================================\n")


if __name__ == "__main__":
    main()
