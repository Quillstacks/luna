"""
luna.latent_map.visualizer
~~~~~~~~~~~~~~~~~~~~~~~~~~
Dimensionality reduction (UMAP/t-SNE) and Plotly interactive dashboard generator
for DINOv3 latent space mapping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None  # type: ignore

try:
    import umap
except ImportError:
    umap = None  # type: ignore

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from luna.screening.pithos import PithosMIDB

log = logging.getLogger(__name__)


class LatentMapVisualizer:
    """
    Fits 2D manifold projections and builds interactive Plotly HTML dashboards.
    """

    def __init__(self, method: str = "umap") -> None:
        self.method = method.lower()

    def reduce_dimensions(
        self, embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1
    ) -> np.ndarray:
        """
        Project N x 384 embeddings down to N x 2 using UMAP, t-SNE, or PCA.
        """
        n_samples = len(embeddings)
        log.info("Projecting %d embeddings to 2D using %s...", n_samples, self.method.upper())

        if self.method == "umap":
            if umap is None:
                log.warning("umap-learn not installed. Falling back to PCA.")
                reducer = PCA(n_components=2)
            else:
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric="cosine",
                    random_state=42,
                )
        elif self.method == "tsne":
            reducer = TSNE(n_components=2, metric="cosine", random_state=42)
        else:
            reducer = PCA(n_components=2)

        coords_2d = reducer.fit_transform(embeddings)
        return coords_2d.astype(np.float32)

    def build_dashboard(
        self,
        anchor_info: List[Dict[str, Any]],
        anchor_embeddings: np.ndarray,
        background_metadata: List[Dict[str, Any]],
        background_embeddings: np.ndarray,
        out_html_path: str | Path = "data/_scratch/luna_latent_map.html",
    ) -> str:
        """
        Build and save a Plotly HTML dashboard visualizing the latent space.
        """
        out_path = Path(out_html_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        n_anchors = len(anchor_embeddings)
        n_bg = len(background_embeddings)
        combined_embeddings = np.vstack([anchor_embeddings, background_embeddings])

        coords_2d = self.reduce_dimensions(combined_embeddings)

        anchor_2d = coords_2d[:n_anchors]
        bg_2d = coords_2d[n_anchors:]

        # Compute nearest anchor distance for each background tile
        binary_anchors = PithosMIDB.binarize(anchor_embeddings)
        binary_bg = PithosMIDB.binarize(background_embeddings)

        bits_anchors = np.unpackbits(binary_anchors.view(np.uint8), axis=1)[:, :384]
        bits_bg = np.unpackbits(binary_bg.view(np.uint8), axis=1)[:, :384]

        # Find min Hamming distance to any anchor for bg tiles
        min_hamming = []
        nearest_anchor_name = []
        for i in range(n_bg):
            dists = np.sum(bits_anchors != bits_bg[i], axis=1)
            min_idx = np.argmin(dists)
            min_hamming.append(int(dists[min_idx]))
            nearest_anchor_name.append(anchor_info[min_idx]["name"])

        # Create Plotly figure
        fig = go.Figure()

        # Trace 1: Background Tiles
        bg_hover = [
            f"<b>Background Tile</b><br>"
            f"Product ID: {m['product_id']}<br>"
            f"Lat / Lon: {m['lat']:.4f}°, {m['lon']:.4f}°<br>"
            f"Offset (X, Y): ({m['x_offset']}, {m['y_offset']})<br>"
            f"Nearest Pit Anchor: {anchor_name}<br>"
            f"Hamming Dist to Anchor: {dist}/384"
            for m, dist, anchor_name in zip(background_metadata, min_hamming, nearest_anchor_name)
        ]

        fig.add_trace(
            go.Scatter(
                x=bg_2d[:, 0],
                y=bg_2d[:, 1],
                mode="markers",
                name="Background Lunar Regolith",
                marker=dict(
                    size=5,
                    color=min_hamming,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Hamming Dist<br>to Closest Pit"),
                    opacity=0.6,
                ),
                text=bg_hover,
                hoverinfo="text",
            )
        )

        # Trace 2: Pit Anchors (Highlighted)
        anchor_categories = [a["category"] for a in anchor_info]
        anchor_names = [a["name"] for a in anchor_info]

        anchor_hover = [
            f"<b>{a['category']}</b><br>"
            f"Anchor Name: {a['name']}<br>"
            f"Family ID: {a['family_id']}"
            for a in anchor_info
        ]

        fig.add_trace(
            go.Scatter(
                x=anchor_2d[:, 0],
                y=anchor_2d[:, 1],
                mode="markers+text",
                name="Pit Anchors (Zero-Shot)",
                marker=dict(
                    size=12,
                    color="red",
                    symbol="star",
                    line=dict(width=1, color="yellow"),
                ),
                text=[a["name"].split("_")[0] for a in anchor_info],
                textposition="top center",
                customdata=anchor_hover,
                hoverinfo="text",
                hovertext=anchor_hover,
            )
        )

        # Update Layout with sleek dark theme
        fig.update_layout(
            title=dict(
                text="<b>LUNA Pipeline: DINOv3 Latent Space Topology Map</b><br><sup>Zero-Shot Pit Anchors vs. Lunar Surface Tiles</sup>",
                x=0.05,
                font=dict(size=20, color="#ffffff"),
            ),
            paper_bgcolor="#111827",
            plot_bgcolor="#1f2937",
            font=dict(color="#f3f4f6"),
            xaxis=dict(
                title=f"{self.method.upper()} Dimension 1",
                gridcolor="#374151",
                zerolinecolor="#4b5563",
            ),
            yaxis=dict(
                title=f"{self.method.upper()} Dimension 2",
                gridcolor="#374151",
                zerolinecolor="#4b5563",
            ),
            legend=dict(
                bgcolor="rgba(31, 41, 55, 0.8)",
                bordercolor="#4b5563",
                borderwidth=1,
            ),
            width=1200,
            height=800,
        )

        fig.write_html(str(out_path))
        log.info("Saved interactive latent space dashboard → %s", out_path)
        return str(out_path)
