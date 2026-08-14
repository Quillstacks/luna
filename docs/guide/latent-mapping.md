# Latent Mapping & Manifolds

`luna.latent_map` provides tools for analyzing the high-dimensional feature landscape of lunar surfaces, clustering geological formations, and visualizing continuous morphological manifolds.

---

## 1. Dimensionality Reduction via UMAP

High-dimensional DINOv3 embeddings ($D=1024$) are projected to 2D/3D Riemannian manifolds via Uniform Manifold Approximation and Projection (UMAP):

- Preserves both local neighbourhood clustering and global topological relationships.
- Separates impact craters, volcanic collapse pits, graben fractures, and boulder fields into distinct topological clusters.

```python
from luna.latent_map import LatentMapProjector

projector = LatentMapProjector(n_neighbors=30, min_dist=0.1, metric="cosine")
coords_2d = projector.fit_transform(feature_matrix)
```

---

## 2. Interactive Feature Space Dashboard

`luna` includes an interactive Dash/Datashader visualizer for exploring millions of patch embeddings:

```bash
python -m luna.latent_map.dashboard --input data/embeddings/ --port 8050
```

Features:
- Live image tile previews on hover.
- Color-coding by terrain classification, solar phase angle, and confidence score.
- Dynamic bounding-box selection to isolate novel morphological candidates.
