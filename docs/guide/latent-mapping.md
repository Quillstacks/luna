# Latent Mapping & Manifold Exploration

`luna.latent_map` provides tools for analyzing the high-dimensional feature landscape ($D=384$) of lunar surface tiles, analyzing topological clustering, measuring metric preservation distortions, and generating interactive dashboards.

---

## 1. Manifold Geometry & Topology

The 384-dimensional DINOv3 feature representations are evaluated and mapped using non-linear manifold projection techniques:

- **Topological Clustering**: Evaluates neighborhood graphs and diffusion components across reference collapse pits and background lunar terrains.
- **Metric Distortion Metrics**:
    - **Johnson-Lindenstrauss (JL) Distortion**: Measures pairwise distance preservation ratio: $\text{Ratio}(u, v) = \frac{\|f(u) - f(v)\|_2}{\|u - v\|_2}$
    - **Laplace-Beltrami Spectrum**: Normalized graph Laplacian $L_{\text{sym}} = I - D^{-1/2} W D^{-1/2}$ for spectral diffusion analysis.
    - **Nash Isometric Strain & Sammon Stress**: Quantifies local vs. global projection error.

### Python API Example

```python
import numpy as np
from luna.latent_map.topology import PitTopologyAnalyzer
from luna.latent_map.parametric_umap import ParametricUMAPEncoder
from luna.latent_map.manifold_math import compute_jl_distortion_ratio

# 1. Analyze reference pit anchor topology
analyzer = PitTopologyAnalyzer()
anchor_info, anchor_embeddings = analyzer.load_and_encode_anchors("data/_scratch/pits/")

# 2. Train or evaluate parametric UMAP projection (384-d -> 2D)
umap_encoder = ParametricUMAPEncoder(input_dim=384, output_dim=2)
coords_2d = umap_encoder.fit_transform(anchor_embeddings)

# 3. Check isometric distance preservation
ratios, mean_preservation = compute_jl_distortion_ratio(anchor_embeddings, coords_2d)
print(f"Mean JL Distance Preservation: {mean_preservation:.2%}")
```

---

## 2. Interactive Dash / Datashader Dashboard

`luna` provides an interactive Dash & Datashader explorer for visualizing millions of embedded lunar tiles:

### CLI Execution

```bash
# Generate standalone interactive HTML report
python -m luna.latent_map.cli \
    --pits-dir data/_scratch/pits/ \
    --indices-dir data/_scratch/indices/ \
    --sample-size 5000 \
    --method umap \
    --out data/_scratch/luna_latent_map.html
```

Or launch the real-time Dash server:

```bash
python -m luna.latent_map.dash_app --port 8050
```

### Dashboard Capabilities
- **WebGL Scatterplot**: Interactive rendering of 2D/3D UMAP manifold projections.
- **Hover Previews**: Instant inspection of raw LROC NAC sub-tiles and associated metadata.
- **Feature Density Contours**: Kernel density estimations (KDE) highlighting high-probability morphological anomaly clusters.

