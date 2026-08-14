# LUNA

> **L.U.N.A.** — *Lots of Unexplained NAC Anomalies*  
> Python library for lunar planetary photogrammetry, NASA PDS data access, and vision model inference on Lunar Reconnaissance Orbiter Camera (LROC) NAC imagery.

`luna` provides standalone photogrammetry without requiring external USGS ISIS3 dependencies, resilient on-demand NASA Planetary Data System (PDS) downloads, DINOv3 feature representations, and candidate screening engines.

---

## Installation

### Core Installation (Lightweight, no PyTorch required)
For pure photogrammetry, SPICE ray-tracing, LOLA elevation lookups, and PDS downloads:
```bash
pip install luna
```

### Modular Extras
Install specialized capabilities based on your workload:
```bash
# Vision Transformers & Deep Learning (DINOv3, Stage-2 Decoders, LoRA)
pip install "luna[ml]"

# Geospatial & Vector Geometry (Shapely, Rasterio, PyProj)
pip install "luna[geo]"

# Latent Manifold Learning & Topological Analysis (UMAP, Scikit-Learn)
pip install "luna[analysis]"

# Interactive Web Dashboard (Dash, Datashader)
pip install "luna[dashboard]"

# Complete Monolithic Installation
pip install "luna[all]"
```

---

## Core Features

- **Pure-Python SPICE Photogrammetry (`luna.io`)**: Sub-pixel pushbroom ray-tracing (`ground_to_image`, `image_to_ground`) via `spiceypy` with automated LOLA DEM parallax correction and ascending/descending orbit detection.
- **PDS Index & Resilient Ingestion (`luna.io`)**: Instant binary search over remote PDS `INDEX.TAB` archives, connection pooling, token-bucket bandwidth limiting, and automatic HTTP 429/503 retry resilience.
- **On-Demand SPICE Kernel Management**: Automatic discovery and downloading of NAIF kernels per observation timestamp, avoiding CSPICE DAF handle table exhaustion.
- **Vision Transformers (`luna.models`)**: DINOv3 ViT-S/16 feature backbones with Matryoshka representations and Stage-2 dense decoders.
- **Candidate Screening Engine (`luna.screening`)**: High-throughput candidate extraction and native C++ binary Hamming index integration (`pithosdb`).
- **Latent Manifold Analysis (`luna.latent_map`)**: Parametric UMAP projections and morphological topology metrics.

---

## Quickstart

### 1. PDS Ingestion & SPICE Coordinate Ray-Tracing
Download an LROC NAC product and map lunar surface coordinates (longitude, latitude, elevation) to exact detector pixel coordinates:

```python
from luna.io.pds_fetch import fetch_nac
from luna.io.spice_project import ground_to_image

# 1. Fetch calibrated NAC .IMG on-demand from NASA PDS
img_path = fetch_nac("M1118880788RC", dest_dir="data/_scratch")

# 2. Project lunar ground coordinates to detector pixel coordinates (sample, line)
# Example: Marius Hills Skylight (alt = -2010 m relative to lunar sphere)
sample_px, line_px = ground_to_image(
    img_path,
    lon_deg=303.228,
    lat_deg=14.091,
    alt_m=-2010.0
)
print(f"Target pixel coordinate: Sample = {sample_px:.2f}, Line = {line_px:.2f}")
```

### 2. High-Speed PDS Archive Index Resolution
Resolve any NAC product ID to its NASA PDS URL in $< 50\text{ ms}$ without downloading whole index volumes:

```python
from luna.io.pds_index import PDSIndex

idx = PDSIndex()
url = idx.url_for("M1118880788RC")
geom = idx.geometry_for("M1118880788RC")

print(f"PDS URL: {url}")
print(f"Resolution: {geom['resolution']} m/px, Incidence Angle: {geom['incidence_angle']}°")
```

### 3. Fast Image Reading & Calibrated Reflectance ($I/F$)
```python
import numpy as np
from luna.io.nac_reader import read_nac

nac = read_nac("data/_scratch/M1118880788RC.IMG", geometry=True)
print(f"Dimensions: {nac.samples} x {nac.lines} px")
print(f"Mean valid reflectance (I/F): {np.nanmean(nac.pixels):.4f}")
```

---

## Package Architecture

```
luna/
├── io/               # PDS index resolver, streaming fetcher, SPICE photogrammetry, NAC reader
├── models/           # DINOv3 vision transformer, Stage-2 decoder, LoRA adapters [luna[ml]]
├── screening/        # Candidate screening, ring-buffer disruptor, Pithos binary search
├── storage/          # Vector indexing backends (FAISS, PithosStore)
├── latent_map/       # Latent space manifold learning, parametric UMAP, topology analyzer
├── labels/           # Lunar Pit Atlas (LPA) catalog parser & ground truth loaders
├── geometry.py       # Bilinear approximations and coordinate utilities
├── pipeline.py       # High-level end-to-end scanning pipeline [luna[ml]]
└── _compat.py        # Dependency checking and lazy loading utilities
```

---

## Data Sources & Attributions

- **LROC NAC Imagery**: NASA Planetary Data System (PDS) Imaging Node / ASU LROC.
- **Ephemeris & Attitude**: NASA NAIF LRO SPICE Kernel Archive.
- **Lunar Pit Atlas (LPA)**: Wagner & Robinson (2014, 2021) catalog.

---

## License

MIT License.
