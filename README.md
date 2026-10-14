# LUNA

> **L.U.N.A.** - *Lots of Unexplained NAC Anomalies*  
> Python library for lunar planetary photogrammetry, NASA PDS data access, and vision model inference on Lunar Reconnaissance Orbiter Camera (LROC) NAC imagery.

`luna` provides standalone photogrammetry without requiring external USGS ISIS3 dependencies, resilient on-demand NASA Planetary Data System (PDS) and SLDEM2015 downloads, DINOv3 feature representations, and candidate screening engines.

---

## Installation

### Core Installation (Lightweight, no PyTorch required)
For pure photogrammetry, SPICE ray-tracing, LOLA/SLDEM2015 elevation lookups, and PDS downloads:
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

- **Pure-Python SPICE Photogrammetry (`luna.io.spice_project`)**: Sub-pixel pushbroom camera model operating in the `MOON_ME` Mean-Earth/Polar-Axis frame (configurable via `LUNA_BODY_FRAME`). Includes 1D bisection root-finding for `ground_to_image`, analytical ray-sphere intersection with iterative DEM refinement for `image_to_ground`, and structured `MissingKernelError` handling.
- **Optional Lunar Topography & DEM Integration (`luna.io.dem`)**: On-demand SLDEM2015 tile resolution (`get_sldem2015_tile_name`), regional downloading (`fetch_dem_for_region`), and fast window-read querying (`LunarDEM`, `default_dem`) with automatic GDAL `.JP2.aux.xml` vertical scale (`0.5 m/DN`) validation. Both `alt_m` and `dem` are optional across all projection routines, defaulting to the 1737.4 km lunar reference sphere.
- **PDS Index & Resilient Ingestion (`luna.io.pds_index`, `luna.io.pds_fetch`)**: Instant HTTP range-request binary search over NASA MCP PDS4 mirrors (`DEFAULT_BASE`), dynamic JPL volume boundary discovery, CSV-first `INDEX.TAB` geometry parsing with fixed-width fallback, connection pooling, token-bucket bandwidth limiting, and HTTP 429/503 retry resilience.
- **SPICE-First Projection & Orbit Handling (`luna.io.projection`)**: `get_image_of_roi` prioritizes exact SPICE `ground_to_image` projection with automatic fallback to `LinearProjection` (2D damped Newton-Raphson solver with ascending/descending orbit detection).
- **On-Demand SPICE Kernel Management (`luna.io.kernel_fetch`)**: Automatic discovery and downloading of NAIF kernels per observation timestamp (`ensure_kernels_for_date`, `furnish_kernels_for_date`), avoiding CSPICE DAF handle table exhaustion.
- **Vision Transformers (`luna.models`)**: DINOv3 ViT-S/16 feature backbones with Matryoshka representations and Stage-2 dense decoders.
- **Candidate Screening Engine (`luna.screening`)**: High-throughput candidate extraction and native C++ binary Hamming index integration (`pithosdb`).
- **Latent Manifold Analysis (`luna.latent_map`)**: Parametric UMAP projections and morphological topology metrics.
- **Physical Verification Suite (`luna.easter_egg`)**: End-to-end orbital photogrammetry and illumination verification utilities (`verify_physics`, `render_sweep`, `render_tufte`).

---

## Quickstart

### 1. PDS Ingestion & SPICE Coordinate Ray-Tracing
Download an LROC NAC product and project between lunar ground coordinates (`MOON_ME` longitude, latitude, optional elevation/DEM) and detector pixel coordinates:

```python
from luna.io.dem import LunarDEM, fetch_dem_for_region
from luna.io.pds_fetch import fetch_nac
from luna.io.spice_project import ground_to_image, image_to_ground

# 1. Fetch calibrated NAC .IMG on-demand from NASA PDS
img_path = fetch_nac("M1118880788RC", dest_dir="data/_scratch")

# 2a. Project using the default 1737.4 km lunar reference sphere (alt_m=0.0)
s_sphere, l_sphere = ground_to_image(img_path, lon_deg=303.228, lat_deg=14.091)

# 2b. Project with explicit scalar elevation (e.g., Marius Hills Skylight at -2010 m)
sample_px, line_px = ground_to_image(
    img_path,
    lon_deg=303.228,
    lat_deg=14.091,
    alt_m=-2010.0,
)
print(f"Target pixel coordinate: Sample = {sample_px:.2f}, Line = {line_px:.2f}")

# 2c. Project forward and inverse with optional SLDEM2015 LunarDEM terrain lookup
tiles = fetch_dem_for_region(min_lat=13.5, max_lat=14.5, min_lon=302.5, max_lon=304.0)
with LunarDEM(tiles) as dem:
    s_dem, l_dem = ground_to_image(img_path, lon_deg=303.228, lat_deg=14.091, dem=dem)
    lon_deg, lat_deg = image_to_ground(img_path, sample=s_dem, line=l_dem, dem=dem)
    print(f"Round-trip ground coordinate: lon={lon_deg:.5f} deg, lat={lat_deg:.5f} deg")
```

### 2. SPICE-First ROI Extraction
Extract a calibrated reflectance crop centered on lunar coordinates (prioritizing exact SPICE projection with automatic fallback to `LinearProjection`):

```python
from luna.io.projection import get_image_of_roi

tile = get_image_of_roi(
    product_id="M1118880788RC",
    lat=14.091,
    lon=303.228,
    width=512,
    height=512,
    use_spice=True,
    alt_m=-2010.0,
)
print(f"Tile shape: {tile.shape}")
```

### 3. High-Speed PDS Archive Index Resolution
Resolve any NAC product ID to its NASA PDS URL in under 50 ms without downloading full index volumes:

```python
from luna.io.pds_index import PDSIndex

idx = PDSIndex()
url = idx.url_for("M1118880788RC")
geom = idx.geometry_for("M1118880788RC")

print(f"PDS URL: {url}")
print(f"Resolution: {geom['resolution']} m/px, Incidence Angle: {geom['incidence_angle']} deg")
```

### 4. Fast Image Reading & Calibrated Reflectance (I/F)
```python
import numpy as np
from luna.io.nac_reader import read_nac

nac = read_nac("data/_scratch/M1118880788RC.IMG", geometry=True)
print(f"Dimensions: {nac.samples} x {nac.lines} px")
print(f"Mean valid reflectance (I/F): {np.nanmean(nac.pixels):.4f}")
```

---

## Package Architecture

```text
luna/
|-- io/               # PDS index resolver, streaming fetcher, SPICE photogrammetry, LunarDEM, NAC reader
|-- models/           # DINOv3 vision transformer, Stage-2 decoder, LoRA adapters [luna[ml]]
|-- screening/        # Candidate screening, ring-buffer disruptor, Pithos binary search
|-- storage/          # Vector indexing backends (FAISS, PithosStore)
|-- latent_map/       # Latent space manifold learning, parametric UMAP, topology analyzer
|-- labels/           # Lunar Pit Atlas (LPA) catalog parser and ground truth loaders
|-- easter_egg/       # Physics verification and Tufte-style SVG rendering utilities
|-- pipeline.py       # High-level end-to-end scanning pipeline [luna[ml]]
|-- ingestor.py       # Multi-threaded orbital strip ingestion and tile streaming
|-- classifier.py     # Candidate classification and false-positive filtering
|-- metrics.py        # Pipeline telemetry, throughput calculation, and band progress tracking
|-- exceptions.py     # Custom exception hierarchy (LunaError, MissingKernelError, IndexError)
`-- _compat.py        # Optional dependency checking and lazy loading utilities
```

---

## Data Sources & Attributions

- **LROC NAC Imagery**: NASA Planetary Data System (PDS) Imaging Node / ASU LROC.
- **Ephemeris & Attitude**: NASA NAIF LRO SPICE Kernel Archive.
- **Topography & Elevation (SLDEM2015)**: LOLA + SELENE Terrain Camera DEM (512 ppd), Barker et al. (2016), *Icarus*, 273, 346-355.
- **Lunar Pit Atlas (LPA)**: Wagner & Robinson (2014, 2021) catalog.

---

## Research Group & Maintainers

`luna` is developed and maintained by the **[Moonatics](https://github.com/moonatics)** research group:

- [`@F1nnSBK`](https://github.com/F1nnSBK)
- [`@schutera`](https://github.com/schutera)

---

## License

MIT License.

