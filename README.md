# luna

Python library for lunar pit detection, feature extraction, and orbital imagery analysis on LROC Narrow Angle Camera (NAC) products.

`luna` provides end-to-end tools for ingesting PDS orbital imagery, evaluating multi-scale vision models (DINOv3, Stage-2 Dense Decoders, Mask R-CNN), mapping high-dimensional feature spaces via UMAP latent projections, and executing candidate screening pipelines.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

Requires Python >= 3.10. PyTorch and CUDA dependencies should be pre-installed according to your environment and hardware specifications.

## Core Features and Architecture

The library is organized into specialized submodules for data ingestion, model inference, screening protocols, and latent analysis:

- `luna.io`: PDS archive resolution, LROC NAC reader, SPICE kernel fetcher, and coordinate projection utilities (`ground_to_image`).
- `luna.models`: DINOv3 feature encoders, Stage-2 dense prediction decoders, and multi-scale embedding refiners.
- `luna.screening`: Candidate generation, multi-family resonant voting, and candidate filtering protocols (`pithos`).
- `luna.storage`: Vector and metadata indexing engines (`FaissStore`, `PithosStore`).
- `luna.latent_map`: Manifold representation, topology analysis, parametric UMAP projection, and interactive feature space inspection.
- `luna.labels`: Catalog parsers for the Wagner & Robinson Lunar Pit Atlas (LPA) and rasterized geometric ground-truth labels.

## Quickstart

### Data Ingestion and SPICE Projection

Fetch LROC NAC products and project surface coordinates to image pixel coordinates:

```python
from luna.io.pds_fetch import fetch_nac
from luna.io.spice_project import ground_to_image

# Fetch NAC product
label_path, image_path = fetch_nac("M1118880788RC")

# Convert ground coordinates (longitude, latitude) to image pixels
pixel_x, pixel_y = ground_to_image(label_path, lon=87.599, lat=58.6979)
```

### Screening and Candidate Generation

Run candidate screening across orbital images:

```python
from luna.screening.pithos import PithosScreener

screener = PithosScreener()
candidates = screener.process_product("M1118880788RC")
```

### Latent Feature Mapping

Export and inspect UMAP projections of DINOv3 feature embeddings:

```bash
python -m luna.latent_map.cli --input data/embeddings/ --output data/latent_map.json
```

## Directory Layout

- `luna/`: Core Python library package.
  - `io/`: Data fetching, PDS index handling, SPICE coordinate transformations.
  - `models/`: DINOv3 encoders, Stage-2 dense decoders, and refiners.
  - `screening/`: Screening algorithms and candidate generation logic.
  - `storage/`: Vector storage backends (FAISS, Pithos metadata).
  - `latent_map/`: Topology math, manifold analysis, and UMAP visualizers.
- `scripts/`: Execution scripts for model training, dataset conversion, evaluation, and dashboard servers.
- `docs/`: API documentation and workflow reference guides.
- `catalogs/`: Lunar Pit Atlas snapshots and product lookup tables.

## Data Sources

- LROC NAC CDR/EDR: NASA Planetary Data System (PDS) LROC Node.
- SPICE Kernels: NASA NAIF LRO ephemeris and attitude archive.
- Lunar Pit Atlas (LPA): Cataloged pit locations and dimensions.

## License

MIT License.
