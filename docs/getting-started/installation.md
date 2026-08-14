# Installation & Environment Setup

This guide walks through setting up `luna` for local development, model training, and SPICE-based orbital photogrammetry.

---

## 1. Prerequisites

- **Python**: `>= 3.10`
- **Package Manager**: [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip` / `conda`
- **C/C++ Build Tools**: `clang` / `gcc` for compiling Cython modules and native C++ Pithos libraries
- **Operating System**: macOS (Apple Silicon / Intel) or Linux (Ubuntu 20.04+)

---

## 2. Virtual Environment Setup

Using `uv`:

```bash
# Create virtual environment
uv venv .venv --python 3.12

# Activate virtual environment
source .venv/bin/activate  # macOS / Linux
# or: .venv\Scripts\activate  # Windows
```

---

## 3. Package Installation

Install `luna` in editable mode with optional extras:

=== "Core Photogrammetry Only"

    ```bash
    pip install luna
    ```

=== "Machine Learning & Vision Models"

    ```bash
    pip install "luna[ml]"
    ```

=== "Full Research Installation"

    ```bash
    pip install "luna[all,dev,docs]"
    ```

### Modular Dependency Groups

| Extra Group | Included Packages | Purpose |
| :--- | :--- | :--- |
| `ml` | `torch`, `torchvision`, `transformers`, `peft`, `pycocotools` | DINOv3 Vision Transformers, LoRA weights, Stage-2 decoders |
| `geo` | `shapely`, `pyproj`, `rasterio` | Geospatial polygon clipping, raster reprojection |
| `analysis` | `umap-learn`, `scikit-learn`, `numba`, `matplotlib` | Parametric UMAP projections, manifold topology |
| `screening` | `psutil` | High-throughput candidate ring-buffer screening |
| `dashboard` | `dash`, `datashader`, `pandas` | Interactive latent space web dashboard |
| `all` | All optional packages bundled | Full monolithic installation |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `mypy` | Unit testing, coverage, linting & type checks |
| `docs` | `mkdocs-material`, `mkdocstrings[python]` | Documentation building & local preview |

### Vector Index Backend (`pithosdb`)

For fast binary embedding indexing, install the standalone Pithos engine:

```bash
pip install pithosdb
```

---

## 4. Hardware & GPU Acceleration

For DINOv3 feature encoding and Stage-2 Mask R-CNN decoders, ensure PyTorch detects your hardware accelerator:

=== "NVIDIA CUDA (Linux)"

    ```bash
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    ```

=== "Apple Silicon MPS (macOS)"

    PyTorch included via standard installation supports Metal Performance Shaders (MPS) natively:
    ```python
    import torch
    print("MPS Available:", torch.backends.mps.is_available())
    ```

---

## 5. SPICE Kernel Archive Setup

`luna` automatically downloads missing NAIF SPICE kernels on-demand via `luna.io.kernel_fetch.ensure_kernels_for_date`. 

The local kernel directory structure is stored under:
```text
data/spice/lro/
├── lsk/          # Leapsecond kernels (.tls)
├── sclk/         # Spacecraft clock kernels (.tsc)
├── pck/          # Planetary constants (.bpc, .tpc)
├── fk/           # Frame definitions (.tf)
├── ik/           # Instrument kernels (.ti)
├── spk/          # Spacecraft & planetary ephemeris (.bsp)
├── ck/           # Spacecraft & instrument attitude (.bc)
└── mk/           # NAIF year metakernels (.tm)
```

You can pre-warm kernels for a specific date:

```python
from datetime import datetime, timezone
from luna.io.kernel_fetch import ensure_kernels_for_date

# Pre-fetch all kernels needed for April 2012
dt = datetime(2012, 4, 15, tzinfo=timezone.utc)
paths = ensure_kernels_for_date(dt)
print(f"Furnished {len(paths)} kernels for {dt.date()}")
```
