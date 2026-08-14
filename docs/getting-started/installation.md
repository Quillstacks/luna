# Installation & Environment Setup

This guide walks through setting up `luna` for local development, high-performance model training, and SPICE-based orbital photogrammetry.

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

=== "Core + Development + Docs"

    ```bash
    uv pip install -e ".[dev,docs,dashboard]"
    ```

=== "Production / Minimal"

    ```bash
    uv pip install -e "."
    ```

### Optional Dependency Groups

| Extra Group | Included Packages | Purpose |
| :--- | :--- | :--- |
| `dev` | `pytest`, `pydeps`, `vulture` | Unit testing and dead code analysis |
| `docs` | `mkdocs-material`, `mkdocstrings[python]` | Documentation building & local preview |
| `dashboard` | `dash`, `datashader` | Interactive latent space visualizers |

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
