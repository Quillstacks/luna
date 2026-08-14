# Release Notes & Changelog

All notable changes to the `luna` library are documented in this file.

---

## [Unreleased] — 2026-08-14

### Photogrammetry & SPICE Architecture Overhaul

#### 1. Targeted On-Demand SPICE Kernel Loading
* **Problem**: When processing multi-year LROC NAC images, loading all `.bc` attitude kernels simultaneously into CSPICE exceeded the internal Double Precision Array File (DAF) handle table limit, triggering:
  `spiceypy.utils.exceptions.SpiceDAFNOSUCHHANDLE: handle = 0`
* **Fix**: Added `furnish_kernels_for_date(dt, kernel_root)` in `luna.io.spice_project`.
  - Clears the kernel pool via `sp.kclear()` when switching date windows.
  - Loads timeless base kernels (`.tls`, `.tsc`, `.tpc`, `.bpc`, `.tf`, `.ti`, `de421.bsp`).
  - Selectively furnishes only the exact `.bsp` and `.bc` files covering timestamp `dt`.
  - Caches the active interval $[t_{\text{start}}, t_{\text{stop}}]$ for zero-overhead repeated queries.
  - Retained `furnish_kernels(kernel_root)` for batch single-phase workloads.

#### 2. Planetocentric Topography & Elevation Parallax Correction
* **Problem**: For off-nadir observations (emission angle $> 5^\circ$), assuming a spherical lunar datum ($alt = 0\text{ m}$) caused horizontal parallax displacements of up to $500\text{ px}$ in deep basins (e.g., Marius Hills at elevation $-2010\text{ m}$).
* **Fix**: Updated `ground_to_image(label_path, lon_deg, lat_deg, alt_m)`:
  - Added support for explicit ground elevation `alt_m` (e.g., from LOLA DEMs).
  - Converted coordinates via planetocentric spherical conversion: $\vec{G}_{bf} = \text{sp.latrec}(R_{\text{moon}} + \text{alt\_km}, \text{lon\_rad}, \text{lat\_rad})$
  - Updated `image_to_ground` with `sp.reclat(point)` for robust reverse planetocentric lookups.

#### 3. Ascending vs. Descending Orbit Handling
* **Problem**: In `LinearProjection.from_nac_geometry`, corner mapping assumed Line 0 was always the Northern edge (`lat_top > lat_bot`). For ascending orbits (South-to-North passes), Line 0 is at the Southern edge, causing negative line offsets or solver divergence in `_solve_uv`.
* **Fix**: Added automated flight-direction detection (`is_ascending`) by evaluating `start_lat`/`stop_lat`, `orbit_direction`, and comparing corner latitudes to ensure Line 0 ($v=0$) always maps to the start of exposure.

---

### Documentation & Developer Experience

* **MkDocs Material Documentation System**:
  - Full technical documentation site configured with Dark/Light mode, syntax highlighting, search, and navigation.
  - MathJax ($LaTeX$) mathematical formula rendering for orbital mechanics and photogrammetry.
  - Mermaid.js diagrams for architecture and data pipelines.
  - Automated Python API reference generation via `mkdocstrings`.
  - Added technical deep dives for SPICE, PDS range-reads, DINOv3 encoders, and the native C++ Pithos search engine.

---

## [0.0.1] — 2026-08-10

### Initial Features
* **PDS Ingestion**: Fast fixed-width `INDEX.TAB` binary search (`PDSIndex`) and automatic product fetching (`fetch_nac`).
* **Pushbroom Camera Model**: Initial pure-Python SPICE projection implementation (`ground_to_image`).
* **Bilinear Projection**: 4-corner Newton-Raphson approximation for fast spatial ROI cropping (`LinearProjection`).
* **Machine Learning Pipelines**:
  * DINOv3 feature backbone integration for lunar terrain representation.
  * Stage-2 dense prediction decoder architecture.
  * Native C++ Pithos SIMD Hamming distance index for fast candidate retrieval.
  * UMAP latent space visualization and metric estimation.
