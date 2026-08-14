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

### PDS Ingestion & Rate-Limit (HTTP 429) Resilience

#### 1. Rate-Limit Handling & Exponential Backoff in `PDSIndex`
* **Problem**: High-frequency HTTP range requests to volume `INDEX.TAB` endpoints occasionally triggered `HTTP 429 Too Many Requests` from NASA PDS / ASU LROC servers. Missing `429` status in retry lists caused unhandled `IndexError` exceptions during binary search lookups.
* **Fix**:
  - Added `429` to `urllib3.util.Retry` `status_forcelist` with `backoff_factor=0.5` and `respect_retry_after_header=False` to prevent extended thread blocking.
  - Implemented exponential backoff in `PDSIndex._read_record` (up to 5 retries with `1.0 * 2^attempt` sleep) when reading range slices.

#### 2. Transient HTTP Error Handling in `fetch_nac`
* **Problem**: `fetch_nac` only caught network-level exceptions (`ConnectionError`, `Timeout`, `ChunkedEncodingError`), causing immediate failure on `r.raise_for_status()` when receiving transient `429` or `503` responses from CDN mirrors.
* **Fix**: Added `requests.exceptions.HTTPError` to the transient error tuple, enabling automatic exponential backoff retries and temporary file cleanup across all HTTP gateway errors.

#### 3. Session Pooling & User-Agent Customization
* Added `get_pds_session(user_agent=...)` to maintain keep-alive connection pools across consecutive product downloads.
* Introduced `DEFAULT_USER_AGENT` with support for `LUNA_USER_AGENT` environment variable overrides to prevent Cloudflare/WAF throttling on default Python `requests` headers.

#### 4. Direct URL Bypass
* Added `url` parameter to `fetch_nac`: When direct archive links are already resolved (e.g. from QuickMap or ASU metadata queries), `fetch_nac` bypasses the volume `INDEX.TAB` binary search entirely.

#### 5. Bandwidth Limiting & Atomic Stream Writes
* Added `BandwidthLimiter` (token-bucket algorithm) with `max_bandwidth_mbps` parameter to control network saturation during high-throughput downloads.
* Hardened download stream with 1MB chunk buffers, differentiated socket timeouts `(connect_timeout, read_timeout)`, console progress bar redraw throttling (`mininterval=0.2s`), and atomic `.IMG.tmp` $\to$ `.IMG` replacement on completion.

#### 6. Resilience Test Suite
* Added automated unit test suite in `luna.tests.test_pds_resilience` covering range-read recovery, retry exhaustion, and HTTP error handling in `fetch_nac`.

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
