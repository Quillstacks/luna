"""Build the projection ground-truth oracle for GPU validation.

Picks N diverse NAC products from ``catalogs/pit_nacs.json``, fetches the CDR
.IMG (which carries the PVL label needed by SPICE), ensures the SPICE kernels
for the exposure date, and runs the existing CPU
``luna.io.spice_project.ground_to_image`` over a regular (lon, lat) grid
inside each footprint. The result is frozen as
``tests/data/projection_ground_truth.npz``.

Every subsequent GPU implementation must reproduce these (sample, line)
arrays to <0.3 px max error. No GPU dependency to build the oracle — runs
on a Mac.

Usage:
    python scripts/build_projection_oracle.py
    python scripts/build_projection_oracle.py --grid 128 --frames 5
    python scripts/build_projection_oracle.py --no-fetch  # use cached IMGs only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ALE/scipy compatibility shim mirrored from build_ellipse_dataset.py.
try:
    from scipy.spatial.transform import Rotation as _R
    if not hasattr(_R, "as_dcm"):
        _R.as_dcm = _R.as_matrix
    if not hasattr(_R, "from_dcm"):
        _R.from_dcm = _R.from_matrix
except Exception:
    pass

from luna.io import fetch_nac, read_nac
from luna.io.spice_project import ensure_kernels_for_label, ground_to_image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "oracle_cdrs"
OUT_PATH = ROOT / "tests" / "data" / "projection_ground_truth.npz"
META_PATH = ROOT / "tests" / "data" / "projection_ground_truth.json"

log = logging.getLogger("oracle")


def pick_diverse_products(catalog_path: Path, n: int) -> list[str]:
    """Pick ``n`` distinct NAC product IDs spanning L/R sides and varied dates.

    NAC product IDs encode acquisition time in the numeric portion (M<digits><L|R>),
    so spread across the digit range buys date diversity for free.
    """
    with open(catalog_path) as f:
        catalog = json.load(f)
    seen: dict[str, str] = {}  # pid -> first pit it appeared under (debug only)
    for pit_id, entries in catalog.items():
        for e in entries:
            pid = e["product"]
            if pid not in seen:
                seen[pid] = pit_id
    pids = sorted(seen.keys())
    # Stratify: alternate L/R, then sample evenly across the sorted list.
    left = [p for p in pids if p.endswith("L")]
    right = [p for p in pids if p.endswith("R")]
    out: list[str] = []
    for pool in (left, right, left, right, left, right):
        if not pool or len(out) >= n:
            continue
        # Take from start/end/middle to spread acquisition dates.
        idx = (len(pool) * len(out)) // max(n, 1)
        cand = pool[idx % len(pool)]
        if cand not in out:
            out.append(cand)
        if len(out) >= n:
            break
    # Fill the rest from any remaining unique pids.
    for p in pids:
        if len(out) >= n:
            break
        if p not in out:
            out.append(p)
    return out[:n]


def fetch_label(product_id: str, fetch: bool) -> Optional[Path]:
    """Return the local CDR .IMG path (which carries the PVL label inside it)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # PDSIndex resolves ``M..R`` to ``M..RC.IMG`` (CDR variant).
    if not fetch:
        # Best-effort: look for an already-cached .IMG.
        existing = list(DATA_DIR.glob(f"{product_id}*.IMG"))
        return existing[0] if existing else None
    try:
        return fetch_nac(product_id, dest_dir=DATA_DIR)
    except Exception as e:
        log.warning("fetch failed for %s: %s", product_id, e)
        return None


def build_grid(footprint: dict, n: int, margin: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Return (lon_grid, lat_grid), each shape (n, n), inside the footprint bbox.

    ``margin`` shrinks each side by that fraction to avoid the 80-iter bisection
    failing on points right at the exposure-window edge.
    """
    lons = np.array([footprint["upper_left"][0], footprint["upper_right"][0],
                     footprint["lower_left"][0], footprint["lower_right"][0]],
                    dtype=np.float64)
    lats = np.array([footprint["upper_left"][1], footprint["upper_right"][1],
                     footprint["lower_left"][1], footprint["lower_right"][1]],
                    dtype=np.float64)
    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    lon0 = lon_min + margin * lon_span
    lon1 = lon_max - margin * lon_span
    lat0 = lat_min + margin * lat_span
    lat1 = lat_max - margin * lat_span
    lon_grid, lat_grid = np.meshgrid(
        np.linspace(lon0, lon1, n),
        np.linspace(lat0, lat1, n),
        indexing="xy",
    )
    return lon_grid.astype(np.float64), lat_grid.astype(np.float64)


def project_grid(label_path: Path, lon_grid: np.ndarray, lat_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Run ground_to_image over the grid. Out-of-window points -> NaN."""
    n_total = lon_grid.size
    samples = np.full(n_total, np.nan, dtype=np.float64)
    lines = np.full(n_total, np.nan, dtype=np.float64)
    n_ok = 0
    flat_lon = lon_grid.ravel()
    flat_lat = lat_grid.ravel()
    t0 = time.time()
    for i in range(n_total):
        try:
            s, l = ground_to_image(label_path, float(flat_lon[i]), float(flat_lat[i]))
            samples[i] = s
            lines[i] = l
            n_ok += 1
        except ValueError:
            pass  # outside exposure window — leave NaN
        if i and i % 1000 == 0:
            rate = i / (time.time() - t0)
            log.info("  %d/%d (%.0f px/s, %.1f%% in-window)", i, n_total, rate, 100 * n_ok / max(i, 1))
    return samples.reshape(lon_grid.shape), lines.reshape(lon_grid.shape), n_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--grid", type=int, default=128,
                    help="grid resolution per side (default 128 → 16384 points/frame)")
    ap.add_argument("--frames", type=int, default=5, help="number of NAC products")
    ap.add_argument("--no-fetch", action="store_true",
                    help="skip download; only use already-cached .IMG files")
    ap.add_argument("--catalog", type=Path, default=ROOT / "catalogs" / "pit_nacs.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    pids = pick_diverse_products(args.catalog, args.frames)
    log.info("selected products: %s", pids)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    products: list[str] = []
    label_paths: list[str] = []
    lon_stack: list[np.ndarray] = []
    lat_stack: list[np.ndarray] = []
    sample_stack: list[np.ndarray] = []
    line_stack: list[np.ndarray] = []
    meta: list[dict] = []

    for pid in pids:
        log.info("=== %s ===", pid)
        path = fetch_label(pid, fetch=not args.no_fetch)
        if path is None:
            log.warning("skipping %s: no local IMG and fetch disabled/failed", pid)
            continue
        try:
            img = read_nac(path, geometry=True)
            ensure_kernels_for_label(path)
        except Exception as e:
            log.warning("skipping %s: setup failed: %s", pid, e)
            continue
        if img.footprint is None or any(v is None for c in img.footprint.values() for v in c):
            log.warning("skipping %s: missing footprint corners", pid)
            continue
        lon_grid, lat_grid = build_grid(img.footprint, args.grid)
        samples, lines, n_ok = project_grid(path, lon_grid, lat_grid)
        if n_ok < 0.5 * lon_grid.size:
            log.warning("only %d/%d in-window — keeping anyway for diagnostics",
                        n_ok, lon_grid.size)
        products.append(pid)
        label_paths.append(str(path.relative_to(ROOT)))
        lon_stack.append(lon_grid)
        lat_stack.append(lat_grid)
        sample_stack.append(samples)
        line_stack.append(lines)
        meta.append({
            "product_id": pid,
            "label_path": str(path.relative_to(ROOT)),
            "lines": img.lines,
            "samples": img.samples,
            "footprint": img.footprint,
            "center_lon": img.center_lon,
            "center_lat": img.center_lat,
            "resolution_m": img.resolution_m,
            "in_window_count": int(n_ok),
            "grid": args.grid,
        })

    if not products:
        log.error("no frames built — nothing to save")
        return 2

    np.savez_compressed(
        OUT_PATH,
        products=np.array(products),
        label_paths=np.array(label_paths),
        lon_grids=np.stack(lon_stack),
        lat_grids=np.stack(lat_stack),
        samples=np.stack(sample_stack),
        lines=np.stack(line_stack),
    )
    META_PATH.write_text(json.dumps(meta, indent=2, default=str))
    log.info("wrote %s (%d frames, %d×%d grid)",
             OUT_PATH, len(products), args.grid, args.grid)
    log.info("wrote %s", META_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
