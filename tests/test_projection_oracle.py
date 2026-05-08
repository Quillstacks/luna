"""Sanity-check the frozen projection oracle.

The oracle (built by ``scripts/build_projection_oracle.py``) is the ground
truth that every GPU/Triton implementation in subsequent phases must match
to <0.3 px max error. This test only validates the oracle's own internal
consistency — the actual GPU comparison lives in ``test_projection_gpu.py``
once Phase 2 lands.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ORACLE_PATH = Path(__file__).resolve().parent / "data" / "projection_ground_truth.npz"


def _load_oracle():
    if not ORACLE_PATH.exists():
        pytest.skip(f"oracle not built; run scripts/build_projection_oracle.py")
    return np.load(ORACLE_PATH, allow_pickle=False)


def test_oracle_shapes_consistent():
    """All per-frame arrays share the same (F, N, N) layout."""
    z = _load_oracle()
    F = len(z["products"])
    assert F >= 1
    for key in ("lon_grids", "lat_grids", "samples", "lines"):
        arr = z[key]
        assert arr.shape[0] == F, f"{key} frame count {arr.shape[0]} != {F}"
        assert arr.ndim == 3 and arr.shape[1] == arr.shape[2], (
            f"{key} not (F, N, N): {arr.shape}"
        )


def test_oracle_grid_strictly_monotonic():
    """Lon/lat grids should be regular meshgrids — no NaN, monotonic per row/col."""
    z = _load_oracle()
    for f, pid in enumerate(z["products"]):
        lon = z["lon_grids"][f]
        lat = z["lat_grids"][f]
        assert np.isfinite(lon).all(), f"{pid}: NaN in lon grid"
        assert np.isfinite(lat).all(), f"{pid}: NaN in lat grid"
        # First row of lon should be strictly increasing left→right (with meshgrid xy).
        assert np.all(np.diff(lon[0, :]) > 0), f"{pid}: lon not monotonic across columns"
        # First column of lat should be strictly increasing top→bottom.
        assert np.all(np.diff(lat[:, 0]) > 0), f"{pid}: lat not monotonic across rows"


def test_oracle_majority_in_window():
    """Most grid points should land inside the NAC exposure window.

    The grid is sampled inside the inner 80% of each footprint, so we expect
    high in-window fractions (typically >70%). Anything below 50% suggests the
    footprint corners and the SPICE pose disagree — worth flagging.
    """
    z = _load_oracle()
    for f, pid in enumerate(z["products"]):
        samples = z["samples"][f]
        in_window = np.isfinite(samples).sum() / samples.size
        assert in_window > 0.5, (
            f"{pid}: only {in_window:.1%} of grid in exposure window — "
            "footprint/pose mismatch?"
        )


def test_oracle_samples_within_detector():
    """Returned (sample, line) coords should fall within the NAC detector bounds.

    NAC detector is 5064 samples wide; line count varies per frame. We allow a
    small over-edge tolerance because the bisection can return slightly outside
    the window before the caller decides to clip.
    """
    z = _load_oracle()
    for f, pid in enumerate(z["products"]):
        samples = z["samples"][f]
        lines = z["lines"][f]
        valid = np.isfinite(samples) & np.isfinite(lines)
        if not valid.any():
            continue
        s_in = samples[valid]
        # NAC detector is 5064 samples; allow ±20 px guardband for edge bisection.
        assert s_in.min() > -20 and s_in.max() < 5084, (
            f"{pid}: sample out of detector range [{s_in.min()}, {s_in.max()}]"
        )
        l_in = lines[valid]
        assert l_in.min() > -20, f"{pid}: line < 0 ({l_in.min()})"
