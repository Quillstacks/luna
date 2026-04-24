"""Lon/lat <-> pixel conversion for LROC NAC frames.

NAC corners from INDEX.TAB are labeled by instrument row/column: ``upper_*``
= line 0 (start of exposure), ``lower_*`` = last line, ``*_left`` = sample 0,
``*_right`` = last sample. Because LRO's orbit is inclined and meridians
converge away from the equator, the image maps to a *trapezoid* in (lon, lat)
for long strips — an affine (parallelogram) fit leaves up to ~180 px residual.

We use a bilinear map parametrised by unit image coords ``(u, v) in [0,1]²``:

    lon(u, v) = (1-u)(1-v) UL + u(1-v) UR + (1-u)v LL + u v LR
    lat(u, v) = (same with lat corners)
    x = u (W-1),   y = v (H-1)

Bilinear is exact for all 4 corners. Forward (u,v) -> (lon,lat) is trivial.
Inverse (lon,lat) -> (u,v) is nonlinear; we solve by 2D Newton starting from
the affine estimate — converges in 2–3 iterations to sub-pixel accuracy at
NAC scales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _affine_guess(
    corners_ll: np.ndarray, corners_uv: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares affine (lon,lat)->(u,v) and (u,v)->(lon,lat) from 4 corners."""
    lon = corners_ll[:, 0]
    lat = corners_ll[:, 1]
    M = np.stack([lon, lat, np.ones(4)], axis=1)
    fwd, *_ = np.linalg.lstsq(M, corners_uv, rcond=None)  # (3, 2)
    u = corners_uv[:, 0]
    v = corners_uv[:, 1]
    Muv = np.stack([u, v, np.ones(4)], axis=1)
    inv, *_ = np.linalg.lstsq(Muv, corners_ll, rcond=None)  # (3, 2)
    return fwd.T, inv.T  # each (2, 3)


@dataclass
class LinearProjection:
    """Bilinear map between NAC pixel and lon/lat using the 4 INDEX corners."""

    # Corner arrays, row order: UL, UR, LL, LR
    lon_corners: np.ndarray  # shape (4,)
    lat_corners: np.ndarray  # shape (4,)
    samples: int
    lines: int
    # Cached affine for warm-starting the Newton solver.
    _affine_fwd: np.ndarray  # (2, 3): (lon,lat,1) -> (u,v)
    _affine_inv: np.ndarray  # (2, 3): (u,v,1) -> (lon,lat)

    @classmethod
    def from_nac_geometry(
        cls,
        geom: dict,
        samples: Optional[int] = None,
        lines: Optional[int] = None,
    ) -> "LinearProjection":
        W = int(samples if samples is not None else geom["line_samples"])
        H = int(lines if lines is not None else geom["image_lines"])
        lon_c = np.array([
            geom["upper_left_longitude"],
            geom["upper_right_longitude"],
            geom["lower_left_longitude"],
            geom["lower_right_longitude"],
        ], dtype=np.float64)
        lat_c = np.array([
            geom["upper_left_latitude"],
            geom["upper_right_latitude"],
            geom["lower_left_latitude"],
            geom["lower_right_latitude"],
        ], dtype=np.float64)
        uv_c = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
        ll_c = np.stack([lon_c, lat_c], axis=1)
        fwd_affine, inv_affine = _affine_guess(ll_c, uv_c)
        return cls(
            lon_corners=lon_c, lat_corners=lat_c,
            samples=W, lines=H,
            _affine_fwd=fwd_affine, _affine_inv=inv_affine,
        )

    # -- forward (u,v)->(lon,lat) ---------------------------------------

    def _bilinear_ll(self, u: float, v: float) -> tuple[float, float]:
        w = np.array([(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v])
        return float(w @ self.lon_corners), float(w @ self.lat_corners)

    def _jacobian(self, u: float, v: float) -> np.ndarray:
        # d(lon,lat)/d(u,v) at (u,v)
        dlon_du = (-(1 - v)) * self.lon_corners[0] + (1 - v) * self.lon_corners[1] + (-v) * self.lon_corners[2] + v * self.lon_corners[3]
        dlon_dv = (-(1 - u)) * self.lon_corners[0] + (-u) * self.lon_corners[1] + (1 - u) * self.lon_corners[2] + u * self.lon_corners[3]
        dlat_du = (-(1 - v)) * self.lat_corners[0] + (1 - v) * self.lat_corners[1] + (-v) * self.lat_corners[2] + v * self.lat_corners[3]
        dlat_dv = (-(1 - u)) * self.lat_corners[0] + (-u) * self.lat_corners[1] + (1 - u) * self.lat_corners[2] + u * self.lat_corners[3]
        return np.array([[dlon_du, dlon_dv], [dlat_du, dlat_dv]])

    def _solve_uv(self, lon: float, lat: float, tol: float = 1e-9, max_iter: int = 20) -> tuple[float, float]:
        # Affine warm start.
        uv = self._affine_fwd @ np.array([lon, lat, 1.0])
        u, v = float(uv[0]), float(uv[1])
        for _ in range(max_iter):
            lon_k, lat_k = self._bilinear_ll(u, v)
            r = np.array([lon - lon_k, lat - lat_k])
            if np.max(np.abs(r)) < tol:
                return u, v
            J = self._jacobian(u, v)
            du, dv = np.linalg.solve(J, r)
            u += du
            v += dv
        return u, v


def lonlat_to_pixel(proj: LinearProjection, lon: float, lat: float) -> tuple[int, int]:
    u, v = proj._solve_uv(lon, lat)
    x = u * (proj.samples - 1)
    y = v * (proj.lines - 1)
    return int(round(x)), int(round(y))


def pixel_to_lonlat(proj: LinearProjection, x: int, y: int) -> tuple[float, float]:
    u = x / (proj.samples - 1)
    v = y / (proj.lines - 1)
    return proj._bilinear_ll(u, v)
