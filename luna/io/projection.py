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
from .nac_reader import NACImage, read_nac
from .pds_fetch import fetch_nac

import numpy as np


def _affine_guess(
    corners_ll: np.ndarray, corners_uv: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lon = corners_ll[:, 0]
    lat = corners_ll[:, 1]
    M = np.stack([lon, lat, np.ones(4)], axis=1)
    fwd, *_ = np.linalg.lstsq(M, corners_uv, rcond=None) 
    u = corners_uv[:, 0]
    v = corners_uv[:, 1]
    Muv = np.stack([u, v, np.ones(4)], axis=1)
    inv, *_ = np.linalg.lstsq(Muv, corners_ll, rcond=None)
    return fwd.T, inv.T


@dataclass
class LinearProjection:
    """Bilinear map between NAC pixel and lon/lat using the 4 INDEX corners."""

    lon_corners: np.ndarray 
    lat_corners: np.ndarray  
    samples: int
    lines: int
    _affine_fwd: np.ndarray
    _affine_inv: np.ndarray

    @classmethod
    def from_nac_geometry(
        cls,
        geom: dict,
        samples: Optional[int] = None,
        lines: Optional[int] = None,
    ) -> "LinearProjection":
        raw_samples = samples if samples is not None else geom.get("line_samples")
        raw_lines = lines if lines is not None else geom.get("image_lines")
        
        if raw_samples is None or raw_lines is None:
            raise ValueError("Missing image dimensions (lines/samples) in geometry.")
            
        W = int(raw_samples)
        H = int(raw_lines)
        
        try:
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
        except (KeyError, TypeError) as e:
            raise ValueError(f"Missing or invalid corner coordinates in geometry: {e}")

        if np.isnan(lon_c).any() or np.isnan(lat_c).any():
            raise ValueError("NaN values found in corner coordinates.")

        uv_c = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
        ll_c = np.stack([lon_c, lat_c], axis=1)
        
        fwd_affine, inv_affine = _affine_guess(ll_c, uv_c)
        
        return cls(
            lon_corners=lon_c, lat_corners=lat_c,
            samples=W, lines=H,
            _affine_fwd=fwd_affine, _affine_inv=inv_affine,
        )

    def _bilinear_ll(self, u: float, v: float) -> tuple[float, float]:
        w = np.array([(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v])
        return float(w @ self.lon_corners), float(w @ self.lat_corners)

    def _jacobian(self, u: float, v: float) -> np.ndarray:
        dlon_du = (-(1 - v)) * self.lon_corners[0] + (1 - v) * self.lon_corners[1] + (-v) * self.lon_corners[2] + v * self.lon_corners[3]
        dlon_dv = (-(1 - u)) * self.lon_corners[0] + (-u) * self.lon_corners[1] + (1 - u) * self.lon_corners[2] + u * self.lon_corners[3]
        dlat_du = (-(1 - v)) * self.lat_corners[0] + (1 - v) * self.lat_corners[1] + (-v) * self.lat_corners[2] + v * self.lat_corners[3]
        dlat_dv = (-(1 - u)) * self.lat_corners[0] + (-u) * self.lat_corners[1] + (1 - u) * self.lat_corners[2] + u * self.lat_corners[3]
        return np.array([[dlon_du, dlon_dv], [dlat_du, dlat_dv]])

    def _solve_uv(self, lon: float, lat: float, tol: float = 1e-9, max_iter: int = 20) -> tuple[float, float]:
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


def get_image_of_roi(
    product_id: str,
    lat: float,
    lon: float,
    width: int = 256,
    height: int = 256,
) -> np.ndarray:
    """Crop a region of interest from a NAC frame centred on a lon/lat point.

    Fetches the NAC CDR from PDS if not cached locally, builds a bilinear
    projection from the INDEX geometry, converts the target coordinate to pixel
    space, and returns the surrounding ``width × height`` tile.

    Pixels outside the image boundary are padded with ``NaN``.

    Parameters
    ----------
    product_id:
        LROC NAC product ID (e.g. ``"M102285549LE"``).
    lat:
        Geodetic latitude of the ROI centre in degrees (positive north).
    lon:
        Longitude of the ROI centre in degrees (positive east, 0–360).
    width:
        Tile width in pixels (default 256).
    height:
        Tile height in pixels (default 256).

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(height, width)``.  Invalid / masked pixels
        are ``NaN``.

    Raises
    ------
    ValueError
        If the projected pixel centre lies entirely outside the NAC frame.
    """
    path = fetch_nac(product_id, dest_dir="data/_scratch")
    img: NACImage = read_nac(path, geometry=True)
    proj = LinearProjection.from_nac_geometry(
        img.geometry, samples=img.samples, lines=img.lines
    )

    u, v = proj._solve_uv(lon, lat)
    col_f = u * (proj.samples - 1)
    row_f = v * (proj.lines - 1)

    col = int(np.floor(col_f))
    row = int(np.floor(row_f))

    if (
        col < -width or col >= img.samples + width or
        row < -height or row >= img.lines + height
    ):
        raise ValueError(
            f"Coordinate (lat={lat}, lon={lon}) projects to ({col_f:.2f}, {row_f:.2f}), "
            f"which is outside the NAC frame ({img.samples} × {img.lines} px)."
        )

    r0 = row - height // 2
    c0 = col - width  // 2
    r1 = r0 + height
    c1 = c0 + width

    # Clamp to image bounds and remember how much we clipped on each side.
    r0_clamped = int(np.clip(r0, 0, img.lines))
    c0_clamped = int(np.clip(c0, 0, img.samples))
    r1_clamped = int(np.clip(r1, 0, img.lines))
    c1_clamped = int(np.clip(c1, 0, img.samples))

    crop = img.pixels[r0_clamped:r1_clamped, c0_clamped:c1_clamped]

    if crop.size == 0:
        return np.full((height, width), np.nan, dtype=np.float32)

    # Fast path: crop fits entirely inside the frame.
    if crop.shape == (height, width):
        return crop

    # Slow path: pad edges that were clipped with NaN.
    tile = np.full((height, width), np.nan, dtype=np.float32)
    dst_r = r0_clamped - r0
    dst_c = c0_clamped - c0
    tile[dst_r : dst_r + crop.shape[0], dst_c : dst_c + crop.shape[1]] = crop
    return tile


def lonlat_to_pixel(proj: LinearProjection, lon: float, lat: float) -> tuple[int, int]:
    u, v = proj._solve_uv(lon, lat)
    x = u * (proj.samples - 1)
    y = v * (proj.lines - 1)
    return int(round(x)), int(round(y))


def pixel_to_lonlat(proj: LinearProjection, x: int, y: int) -> tuple[float, float]:
    u = x / (proj.samples - 1)
    v = y / (proj.lines - 1)
    return proj._bilinear_ll(u, v)