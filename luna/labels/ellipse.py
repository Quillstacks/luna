"""Ellipse-approximated pit masks from Wagner/LPA catalog dimensions.

LPA carries ``funnel_max_m`` (long axis), ``funnel_min_m`` (short axis), and
``azimuth_deg`` (compass bearing of the long axis, 0 = north / +lat, measured
clockwise). A pit's collapse funnel is rarely a clean ellipse, but the three
numbers are the best free label we have — good enough to teach Mask R-CNN
"roughly-here, roughly-this-size, roughly-this-shape". The human-in-the-loop
relabel pass (step 4 of the main loop) is what turns this into truth.

Image convention: y grows downward, x grows rightward. Azimuth is converted
so the major axis tilts correctly under that convention (north = -y).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def ellipse_mask(
    shape: tuple[int, int],
    center_xy: tuple[float, float],
    semi_major_px: float,
    semi_minor_px: float,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """Rasterise a filled ellipse to a boolean mask of ``(H, W) = shape``.

    ``rotation_deg`` is the clockwise rotation of the major axis from the +y
    (image-down / compass-north) axis. This matches LPA's ``azimuth_deg``.
    """
    H, W = shape
    cx, cy = center_xy
    a = max(float(semi_major_px), 0.5)
    b = max(float(semi_minor_px), 0.5)

    # Rect around the rotated ellipse to avoid rasterising the whole image.
    r = max(a, b) + 1
    x0 = max(0, int(math.floor(cx - r)))
    x1 = min(W, int(math.ceil(cx + r)) + 1)
    y0 = max(0, int(math.floor(cy - r)))
    y1 = min(H, int(math.ceil(cy + r)) + 1)
    if x0 >= x1 or y0 >= y1:
        return np.zeros(shape, dtype=bool)

    ys, xs = np.mgrid[y0:y1, x0:x1]
    dx = xs - cx
    dy = ys - cy

    # Azimuth: 0 = north (image up, -y), clockwise positive. Rotate so the
    # major axis lies along the "u" coordinate below.
    theta = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # u = along major axis (starts pointing up, i.e. -y at azimuth=0)
    # v = perpendicular
    u = -dy * cos_t + dx * sin_t
    v = dy * sin_t + dx * cos_t

    mask_local = (u * u) / (a * a) + (v * v) / (b * b) <= 1.0
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = mask_local
    return mask


def pit_mask_from_lpa(
    shape: tuple[int, int],
    center_px: tuple[float, float],
    funnel_max_m: float,
    resolution_m: float,
    funnel_min_m: Optional[float] = None,
    azimuth_deg: Optional[float] = None,
) -> np.ndarray:
    """Wrapper: catalog metric dimensions -> pixel-space ellipse mask."""
    semi_major_px = 0.5 * funnel_max_m / resolution_m
    minor_m = funnel_min_m if funnel_min_m else funnel_max_m
    semi_minor_px = 0.5 * minor_m / resolution_m
    return ellipse_mask(
        shape=shape,
        center_xy=center_px,
        semi_major_px=semi_major_px,
        semi_minor_px=semi_minor_px,
        rotation_deg=azimuth_deg or 0.0,
    )
