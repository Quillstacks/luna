"""Read a LROC NAC CDR ``.IMG`` file into a numpy array.

NAC CDRs ship with an attached PVL label at the head of the file followed by
the pixel raster. The label tells us record size, line count, sample count,
sample bit depth, and sentinel values for NULL / saturation. CDRs are 16-bit
signed integers; multiplying by ``SCALING_FACTOR`` gives I/F reflectance.

Geometry (center lat/lon, ground resolution, footprint corners) is NOT in the
CDR label — it's only available from the per-volume ``INDEX.TAB`` or from
SPICE. If ``geometry=True``, :class:`luna.io.pds_index.PDSIndex` is used to
fetch the geometry row over HTTP (one ~17-step range-read).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
from luna.config import LROC_VALID_MIN

import requests
import numpy as np
import pvl


# CDR sentinel DN values (per INDEX.LBL / SIS §3.3). Any pixel <= VALID_MINIMUM
# is not observational data and must be masked before normalisation.
_NULL = -32768
_VALID_MINIMUM = LROC_VALID_MIN

_QUICKMAP_NAC_SEARCH_ENDPOINT = "https://lroc-tiles.quickmap.io/fcgi-bin/fprovweb.exe"

@dataclass
class NACImage:
    """A NAC CDR frame in memory."""

    pixels: np.ndarray          # shape (lines, samples), float32; invalid pixels = NaN
    lines: int
    samples: int
    product_id: str
    center_lon: Optional[float]
    center_lat: Optional[float]
    resolution_m: Optional[float]   # meters per pixel on the ground
    footprint: Optional[dict] = None  # upper_left/right, lower_left/right lat/lon
    geometry: dict = field(default_factory=dict)  # full INDEX row if fetched
    label: dict = field(default_factory=dict)     # raw PVL, for debugging


def _label_byte_count(label: pvl.PVLModule) -> int:
    record_bytes = int(label["RECORD_BYTES"])
    label_records = int(label.get("LABEL_RECORDS", 1))
    return record_bytes * label_records


def _attach_geometry(img: NACImage, geom: dict) -> None:
    img.geometry = geom
    img.center_lon = geom.get("center_longitude")
    img.center_lat = geom.get("center_latitude")
    img.resolution_m = geom.get("resolution")
    img.footprint = {
        "upper_left": (geom.get("upper_left_longitude"), geom.get("upper_left_latitude")),
        "upper_right": (geom.get("upper_right_longitude"), geom.get("upper_right_latitude")),
        "lower_left": (geom.get("lower_left_longitude"), geom.get("lower_left_latitude")),
        "lower_right": (geom.get("lower_right_longitude"), geom.get("lower_right_latitude")),
    }

def _query_quickmap(spoly: str) -> list[dict[str, Any]]:
    """Send a polygon footprint query to the QuickMap COGNAC16 service.

    Parameters
    ----------
    spoly:
        Flat, comma-separated coordinate string in the form
        ``lon0,lat0,lon1,lat1,...,lon0,lat0`` (closed ring, longitude first).

    Returns
    -------
    list[dict]
        The ``"features"`` array from the QuickMap response.

    Raises
    ------
    requests.HTTPError
        If the service returns a non-2xx status code.
    """
    response = requests.get(
        _QUICKMAP_NAC_SEARCH_ENDPOINT,
        params={
            "_xtype": "text/plain",
            "dsource": "cognac16",
            "spoly": spoly,
            "bodyview": "lunar-lonlat",
            "cmd_script": "searchpoly_v0",
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise ValueError(f"QuickMap API error: {data['error']}. Please try drawing a smaller region of interest.")
    if "features" not in data:
        raise KeyError(f"Unexpected response format from QuickMap: {data}")
    return data["features"]


def get_nacs_from_polygon(
    coords: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    """Return QuickMap features intersecting an arbitrary polygon on the Moon.

    Parameters
    ----------
    coords:
        Sequence of ``(lon, lat)`` tuples defining the polygon vertices.
        At least three points are required. The ring is closed automatically
        if the first and last points differ.

    Raises
    ------
    ValueError
        If fewer than three coordinate pairs are supplied.
    """
    if len(coords) < 3:
        raise ValueError(f"A polygon requires at least 3 points, got {len(coords)}.")

    import logging
    from shapely.geometry import Polygon, box
    from shapely.validation import make_valid
    import numpy as np

    log = logging.getLogger("luna.io.nac_reader")

    # Normalize longitudes to -180 to 180 to prevent wrap-around issues
    norm_coords = []
    for lon, lat in coords:
        norm_lon = (lon + 180) % 360 - 180
        norm_coords.append((norm_lon, lat))

    roi_poly = Polygon(norm_coords)
    if not roi_poly.is_valid:
        roi_poly = make_valid(roi_poly)

    # Calculate bounding box dimensions
    min_lon, min_lat, max_lon, max_lat = roi_poly.bounds
    width = max_lon - min_lon
    height = max_lat - min_lat

    # Subdivide if the region spans more than 2.0 degrees in either dimension
    GRID_CELL_SIZE = 2.0
    if width > GRID_CELL_SIZE or height > GRID_CELL_SIZE:
        log.info(
            "ROI bounding box is large (%.2f° x %.2f°). Subdividing into %s° sub-grid cells to bypass API limits...",
            width,
            height,
            GRID_CELL_SIZE
        )

        lon_steps = int(np.ceil(width / GRID_CELL_SIZE))
        lat_steps = int(np.ceil(height / GRID_CELL_SIZE))

        all_features = []
        seen_pids = set()

        for i in range(lon_steps):
            cell_min_lon = min_lon + i * GRID_CELL_SIZE
            cell_max_lon = min(min_lon + (i + 1) * GRID_CELL_SIZE, max_lon)

            for j in range(lat_steps):
                cell_min_lat = min_lat + j * GRID_CELL_SIZE
                cell_max_lat = min(min_lat + (j + 1) * GRID_CELL_SIZE, max_lat)

                cell_poly = box(cell_min_lon, cell_min_lat, cell_max_lon, cell_max_lat)
                if not cell_poly.intersects(roi_poly):
                    continue

                intersect_poly = cell_poly.intersection(roi_poly)
                if intersect_poly.is_empty:
                    continue

                # Extract coordinates for QuickMap query
                if intersect_poly.geom_type == 'Polygon':
                    sub_coords = list(intersect_poly.exterior.coords)
                else:
                    sub_coords = [
                        (cell_min_lon, cell_min_lat),
                        (cell_max_lon, cell_min_lat),
                        (cell_max_lon, cell_max_lat),
                        (cell_min_lon, cell_max_lat)
                    ]

                spoly = ",".join(f"{lon},{lat}" for lon, lat in sub_coords)
                try:
                    features = _query_quickmap(spoly)
                    for feat in features:
                        pid = feat.get("properties", {}).get("label")
                        if pid and pid not in seen_pids:
                            seen_pids.add(pid)
                            all_features.append(feat)
                except Exception as e:
                    log.warning(
                        "Sub-grid QuickMap query failed for cell [%.2f, %.2f]: %s",
                        cell_min_lon,
                        cell_min_lat,
                        e
                    )

        log.info(
            "Sub-grid query complete. Aggregated %d unique candidate features.",
            len(all_features)
        )
        return all_features

    # Single-cell query path
    ring = list(norm_coords)
    if ring[0] != ring[-1]:
        ring.append(ring[0])

    spoly = ",".join(f"{lon},{lat}" for lon, lat in ring)
    return _query_quickmap(spoly)


def read_nac(path: str | Path, geometry: bool = False) -> NACImage:
    """Parse a NAC CDR .IMG file into a NACImage.

    If ``geometry=True``, fetches footprint + resolution from PDS INDEX.TAB
    via :class:`luna.io.pds_index.PDSIndex`.
    """
    path = Path(path)
    with open(path, "rb") as f:
        label = pvl.load(f)
        f.seek(_label_byte_count(label))
        img_block = label["IMAGE"]
        lines = int(img_block["LINES"])
        samples = int(img_block["LINE_SAMPLES"])
        sample_bits = int(img_block["SAMPLE_BITS"])
        sample_type = str(img_block["SAMPLE_TYPE"])
        dtype = _numpy_dtype(sample_bits, sample_type)
        raw = np.frombuffer(f.read(lines * samples * dtype.itemsize), dtype=dtype)
        raw = raw.reshape(lines, samples)

    invalid = raw <= _VALID_MINIMUM  # NULL / saturation sentinels
    pixels = raw.astype(np.float32)
    pixels[invalid] = np.nan

    scaling = img_block.get("SCALING_FACTOR")
    offset = img_block.get("OFFSET")
    if scaling is not None:
        pixels = pixels * float(getattr(scaling, "value", scaling))
    if offset is not None:
        pixels = pixels + float(getattr(offset, "value", offset))
    # Normalise to [0, 1] on valid pixels only, so sentinels can't skew scale.
    p_max = float(np.nanmax(pixels)) if np.isfinite(pixels).any() else 1.0
    if p_max > 0:
        pixels = pixels / p_max

    product_id = str(label.get("PRODUCT_ID", path.stem))
    img = NACImage(
        pixels=pixels,
        lines=lines,
        samples=samples,
        product_id=product_id,
        center_lon=None,
        center_lat=None,
        resolution_m=None,
        label=dict(label),
    )
    if geometry:
        from .pds_index import PDSIndex  # lazy to avoid import cycle
        _attach_geometry(img, PDSIndex().geometry_for(product_id))
    return img


def _numpy_dtype(bits: int, sample_type: str) -> np.dtype:
    sample_type = sample_type.upper()
    signed = "SIGNED" in sample_type or "INTEGER" in sample_type and "UNSIGNED" not in sample_type
    msb = "MSB" in sample_type
    kind = "i" if signed else "u"
    byte_order = ">" if msb else "<"
    return np.dtype(f"{byte_order}{kind}{bits // 8}")
