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
from typing import Optional

import numpy as np
import pvl


# CDR sentinel DN values (per INDEX.LBL / SIS §3.3). Any pixel <= VALID_MINIMUM
# is not observational data and must be masked before normalisation.
_NULL = -32768
_VALID_MINIMUM = -32752


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
