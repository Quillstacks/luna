"""DEM (Digital Elevation Model) fetching and querying for lunar surface coordinates.

Provides automated lookup, download, and georeferenced querying of NASA LOLA /
Kaguya SLDEM2015 tiles (512 pixels/degree, ~59 m/pixel) directly from the PDS
archive. All DEM utilities are optional and only required when terrain-corrected
elevations are desired over the standard 1737.4 km lunar reference sphere.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import requests
from tqdm import tqdm

from luna._compat import require_extra

log = logging.getLogger(__name__)

SLDEM2015_BASE_URL = "http://imbrium.mit.edu/DATA/SLDEM2015/TILES/JP2"
# The JP2 tiles store int16 DN with a 0.5 m scale factor that is ONLY declared in the
# GDAL sidecar (``*_AUX.XML``, <Scale>0.5</Scale>). Without it rasterio reports scale=1.0
# and every elevation comes out 2x too large.
SLDEM2015_AUX_BASE_URL = (
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
    "lrolol_1xxx/data/sldem2015/tiles/jp2"
)
DEFAULT_DEM_DIR = Path("data/dem")

_DEFAULT_DEM: LunarDEM | None = None


def get_sldem2015_tile_name(lat_deg: float, lon_deg: float) -> str:
    """Determine the SLDEM2015 512 ppd tile name for given planetocentric coordinates.

    Args:
        lat_deg: Planetocentric latitude in degrees (-60 to +60 for SLDEM2015 tiles).
        lon_deg: Planetocentric longitude in degrees (0 to 360, or -180 to +180).

    Returns:
        Tile filename stem without extension (e.g., ``"SLDEM2015_512_00N_30N_270_315"``).

    Raises:
        ValueError: If ``lat_deg`` is outside the ``[-60, +60]`` SLDEM2015 coverage bounds
            or if either coordinate is NaN.

    Example:
        >>> from luna.io.dem import get_sldem2015_tile_name
        >>> get_sldem2015_tile_name(14.09, 303.23)
        'SLDEM2015_512_00N_30N_270_315'
    """
    if math.isnan(lat_deg) or math.isnan(lon_deg):
        raise ValueError(f"Coordinates cannot be NaN (lat={lat_deg}, lon={lon_deg})")

    lon_norm = float(lon_deg) % 360.0

    # Determine latitude band (30-degree bins)
    if -60.0 <= lat_deg < -30.0:
        lat_str = "60S_30S"
    elif -30.0 <= lat_deg < 0.0:
        lat_str = "30S_00S"
    elif 0.0 <= lat_deg < 30.0:
        lat_str = "00N_30N"
    elif 30.0 <= lat_deg <= 60.0:
        lat_str = "30N_60N"
    else:
        raise ValueError(f"Latitude {lat_deg}° out of SLDEM2015 tile bounds [-60, +60]")

    # Determine longitude band (45-degree bins: 000_045, 045_090, ..., 315_360)
    lon_bin = int(lon_norm // 45)
    lon_bin = min(max(lon_bin, 0), 7)
    lon_start = lon_bin * 45
    lon_end = lon_start + 45
    lon_str = f"{lon_start:03d}_{lon_end:03d}"

    return f"SLDEM2015_512_{lat_str}_{lon_str}"


def fetch_dem_for_region(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    out_dir: Path | str = DEFAULT_DEM_DIR,
    force: bool = False,
    timeout: tuple[float, float] = (10.0, 60.0),
) -> list[Path]:
    """Find and download all required SLDEM2015 tiles covering a bounding box.

    Downloads both the ``.JP2`` elevation raster and its required GDAL ``.JP2.aux.xml``
    sidecar metadata (which declares the 0.5 m/DN vertical scaling factor).

    Args:
        min_lat: Minimum planetocentric latitude in degrees.
        max_lat: Maximum planetocentric latitude in degrees.
        min_lon: Minimum planetocentric longitude in degrees.
        max_lon: Maximum planetocentric longitude in degrees.
        out_dir: Destination directory to store the DEM JP2 tiles.
        force: If True, re-download even if the tile already exists on disk.
        timeout: Tuple of ``(connect_timeout, read_timeout)`` in seconds.

    Returns:
        List of local ``Path`` objects to the downloaded or cached DEM ``.JP2`` tiles.

    Raises:
        ValueError: If ``min_lat > max_lat`` or ``min_lon > max_lon``.
    """
    if min_lat > max_lat:
        raise ValueError(f"min_lat ({min_lat}) must be <= max_lat ({max_lat})")
    if min_lon > max_lon:
        raise ValueError(f"min_lon ({min_lon}) must be <= max_lon ({max_lon})")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sample endpoints, midpoint, and intermediate 15-degree steps to collect all intersecting tiles
    n_lat = max(3, int(math.ceil((max_lat - min_lat) / 15.0)) + 1)
    n_lon = max(3, int(math.ceil((max_lon - min_lon) / 15.0)) + 1)
    sample_lats = np.linspace(min_lat, max_lat, n_lat)
    sample_lons = np.linspace(min_lon, max_lon, n_lon)

    required_tiles: set[str] = set()
    for lat in sample_lats:
        for lon in sample_lons:
            try:
                tname = get_sldem2015_tile_name(float(lat), float(lon))
                required_tiles.add(tname)
            except ValueError:
                pass

    downloaded_paths: list[Path] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "luna/0.0.1 (Lunar Elevation Fetcher)"})

    for tile in sorted(required_tiles):
        jp2_name = f"{tile}.JP2"
        out_path = out_dir / jp2_name
        temp_path = out_dir / f"{jp2_name}.part"
        aux_path = out_dir / f"{jp2_name}.aux.xml"

        if not aux_path.exists() or force:
            aux_url = f"{SLDEM2015_AUX_BASE_URL}/{tile}_AUX.XML"
            r = session.get(aux_url, timeout=timeout)
            r.raise_for_status()
            aux_path.write_bytes(r.content)

        if out_path.exists() and not force:
            downloaded_paths.append(out_path)
            continue

        url = f"{SLDEM2015_BASE_URL}/{jp2_name}"
        log.info("Fetching lunar DEM tile: %s -> %s", url, out_path)

        with session.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            total_size = int(resp.headers.get("Content-Length", 0))

            with open(temp_path, "wb") as f, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=tile,
                mininterval=0.2,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        temp_path.replace(out_path)
        downloaded_paths.append(out_path)

    return downloaded_paths


def default_dem(dem_dir: Path | str | None = None) -> LunarDEM:
    """Return a process-wide ``LunarDEM`` over all ``*.JP2`` tiles in ``dem_dir``.

    Args:
        dem_dir: Optional directory containing SLDEM2015 ``.JP2`` tiles. Defaults
            to ``$LUNA_DEM_DIR`` if set, or ``data/dem``.

    Returns:
        Configured :class:`LunarDEM` instance.

    Raises:
        FileNotFoundError: If no ``.JP2`` tiles exist in the target directory.
    """
    global _DEFAULT_DEM
    if _DEFAULT_DEM is None or dem_dir is not None:
        root = Path(dem_dir or os.environ.get("LUNA_DEM_DIR", DEFAULT_DEM_DIR))
        tiles = sorted(root.glob("*.JP2"))
        if not tiles:
            raise FileNotFoundError(
                f"no SLDEM JP2 tiles in {root.resolve()} (see fetch_dem_for_region)"
            )
        _DEFAULT_DEM = LunarDEM(tiles)
    return _DEFAULT_DEM


class LunarDEM:
    """Fast georeferenced DEM reader for lunar surface elevation queries.

    Returns elevations in meters relative to the 1737.4 km lunar reference sphere,
    with each raster's scale and offset applied. Points outside loaded tiles or on
    nodata raise ``ValueError`` in :meth:`get_elevation_m` and return ``NaN`` in
    :meth:`query_grid`.

    Args:
        dem_paths: A list of paths to DEM ``.JP2`` / GeoTIFF files, a single file path,
            or a directory containing ``*.JP2`` files.

    Raises:
        ImportError: If ``rasterio`` (from ``luna[geo]``) is not installed.
        RuntimeError: If an SLDEM2015 ``.JP2`` tile is opened without its ``.aux.xml``
            GDAL sidecar declaring the 0.5 m/DN scale factor.
    """

    def __init__(self, dem_paths: Sequence[Path | str] | Path | str) -> None:
        require_extra("geo", "LunarDEM", ["rasterio"])
        import rasterio

        if isinstance(dem_paths, (str, Path)):
            p_obj = Path(dem_paths)
            if p_obj.is_dir():
                resolved_paths = sorted(
                    {
                        *p_obj.glob("*.JP2"),
                        *p_obj.glob("*.jp2"),
                        *p_obj.glob("*.tif"),
                        *p_obj.glob("*.tiff"),
                    }
                )
            else:
                resolved_paths = [p_obj]
        else:
            resolved_paths = [Path(p) for p in dem_paths]

        if not resolved_paths:
            raise ValueError("dem_paths must contain at least one DEM file")

        self.dem_paths: list[Path] = resolved_paths
        self.datasets: list[Any] = [rasterio.open(str(p)) for p in self.dem_paths]
        self._scale_offset: list[tuple[float, float, float | None]] = []
        for p, ds in zip(self.dem_paths, self.datasets):
            scale = float(ds.scales[0]) if ds.scales else 1.0
            offset = float(ds.offsets[0]) if ds.offsets else 0.0
            if (
                p.name.upper().startswith("SLDEM2015")
                and p.suffix.upper() == ".JP2"
                and scale == 1.0
            ):
                self.close()
                raise RuntimeError(
                    f"{p.name}: SLDEM2015 JP2 without GDAL sidecar '{p.name}.aux.xml' "
                    f"(scale 0.5 m/DN missing). Fetch {SLDEM2015_AUX_BASE_URL}/{p.stem}_AUX.XML "
                    f"or call fetch_dem_for_region()."
                )
            self._scale_offset.append((scale, offset, ds.nodata))

    def close(self) -> None:
        """Close all underlying rasterio datasets."""
        for ds in getattr(self, "datasets", []):
            try:
                ds.close()
            except Exception:
                pass

    def __enter__(self) -> LunarDEM:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def get_elevation_m(self, lon_deg: float, lat_deg: float) -> float:
        """Query surface altitude in meters relative to the 1737.4 km reference sphere.

        Args:
            lon_deg: Planetocentric longitude in degrees (wrapped automatically to ``[0, 360)``).
            lat_deg: Planetocentric latitude in degrees (``[-90, +90]``).

        Returns:
            Surface elevation in meters relative to the reference sphere.

        Raises:
            ValueError: If the coordinate is NaN, falls on a nodata pixel, or is not
                covered by any loaded DEM tile.
        """
        if math.isnan(lon_deg) or math.isnan(lat_deg):
            raise ValueError(f"Coordinates cannot be NaN (lon={lon_deg}, lat={lat_deg})")

        import rasterio

        lon_norm = float(lon_deg) % 360.0
        saw_nodata = False

        for ds, (scale, offset, nodata) in zip(self.datasets, self._scale_offset):
            b = ds.bounds
            if b.left <= lon_norm <= b.right and b.bottom <= lat_deg <= b.top:
                transform = getattr(ds, "transform", None)
                height = getattr(ds, "height", None)
                width = getattr(ds, "width", None)
                if isinstance(transform, rasterio.Affine) and isinstance(height, int) and isinstance(width, int):
                    r_idx, c_idx = rasterio.transform.rowcol(transform, lon_norm, lat_deg)
                    r_clamped = int(np.clip(int(r_idx), 0, height - 1))
                    c_clamped = int(np.clip(int(c_idx), 0, width - 1))
                    win = rasterio.windows.Window(c_clamped, r_clamped, 1, 1)
                    raw = ds.read(1, window=win)[0, 0]
                else:
                    raw = next(ds.sample([(lon_norm, lat_deg)]))[0]

                raw_f = float(raw)
                is_nodata = math.isnan(raw_f) or (
                    nodata is not None
                    and (raw == nodata or (math.isnan(float(nodata)) and math.isnan(raw_f)))
                )
                if is_nodata:
                    saw_nodata = True
                    continue
                return raw_f * scale + offset

        if saw_nodata:
            raise ValueError(f"DEM nodata at lon={lon_deg}, lat={lat_deg}")
        raise ValueError(f"lon={lon_deg}, lat={lat_deg} not covered by any loaded DEM tile")

    def query_grid(self, lon_grid: np.ndarray, lat_grid: np.ndarray) -> np.ndarray:
        """Query elevations in meters for arrays of longitude and latitude.

        Args:
            lon_grid: NumPy array of longitudes in degrees.
            lat_grid: NumPy array of latitudes in degrees (same shape as ``lon_grid``).

        Returns:
            Float32 NumPy array of elevations in meters, with ``NaN`` where uncovered
            or nodata.

        Raises:
            ValueError: If ``lon_grid`` and ``lat_grid`` do not have the same shape.
        """
        require_extra("geo", "LunarDEM.query_grid", ["rasterio"])
        import rasterio

        lon_arr = np.asarray(lon_grid, dtype=np.float64)
        lat_arr = np.asarray(lat_grid, dtype=np.float64)
        if lon_arr.shape != lat_arr.shape:
            raise ValueError(
                f"lon_grid shape {lon_arr.shape} must match lat_grid shape {lat_arr.shape}"
            )

        out_alt = np.full(lon_arr.shape, np.nan, dtype=np.float32)
        lon_norm = lon_arr % 360.0

        for ds, (scale, offset, nodata) in zip(self.datasets, self._scale_offset):
            b = ds.bounds
            in_tile = (
                (lon_norm >= b.left)
                & (lon_norm <= b.right)
                & (lat_arr >= b.bottom)
                & (lat_arr <= b.top)
            )
            if np.any(in_tile):
                sub_lons = lon_norm[in_tile]
                sub_lats = lat_arr[in_tile]
                min_c_lon, max_c_lon = float(np.min(sub_lons)), float(np.max(sub_lons))
                min_c_lat, max_c_lat = float(np.min(sub_lats)), float(np.max(sub_lats))

                r_top, c_left = rasterio.transform.rowcol(ds.transform, min_c_lon, max_c_lat)
                r_bot, c_right = rasterio.transform.rowcol(ds.transform, max_c_lon, min_c_lat)

                r_start = max(0, min(int(r_top), int(r_bot)) - 2)
                r_end = min(ds.height, max(int(r_top), int(r_bot)) + 2)
                c_start = max(0, min(int(c_left), int(c_right)) - 2)
                c_end = min(ds.width, max(int(c_left), int(c_right)) + 2)

                window = rasterio.windows.Window(
                    c_start, r_start, max(1, c_end - c_start), max(1, r_end - r_start)
                )
                win_data = ds.read(1, window=window)

                row_idx, col_idx = rasterio.transform.rowcol(ds.transform, sub_lons, sub_lats)
                rel_r = np.clip(np.asarray(row_idx) - r_start, 0, win_data.shape[0] - 1)
                rel_c = np.clip(np.asarray(col_idx) - c_start, 0, win_data.shape[1] - 1)

                raw = win_data[rel_r, rel_c].astype(np.float32)
                vals = raw * scale + offset
                if nodata is not None and not math.isnan(float(nodata)):
                    vals[raw == nodata] = np.nan

                existing = out_alt[in_tile]
                valid_mask = ~np.isnan(vals)
                existing[valid_mask] = vals[valid_mask]
                out_alt[in_tile] = existing

        return out_alt
