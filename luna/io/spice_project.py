"""Pure-Python SPICE projection for LROC NAC: (lon, lat) -> (sample, line).

Replaces the bilinear-from-INDEX-corners approximation in ``luna.io.projection``
with the full pushbroom camera model, using ``spiceypy`` + NAIF kernels
directly. Equivalent to ISIS ``campt type=ground`` but with no ISIS/Docker/WSL
round-trip.

Camera model (LROC NAC, per ``lro_lroc_v20.ti``):
- NAC-L ikid=-85600, frame ``LRO_LROCNACL``, boresight sample 2548, f=699.62 mm
- NAC-R ikid=-85610, frame ``LRO_LROCNACR``, boresight sample 2496, f=701.57 mm
- Linear detector, 5064 samples wide, 1 line; detector runs along focal-plane
  Y axis (TRANSY=[0,+/-0.007,0]). Detector x=0 in the focal plane, so the
  across-track dimension (line) is encoded purely by time: each output line is
  a different ET snapshot of the same linear array.
- Pixel pitch 0.007 mm; 1 / 0.007 = 142.857 px/mm (appears as ITRANSS/ITRANSL).

Ground-to-image algorithm (pure 1D root-find):
1. Convert (lon, lat, 0) to body-fixed (IAU_MOON) XYZ via ``spice.georec``.
2. For a trial ET in [t_start, t_stop], rotate G into J2000 and form the
   look vector from the spacecraft to G.
3. Rotate into camera frame; project via pinhole.
4. The detector-line condition at ET is ``x_mm_focal = 0`` — equivalent to the
   look-vector x-component being zero in the camera frame. Bracket-and-bisect
   on that residual over the exposure window to find the ET at which G is
   on the detector.
5. Compute line from ET (``(et - t_start) / line_rate``) and sample from the
   focal-plane y-coord via ITRANSS.

This is the same math USGS CSM's ``UsgsAstroLsSensorModel::groundToImage``
runs internally; we do it in ~100 lines because NAC has no lens distortion
to speak of and a trivial TRANS matrix.
"""

from __future__ import annotations

import glob
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pvl
import spiceypy as sp

log = logging.getLogger(__name__)

# Kernel root. Layout mirrors NAIF's archive: lsk/ sclk/ pck/ fk/ ik/ spk/ ck/.
DEFAULT_KERNEL_ROOT = Path(__file__).resolve().parents[2] / "data" / "spice" / "lro"

_NAC_PARAMS = {
    "L": {"ikid": -85600, "frame": "LRO_LROCNACL", "focal_mm": 699.62,
          "boresight_sample": 2548.0, "pitch_mm": 0.007, "px_per_mm": 142.857},
    "R": {"ikid": -85610, "frame": "LRO_LROCNACR", "focal_mm": 701.57,
          "boresight_sample": 2496.0, "pitch_mm": 0.007, "px_per_mm": -142.857},
}

_FURNISHED: set[str] = set()
_LOADED_WINDOW: tuple[datetime, datetime] | None = None


def furnish_kernels(kernel_root: Path | str | None = None) -> int:
    """Load every SPICE kernel under ``kernel_root`` into the pool.

    Idempotent — re-calls skip kernels already furnished. Preserved for
    high-throughput batch workloads within a single mission phase. Returns the
    total number in the pool afterwards.
    """
    global _LOADED_WINDOW
    root = Path(kernel_root or DEFAULT_KERNEL_ROOT)
    exts = ("tls", "tsc", "tpc", "bpc", "tf", "ti", "bsp", "bc")
    for ext in exts:
        for f in sorted(glob.glob(str(root / "**" / f"*.{ext}"), recursive=True)):
            if f not in _FURNISHED:
                sp.furnsh(f)
                _FURNISHED.add(f)
    _LOADED_WINDOW = None
    total = sp.ktotal("ALL")
    log.debug("kernel pool: %d loaded", total)
    return total


def furnish_kernels_for_date(
    dt: datetime,
    kernel_root: Path | str | None = None,
    force: bool = False,
) -> int:
    """Load untimed base kernels + only the SPK and CK kernels covering ``dt`` into the pool.

    Clears the SPICE kernel pool via ``sp.kclear()`` before loading if the active
    kernel pool does not already cover ``dt``, preventing DAF handle table
    overflow when querying across multiple mission years.
    """
    global _LOADED_WINDOW
    from .kernel_fetch import _kernel_covers, _WINDOW_RE, _ydoy_to_date

    dt_utc = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

    if not force and _LOADED_WINDOW is not None and len(_FURNISHED) > 0:
        win_start, win_end = _LOADED_WINDOW
        if win_start <= dt_utc <= win_end and sp.ktotal("ALL") > 0:
            return sp.ktotal("ALL")

    sp.kclear()
    _FURNISHED.clear()

    root = Path(kernel_root or DEFAULT_KERNEL_ROOT)
    exts = ("tls", "tsc", "tpc", "bpc", "tf", "ti", "bsp", "bc")
    timed_starts: list[datetime] = []
    timed_ends: list[datetime] = []

    for ext in exts:
        for f in sorted(glob.glob(str(root / "**" / f"*.{ext}"), recursive=True)):
            fname = Path(f).name
            if _kernel_covers(fname, dt_utc):
                if f not in _FURNISHED:
                    sp.furnsh(f)
                    _FURNISHED.add(f)
                m = _WINDOW_RE.search(fname)
                if m:
                    y0, d0, y1, d1 = map(int, m.groups())
                    timed_starts.append(_ydoy_to_date(y0, d0))
                    timed_ends.append(_ydoy_to_date(y1, d1))

    if timed_starts and timed_ends:
        _LOADED_WINDOW = (max(timed_starts), min(timed_ends))
    else:
        _LOADED_WINDOW = None

    total = sp.ktotal("ALL")
    log.debug("kernel pool (for %s): %d loaded", dt_utc.isoformat(), total)
    return total


def ensure_kernels_for_label(label_path: Path | str, kernel_root: Path | str | None = None) -> None:
    """Make sure all SPICE kernels needed to project from ``label_path`` are present and furnished.

    Looks up the NAC START_TIME, ensures missing kernels are downloaded via
    :func:`luna.io.kernel_fetch.ensure_kernels_for_date`, then selectively
    furnishes kernels covering that timestamp.
    """
    from .kernel_fetch import ensure_kernels_for_date  # local import to avoid cycles
    lbl = pvl.load(str(label_path))
    start_dt = lbl["START_TIME"].replace(tzinfo=None)
    ensure_kernels_for_date(start_dt, root=kernel_root)
    furnish_kernels_for_date(start_dt, kernel_root=kernel_root)


def _nac_side_from_pid(product_id: str) -> str:
    """Return 'L' or 'R' from a LROC NAC product ID."""
    pid = product_id.upper().strip()
    # IDs like M126710873RE / M126710873LC -> take the letter before trailing C/E.
    if len(pid) >= 2 and pid[-1] in "CE" and pid[-2] in "LR":
        return pid[-2]
    if pid.endswith(("L", "R")):
        return pid[-1]
    raise ValueError(f"cannot infer NAC side (L/R) from product_id {product_id!r}")


def _read_times(label_path: Path) -> tuple[float, float, float, int, int]:
    """Return (et_start, et_stop, line_rate_s, n_lines, n_samples) from PDS3 label."""
    lbl = pvl.load(str(label_path))
    # pvl returns timezone-aware datetimes; SPICE utc2et rejects the "+00:00"
    # suffix, so strip tz and format as plain UTC.
    start_dt = lbl["START_TIME"].replace(tzinfo=None)
    stop_dt = lbl["STOP_TIME"].replace(tzinfo=None)
    et_start = sp.utc2et(start_dt.isoformat())
    et_stop = sp.utc2et(stop_dt.isoformat())
    # LROC NAC is a line-scan camera; rows are time-indexed.
    n_lines = int(lbl["IMAGE"]["LINES"])
    n_samples = int(lbl["IMAGE"]["LINE_SAMPLES"])
    # Line exposure duration: prefer the explicit field if present, else derive.
    if "LRO:LINE_EXPOSURE_DURATION" in lbl:
        # Typically ms, as PVL Units; convert to seconds.
        v = lbl["LRO:LINE_EXPOSURE_DURATION"]
        line_rate = float(v.value) * 1e-3 if getattr(v, "value", None) is not None else float(v) * 1e-3
    else:
        line_rate = (et_stop - et_start) / max(n_lines - 1, 1)
    return et_start, et_stop, line_rate, n_lines, n_samples


def _look_cam(et: float, G_bf: np.ndarray, frame: str) -> np.ndarray:
    """Unit look vector from spacecraft to body-fixed ground point, in camera frame at ET."""
    # SC position in J2000, relative to Moon center, no aberration correction:
    # lon/lat is a point on the body-fixed surface — we rotate it to J2000 at the
    # same ET, then subtract the SC position there.
    sc_pos, _ = sp.spkpos("LRO", et, "J2000", "NONE", "MOON")
    rot_bf_to_j2k = sp.pxform("IAU_MOON", "J2000", et)
    G_j2k = rot_bf_to_j2k @ G_bf
    look_j2k = G_j2k - sc_pos
    rot_j2k_to_cam = sp.pxform("J2000", frame, et)
    return rot_j2k_to_cam @ look_j2k


def ground_to_image(
    label_path: Path | str,
    lon_deg: float,
    lat_deg: float,
    alt_m: float = 0.0,
    kernel_root: Path | str | None = None,
) -> tuple[float, float]:
    """(lon, lat) on the Moon -> (sample, line) 0-indexed on the NAC detector.

    Uses full SPICE geometry: spacecraft ephemeris, attitude CKs, and the NAC
    IK. Returns sub-pixel coords; caller may round as needed. Raises
    ``ValueError`` if the point is outside the exposure window.

    Parameters
    ----------
    label_path:
        Path to PDS3 .LBL / .IMG file.
    lon_deg:
        Planetocentric longitude in degrees (0–360).
    lat_deg:
        Planetocentric latitude in degrees (-90 to +90).
    alt_m:
        Surface altitude in meters relative to the Moon reference sphere
        (e.g., from LOLA DEM). Crucial for off-nadir observations to prevent
        parallax displacement.
    kernel_root:
        Optional root directory of SPICE kernels.
    """
    label_path = Path(label_path)
    lbl = pvl.load(str(label_path))
    start_dt = lbl["START_TIME"].replace(tzinfo=None)
    from .kernel_fetch import ensure_kernels_for_date
    ensure_kernels_for_date(start_dt, root=kernel_root)
    furnish_kernels_for_date(start_dt, kernel_root)

    pid = str(lbl["PRODUCT_ID"])
    side = _nac_side_from_pid(pid)
    p = _NAC_PARAMS[side]

    et_start, et_stop, line_rate, n_lines, n_samples = _read_times(label_path)

    # Ground point in body-fixed IAU_MOON via planetocentric coordinates.
    radii = sp.bodvrd("MOON", "RADII", 3)[1]
    r_moon = float(radii[0])
    alt_km = alt_m / 1000.0
    G_bf = np.asarray(sp.latrec(
        r_moon + alt_km, np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    ))

    # Residual is the x-component of the camera-frame look vector. NAC's detector
    # is a 1D array at x_focal=0, so the ET that images G is the one where
    # look_cam[0]=0. Monotonic over a NAC exposure.
    def resid(et: float) -> float:
        u = _look_cam(et, G_bf, p["frame"])
        return float(u[0] / np.linalg.norm(u))

    r0, r1 = resid(et_start), resid(et_stop)
    if r0 * r1 > 0:
        raise ValueError(
            f"point (lon={lon_deg}, lat={lat_deg}) outside NAC exposure window "
            f"[{et_start:.3f}, {et_stop:.3f}] — residuals {r0:.3e}, {r1:.3e}"
        )
    lo, hi = et_start, et_stop
    for _ in range(80):  # ~1e-24 convergence on a ~20 s window; well over-budget
        mid = 0.5 * (lo + hi)
        rm = resid(mid)
        if r0 * rm <= 0:
            hi, r1 = mid, rm
        else:
            lo, r0 = mid, rm
        if hi - lo < 1e-9:
            break
    et_hit = 0.5 * (lo + hi)

    # At et_hit, compute sample from the focal-plane Y coordinate of the look vector.
    u = _look_cam(et_hit, G_bf, p["frame"])
    y_mm = p["focal_mm"] * u[1] / u[2]
    # ITRANSS for NAC-L is [0,0,142.857]; NAC-R is [0,0,-142.8571]. Thus
    # sample_offset_from_boresight = px_per_mm * y_mm.
    sample = p["boresight_sample"] + p["px_per_mm"] * y_mm
    # Line is determined by the ET offset from start.
    line = (et_hit - et_start) / line_rate

    return float(sample), float(line)


def image_to_ground(
    label_path: Path | str,
    sample: float,
    line: float,
    kernel_root: Path | str | None = None,
) -> tuple[float, float]:
    """(sample, line) 0-indexed -> (lon, lat) via SPICE."""
    label_path = Path(label_path)
    lbl = pvl.load(str(label_path))
    start_dt = lbl["START_TIME"].replace(tzinfo=None)
    from .kernel_fetch import ensure_kernels_for_date
    ensure_kernels_for_date(start_dt, root=kernel_root)
    furnish_kernels_for_date(start_dt, kernel_root)

    side = _nac_side_from_pid(str(lbl["PRODUCT_ID"]))
    p = _NAC_PARAMS[side]

    et_start, _, line_rate, _, _ = _read_times(label_path)
    
    et = et_start + (line * line_rate)
    
    y_focal = (sample - p["boresight_sample"]) / p["px_per_mm"]
    look_cam = np.array([0.0, y_focal, p["focal_mm"]])
    
    try:
        point, _, _ = sp.sincpt("Ellipsoid", "MOON", et, "IAU_MOON", "NONE", "LRO", p["frame"], look_cam)
        _, lon_rad, lat_rad = sp.reclat(point)
        lon_deg = float(np.rad2deg(lon_rad)) % 360.0
        return lon_deg, float(np.rad2deg(lat_rad))
    except Exception:
        return 0.0, 0.0


def lonlat_to_pixel_spice(
    label_path: Path | str,
    lon_deg: float,
    lat_deg: float,
    alt_m: float = 0.0,
    kernel_root: Path | str | None = None,
) -> tuple[int, int]:
    """Integer (sample_idx, line_idx), 0-indexed. Convenience wrapper."""
    s, l = ground_to_image(label_path, lon_deg, lat_deg, alt_m, kernel_root)
    return int(round(s)), int(round(l))

