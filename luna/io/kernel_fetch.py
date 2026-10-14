"""On-demand NAIF SPICE kernel fetcher for LRO.

Given a NAC exposure time, ensures the minimum set of kernels needed to run
``luna.io.spice_project.ground_to_image`` and ``image_to_ground`` is present
under ``data/spice/lro/`` (or ``$LUNA_SPICE_ROOT``). Parses the year-specific
NAIF metakernel, filters the ``KERNELS_TO_LOAD`` list to entries whose filename
time-window covers the ET, and downloads the missing files in parallel.

Per-year cold fetch is ~1–2 GB; subsequent frames in the same window are
cache hits.
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from luna.exceptions import IndexError

log = logging.getLogger(__name__)

NAIF_BASE = "https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000"
DEFAULT_ROOT = Path(
    os.environ.get(
        "LUNA_SPICE_ROOT",
        Path(__file__).resolve().parents[2] / "data" / "spice" / "lro",
    )
).resolve()

# Kernel filenames carry a time range like ``lrolc_2014059_2014091_v06.bc`` —
# (YYYY DOY) pairs. Untimed kernels (lsk, pck, generic ik/fk) lack this pattern.
_WINDOW_RE = re.compile(r"(\d{4})(\d{3})_(\d{4})(\d{3})")


def _ydoy_to_date(y: int, doy: int) -> datetime:
    return datetime(y, 1, 1, tzinfo=timezone.utc).replace(day=1) + _doy_offset(doy)


def _doy_offset(doy: int) -> timedelta:
    return timedelta(days=doy - 1)


def _date_to_ydoy(dt: datetime) -> tuple[int, int]:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.year, int(dt.strftime("%j"))


def _kernel_covers(filename: str, dt: datetime) -> bool:
    """Return True if the filename's encoded time window contains ``dt``, or if untimed."""
    m = _WINDOW_RE.search(filename)
    if not m:
        return True
    y0, d0, y1, d1 = map(int, m.groups())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    start = _ydoy_to_date(y0, d0)
    end = _ydoy_to_date(y1, d1)
    return start <= dt <= end


def _fetch_metakernel(year: int, root: Path, session: requests.Session) -> Path:
    """Return a local path to ``lro_<year>_v??.tm``, fetching the highest version available."""
    mk_dir = root / "mk"
    mk_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(mk_dir.glob(f"lro_{year}_v*.tm"))
    if existing:
        return existing[-1]
    for v in range(10, 0, -1):
        name = f"lro_{year}_v{v:02d}.tm"
        url = f"{NAIF_BASE}/extras/mk/{name}"
        r = session.get(url, timeout=10, verify=False)
        if r.status_code == 200 and len(r.content) > 1024:
            out = mk_dir / name
            out.write_bytes(r.content)
            log.info("fetched metakernel %s", name)
            return out
    raise IndexError(f"no metakernel found on NAIF for year {year}")


def _parse_kernels_to_load(mk_path: Path) -> list[str]:
    """Return relative-to-data/ kernel paths listed in a NAIF metakernel."""
    text = mk_path.read_text()
    in_block = False
    kernels: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("KERNELS_TO_LOAD"):
            in_block = True
            continue
        if in_block and ")" in s:
            in_block = False
        if in_block:
            m = re.search(r"\$KERNELS/([\w./-]+)", s)
            if m:
                kernels.append(m.group(1))
    return kernels


def _download_one(url: str, dest: Path, session: requests.Session) -> tuple[str, bool, int]:
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest), True, dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(f"{dest.suffix}.tmp.{os.getpid()}")
    try:
        r = session.get(url, timeout=60, verify=False, stream=True)
        if r.status_code != 200:
            return url, False, 0
        with open(tmp_dest, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
        if tmp_dest.exists() and tmp_dest.stat().st_size > 0:
            tmp_dest.replace(dest)
        return url, True, dest.stat().st_size
    except Exception as e:
        if tmp_dest.exists():
            tmp_dest.unlink(missing_ok=True)
        log.warning("Download failed for %s: %s", url, e)
        return url, False, 0


def ensure_kernels_for_date(
    dt: datetime,
    root: Path | str | None = None,
    max_workers: int = 6,
) -> list[Path]:
    """Ensure all SPICE kernels covering timestamp ``dt`` are present locally.

    Parses the NAIF annual metakernel for ``dt.year``, filters the ``KERNELS_TO_LOAD``
    manifest to entries whose filename time window contains ``dt`` (plus timeless base
    kernels), and downloads any missing kernels in parallel from the NASA NAIF archive.

    Args:
        dt: Observation timestamp (datetime object).
        root: Local directory to store kernels. Defaults to :data:`DEFAULT_ROOT`.
        max_workers: Number of parallel download threads.

    Returns:
        List of local ``Path`` objects to all verified kernels covering the timestamp.
    """
    resolved_root = Path(root or DEFAULT_ROOT)
    session = requests.Session()
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    year, _ = _date_to_ydoy(dt)
    mk_path = _fetch_metakernel(year, resolved_root, session)
    rel_paths = _parse_kernels_to_load(mk_path)
    needed = [p for p in rel_paths if _kernel_covers(p, dt)]

    to_fetch: list[tuple[str, Path]] = []
    existing: list[Path] = []
    for rel in needed:
        local = resolved_root / rel
        if local.exists() and local.stat().st_size > 0:
            existing.append(local)
        else:
            to_fetch.append((f"{NAIF_BASE}/data/{rel}", local))

    if to_fetch:
        log.info("fetching %d kernels (%d already cached)", len(to_fetch), len(existing))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_download_one, url, dest, session): (url, dest)
                for url, dest in to_fetch
            }
            for fut in as_completed(futures):
                url, ok, _sz = fut.result()
                if not ok:
                    log.warning("kernel fetch failed: %s", url)

    return [resolved_root / rel for rel in needed]
