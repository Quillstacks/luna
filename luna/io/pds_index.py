"""Resolve LROC NAC product IDs to PDS download URLs on demand.

The LROC CDR archive is partitioned into ~35 volumes (``LROLRC_10xx``) with a
fixed-width ``INDEX/INDEX.TAB`` per volume (901 bytes per record incl. CRLF,
sorted by PRODUCT_ID, ~60k-100k rows per volume). This module does no bulk
download — it characterizes each volume by its first/last PRODUCT_ID via a
single HTTP range request, then binary-searches the relevant volume's index
when asked to find a specific product.

Typical cost per product resolution: one range-read to characterize the target
volume (cached), plus ~17 range-reads of 901 bytes each (the binary search).

Usage:
    from luna.io.pds_index import PDSIndex
    idx = PDSIndex()
    url = idx.url_for("M126710873RE")
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger("luna.io.pds_index")

DEFAULT_BASE = "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0"
EDR_BASE = "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-2-EDR-V1.0"
RECORD_BYTES = 901  # fixed-width record incl. CRLF, per INDEX.LBL
# Fixed-width column offsets (1-indexed START_BYTE from LBL, converted to 0-indexed).
# VOLUME_ID: bytes 2-12 (11 chars, quoted)  -> indices 1..12
# FILE_SPECIFICATION_NAME: bytes 16-90 (75 chars, quoted) -> indices 15..90
# PRODUCT_ID: bytes 122-134 (13 chars, quoted) -> indices 121..134
VOLUME_SLICE = slice(1, 12)
FSN_SLICE = slice(15, 90)
PID_SLICE = slice(121, 134)
# Geometry columns (START_BYTE - 1, START_BYTE - 1 + BYTES) from INDEX.LBL.
GEOMETRY_SLICES = {
    "image_lines": slice(686, 692),
    "line_samples": slice(693, 697),
    "resolution": slice(716, 723),
    "emission_angle": slice(724, 729),
    "incidence_angle": slice(730, 736),
    "phase_angle": slice(737, 743),
    "sub_solar_latitude": slice(758, 764),
    "sub_solar_longitude": slice(765, 771),
    "sub_spacecraft_latitude": slice(772, 778),
    "sub_spacecraft_longitude": slice(779, 785),
    "center_latitude": slice(805, 811),
    "center_longitude": slice(812, 818),
    "upper_right_latitude": slice(819, 825),
    "upper_right_longitude": slice(826, 832),
    "lower_right_latitude": slice(833, 839),
    "lower_right_longitude": slice(840, 846),
    "lower_left_latitude": slice(847, 853),
    "lower_left_longitude": slice(854, 860),
    "upper_left_latitude": slice(861, 867),
    "upper_left_longitude": slice(868, 874),
    "spacecraft_altitude": slice(875, 882),
}
CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "pds_volume_index.json"


def _normalize_product_id(pid: str, suffix: str = "C") -> str:
    pid = pid.upper().strip()
    if pid.endswith(".IMG"):
        pid = pid[:-4]
    # INDEX.TAB uses 12-char PRODUCT_IDs with an archive-specific trailing
    # letter: 'C' for CDR (calibrated), 'E' for EDR (raw). User-facing IDs
    # like 'M126710873R' get the suffix appended here.
    if re.fullmatch(r"M\d{7,10}[LR]", pid):
        pid = pid + suffix
    # Tolerate an already-suffixed id being reused across archives.
    if re.fullmatch(r"M\d{7,10}[LR][CE]", pid):
        pid = pid[:-1] + suffix
    return pid


def _strip_quoted(field: str) -> str:
    return field.strip().strip('"').strip()


def _parse_record(text: str) -> tuple[str, str, str]:
    """Return (volume_id, product_id, file_spec_name) from one fixed-width line."""
    return (
        _strip_quoted(text[VOLUME_SLICE]),
        _strip_quoted(text[PID_SLICE]),
        _strip_quoted(text[FSN_SLICE]),
    )


@dataclass
class VolumeInfo:
    volume_id: str
    index_url: str
    row_count: int
    first_pid: str
    last_pid: str


class PDSIndex:
    """Binary-search resolver for LROC CDR product IDs."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE,
        cache_path: Optional[Path | str] = None,
        session: Optional[requests.Session] = None,
        archive: str = "CDR",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.archive = archive.upper()
        # CDR → 'C', EDR → 'E'. Mirrors the archive's file naming.
        self.suffix = {"CDR": "C", "EDR": "E"}.get(self.archive, "C")
        if cache_path is None:
            cache_path = CACHE_PATH.with_name(f"pds_volume_index_{self.archive.lower()}.json")
        self.cache_path = Path(cache_path)
        self.session = session or requests.Session()
        self._volumes: Optional[list[VolumeInfo]] = None

    # -- volume characterization ---------------------------------------

    def _list_volumes(self) -> list[str]:
        r = self.session.get(self.base_url + "/", timeout=60)
        r.raise_for_status()
        vols = sorted(set(re.findall(r'href="(LROLRC_\d+)/"', r.text)))
        return [v for v in vols]

    def _read_record(self, url: str, row: int) -> str:
        start = row * RECORD_BYTES
        end = start + RECORD_BYTES - 1
        r = self.session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=30)
        if r.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {r.status_code} on range read {url}")
        return r.content.decode("ascii", errors="replace")

    def _characterize_volume(self, vol: str) -> VolumeInfo:
        index_url = f"{self.base_url}/{vol}/INDEX/INDEX.TAB"
        head = self.session.head(index_url, timeout=30, allow_redirects=True)
        head.raise_for_status()
        total = int(head.headers["Content-Length"])
        rows = total // RECORD_BYTES
        first = _parse_record(self._read_record(index_url, 0))[1]
        last = _parse_record(self._read_record(index_url, rows - 1))[1]
        return VolumeInfo(volume_id=vol, index_url=index_url, row_count=rows,
                          first_pid=first, last_pid=last)

    def volumes(self, force: bool = False) -> list[VolumeInfo]:
        if self._volumes is not None and not force:
            return self._volumes
        if self.cache_path.exists() and not force:
            raw = json.loads(self.cache_path.read_text())
            self._volumes = [VolumeInfo(**v) for v in raw]
            return self._volumes
        vols = self._list_volumes()
        log.info("characterizing %d CDR volumes (one-time)...", len(vols))
        out: list[VolumeInfo] = []
        for v in vols:
            try:
                info = self._characterize_volume(v)
                out.append(info)
                log.info("  %s  rows=%d  pids %s..%s", info.volume_id, info.row_count,
                         info.first_pid, info.last_pid)
            except Exception as e:  # noqa: BLE001
                log.warning("  %s: %s", v, e)
        self._volumes = out
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps([v.__dict__ for v in out], indent=2))
        return out

    # -- product lookup -------------------------------------------------

    def _binary_search(self, vol: VolumeInfo, target: str) -> Optional[tuple[str, str, str]]:
        """Return (pid, fsn, raw_record) for the matching row, else None."""
        lo, hi = 0, vol.row_count - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            rec = self._read_record(vol.index_url, mid)
            _, pid, fsn = _parse_record(rec)
            if pid == target:
                return pid, fsn, rec
            if pid < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    def _parse_geometry(self, record: str) -> dict:
        """Extract geometry fields from an INDEX.TAB record (numeric strings -> float)."""
        out: dict = {}
        for key, sl in GEOMETRY_SLICES.items():
            raw = _strip_quoted(record[sl])
            try:
                out[key] = float(raw) if raw else None
            except ValueError:
                out[key] = None
        return out

    def _lookup(self, product_id: str) -> tuple[VolumeInfo, str, str, str]:
        """Return (vol, pid, fsn, raw_record) for a product, or raise KeyError."""
        pid = _normalize_product_id(product_id, self.suffix)
        for v in self.volumes():
            if v.first_pid <= pid <= v.last_pid:
                hit = self._binary_search(v, pid)
                if hit is not None:
                    pid, fsn, rec = hit
                    return v, pid, fsn, rec
        raise KeyError(f"product {product_id!r} not found in any volume")

    def url_for(self, product_id: str) -> str:
        """Return the absolute archive URL for a NAC product, or raise KeyError."""
        _, _, fsn, _ = self._lookup(product_id)
        # fsn is e.g. "LRO-L-LROC-3-CDR-V1.0/LROLRC_1001/DATA/SCI/.../Mxxxx.IMG"
        root_marker = f"/LRO-L-LROC-{'2-EDR' if self.archive == 'EDR' else '3-CDR'}-V1.0"
        return self.base_url.rsplit(root_marker, 1)[0] + "/" + fsn

    def geometry_for(self, product_id: str) -> dict:
        """Return geometry fields (center lat/lon, resolution, corners, ...) for a product."""
        _, pid, fsn, rec = self._lookup(product_id)
        geom = self._parse_geometry(rec)
        geom["product_id"] = pid
        geom["file_spec_name"] = fsn
        return geom
