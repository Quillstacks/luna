"""Resolve LROC NAC product IDs to PDS download URLs on demand.

The LROC CDR archive is partitioned into ~35 volumes (LROLRC_10xx) with a
fixed-width INDEX/INDEX.TAB per volume (sorted by PRODUCT_ID). This module
characterizes each volume by its first/last PRODUCT_ID via HTTP range requests, 
then binary-searches the relevant volume's index to find a specific product.

Usage:
    from luna.io.pds_index import PDSIndex
    idx = PDSIndex()
    url = idx.url_for("M126710873RE")
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("luna.io.pds_index")

DEFAULT_BASE = "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0"
EDR_BASE = "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-2-EDR-V1.0"
PID_PATTERN = re.compile(r"^M\d{7,10}[A-Z]{1,2}[CE]$")

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


def _is_valid_pid(pid: str) -> bool:
    return PID_PATTERN.match(pid) is not None


def _normalize_product_id(pid: str, suffix: str = "C") -> str:
    pid = pid.upper().strip()
    if pid.endswith(".IMG"):
        pid = pid[:-4]
    
    if re.fullmatch(r"M\d{7,10}[LR]", pid):
        pid = pid + suffix
    
    if re.fullmatch(r"M\d{7,10}[LR][CE]", pid):
        pid = pid[:-1] + suffix
    return pid


def _strip_quoted(field: str) -> str:
    return field.strip().strip('"').strip()


def _parse_record(text: str) -> tuple[str, str, str]:
    text = text.strip('\r\n')
    try:
        reader = csv.reader(io.StringIO(text))
        row = next(reader)
        
        vol = row[0].strip() if len(row) > 0 else ""
        fsn = row[1].strip() if len(row) > 1 else ""
        pid = row[5].strip() if len(row) > 5 else ""
        
        return vol, pid, fsn
    except Exception:
        return "", "", ""


@dataclass
class VolumeInfo:
    volume_id: str
    index_url: str
    record_bytes: int
    row_count: int
    first_row_index: int
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
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Luna-PDS-Tool/1.0 (Academic/Research)"
        })

        retries = Retry(
            total=3, 
            backoff_factor=1, 
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        self.base_url = base_url.rstrip("/")
        self.archive = archive.upper()
        self.suffix = {"CDR": "C", "EDR": "E"}.get(self.archive, "C")
        
        if cache_path is None:
            cache_path = CACHE_PATH.with_name(f"pds_volume_index_{self.archive.lower()}.json")
        
        self.cache_path = Path(cache_path)
        self.session = session or requests.Session()
        self._volumes: Optional[list[VolumeInfo]] = None

    def _list_volumes(self) -> list[str]:
        r = self.session.get(self.base_url + "/", timeout=60)
        r.raise_for_status()
        vols = sorted(set(re.findall(r'href="(LROLRC_\d+\w*)/"', r.text)))
        return [v for v in vols]

    def _get_record_bytes(self, url: str) -> int:
        r = self.session.get(url, headers={"Range": "bytes=0-2048"}, timeout=30)
        r.raise_for_status()
        
        idx = r.content.find(b"\r\n")
        if idx != -1:
            return idx + 2
            
        idx = r.content.find(b"\n")
        if idx != -1:
            return idx + 1
            
        raise ValueError(f"Could not determine record length for {url}")

    def _read_record(self, url: str, row: int, record_bytes: int) -> str:
        start = row * record_bytes
        end = start + record_bytes - 1
        r = self.session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=30)
        if r.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {r.status_code} on range read {url}")
        return r.content.decode("ascii", errors="replace")

    def _characterize_volume(self, volume_id: str) -> VolumeInfo:
        index_url = f"{self.base_url}/{volume_id}/INDEX/INDEX.TAB"
        
        record_bytes = self._get_record_bytes(index_url)
        
        resp = self.session.head(index_url, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        total_bytes = int(resp.headers["Content-Length"])
        est_rows = total_bytes // record_bytes
        
        first_pid = ""
        first_row_index = 0

        for i in range(50):
            try:
                rec = self._read_record(index_url, i, record_bytes)
                _, candidate_pid, _ = _parse_record(rec)

                if _is_valid_pid(candidate_pid):
                    first_pid = candidate_pid
                    first_row_index = i
                    break
            except Exception:
                continue

        if not first_pid:
            raise ValueError(f"No valid starting PID found in volume {volume_id}")

        last_pid = ""
        actual_row_count = est_rows
        
        for i in range(1, 51):
            row_idx = est_rows - i
            if row_idx < 0: 
                break
            
            try:
                rec = self._read_record(index_url, row_idx, record_bytes)
                _, candidate_pid, _ = _parse_record(rec)
                
                if _is_valid_pid(candidate_pid):
                    last_pid = candidate_pid
                    actual_row_count = row_idx + 1
                    break
            except (ValueError, RuntimeError):
                continue
        
        if not last_pid:
            raise ValueError(f"No valid ending PID found in volume {volume_id}")

        return VolumeInfo(
            volume_id=volume_id,
            index_url=index_url,
            record_bytes=record_bytes,
            row_count=actual_row_count,
            first_row_index=first_row_index,
            first_pid=first_pid,
            last_pid=last_pid
        )

    def volumes(self, force: bool = False) -> list[VolumeInfo]:
        if self._volumes is not None and not force:
            return self._volumes
        
        if self.cache_path.exists() and not force:
            raw = json.loads(self.cache_path.read_text())
            self._volumes = [VolumeInfo(**v) for v in raw]
            return self._volumes
            
        vols = self._list_volumes()
        log.info("Characterizing %d CDR volumes from ASU Master...", len(vols))
        out: list[VolumeInfo] = []
        
        for v in vols:
            try:
                info = self._characterize_volume(v)
                out.append(info)
                log.info(f"  {info.volume_id} OK: {info.first_pid} -> {info.last_pid}")
            except Exception as e:
                log.warning(f"  Skipping {v}: {e}")
                
        self._volumes = out
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps([v.__dict__ for v in out], indent=2))
        return out

    def _binary_search(self, vol: VolumeInfo, target: str) -> Optional[tuple[str, str, str]]:
        lo = vol.first_row_index
        hi = vol.row_count - 1
        
        while lo <= hi:
            mid = (lo + hi) // 2
            rec = self._read_record(vol.index_url, mid, vol.record_bytes)
            _, pid, fsn = _parse_record(rec)
            
            if pid == target:
                return pid, fsn, rec
            if pid < target:
                lo = mid + 1
            else:
                hi = mid - 1
                
        return None

    def _parse_geometry(self, record: str) -> dict:
        out: dict = {}
        for key, sl in GEOMETRY_SLICES.items():
            raw = _strip_quoted(record[sl])
            try:
                out[key] = float(raw) if raw else None
            except ValueError:
                out[key] = None
        return out

    def _lookup(self, product_id: str) -> tuple[VolumeInfo, str, str, str]:
        pid = _normalize_product_id(product_id, self.suffix)
        
        for v in self.volumes():
            if v.first_pid <= pid <= v.last_pid:
                hit = self._binary_search(v, pid)
                if hit is not None:
                    _, fsn, rec = hit
                    return v, pid, fsn, rec
                    
        raise KeyError(f"Product {product_id!r} not found in any volume")

    def url_for(self, product_id: str) -> str:
        _, _, fsn, _ = self._lookup(product_id)
        root_marker = f"/LRO-L-LROC-{'2-EDR' if self.archive == 'EDR' else '3-CDR'}-V1.0"
        return self.base_url.rsplit(root_marker, 1)[0] + "/" + fsn

    def geometry_for(self, product_id: str) -> dict:
        _, pid, fsn, rec = self._lookup(product_id)
        geom = self._parse_geometry(rec)
        geom["product_id"] = pid
        geom["file_spec_name"] = fsn
        return geom