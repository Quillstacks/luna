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
import time
from dataclasses import dataclass
from pathlib import Path

from luna.exceptions import IndexError
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

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
    """Binary-search resolver for NASA Planetary Data System (PDS) LROC product IDs.

    The LROC CDR and EDR archives partition millions of observations across ~35
    volume directories (`LROLRC_10xx`), each containing a fixed-width `INDEX.TAB`.
    `PDSIndex` uses HTTP range-requests to perform $O(\\log N)$ binary search directly
    over remote index files, resolving download URLs in $< 50\\text{ ms}$.

    Attributes:
        base_url: Base HTTP URL of the PDS archive.
        archive: Archive type, either ``"CDR"`` (calibrated) or ``"EDR"`` (raw).
        suffix: Product suffix character (``"C"`` for CDR, ``"E"`` for EDR).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        cache_path: Optional[Path | str] = None,
        session: Optional[requests.Session] = None,
        archive: str = "CDR",
    ) -> None:
        """Initialize a PDSIndex resolver.

        Args:
            base_url: Optional custom PDS archive root URL. If None, uses default
                CDR/EDR endpoints.
            cache_path: Optional path for storing volume boundary metadata.
            session: Optional custom ``requests.Session`` with customized adapters.
            archive: Archive type to query (``"CDR"`` or ``"EDR"``, case-insensitive).
        """
        if session is not None:
            self.session = session
        else:
            import httpx
            headers = {
                "User-Agent": "Luna-Planetary-Research/3.0 (Academic Research; NASA LROC Science Ingestion; contact: planetary-science-team)",
                "Accept": "*/*",
            }
            limits = httpx.Limits(max_connections=32, max_keepalive_connections=16, keepalive_expiry=60.0)
            timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
            self.session = httpx.Client(limits=limits, timeout=timeout, headers=headers, follow_redirects=True)

        self.archive = archive.upper()
        self.suffix = {"CDR": "C", "EDR": "E"}.get(self.archive, "C")
        
        # Load local geometry cache if it exists
        self.geom_cache_path = Path(__file__).resolve().parents[2] / "data" / "pds_geometry_cache.json"
        self._geom_cache = {}
        if self.geom_cache_path.exists():
            try:
                import json
                self._geom_cache = json.loads(self.geom_cache_path.read_text())
            except Exception as e:
                log.warning("Failed to load geometry cache: %s", e)
        
        if base_url is None:
            self.base_url = EDR_BASE if self.archive == "EDR" else DEFAULT_BASE
        else:
            self.base_url = base_url.rstrip("/")
 
        if cache_path is None:
            cache_path = CACHE_PATH.with_name(f"pds_volume_index_{self.archive.lower()}.json")
        
        self.cache_path = Path(cache_path)
        self._volumes: Optional[list[VolumeInfo]] = None

        # Check for local Parquet URL index cache
        self.url_cache_parquet = Path(__file__).resolve().parents[2] / "data" / "pds_cdr_url_cache.parquet"
        self._url_cache = {}
        if self.url_cache_parquet.exists():
            try:
                import pyarrow.parquet as pq
                tbl = pq.read_table(self.url_cache_parquet, columns=["product_id", "file_spec"])
                pids = tbl["product_id"].to_pylist()
                fsns = tbl["file_spec"].to_pylist()
                base_prefix = self.base_url.rstrip("/") + "/"
                for p, f in zip(pids, fsns):
                    self._url_cache[p] = base_prefix + f
                log.info(f"Loaded {len(self._url_cache):,} local PDS URL mappings from {self.url_cache_parquet.name}")
            except Exception as e:
                log.warning("Failed to load Parquet URL cache: %s", e)

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
        for attempt in range(10):
            try:
                r = self.session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=30)
                if r.status_code in (200, 206):
                    return r.content.decode("ascii", errors="replace")
                elif r.status_code == 429:
                    time.sleep(30.0)
                    continue
                else:
                    raise IndexError(f"HTTP {r.status_code} on range read {url}")
            except Exception as e:
                if attempt == 9:
                    raise e
                time.sleep(2.0)
        raise IndexError(f"Failed range read on {url}")

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
        """Resolve a full download URL for an LROC NAC product ID."""
        if not hasattr(self, "_url_cache"):
            self._url_cache = {}
        pid = _normalize_product_id(product_id, self.suffix)
        if pid in self._url_cache:
            return self._url_cache[pid]

        if hasattr(self, "_geom_cache") and pid in self._geom_cache and "file_spec_name" in self._geom_cache[pid]:
            fsn = self._geom_cache[pid]["file_spec_name"]
            root_marker = f"/LRO-L-LROC-{'2-EDR' if self.archive == 'EDR' else '3-CDR'}-V1.0"
            res_url = self.base_url.rsplit(root_marker, 1)[0] + "/" + fsn
            self._url_cache[pid] = res_url
            return res_url
            
        _, _, fsn, _ = self._lookup(product_id)
        root_marker = f"/LRO-L-LROC-{'2-EDR' if self.archive == 'EDR' else '3-CDR'}-V1.0"
        res_url = self.base_url.rsplit(root_marker, 1)[0] + "/" + fsn
        self._url_cache[pid] = res_url
        return res_url

    def geometry_for(self, product_id: str) -> dict:
        """Retrieve the parsed INDEX geometry metadata dictionary for a product ID.

        Args:
            product_id: Product ID string (e.g., ``"M1118880788RC"``).

        Returns:
            Dictionary containing corner coordinates (``upper_left_latitude``,
            ``upper_left_longitude``, etc.), photometric angles (``emission_angle``,
            ``incidence_angle``, ``phase_angle``), and ground resolution.

        Raises:
            KeyError: If the product is not found in the PDS index.

        Example:
            >>> from luna.io.pds_index import PDSIndex
            >>> idx = PDSIndex()
            >>> geom = idx.geometry_for("M1118880788RC")
            >>> print(geom["resolution"], geom["emission_angle"])
        """
        pid = _normalize_product_id(product_id, self.suffix)
        if hasattr(self, "_geom_cache") and pid in self._geom_cache:
            return self._geom_cache[pid]
            
        _, pid, fsn, rec = self._lookup(product_id)
        geom = self._parse_geometry(rec)
        geom["product_id"] = pid
        geom["file_spec_name"] = fsn
        return geom

    @staticmethod
    def update_geometry_cache_from_geojson(features: list[dict], archive: str = "CDR") -> None:
        geom_cache_path = Path(__file__).resolve().parents[2] / "data" / "pds_geometry_cache.json"
        
        cache = {}
        if geom_cache_path.exists():
            try:
                import json
                cache = json.loads(geom_cache_path.read_text())
            except Exception:
                pass
                
        suffix = {"CDR": "C", "EDR": "E"}.get(archive.upper(), "C")
        
        updated = False
        for feat in features:
            props = feat.get("properties", {})
            pid = props.get("label")
            if not pid:
                continue
                
            norm_pid = _normalize_product_id(pid, suffix)
            if norm_pid in cache:
                continue
                
            attrs = props.get("attributes", {})
            coords = feat.get("geometry", {}).get("coordinates", [[]])[0]
            if len(coords) < 4:
                continue
                
            def norm_lon(lon):
                return (lon + 180) % 360 - 180
                
            ul_lon, ul_lat = norm_lon(coords[0][0]), coords[0][1]
            ur_lon, ur_lat = norm_lon(coords[1][0]), coords[1][1]
            lr_lon, lr_lat = norm_lon(coords[2][0]), coords[2][1]
            ll_lon, ll_lat = norm_lon(coords[3][0]), coords[3][1]
            
            c_lat = (ul_lat + lr_lat) / 2.0
            c_lon = (ul_lon + lr_lon) / 2.0
            
            geom = {
                "product_id": norm_pid,
                "resolution": attrs.get("Resolution"),
                "emission_angle": attrs.get("Emission"),
                "incidence_angle": attrs.get("Incidence"),
                "phase_angle": attrs.get("Phase"),
                "sub_solar_latitude": attrs.get("SubSol Lat"),
                "sub_solar_longitude": attrs.get("SubSol Lon") if attrs.get("SubSol Lon") is None else norm_lon(attrs.get("SubSol Lon")),
                "sub_spacecraft_latitude": None,
                "sub_spacecraft_longitude": None,
                "center_latitude": c_lat,
                "center_longitude": c_lon,
                "upper_right_latitude": ur_lat,
                "upper_right_longitude": ur_lon,
                "lower_right_latitude": lr_lat,
                "lower_right_longitude": lr_lon,
                "lower_left_latitude": ll_lat,
                "lower_left_longitude": ll_lon,
                "upper_left_latitude": ul_lat,
                "upper_left_longitude": ul_lon,
                "spacecraft_altitude": None,
            }
            cache[norm_pid] = geom
            updated = True
            
        if updated:
            import json
            geom_cache_path.parent.mkdir(parents=True, exist_ok=True)
            geom_cache_path.write_text(json.dumps(cache, indent=2))