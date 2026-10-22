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
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from luna.exceptions import IndexError
from luna.paths import candidate_paths, find_file, writable_path

log = logging.getLogger(__name__)

DEFAULT_BASE = (
    "https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/"
    "pds4/lroc/lro-l-lroc-3-cdr"
)
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

# NOTE: data locations are resolved by luna.paths (env > cwd > bundled > ~/.cache/luna),
# never relative to this file (that would point into site-packages when installed).


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
    text = text.strip("\r\n")
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
    volume directories (``LROLRC_10xx``), each containing a fixed-width ``INDEX.TAB``.
    ``PDSIndex`` uses HTTP range-requests to perform logarithmic binary search directly
    over remote index files, resolving download URLs and observation geometry in
    milliseconds.

    Attributes:
        base_url: Base HTTP URL of the PDS archive.
        archive: Archive type, either ``"CDR"`` (calibrated) or ``"EDR"`` (raw).
        suffix: Product suffix character (``"C"`` for CDR, ``"E"`` for EDR).
    """

    def __init__(
        self,
        base_url: str | None = None,
        cache_path: Path | str | None = None,
        session: requests.Session | None = None,
        archive: str = "CDR",
        allow_remote_characterization: bool = False,
        cache_dir: Path | str | None = None,
    ) -> None:
        """Initialize a PDSIndex resolver.

        Args:
            base_url: Optional custom PDS archive root URL. If None, uses default
                CDR/EDR endpoints.
            cache_path: Optional explicit path of the volume boundary index JSON.
            session: Optional custom ``requests.Session`` with customized adapters.
            archive: Archive type to query (``"CDR"`` or ``"EDR"``, case-insensitive).
            allow_remote_characterization: If False (default), a missing volume
                index raises ``RuntimeError`` instead of silently scraping every
                volume from PDS. Set True or call :meth:`build_volume_index`.
            cache_dir: Optional directory searched first (and written to) for
                the volume index, URL cache and geometry cache.
        """
        self.allow_remote_characterization = allow_remote_characterization
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
        if session is not None:
            self.session = session
        else:
            self.session = requests.Session()
            self.session.headers.update(
                {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept": "*/*",
                    "Referer": "https://data.lroc.im-ldi.com/",
                }
            )
            retries = Retry(
                total=5,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "HEAD"],
                respect_retry_after_header=False,
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retries)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)

        self.archive = archive.upper()
        self.suffix = {"CDR": "C", "EDR": "E"}.get(self.archive, "C")

        # Load local geometry cache if it exists
        self.geom_cache_path = self._locate("pds_geometry_cache.json")
        self._geom_cache: dict[str, dict[str, Any]] = {}
        if self.geom_cache_path.exists():
            try:
                self._geom_cache = json.loads(self.geom_cache_path.read_text())
            except Exception as e:
                log.warning("Failed to load geometry cache: %s", e)

        if base_url is None:
            self.base_url = EDR_BASE if self.archive == "EDR" else DEFAULT_BASE
        else:
            self.base_url = base_url.rstrip("/")

        volume_index_name = f"pds_volume_index_{self.archive.lower()}.json"
        self.searched_paths: list[Path] = []
        if self.cache_dir is not None:
            self.searched_paths.append(self.cache_dir / volume_index_name)
        self.searched_paths.extend(candidate_paths(volume_index_name, "data"))
        if cache_path is None:
            cache_path = self._locate(volume_index_name)
        else:
            self.searched_paths.insert(0, Path(cache_path))

        self.cache_path = Path(cache_path)
        self._volumes: list[VolumeInfo] | None = None

        # Optional local Parquet URL index cache
        self.url_cache_parquet = self._locate("pds_cdr_url_cache.parquet")
        self._url_cache: dict[str, str] = {}
        if self.url_cache_parquet.exists():
            try:
                import pyarrow.parquet as pq

                tbl = pq.read_table(
                    self.url_cache_parquet, columns=["product_id", "file_spec"]
                )
                pids = tbl["product_id"].to_pylist()
                fsns = tbl["file_spec"].to_pylist()
                for p, f in zip(pids, fsns):
                    self._url_cache[p] = self._format_url_from_fsn(f)
                log.info(
                    "Loaded %d local PDS URL mappings from %s",
                    len(self._url_cache),
                    self.url_cache_parquet.name,
                )
            except Exception as e:
                log.warning("Failed to load Parquet URL cache: %s", e)

    def _locate(self, name: str) -> Path:
        """Find ``name`` (existing) or return the path where it should be written."""
        if self.cache_dir is not None:
            p = self.cache_dir / name
            if p.exists():
                return p
            found = find_file(name, "data")
            return found if found is not None else p
        return find_file(name, "data") or writable_path(name, "data")

    def _format_url_from_fsn(self, fsn: str) -> str:
        """Convert a PDS FILE_SPECIFICATION_NAME into a direct download URL."""
        fsn_clean = fsn.strip()
        if "pds.mcp.nasa.gov" in self.base_url:
            if "/" in fsn_clean and "LRO-L-LROC" in fsn_clean.split("/")[0]:
                fsn_clean = fsn_clean.split("/", 1)[1]
            return f"{self.base_url.rstrip('/')}/{fsn_clean}"
        root_marker = f"/LRO-L-LROC-{'2-EDR' if self.archive == 'EDR' else '3-CDR'}-V1.0"
        return self.base_url.rsplit(root_marker, 1)[0] + "/" + fsn_clean

    def _list_volumes(self) -> list[str]:
        prefix = "1" if self.archive == "CDR" else "0"
        try:
            r = self.session.get(
                "https://pds-imaging.jpl.nasa.gov/volumes/lro.html", timeout=30
            )
            if r.status_code == 200:
                vols = sorted(
                    set(re.findall(rf"\b(LROLRC_{prefix}\d{{3}}[A-Z]*)\b", r.text))
                )
                if vols:
                    return vols
        except Exception as e:
            log.warning("Failed to fetch volumes from JPL: %s", e)

        try:
            r = self.session.get(self.base_url + "/", timeout=60)
            r.raise_for_status()
            vols = sorted(set(re.findall(r'href="(LROLRC_\d+\w*)/"', r.text)))
            if vols:
                return vols
        except Exception as e:
            log.warning("Failed to fetch volumes from base_url: %s", e)
        return []

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
        for attempt in range(5):
            r = self.session.get(
                url, headers={"Range": f"bytes={start}-{end}"}, timeout=20
            )
            if r.status_code in (200, 206):
                return r.content.decode("ascii", errors="replace")
            if r.status_code == 429:
                sleep_s = 1.0 * (2**attempt)
                log.debug(
                    "HTTP 429 on index read %s, retrying in %.1fs...", url, sleep_s
                )
                time.sleep(sleep_s)
                continue
            raise IndexError(f"HTTP {r.status_code} on range read {url}")
        raise IndexError(f"HTTP 429 on range read {url} after 5 retries")

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
            last_pid=last_pid,
        )

    def volumes(self, force: bool = False) -> list[VolumeInfo]:
        """Return the volume boundary index.

        Loads the local JSON index. If it is missing, raises ``RuntimeError``
        unless ``allow_remote_characterization`` is True (or ``force=True`` /
        :meth:`build_volume_index` is called explicitly).
        """
        if self._volumes is not None and not force:
            return self._volumes

        if self.cache_path.exists() and not force:
            raw = json.loads(self.cache_path.read_text())
            self._volumes = [VolumeInfo(**v) for v in raw]
            return self._volumes

        if not force and not self.allow_remote_characterization:
            searched = "\n  ".join(str(p) for p in self.searched_paths)
            raise RuntimeError(
                f"PDS volume index for {self.archive} not found. Searched:\n  {searched}\n"
                "Refusing to scrape all PDS volumes automatically (risks HTTP 429). "
                "Provide the file via LUNA_DATA_DIR / ./data / cache_dir / cache_path, "
                "call PDSIndex.build_volume_index() explicitly, or construct "
                "PDSIndex(allow_remote_characterization=True)."
            )
        return self.build_volume_index()

    def build_volume_index(self) -> list[VolumeInfo]:
        """Explicitly characterize every PDS volume remotely and cache the result."""
        vols = self._list_volumes()
        log.info("Characterizing %d CDR volumes from PDS...", len(vols))
        out: list[VolumeInfo] = []

        for v in vols:
            try:
                info = self._characterize_volume(v)
                out.append(info)
                log.info("  %s OK: %s -> %s", info.volume_id, info.first_pid, info.last_pid)
            except Exception as e:
                log.warning("  Skipping %s: %s", v, e)

        self._volumes = out
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps([v.__dict__ for v in out], indent=2))
        return out

    def _binary_search(
        self, vol: VolumeInfo, target: str
    ) -> tuple[str, str, str] | None:
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

    def _parse_geometry(self, record: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        try:
            row = next(csv.reader(io.StringIO(record.strip("\r\n"))))
            col_map = {
                "image_lines": 52,
                "line_samples": 53,
                "resolution": 57,
                "emission_angle": 58,
                "incidence_angle": 59,
                "phase_angle": 60,
                "sub_solar_latitude": 63,
                "sub_solar_longitude": 64,
                "sub_spacecraft_latitude": 65,
                "sub_spacecraft_longitude": 66,
                "center_latitude": 69,
                "center_longitude": 70,
                "upper_right_latitude": 71,
                "upper_right_longitude": 72,
                "lower_right_latitude": 73,
                "lower_right_longitude": 74,
                "lower_left_latitude": 75,
                "lower_left_longitude": 76,
                "upper_left_latitude": 77,
                "upper_left_longitude": 78,
                "spacecraft_altitude": 79,
            }
            if len(row) >= 80:
                for key, idx in col_map.items():
                    raw = _strip_quoted(row[idx]) if len(row) > idx else ""
                    try:
                        out[key] = float(raw) if raw else None
                    except (ValueError, TypeError):
                        out[key] = None
                return out
        except Exception:
            pass

        for key, sl in GEOMETRY_SLICES.items():
            raw = _strip_quoted(record[sl]) if len(record) > sl.stop else ""
            try:
                out[key] = float(raw) if raw else None
            except ValueError:
                out[key] = None
        return out

    def _lookup(self, product_id: str) -> tuple[VolumeInfo, str, str, str]:
        pid = _normalize_product_id(product_id, self.suffix)

        log.warning(
            "No local URL/geometry cache entry for %s; performing remote INDEX.TAB "
            "binary search against PDS.",
            pid,
        )
        for v in self.volumes():
            if v.first_pid <= pid <= v.last_pid:
                hit = self._binary_search(v, pid)
                if hit is not None:
                    _, fsn, rec = hit
                    return v, pid, fsn, rec

        raise KeyError(f"Product {product_id!r} not found in any volume")

    def url_for(self, product_id: str) -> str:
        """Resolve a full download URL for an LROC NAC product ID.

        Args:
            product_id: Product ID string (e.g., ``"M1118880788RC"`` or ``"M1118880788R"``).

        Returns:
            Direct HTTPS download URL to the product raster file on the PDS server.

        Raises:
            KeyError: If the product ID is not found in any volume index.
        """
        if not hasattr(self, "_url_cache"):
            self._url_cache = {}
        pid = _normalize_product_id(product_id, self.suffix)
        if pid in self._url_cache:
            return self._url_cache[pid]

        if (
            hasattr(self, "_geom_cache")
            and pid in self._geom_cache
            and "file_spec_name" in self._geom_cache[pid]
        ):
            fsn = self._geom_cache[pid]["file_spec_name"]
        else:
            _, _, fsn, _ = self._lookup(product_id)

        res_url = self._format_url_from_fsn(fsn)
        self._url_cache[pid] = res_url
        return res_url

    def geometry_for(self, product_id: str) -> dict[str, Any]:
        """Retrieve the parsed INDEX geometry metadata dictionary for a product ID.

        Args:
            product_id: Product ID string (e.g., ``"M1118880788RC"``).

        Returns:
            Dictionary containing corner coordinates (``upper_left_latitude``,
            ``upper_left_longitude``, etc.), photometric angles (``emission_angle``,
            ``incidence_angle``, ``phase_angle``), and ground resolution.

        Raises:
            KeyError: If the product is not found in the PDS index.
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
    def update_geometry_cache_from_geojson(
        features: list[dict[str, Any]], archive: str = "CDR"
    ) -> None:
        """Persist footprint geometry from QuickMap GeoJSON features into the local cache."""
        geom_cache_path = writable_path("pds_geometry_cache.json", "data")

        cache: dict[str, Any] = {}
        existing = find_file("pds_geometry_cache.json", "data")
        if existing is not None:
            try:
                cache = json.loads(existing.read_text())
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

            def norm_lon(lon: float) -> float:
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
                "sub_solar_longitude": (
                    None
                    if attrs.get("SubSol Lon") is None
                    else norm_lon(attrs["SubSol Lon"])
                ),
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
            geom_cache_path.parent.mkdir(parents=True, exist_ok=True)
            geom_cache_path.write_text(json.dumps(cache, indent=2))
