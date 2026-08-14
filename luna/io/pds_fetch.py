"""Fetch LROC NAC CDR products from the PDS archive by product ID.

A NAC product ID looks like ``M126710873R`` (L/R = NAC-left / NAC-right). The
archived PRODUCT_ID gets a trailing ``C`` (CDR) — e.g. ``M126710873RC.IMG``.
Resolution goes through :class:`luna.io.pds_index.PDSIndex`, which binary-
searches the per-volume ``INDEX/INDEX.TAB`` over HTTP range requests.

Only CDR (calibrated) products are used here; EDR handling is out of scope.
"""

from __future__ import annotations

import logging
import os
import time

from luna.exceptions import NACNotFoundError
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from .pds_index import PDSIndex, _normalize_product_id

log = logging.getLogger(__name__)

DEFAULT_USER_AGENT = os.getenv(
    "LUNA_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
)

DEFAULT_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://data.lroc.im-ldi.com/",
}

_index: Optional[PDSIndex] = None
_session: Optional[requests.Session] = None


def get_pds_session(user_agent: Optional[str] = None) -> requests.Session:
    """Creates or returns a configured requests.Session with proper User-Agent and connection pooling."""
    global _session
    if user_agent is not None:
        s = requests.Session()
        headers = dict(DEFAULT_HEADERS)
        headers["User-Agent"] = user_agent
        s.headers.update(headers)
        return s
    if _session is None:
        _session = requests.Session()
        _session.headers.update(DEFAULT_HEADERS)
    return _session


def _shared_index() -> PDSIndex:
    global _index
    if _index is None:
        _index = PDSIndex()
    return _index


class BandwidthLimiter:
    """Rate limiter for controlling download bandwidth usage.
    
    Uses a token bucket algorithm to enforce a maximum byte rate.
    Note: The input parameter is in MB/s (Megabytes per second), not Mbps (Megabits).
    
    Attributes:
        max_bytes_per_second: Maximum bytes per second allowed (None = unlimited).
        tokens: Current available tokens (bytes) in the bucket.
        last_update: Timestamp of last token addition.
    """
    
    def __init__(self, max_bytes_per_second: Optional[float] = None) -> None:
        self.max_bytes_per_second = max_bytes_per_second
        self.tokens: float = 0.0
        self.last_update: float = time.perf_counter()
        if self.max_bytes_per_second is not None:
            self.tokens = float(self.max_bytes_per_second)
    
    def acquire(self, num_bytes: int) -> float:
        """Acquire permission to transfer num_bytes.
        
        Uses a token bucket algorithm to enforce bandwidth limiting.
        
        Args:
            num_bytes: Number of bytes to be transferred.
            
        Returns:
            Time in seconds to sleep before transferring, or 0.0 if no wait needed.
        """
        if self.max_bytes_per_second is None:
            return 0.0
            
        now = time.perf_counter()
        elapsed = now - self.last_update
        self.last_update = now
        
        self.tokens += elapsed * self.max_bytes_per_second
        self.tokens = min(self.tokens, float(self.max_bytes_per_second))
        
        self.tokens -= num_bytes
        
        if self.tokens < 0:
            wait_time = -self.tokens / self.max_bytes_per_second
            self.tokens = 0
            return wait_time
        
        return 0.0


def fetch_nac(
    product_id: str,
    dest_dir: str | os.PathLike = "data",
    url: Optional[str] = None,
    force: bool = False,
    retries: int = 3,
    backoff_s: float = 2.0,
    chunk_size: int = 1 << 20,
    timeout: tuple[float, float] | float = (10.0, 30.0),
    max_bandwidth_mbps: Optional[float] = None,
    session: Optional[requests.Session] = None,
    user_agent: Optional[str] = None,
) -> Path:
    """Download a NAC CDR ``.IMG`` into ``dest_dir``. Returns the local path.

    If ``url`` is given, it is used verbatim (preferred for reproducibility).
    Otherwise PDSIndex resolves the product ID to its archive URL.
    Skips download if the file already exists unless ``force`` is True.

    Downloads to a temporary ``.IMG.tmp`` file and atomically renames it
    upon successful completion to prevent corrupted or partial files on disk.

    Retries on transient network errors (``ConnectionError``, ``Timeout``,
    ``ChunkedEncodingError``, ``HTTPError``) with exponential backoff; partial temporary files
    are discarded between attempts.
    
    Args:
        product_id: LROC NAC product ID to download.
        dest_dir: Destination directory for the downloaded file.
        url: Optional direct URL to download from.
        force: If True, re-download even if file exists.
        retries: Number of retry attempts on network errors.
        backoff_s: Base backoff time in seconds for retries.
        chunk_size: Stream buffer chunk size in bytes (default: 1 MB / 1<<20).
        timeout: Socket timeout configuration; tuple of (connect_timeout, read_timeout)
            or single float in seconds.
        max_bandwidth_mbps: Optional bandwidth limit in MB/s (Megabytes per second).
            IMPORTANT: This is Megabytes, NOT Megabits! If None, no limit is applied.
            If specified, download speed will be capped at this rate to prevent network
            saturation. Example: 10 = 10 MB/s = 80 Mbps.
        session: Optional pre-configured requests.Session (defaults to get_pds_session()).
        user_agent: Optional custom User-Agent string.
    
    Returns:
        Path to the downloaded .IMG file.
    
    Raises:
        NACNotFoundError: If download fails after all retry attempts.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    pid = _normalize_product_id(product_id)
    out = dest_dir / f"{pid}.IMG"
    temp_out = dest_dir / f"{pid}.IMG.tmp"

    if out.exists() and not force:
        return out

    resolved = url or _shared_index().url_for(product_id)
    http_session = session or get_pds_session(user_agent)
    
    # Initialize bandwidth limiter if limit is specified
    bandwidth_limiter = BandwidthLimiter(
        max_bytes_per_second=(max_bandwidth_mbps * (1024 ** 2)) if max_bandwidth_mbps is not None else None
    )
    
    transient = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.HTTPError,
    )
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            # Differentiated timeout (connect_timeout, read_timeout) with configured session
            with http_session.get(resolved, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                with open(temp_out, "wb") as f, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=pid,
                    mininterval=0.2,  # Throttle console redraws to save CPU cycles
                ) as bar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        # Apply bandwidth limiting
                        if bandwidth_limiter.max_bytes_per_second is not None:
                            wait_time = bandwidth_limiter.acquire(len(chunk))
                            if wait_time > 0:
                                time.sleep(wait_time)
                        
                        f.write(chunk)
                        bar.update(len(chunk))

            # Atomic rename once download is completely validated
            temp_out.replace(out)
            return out
        except transient as e:
            last_err = e
            temp_out.unlink(missing_ok=True)
            if attempt == retries:
                break
            wait = backoff_s * (2 ** (attempt - 1))
            log.warning("fetch %s attempt %d/%d failed (%s); retrying in %.1fs",
                        pid, attempt, retries, e, wait)
            time.sleep(wait)
    raise NACNotFoundError(f"fetch_nac({pid}) failed after {retries} attempts: {last_err}")

