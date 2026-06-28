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

log = logging.getLogger("luna.io.pds_fetch")

_index: Optional[PDSIndex] = None


def _shared_index() -> PDSIndex:
    global _index
    if _index is None:
        _index = PDSIndex()
    return _index


def fetch_nac(
    product_id: str,
    dest_dir: str | os.PathLike = "data",
    url: Optional[str] = None,
    force: bool = False,
    retries: int = 3,
    backoff_s: float = 4.0,
) -> Path:
    """Download a NAC CDR ``.IMG`` into ``dest_dir``. Returns the local path.

    If ``url`` is given, it is used verbatim (preferred for reproducibility).
    Otherwise PDSIndex resolves the product ID to its archive URL.
    Skips download if the file already exists unless ``force`` is True.

    Retries on transient network errors (``ConnectionError``, ``Timeout``,
    ``ChunkedEncodingError``) with exponential backoff; partial files are
    discarded between attempts.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    pid = _normalize_product_id(product_id)
    out = dest_dir / f"{pid}.IMG"
    if out.exists() and not force:
        return out

    resolved = url or _shared_index().url_for(product_id)
    transient = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
    )
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(resolved, stream=True, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                with open(out, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, desc=pid
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
                        bar.update(len(chunk))
            return out
        except transient as e:
            last_err = e
            out.unlink(missing_ok=True)
            if attempt == retries:
                break
            wait = backoff_s * (2 ** (attempt - 1))
            log.warning("fetch %s attempt %d/%d failed (%s); retrying in %.1fs",
                        pid, attempt, retries, e, wait)
            time.sleep(wait)
    raise NACNotFoundError(f"fetch_nac({pid}) failed after {retries} attempts: {last_err}")
