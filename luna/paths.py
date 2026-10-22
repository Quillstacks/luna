"""Deterministic resolution of Luna's ``data/`` and ``catalogs/`` files.

Never derive these locations from ``__file__`` parent traversal: when Luna is
installed into ``site-packages`` that points at ``site-packages/data`` (which
does not exist and must never be written to).

Lookup priority for *reading* a file named ``name`` of kind ``data`` or
``catalogs``:

1. Explicit environment variables ``LUNA_DATA_DIR`` / ``LUNA_CATALOG_DIR``
   (``LUNA_PDS_CACHE_DIR`` is additionally honoured for ``data``).
2. The caller's working directory: ``Path.cwd() / "data"`` / ``"catalogs"``.
3. Resources bundled inside the ``luna`` package (``luna/resources/<kind>``).
4. The user cache directory ``~/.cache/luna`` (writable runtime caches).

Writes never go to the package directory; see :func:`writable_path`.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "BUNDLED_DIR",
    "candidate_paths",
    "find_file",
    "user_cache_dir",
    "writable_path",
]

DATA = "data"
CATALOGS = "catalogs"
_ENV = {DATA: "LUNA_DATA_DIR", CATALOGS: "LUNA_CATALOG_DIR"}

BUNDLED_DIR = Path(__file__).resolve().parent / "resources"


def user_cache_dir() -> Path:
    """Return the per-user writable cache directory (``~/.cache/luna``)."""
    return Path.home() / ".cache" / "luna"


def _env_dirs(kind: str) -> list[Path]:
    dirs = []
    for var in (_ENV[kind], "LUNA_PDS_CACHE_DIR" if kind == DATA else ""):
        value = os.environ.get(var) if var else None
        if value:
            dirs.append(Path(value).expanduser())
    return dirs


def candidate_paths(name: str, kind: str = DATA) -> list[Path]:
    """All locations searched for ``name``, in priority order."""
    if kind not in _ENV:
        raise ValueError(f"kind must be 'data' or 'catalogs', got {kind!r}")
    paths = [
        *(d / name for d in _env_dirs(kind)),
        Path.cwd() / kind / name,
    ]
    if kind == DATA:
        # Some projects keep index files (e.g. pds_volume_index_cdr.json) in catalogs/.
        paths.append(Path.cwd() / CATALOGS / name)
    paths.append(BUNDLED_DIR / kind / name)
    if kind == DATA:
        paths.append(BUNDLED_DIR / CATALOGS / name)
    paths.append(user_cache_dir() / name)
    return paths


def find_file(name: str, kind: str = DATA) -> Path | None:
    """Return the first existing candidate for ``name`` or ``None``."""
    for p in candidate_paths(name, kind):
        if p.is_file():
            return p
    return None


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def writable_path(name: str, kind: str = DATA) -> Path:
    """Where a runtime cache file ``name`` should be written.

    Prefers an existing file's location (unless it is bundled, i.e. read-only
    package data), then the env dir, then ``cwd/<kind>`` if that directory
    already exists, and finally ``~/.cache/luna``.
    """
    existing = find_file(name, kind)
    if (
        existing is not None
        and not _is_inside(existing, BUNDLED_DIR)
        and os.access(existing, os.W_OK)
    ):
        return existing
    for d in _env_dirs(kind)[:1]:
        return d / name
    cwd_dir = Path.cwd() / kind
    if cwd_dir.is_dir() and os.access(cwd_dir, os.W_OK):
        return cwd_dir / name
    return user_cache_dir() / name
