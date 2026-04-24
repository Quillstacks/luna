"""Pure-stdlib LPA catalog loader. No heavy deps so the discovery plumbing
stays importable on a bare-numpy environment.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LPAEntry:
    id: str
    name: str
    host: str
    host_type: str
    latitude: float
    longitude: float
    funnel_max_m: Optional[float]
    funnel_min_m: Optional[float]
    inner_max_m: Optional[float]
    azimuth_deg: Optional[float]
    depth_m: Optional[float]
    reference_nac: list[str]


def read_lpa_csv(path: str | Path) -> list[LPAEntry]:
    out: list[LPAEntry] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except (KeyError, ValueError):
                continue
            refs = [r for r in (row.get("reference_nac") or "").split(";") if r]
            out.append(LPAEntry(
                id=row.get("id", ""),
                name=row.get("name", ""),
                host=row.get("host", ""),
                host_type=row.get("host_type", ""),
                latitude=lat,
                longitude=lon,
                funnel_max_m=_maybe_float(row.get("funnel_max_m")),
                funnel_min_m=_maybe_float(row.get("funnel_min_m")),
                inner_max_m=_maybe_float(row.get("inner_max_m")),
                azimuth_deg=_maybe_float(row.get("azimuth_deg")),
                depth_m=_maybe_float(row.get("depth_m")),
                reference_nac=refs,
            ))
    return out


def _maybe_float(v: Optional[str]) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None
