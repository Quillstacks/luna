"""ROI coverage solver (mosaicking) for adjacent NAC images.

Selects a minimal set of adjacent/overlapping LROC NAC images to cover
a Region of Interest (ROI) while maintaining lighting consistency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from shapely.geometry import Polygon
from shapely.validation import make_valid
from shapely.ops import unary_union

from luna.io.nac_reader import get_nacs_from_polygon

log = logging.getLogger(__name__)


def select_coverage_nacs(
    roi_coords: list[tuple[float, float]],
    max_resolution: float = 1.5,
    min_incidence: float = 30.0,
    max_incidence: float = 60.0,
    min_overlap_ratio: float = 0.01,
) -> list[str]:
    """Select a minimal covering set of LROC NAC images to scan an ROI.

    Enforces lighting consistency by grouping candidate images by orbit track
    (along-track) and prioritizing them over cross-track jumps.

    Parameters
    ----------
    roi_coords :
        Vertices of the target Region of Interest (ROI) as (lon, lat) tuples.
    max_resolution :
        Maximum ground resolution in meters per pixel.
    min_incidence :
        Minimum solar incidence angle (degrees).
    max_incidence :
        Maximum solar incidence angle (degrees).
    min_overlap_ratio :
        Stop coverage loop when the remaining uncovered ROI is less than this
        fraction of the original ROI area.

    Returns
    -------
    list[str]
        List of LROC NAC product IDs forming the mosaic.
    """
    if len(roi_coords) < 3:
        raise ValueError("ROI requires at least 3 points defining a closed polygon.")

    # Normalize ROI coordinates to -180 to 180 range
    norm_roi_coords = []
    for lon, lat in roi_coords:
        norm_lon = (lon + 180) % 360 - 180
        norm_roi_coords.append((norm_lon, lat))

    roi_poly = Polygon(norm_roi_coords)
    if not roi_poly.is_valid:
        roi_poly = make_valid(roi_poly)

    log.info("Querying QuickMap for ROI polygon...")
    features = get_nacs_from_polygon(roi_coords)
    log.info("QuickMap returned %d candidate images.", len(features))

    # Parse and filter candidate features
    candidates = []
    for feat in features:
        props = feat.get("properties", {})
        pid = props.get("label")
        attrs = props.get("attributes", {})

        # 1. Filter by resolution
        res = attrs.get("Resolution")
        if res is not None and res > max_resolution:
            continue

        # 2. Filter by incidence angle
        inc = attrs.get("Incidence")
        if inc is not None and (inc < min_incidence or inc > max_incidence):
            continue

        # 3. Parse footprint geometry
        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue

        coords = geom.get("coordinates", [[]])[0]
        if len(coords) < 3:
            continue

        # Normalize candidate footprint coordinates to -180 to 180 range
        norm_coords = []
        for lon, lat in coords:
            norm_lon = (lon + 180) % 360 - 180
            norm_coords.append((norm_lon, lat))

        poly = Polygon(norm_coords)
        if not poly.is_valid:
            poly = make_valid(poly)

        # Check intersection with ROI
        intersection = roi_poly.intersection(poly)
        if intersection.is_empty:
            continue

        candidates.append({
            "product_id": pid,
            "polygon": poly,
            "intersection": intersection,
            "orbit": attrs.get("Orbit", 0),
            "resolution": res or 1.0,
            "incidence": inc or 45.0,
        })

    log.info("%d candidates passed initial resolution and incidence filtering.", len(candidates))
    if not candidates:
        return []

    # Group by Orbit to solve domain gap / lighting consistency
    orbit_groups = {}
    for c in candidates:
        orbit = c["orbit"]
        if orbit not in orbit_groups:
            orbit_groups[orbit] = []
        orbit_groups[orbit].append(c)

    # Sort orbits by how much they cover the ROI
    # For each orbit, compute the union of the candidate intersections with ROI
    orbit_coverage = []
    for orbit, group in orbit_groups.items():
        union_poly = unary_union([c["intersection"] for c in group])
        orbit_coverage.append((orbit, union_poly.area, union_poly))

    # Sort descending by coverage area
    orbit_coverage.sort(key=lambda x: x[1], reverse=True)

    selected_pids = []
    uncovered = roi_poly

    log.info("Starting greedy along-track coverage selection...")

    # Phase 1: Try to cover as much as possible using orbits in order of their coverage capacity
    for orbit, cover_area, union_poly in orbit_coverage:
        if uncovered.area / roi_poly.area < min_overlap_ratio:
            break

        # Filter orbit candidates that still intersect uncovered ROI
        orbit_candidates = [
            c for c in orbit_groups[orbit]
            if c["product_id"] not in selected_pids
        ]

        while orbit_candidates:
            # Select the one in this orbit that covers the most of the remaining uncovered ROI
            best_cand = None
            best_area = 0.0

            for c in orbit_candidates:
                cand_cover = c["polygon"].intersection(uncovered)
                if cand_cover.area > best_area:
                    best_area = cand_cover.area
                    best_cand = c

            # If the best candidate adds virtually no coverage, stop selecting from this orbit
            if best_cand is None or best_area / roi_poly.area < 1e-4:
                break

            selected_pids.append(best_cand["product_id"])
            uncovered = uncovered.difference(best_cand["polygon"])
            orbit_candidates.remove(best_cand)

    log.info("Selected %d images covering %.1f%% of ROI.",
             len(selected_pids), (1.0 - uncovered.area / roi_poly.area) * 100)

    return selected_pids
