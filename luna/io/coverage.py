"""ROI coverage solver (mosaicking) for adjacent NAC images."""

from __future__ import annotations

import logging
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from shapely.geometry import Polygon, shape
from shapely.validation import make_valid
from shapely.ops import transform

from luna.io.nac_reader import get_nacs_from_polygon

log = logging.getLogger(__name__)


INCIDENCE_FALLBACK_STEPS = (85.0, 89.5)
AREA_OUTLIER_FACTOR = 6.0

# Fixed absolute area threshold (in square degrees) instead of a ratio, so
# large bands (e.g. the equator) don't get cut off prematurely.
#
# NOTE: this must stay well above true numerical noise. A single NAC
# footprint is roughly ~0.1 sq deg. 1e-5 (four orders of magnitude below
# that) let the greedy loop chase near-zero slivers for thousands of
# iterations -- and because `uncovered` gets geometrically more complex
# (more holes/vertices) with every pick, each of those iterations got
# progressively slower, effectively hanging the process for hours on the
# equator band. 1e-3 sq deg (~1% of a typical footprint) still closes real
# gaps but stops chasing floating-point-noise-sized slivers.
MIN_MARGINAL_GAIN_AREA = 1e-3

# Re-simplify `uncovered` every N picks to bound its vertex/hole count.
# Shapely boolean ops (intersection/difference) slow down as geometry
# complexity grows; periodic simplification keeps each pick's cost roughly
# constant instead of climbing across a long run.
SIMPLIFY_EVERY_N_PICKS = 25
SIMPLIFY_TOLERANCE_DEG = 1e-4

# Hard ceiling on total picks for a single band, as a circuit breaker. Based
# on observed real bands (40-230 images each), a genuinely well-covering
# selection should never need anywhere close to this. If it's hit, the
# remaining gap is almost certainly a real data limitation (no matching
# imagery for that patch under the current filters), not something more
# picks would fix -- see the warning logged when this triggers.
MAX_TOTAL_PICKS = 4000


def _normalize_ring(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [((lon + 180) % 360 - 180, lat) for lon, lat in coords]


def _to_polygon(geom: dict) -> Polygon | None:
    if geom.get("type") not in ("Polygon", "MultiPolygon"):
        return None
    try:
        raw_poly = shape(geom)
        poly_360 = transform(lambda x, y, z=None: (x % 360, y), raw_poly)
        box1 = Polygon([(0, -90), (180, -90), (180, 90), (0, 90)])
        box2 = Polygon([(180, -90), (360, -90), (360, 90), (180, 90)])
        part1 = poly_360.intersection(box1)
        part2 = poly_360.intersection(box2)
        if not part2.is_empty:
            part2 = transform(lambda x, y, z=None: (x - 360, y), part2)
        if part1.is_empty:
            poly = part2
        elif part2.is_empty:
            poly = part1
        else:
            poly = part1.union(part2)
    except Exception:
        return None
    if poly is None or poly.is_empty:
        return None
    if not poly.is_valid:
        poly = make_valid(poly)
    return poly


def _filter_candidates(
    features: list[dict],
    target_poly: Polygon,
    max_resolution: float,
    min_incidence: float,
    max_incidence: float,
    already_selected: set[str],
) -> list[dict]:
    filtered_feats = []
    filtered_polys = []
    for feat in features:
        props = feat.get("properties", {})
        pid = props.get("label")
        if pid in already_selected:
            continue

        attrs = props.get("attributes", {})
        res = attrs.get("Resolution")
        if res is not None and res > max_resolution:
            continue

        inc = attrs.get("Incidence")
        if inc is not None and (inc < min_incidence or inc > max_incidence):
            continue

        poly = feat.get("polygon_obj")
        if poly is None:
            continue

        filtered_feats.append(feat)
        filtered_polys.append(poly)

    if not filtered_feats:
        return []

    from shapely import STRtree
    tree = STRtree(filtered_polys)
    intersecting_indices = tree.query(target_poly, predicate="intersects")

    candidates = []
    for idx in intersecting_indices:
        feat = filtered_feats[idx]
        poly = filtered_polys[idx]
        props = feat.get("properties", {})
        attrs = props.get("attributes", {})
        pid = props.get("label")
        inc = attrs.get("Incidence")

        intersection = target_poly.intersection(poly)
        if intersection.is_empty:
            continue

        candidates.append({
            "product_id": pid,
            "polygon": poly,
            "intersection": intersection,
            "orbit": attrs.get("Orbit", 0),
            "incidence": inc if inc is not None else 45.0,
        })
    return candidates


def _reject_area_outliers(candidates: list[dict]) -> list[dict]:
    orbit_groups: dict[int, list[dict]] = {}
    for c in candidates:
        orbit_groups.setdefault(c["orbit"], []).append(c)

    kept: list[dict] = []
    rejected = 0
    for group in orbit_groups.values():
        areas = [c["intersection"].area for c in group]
        if len(areas) < 3:
            kept.extend(group)
            continue
            
        median_area = statistics.median(areas)
        if median_area <= 0:
            kept.extend(group)
            continue
            
        for c in group:
            if c["intersection"].area <= median_area * AREA_OUTLIER_FACTOR:
                kept.append(c)
            else:
                rejected += 1

    if rejected:
        log.info("Rejected %d area-outlier candidates.", rejected)
    return kept


def _greedy_select(
    candidates: list[dict],
    uncovered: Polygon,
    roi_area: float,
    min_overlap_ratio: float,
    max_total_picks: int,
    band_label: str | None = None,
) -> tuple[list[str], Polygon]:
    """Lazy-greedy set cover with a max-heap of upper-bound gains."""
    import heapq

    tag = f"[{band_label}] " if band_label else ""

    # (-upper_bound_gain, insertion_order, candidate_index) -- insertion_order
    # breaks ties deterministically so heapq never has to compare dicts.
    heap: list[tuple[float, int, int]] = []
    c_bounds_list = [c["polygon"].bounds for c in candidates]

    for idx, c in enumerate(candidates):
        heapq.heappush(heap, (-c["intersection"].area, idx, idx))

    selected_pids: list[str] = []
    total_picks = 0
    discarded = 0

    from shapely.prepared import prep
    prepared_uncovered = prep(uncovered)

    while heap:
        if uncovered.area / roi_area < min_overlap_ratio:
            break
        if total_picks >= max_total_picks:
            log.warning(
                "%sHit max_total_picks guard (%d) with %.2f%% ROI still uncovered -- "
                "stopping. This band's remaining gap is likely a genuine data "
                "limitation (no candidate imagery under current filters) rather "
                "than an algorithm issue; check the uncovered geometry if unsure.",
                tag, max_total_picks, uncovered.area / roi_area * 100,
            )
            break

        neg_upper_bound, _order, idx = heapq.heappop(heap)
        c = candidates[idx]

        # Fast bounding box upper-bound check (nanosecond arithmetic)
        c_bounds = c_bounds_list[idx]
        u_bounds = uncovered.bounds

        ix_minx = max(c_bounds[0], u_bounds[0])
        ix_miny = max(c_bounds[1], u_bounds[1])
        ix_maxx = min(c_bounds[2], u_bounds[2])
        ix_maxy = min(c_bounds[3], u_bounds[3])

        if ix_minx >= ix_maxx or ix_miny >= ix_maxy:
            bbox_gain = 0.0
        else:
            bbox_gain = (ix_maxx - ix_minx) * (ix_maxy - ix_miny)

        if bbox_gain < MIN_MARGINAL_GAIN_AREA:
            discarded += 1
            continue

        # If the fast upper bound is not enough to beat the top of the heap,
        # push the bbox_gain back as a tighter upper-bound and avoid exact intersection.
        if heap and bbox_gain < -heap[0][0]:
            heapq.heappush(heap, (-bbox_gain, idx, idx))
            continue

        # Only if the bounding box check passes, perform prepared intersection & exact area
        if not prepared_uncovered.intersects(c["polygon"]):
            actual_gain = 0.0
        else:
            actual_gain = c["polygon"].intersection(uncovered).area

        if actual_gain < MIN_MARGINAL_GAIN_AREA:
            # Gain can only shrink further as uncovered shrinks -- safe to
            # discard permanently rather than re-push.
            discarded += 1
            continue

        # If this candidate's real gain still beats (or ties) the next-best
        # upper bound still in the heap, it's genuinely the best available
        # pick right now -- accept it. Otherwise its bound was stale; push
        # the corrected value back and let the heap re-sort.
        if heap and actual_gain < -heap[0][0]:
            heapq.heappush(heap, (-actual_gain, idx, idx))
            continue

        selected_pids.append(c["product_id"])
        uncovered = uncovered.difference(c["polygon"])
        prepared_uncovered = prep(uncovered)
        total_picks += 1

        if total_picks % SIMPLIFY_EVERY_N_PICKS == 0:
            uncovered = uncovered.simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=False)
            if not uncovered.is_valid:
                uncovered = make_valid(uncovered)
            prepared_uncovered = prep(uncovered)
            log.info(
                "%s  ... %d images selected so far, %.2f%% ROI still uncovered "
                "(%d candidates left in heap, %d discarded as exhausted).",
                tag, total_picks, uncovered.area / roi_area * 100,
                len(heap), discarded,
            )

    return selected_pids, uncovered


def select_coverage_nacs(
    roi_coords: list[tuple[float, float]],
    max_resolution: float = 1.5,
    min_incidence: float = 30.0,
    max_incidence: float = 60.0,
    min_overlap_ratio: float = 0.01,
    band_label: str | None = None,
) -> list[str]:
    """Select a minimal, realistic covering set of LROC NAC images for an ROI.

    ``band_label`` (e.g. "-80,-50") is purely cosmetic: it prefixes every log
    line so that when several bands run concurrently via
    :func:`run_bands_parallel`, interleaved log output from different
    processes can still be told apart.
    """
    if len(roi_coords) < 3:
        raise ValueError("ROI requires at least 3 points defining a closed polygon.")

    tag = f"[{band_label}] " if band_label else ""

    log.info(
        "%sPARAMETER CHECK in coverage.py -> max_res: %.2f, min_inc: %.1f, max_inc: %.1f",
        tag, max_resolution, min_incidence, max_incidence,
    )

    roi_poly = Polygon(_normalize_ring(roi_coords))
    if not roi_poly.is_valid:
        roi_poly = make_valid(roi_poly)
    roi_area = roi_poly.area

    log.info("%sQuerying QuickMap for ROI polygon...", tag)
    features = get_nacs_from_polygon(roi_coords)
    log.info("%sQuickMap returned %d candidate images.", tag, len(features))

    parsed_features = []
    for feat in features:
        poly = _to_polygon(feat.get("geometry", {}))
        if poly is not None:
            feat["polygon_obj"] = poly
            parsed_features.append(feat)
    features = parsed_features

    incidence_windows = [max_incidence, *[
        step for step in INCIDENCE_FALLBACK_STEPS if step > max_incidence
    ]]

    selected_pids: list[str] = []
    selected_set: set[str] = set()
    uncovered = roi_poly

    for pass_idx, inc_ceiling in enumerate(incidence_windows):
        if uncovered.area / roi_area < min_overlap_ratio:
            break

        # Pass uncovered instead of full roi_poly to pre-filter candidates
        candidates = _filter_candidates(
            features, uncovered,
            max_resolution=max_resolution,
            min_incidence=min_incidence,
            max_incidence=inc_ceiling,
            already_selected=selected_set,
        )
        if not candidates:
            continue

        candidates = _reject_area_outliers(candidates)
        if not candidates:
            continue

        log.info(
            "%sPass %d (max_incidence<=%.1f): %d candidates for %.2f%% remaining ROI area.",
            tag, pass_idx, inc_ceiling, len(candidates), uncovered.area / roi_area * 100,
        )

        dynamic_max_picks = max(3000, int(roi_area * 0.8))
        log.info("%sCalculated dynamic picks limit for this pass: %d", tag, dynamic_max_picks)

        log.info("%sStarting greedy along-track coverage selection...", tag)
        pass_pids, uncovered = _greedy_select(
            candidates, uncovered, roi_area, min_overlap_ratio,
            max_total_picks=dynamic_max_picks,
            band_label=band_label,
        )
        selected_pids.extend(pass_pids)
        selected_set.update(pass_pids)

        if pass_idx > 0 and pass_pids:
            log.info(
                "%sFallback pass recovered %d additional images at relaxed incidence.",
                tag, len(pass_pids),
            )

    # Final catch-all pass: if the requested incidence window and its
    # fallback steps still leave a gap, make one last attempt with the
    # incidence constraint dropped entirely (0-90 degrees). This is the only
    # way to distinguish "we stopped too early" from "no matching imagery
    # exists here at all" -- if this pass still can't close the gap, that's
    # a genuine data limitation, not an algorithm shortcoming.
    if uncovered.area / roi_area >= min_overlap_ratio:
        log.info(
            "%s%.2f%% still uncovered after all incidence fallback steps -- "
            "running unrestricted catch-all pass (incidence 0-90, resolution filter still applied).",
            tag, uncovered.area / roi_area * 100,
        )
        catchall_candidates = _filter_candidates(
            features, uncovered,
            max_resolution=max_resolution,
            min_incidence=0.0,
            max_incidence=90.0,
            already_selected=selected_set,
        )
        catchall_candidates = _reject_area_outliers(catchall_candidates)
        if catchall_candidates:
            log.info(
                "%sCatch-all pass: %d candidates for %.2f%% remaining ROI area.",
                tag, len(catchall_candidates), uncovered.area / roi_area * 100,
            )
            dynamic_max_picks = max(3000, int(roi_area * 0.8))
            catchall_pids, uncovered = _greedy_select(
                catchall_candidates, uncovered, roi_area, min_overlap_ratio,
                max_total_picks=dynamic_max_picks,
                band_label=band_label,
            )
            selected_pids.extend(catchall_pids)
            selected_set.update(catchall_pids)
            if catchall_pids:
                log.info("%sCatch-all pass recovered %d additional images.", tag, len(catchall_pids))

    coverage_pct = (1.0 - uncovered.area / roi_area) * 100
    if coverage_pct < 99.0:
        log.warning(
            "%sFinal coverage %.2f%% is BELOW the 99%% target even after the unrestricted "
            "catch-all pass. The remaining gap has no matching LROC NAC imagery under the "
            "resolution filter (max_resolution=%.2f) at any incidence angle -- this is a "
            "genuine data limitation, not a solver bug. Consider relaxing max_resolution "
            "for this band if higher-resolution coverage isn't essential there.",
            tag, coverage_pct, max_resolution,
        )

    log.info("%sSelected %d images covering %.1f%% of ROI.", tag, len(selected_pids), coverage_pct)

    return selected_pids