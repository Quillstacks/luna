#!/usr/bin/env python3
"""
# ruff: noqa
luna_analyze.py - Live log analyzer + Dry-Run NAC planner for the Luna pipeline.

Two modes:
  1. analyze  - Parse an existing full_scan.log and print hard numbers.
  2. plan     - Run only the QuickMap + coverage-solver phase (no download,
                no GPU) for each band and report expected NAC counts.

Usage:
  # Analyse the current / past scan log (non-blocking, read-only):
  python luna_analyze.py analyze [--log full_scan.log]

  # Dry-run: query QuickMap & run greedy solver for all bands (no tmux impact):
  python luna_analyze.py plan [--max-incidence 70.0] [--max-resolution 1.5]
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

BANDS = [
    ("-89.99900", "-80.00000"),
    ("-80.00000", "-50.00000"),
    ("-50.00000", "-20.00000"),
    ("-20.00000", "-10.00000"),
    ("-10.00000",  "0.00000"),
    (  "0.00000",  "10.00000"),
    ( "10.00000",  "20.00000"),
    ( "20.00000",  "50.00000"),
    ( "50.00000",  "80.00000"),
    ( "80.00000",  "89.99900"),
]

ANSI_RESET  = "\033[0m"
ANSI_BOLD   = "\033[1m"
ANSI_CYAN   = "\033[1;36m"
ANSI_GREEN  = "\033[1;32m"
ANSI_YELLOW = "\033[1;33m"
ANSI_RED    = "\033[1;31m"
ANSI_DIM    = "\033[2m"


def c(text: str, color: str) -> str:
    return f"{color}{text}{ANSI_RESET}"


def fmt_time(seconds: float) -> str:
    if seconds < 0:
        return "N/A"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


def fmt_time_days(seconds: float) -> str:
    if seconds < 0:
        return "N/A"
    days = int(seconds // 86400)
    rest = seconds % 86400
    h = int(rest // 3600)
    m = int((rest % 3600) // 60)
    s = int(rest % 60)
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if h > 0:
        parts.append(f"{h}h")
    if m > 0:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# ASCII Moon Globe
# ---------------------------------------------------------------------------

# ANSI 256-colour helpers
ANSI_GREY      = "\033[38;5;245m"
ANSI_DARK_GREY = "\033[38;5;238m"
ANSI_BG_RESET  = "\033[49m"


def _band_color_for_status(status: str, conf: str = "") -> str:
    """Return ANSI fg colour for a globe row based on band status."""
    if status == "done":
        return ANSI_GREEN
    if status == "running":
        return ANSI_YELLOW
    if conf == "HIGH":
        return ANSI_GREY
    if conf == "MED":
        return "\033[38;5;243m"   # slightly brighter grey
    if conf == "LOW":
        return "\033[38;5;160m"   # muted red/orange
    return ANSI_DARK_GREY


def print_moon_globe(data: dict) -> None:
    """
    Render an ASCII sphere whose latitude rows are colour-coded by scan status.
    Each terminal row maps to a latitude; within a running band the left portion
    is filled with a brighter character to show % completion.
    """
    WIDTH  = 60   # total character width of the sphere
    HEIGHT = 30   # number of rows (latitude lines)

    # Build status lookup from data['bands']
    bands_dict = {b["band_idx"]: b for b in data["bands"]}

    def lat_info(lat_center: float):
        for bi, (mn, mx) in enumerate(BANDS):
            mn_f, mx_f = float(mn), float(mx)
            if mn_f <= lat_center <= mx_f:
                b = bands_dict.get(bi + 1)
                if b:
                    prog = b["nodes_completed"] / b["nac_count"] if b["nac_count"] > 0 else 0
                    return b["status"], prog, b.get("confidence", "LOW")
        return "pending", 0.0, "LOW"

    # Determine bounding box legend items
    legend = [
        (ANSI_GREEN,   "done          "),
        (ANSI_YELLOW,  "running       "),
        (ANSI_GREY,    "pending/HIGH  "),
        ("\033[38;5;243m", "pending/MED"),
        ("\033[38;5;160m", "pending/LOW"),
    ]

    rows: list[tuple[str, str]] = []
    printed_boundaries = set()

    for row_i in range(HEIGHT):
        # row_i=0 -> top = +90 lat, row_i=HEIGHT-1 -> -90 lat
        lat = 90.0 - (row_i / (HEIGHT - 1)) * 180.0
        # Sphere geometry: half-width of ellipse at this latitude
        half_w = int(round(WIDTH / 2 * math.cos(math.radians(lat))))
        if half_w <= 0:
            rows.append((" " * WIDTH, ""))
            continue

        status, progress, conf = lat_info(lat)
        color = _band_color_for_status(status, conf)

        sphere_char  = "█"
        filled_char  = "▓"   # shown in running band for completed fraction
        partial_char = "░"   # remainder of running band

        left_pad  = WIDTH // 2 - half_w
        right_pad = WIDTH - left_pad - half_w * 2

        inner_width = half_w * 2
        if status == "running" and progress > 0:
            done_cols = int(round(progress * inner_width))
            inner = (
                f"\033[1;32m" + filled_char * done_cols + ANSI_RESET +
                color + partial_char * (inner_width - done_cols) + ANSI_RESET
            )
        else:
            inner = color + sphere_char * inner_width + ANSI_RESET

        # Add latitude tick labels on right edge for band boundaries
        tick = ""
        for mn, mx in BANDS:
            for boundary in (float(mn), float(mx)):
                if boundary in printed_boundaries:
                    continue
                if abs(lat - boundary) < (180 / HEIGHT * 0.55):
                    if abs(boundary) < 0.1:
                        tick = "  0°"
                    else:
                        tick = f" {boundary:+.0f}°"
                    printed_boundaries.add(boundary)
                    break
            if tick:
                break

        rows.append((" " * left_pad + inner + " " * right_pad, tick))

    print()
    print(c("  LUNAR SURFACE SCAN — PROGRESS GLOBE", ANSI_BOLD))
    print()

    # Interleave legend on the right side
    legend_start = HEIGHT // 2 - len(legend) // 2
    for i, item in enumerate(rows):
        prefix = "  "
        leg_str = ""
        leg_idx = i - legend_start
        if 0 <= leg_idx < len(legend):
            lcol, ltext = legend[leg_idx]
            leg_str = f"   {lcol}█{ANSI_RESET} {ltext}"
        
        globe_part, tick = item
        tick_part = f"{tick:<8}"
        print(prefix + globe_part + "   " + tick_part + leg_str)

    print()


# ---------------------------------------------------------------------------
# MODE 1: Log analyser
# ---------------------------------------------------------------------------

@dataclass
class BandStats:
    index: int
    min_lat: str
    max_lat: str
    nac_count: int = 0
    coverage_pct: float = 0.0
    pipeline_duration_s: float = 0.0   # sum of all worker times (reported by Rich)
    wall_clock_s: float = 0.0          # actual wall-clock time (end_ts - start_ts)
    nodes_completed: int = 0
    quickmap_candidates: int = 0
    start_time_str: str = ""
    end_time_str: str = ""
    status: str = "pending"   # pending | running | done
    # Coverage-solver progress (only meaningful while status==running, pre-Selected)
    solver_selected: int = 0           # images selected so far by greedy solver
    solver_uncovered_pct: float = 100.0  # % of ROI still uncovered
    start_dt: Optional[datetime] = None
    last_dt: Optional[datetime] = None


class LogParserState:
    def __init__(self):
        self.last_offset = 0
        self.last_size = 0
        self.bands_data: list[BandStats] = []
        self.current_band_idx = -1
        self.param_check_count = 0
        self._in_roi_rows = False
        self._pending_lats: list[float] = []
        self._last_roi_min: Optional[str] = None
        self._last_roi_max: Optional[str] = None
        self.line_buffer = ""
        self.cached_dict: Optional[dict] = None
        self.log_lines = 0

_parser_states: dict[str, LogParserState] = {}


def get_metrics_data(log_path: Path) -> dict:
    if not log_path.exists():
        return {"error": f"Log file not found: {log_path}"}

    global _parser_states
    state_key = str(log_path.resolve())
    if state_key not in _parser_states:
        _parser_states[state_key] = LogParserState()
    state = _parser_states[state_key]

    try:
        current_size = log_path.stat().st_size
    except Exception:
        current_size = 0

    # If file was truncated/cleared, reset parser
    if current_size < state.last_offset:
        state = LogParserState()
        _parser_states[state_key] = state

    # If the file size is identical and we already parsed it once, return cache
    if current_size == state.last_size and state.cached_dict is not None:
        return state.cached_dict

    new_data = ""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(state.last_offset)
            new_data = f.read()
            state.last_offset = f.tell()
            state.last_size = current_size
    except Exception as e:
        return {"error": f"Failed to read log file: {e}"}

    raw_lines = (state.line_buffer + new_data).splitlines()
    if new_data and not new_data.endswith("\n"):
        state.line_buffer = raw_lines.pop() if raw_lines else ""
    else:
        state.line_buffer = ""

    state.log_lines += len(raw_lines)

    re_selected   = re.compile(r"Selected (\d+) images covering ([\d.]+)% of ROI")
    re_pipeline   = re.compile(r"Total Pipeline Execution Time\s+[\u2502|]+\s+([\d.]+)\s+s")
    re_quickmap   = re.compile(r"QuickMap returned (\d+) candidate images")
    re_paramcheck = re.compile(r"PARAMETER CHECK.*max_inc:\s*([\d.]+)")
    re_node       = re.compile(r"NODE COMPLETE.*lattice frame (\S+)")
    re_solver     = re.compile(r"(\d+) images selected so far, ([\d.]+)% ROI still uncovered")
    re_ts_full    = re.compile(r"\[?(\d{4}-\d{2}-\d{2}\s+)?(\d{2}:\d{2}:\d{2})\]?")
    re_lonlat_pair  = re.compile(r"(-?\d+\.\d+),(-?\d+\.\d+)")
    re_roi_row      = re.compile(r"Target ROI")
    re_table_open   = re.compile(r"\u250c")
    re_table_close  = re.compile(r"\u2514")

    bands_data = state.bands_data
    current_band_idx = state.current_band_idx
    param_check_count = state.param_check_count
    _in_roi_rows = state._in_roi_rows
    _pending_lats = state._pending_lats
    _last_roi_min = state._last_roi_min
    _last_roi_max = state._last_roi_max

    for line in raw_lines:
        ts_m = re_ts_full.search(line)
        line_dt = None
        ts_str = ""
        if ts_m:
            dt_part = ts_m.group(1)
            time_part = ts_m.group(2)
            ts_str = time_part
            if dt_part:
                try:
                    line_dt = datetime.strptime(f"{dt_part.strip()} {time_part}", "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass
            if not line_dt:
                try:
                    t = datetime.strptime(time_part, "%H:%M:%S")
                    line_dt = datetime(2026, 7, 16, t.hour, t.minute, t.second)
                except ValueError:
                    pass

        if re_table_open.search(line):
            _in_roi_rows  = False
            _pending_lats = []

        if re_roi_row.search(line):
            _in_roi_rows = True
        if _in_roi_rows:
            for m_pair in re_lonlat_pair.finditer(line):
                _pending_lats.append(float(m_pair.group(2)))

        if re_table_close.search(line) and _pending_lats:
            _last_roi_min = f"{min(_pending_lats):.5f}"
            _last_roi_max = f"{max(_pending_lats):.5f}"
            _in_roi_rows  = False

        if re_paramcheck.search(line):
            param_check_count += 1
            band_idx = param_check_count - 1
            if _last_roi_min is not None:
                b_min, b_max = _last_roi_min, _last_roi_max
            elif band_idx < len(BANDS):
                b_min, b_max = BANDS[band_idx]
            else:
                b_min, b_max = "?", "?"

            matched_idx = -1
            try:
                min_f = float(b_min)
                max_f = float(b_max)
                for idx, (mn, mx) in enumerate(BANDS):
                    if abs(float(mn) - min_f) < 0.1 and abs(float(mx) - max_f) < 0.1:
                        matched_idx = idx
                        break
            except Exception:
                pass

            if matched_idx == -1:
                matched_idx = band_idx % len(BANDS)

            b = BandStats(
                index=matched_idx + 1,
                min_lat=BANDS[matched_idx][0] if matched_idx < len(BANDS) else b_min,
                max_lat=BANDS[matched_idx][1] if matched_idx < len(BANDS) else b_max,
                start_time_str=ts_str,
                status="running",
                start_dt=line_dt,
            )

            existing_idx = next((i for i, x in enumerate(bands_data) if x.index == b.index), -1)
            if existing_idx != -1:
                bands_data[existing_idx] = b
                current_band_idx = existing_idx
            else:
                bands_data.append(b)
                current_band_idx = len(bands_data) - 1
            continue

        if current_band_idx < 0:
            continue

        bd = bands_data[current_band_idx]
        if line_dt:
            bd.last_dt = line_dt

        m = re_quickmap.search(line)
        if m:
            bd.quickmap_candidates = int(m.group(1))
            continue

        m = re_selected.search(line)
        if m:
            bd.nac_count = int(m.group(1))
            bd.coverage_pct = float(m.group(2))
            continue

        m = re_node.search(line)
        if m:
            bd.nodes_completed += 1
            if bd.nac_count > 0 and bd.nodes_completed >= bd.nac_count:
                bd.status = "done"
            continue

        m = re_solver.search(line)
        if m:
            bd.solver_selected = int(m.group(1))
            bd.solver_uncovered_pct = float(m.group(2))
            continue

        m = re_pipeline.search(line)
        if m:
            bd.pipeline_duration_s = float(m.group(1))
            bd.status = "done"
            if line_dt:
                bd.end_time_str = ts_str
            continue

    state.bands_data = bands_data
    state.current_band_idx = current_band_idx
    state.param_check_count = param_check_count
    state._in_roi_rows = _in_roi_rows
    state._pending_lats = _pending_lats
    state._last_roi_min = _last_roi_min
    state._last_roi_max = _last_roi_max

    # Clean up status for running bands that completed all nodes
    for bd in bands_data:
        if bd.status == "running" and bd.nac_count > 0 and bd.nodes_completed >= bd.nac_count:
            bd.status = "done"

    # Detect global scan vs rescan
    detected_lats = []
    for bd in bands_data:
        try:
            detected_lats.extend([float(bd.min_lat), float(bd.max_lat)])
        except (ValueError, TypeError):
            pass
    lat_span = (max(detected_lats) - min(detected_lats)) if len(detected_lats) >= 2 else 0.0
    is_global_scan = lat_span > 80.0 or "full_scan" in log_path.name

    # Determine workers and speeds
    workers_hint = 4
    completed = [bd for bd in bands_data if bd.status == "done" and bd.nac_count > 0]
    total_nacs_done = sum(bd.nac_count for bd in completed)
    total_wall_done = sum(bd.pipeline_duration_s / workers_hint for bd in completed if bd.pipeline_duration_s > 0)
    
    speed_s_per_nac = total_wall_done / total_nacs_done if total_nacs_done > 0 else 18.0
    speed_nac_per_s = 1.0 / speed_s_per_nac if speed_s_per_nac > 0 else (193.5 / 3600.0)

    # 1-Bit density and extrapolation
    known = {bd.index - 1: bd.nac_count for bd in bands_data if bd.nac_count > 0}
    def _sphere_area(mn: str, mx: str) -> float:
        return abs(math.sin(math.radians(float(mx))) - math.sin(math.radians(float(mn))))
    all_areas = [_sphere_area(mn, mx) for mn, mx in BANDS]
    mid_lat_densities = [
        known[bi] / all_areas[bi]
        for bi in known
        if bi not in (0, 9)
    ]
    d_mid_lat = sum(mid_lat_densities) / len(mid_lat_densities) if mid_lat_densities else None

    band_estimates: dict[int, int] = {}
    for bi, (mn, mx) in enumerate(BANDS):
        if bi in known:
            band_estimates[bi] = known[bi]
        elif bi in (0, 9):
            band_estimates[bi] = 545  # polar cap estimate
        elif d_mid_lat is not None:
            band_estimates[bi] = int(round(d_mid_lat * all_areas[bi] / 25) * 25)
        else:
            band_estimates[bi] = 8886  # mid-lat fallback

    def _confidence(bi: int) -> str:
        if bi in known:
            return "REAL"
        mirror = 9 - bi
        if mirror in known:
            return "HIGH"
        if d_mid_lat is not None:
            return "MED"
        return "LOW"

    # Identify remaining bands
    remaining_band_indices = [i for i in range(len(BANDS)) if i not in known] if is_global_scan else []

    # Calculate active running band metrics
    running_band = next((bd for bd in bands_data if bd.status == "running"), None)
    running_nacs_remaining = 0
    measured_speed_nac_h = None
    measured_eta_s = None

    if running_band:
        if running_band.nac_count > 0:
            running_nacs_remaining = max(0, running_band.nac_count - running_band.nodes_completed)
            if running_band.start_dt and running_band.last_dt:
                elapsed = (running_band.last_dt - running_band.start_dt).total_seconds()
                if elapsed > 300 and running_band.nodes_completed > 10:
                    measured_speed_nac_h = (running_band.nodes_completed / elapsed) * 3600
                    measured_eta_s = running_nacs_remaining / (measured_speed_nac_h / 3600)

    # Return structured dict of everything
    bands_list = []
    for bi in range(10):
        mn, mx = BANDS[bi]
        bd = next((b for b in bands_data if b.index == bi + 1), None)
        if bd:
            bands_list.append({
                "band_idx": bi + 1,
                "min_lat": float(mn),
                "max_lat": float(mx),
                "nac_count": bd.nac_count,
                "coverage_pct": bd.coverage_pct,
                "pipeline_duration_s": bd.pipeline_duration_s,
                "nodes_completed": bd.nodes_completed,
                "status": bd.status,
                "confidence": "REAL"
            })
        elif is_global_scan:
            bands_list.append({
                "band_idx": bi + 1,
                "min_lat": float(mn),
                "max_lat": float(mx),
                "nac_count": band_estimates.get(bi, 0),
                "coverage_pct": 0.0,
                "pipeline_duration_s": 0.0,
                "nodes_completed": 0,
                "status": "pending",
                "confidence": _confidence(bi)
            })

    total_remaining_nacs = running_nacs_remaining
    for bi in remaining_band_indices:
        total_remaining_nacs += band_estimates.get(bi, 0)
    
    est_total_remaining_time = total_remaining_nacs * speed_s_per_nac
    grand_total_est = sum(band_estimates.values())
    grand_total_time = total_nacs_done * speed_s_per_nac + total_remaining_nacs * speed_s_per_nac

    state.cached_dict = {
        "source_log": log_path.name,
        "log_size_mb": round(current_size / 1e6, 2),
        "log_lines": state.log_lines,
        "is_global_scan": is_global_scan,
        "nominal_throughput_nac_h": round(speed_nac_per_s * 3600, 2),
        "nominal_time_per_nac_s": round(speed_s_per_nac, 2),
        "bands_completed_count": len(completed),
        "bands_running_count": sum(1 for b in bands_data if b.status == 'running'),
        "bands_pending_count": len(remaining_band_indices),
        "nacs_confirmed_done": total_nacs_done,
        "bands": bands_list,
        "running_band": {
            "band_idx": running_band.index,
            "min_lat": float(running_band.min_lat),
            "max_lat": float(running_band.max_lat),
            "nac_count": running_band.nac_count,
            "nodes_completed": running_band.nodes_completed,
            "remaining": running_nacs_remaining,
            "nominal_eta_s": running_nacs_remaining * speed_s_per_nac,
            "measured_speed_nac_h": round(measured_speed_nac_h, 2) if measured_speed_nac_h else None,
            "measured_eta_s": measured_eta_s
        } if running_band else None,
        "global_projection": {
            "total_remaining_nacs": total_remaining_nacs,
            "est_remaining_time_s": est_total_remaining_time,
            "grand_total_nacs": grand_total_est,
            "grand_total_time_s": grand_total_time
        } if (remaining_band_indices or running_nacs_remaining > 0) else None
    }
    
    return state.cached_dict


def analyze_log(log_path: Path) -> None:
    data = get_metrics_data(log_path)
    if "error" in data:
        print(c(f"[ERROR] {data['error']}", ANSI_RED))
        sys.exit(1)

    # 1. SYSTEM STATUS & PIPELINE CONFIGURATION CARD
    print(c("┌────────────────────────────────────────────────────────────────────────────┐", ANSI_CYAN))
    print(c("│ " + "LUNA PIPELINE STATUS & DIAGNOSTICS".center(74) + " │", ANSI_BOLD + ANSI_CYAN))
    print(c("├────────────────────────────────────────────────────────────────────────────┤", ANSI_CYAN))
    print(f"│  Source Log  : {data['source_log']:<57} │")
    print(f"│  File Size   : {f'{data['log_size_mb']} MB ({data['log_lines']:,} lines)':<57} │")
    print(f"│  Scan Mode   : {('Global Full-Moon Scan' if data['is_global_scan'] else 'Targeted Regional Rescan'):<57} │")
    print(f"│  Throughput  : {f'{data['nominal_throughput_nac_h']} NACs/h ({data['nominal_time_per_nac_s']}s / NAC nominal)':<57} │")
    print(c("└────────────────────────────────────────────────────────────────────────────┘", ANSI_CYAN))
    print()

    # 2. PROGRESS GLOBE
    if data['is_global_scan']:
        print_moon_globe(data)

    # 3. LATITUDE BAND REGISTRY TABLE
    print(c("┌──────┬────────────────────────┬─────────┬──────────┬────────────────┬────────────┬──────────────┐", ANSI_CYAN))
    print(c("│ Band │     Latitude Range     │  NACs   │ Coverage │ Pipeline Time  │  Progress  │    Status    │", ANSI_BOLD + ANSI_CYAN))
    print(c("├──────┼────────────────────────┼─────────┼──────────┼────────────────┼────────────┼──────────────┤", ANSI_CYAN))

    # Print all bands in order
    for b in data["bands"]:
        lat_str = f"{b['min_lat']:.5f} to {b['max_lat']:.5f}"
        status_color = {"done": ANSI_GREEN, "running": ANSI_YELLOW}.get(b["status"], ANSI_DIM)
        
        dur_str = fmt_time(b["pipeline_duration_s"]) if b["pipeline_duration_s"] > 0 else ("running..." if b["status"] == "running" else "-")
        nac_str = f"{b['nac_count']}" if b["status"] == "done" or b["status"] == "running" else f"~{b['nac_count']}"
        nodes_str = f"{b['nodes_completed']}/{b['nac_count']}" if b["nac_count"] > 0 else f"{b['nodes_completed']}/?"
        cov_str = f"{b['coverage_pct']:.1f}%" if b["coverage_pct"] > 0 else "-"
        status_disp = f"pending/{b['confidence']}" if b["status"] == "pending" else b["status"]

        row = f"│ {b['band_idx']:^4} │ {lat_str:^22} │ {nac_str:>7} │ {cov_str:>8} │ {dur_str:>14} │ {nodes_str:>10} │ {status_disp:^12} │"
        print(c(row, status_color))
        
    print(c("└──────┴────────────────────────┴─────────┴──────────┴────────────────┴────────────┴──────────────┘", ANSI_CYAN))
    print()

    # 4. PROGRESS SUMMARY & PROJECTIONS CARD
    print(c("┌────────────────────────────────────────────────────────────────────────────┐", ANSI_CYAN))
    print(c("│ " + "PROGRESS SUMMARY & PROJECTIONS".center(74) + " │", ANSI_BOLD + ANSI_CYAN))
    print(c("├────────────────────────────────────────────────────────────────────────────┤", ANSI_CYAN))
    
    print(f"│  Bands Complete : {data['bands_completed_count']:<2} / {len(data['bands']):<2}   Bands Running : {data['bands_running_count']:<2}   Bands Pending : {data['bands_pending_count']:<2}        │")
    print(f"│  NACs Confirmed (completed bands): {data['nacs_confirmed_done']:<39} │")
    print(c("├────────────────────────────────────────────────────────────────────────────┤", ANSI_CYAN))
    
    rb = data["running_band"]
    if rb:
        print(f"│  ACTIVE BAND PROGRESS: Band {rb['band_idx']} [{rb['min_lat']}° to {rb['max_lat']}°]                        │")
        if rb["nac_count"] > 0:
            pct = (rb["nodes_completed"] / rb["nac_count"]) * 100
            print(f"│  - Processed : {rb['nodes_completed']:,} / {rb['nac_count']:,} ({pct:.1f}% done)                            │")
            print(f"│  - Nominal ETA   : {fmt_time_days(rb['nominal_eta_s']):<18} (at {data['nominal_throughput_nac_h']:.1f} NACs/h avg)             │")
            if rb["measured_speed_nac_h"] and rb["measured_eta_s"]:
                print(f"│  - Real-time ETA : {fmt_time_days(rb['measured_eta_s']):<18} (at {rb['measured_speed_nac_h']:.1f} NACs/h actual)          │")
        else:
            print(f"│  - Coverage Solver Phase is active                                         │")
        print(c("├────────────────────────────────────────────────────────────────────────────┤", ANSI_CYAN))

    gp = data["global_projection"]
    if gp:
        print(f"│  GLOBAL MISSION PROJECTION:                                                │")
        print(f"│  - Total Remaining NACs : {gp['total_remaining_nacs']:<48} │")
        print(f"│  - Est. Remaining Time  : {fmt_time_days(gp['est_remaining_time_s']):<48} │")
        print(f"│  - Grand Total Est.     : ~{gp['grand_total_nacs']:<47} │")
        print(f"│  - Grand Total Time     : {fmt_time_days(gp['grand_total_time_s']):<48} │")
        print(c("│  (Confidence: HIGH=mirror band known, MED=adjacent band, LOW=equatorial)   │", ANSI_DIM))
    else:
        print(f"│  All target bands successfully completed!                                  │")
        
    print(c("└────────────────────────────────────────────────────────────────────────────┘", ANSI_CYAN))
    print()


def run_server(log_path: Path, host: str, port: int) -> None:
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import StreamingResponse
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
        import asyncio
        import json
    except ImportError as e:
        print(c(f"[ERROR] Required packages missing: {e}", ANSI_RED))
        print("Install them first: .venv/bin/pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI(title="Luna Pipeline Metrics Server")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/metrics")
    def get_metrics():
        return get_metrics_data(log_path)

    @app.get("/api/latent/anchors")
    def get_latent_anchors():
        cache_path = Path("data/_scratch/dash_cache_v2.pkl")
        if not cache_path.exists():
            cache_path = Path("data/_scratch/dash_cache.pkl")
        if not cache_path.exists():
            return {"error": "Cache not found"}
        import pickle
        with open(cache_path, "rb") as f:
            d = pickle.load(f)
        df_anchors = d["df_anchors"].fillna(0)
        return df_anchors.to_dict(orient="records")

    @app.get("/api/latent/topology")
    def get_latent_topology():
        cache_path = Path("data/_scratch/dash_cache_v2.pkl")
        if not cache_path.exists():
            cache_path = Path("data/_scratch/dash_cache.pkl")
        if not cache_path.exists():
            return {"error": "Cache not found"}
        import pickle
        with open(cache_path, "rb") as f:
            d = pickle.load(f)
        df = d["df_anchors"]
        labels = df["label"].tolist() if "label" in df.columns else (df["name"].tolist() if "name" in df.columns else df["anchor_name"].tolist())
        clusters = df["cluster_id"].tolist() if "cluster_id" in df.columns else [0]*len(df)
        return {
            "labels": labels,
            "clusters": clusters,
            "matrix": d["cos_matrix"].tolist()
        }

    @app.get("/api/live")
    def stream_live():
        async def event_generator():
            while True:
                try:
                    metrics = get_metrics_data(log_path)
                    latest_nac = None
                    if log_path.exists():
                        with open(log_path, "rb") as f:
                            try:
                                f.seek(-20480, 2)
                            except OSError:
                                pass
                            last_chunk = f.read().decode("utf-8", errors="ignore")
                            matches = re.findall(r"\b(M\d+[LR]C)\b", last_chunk)
                            if matches:
                                latest_nac = matches[-1]

                    payload = {
                        "latest_nac": latest_nac,
                        "metrics": metrics
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    print(c(f"[INFO] Starting LUNA Metrics Server on http://{host}:{port}", ANSI_GREEN))
    print(c(f"[INFO] Watching log file: {log_path}", ANSI_YELLOW))
    uvicorn.run(app, host=host, port=port)


# ---------------------------------------------------------------------------
# MODE 2: Dry-Run planner
# ---------------------------------------------------------------------------

def roi_polygon(min_lat: str, max_lat: str) -> list:
    mn = float(min_lat)
    mx = float(max_lat)
    return [
        (-179.999, mx),
        ( 179.999, mx),
        ( 179.999, mn),
        (-179.999, mn),
        (-179.999, mx),
    ]


def run_plan(max_incidence: float, max_resolution: float) -> None:
    try:
        from luna.io.coverage import select_coverage_nacs
    except ImportError as e:
        print(c(f"[ERROR] Cannot import luna package: {e}", ANSI_RED))
        print("Run from the luna repo root with the venv active.")
        sys.exit(1)

    print(c("=" * 74, ANSI_CYAN))
    print(c("  LUNA PIPELINE - DRY-RUN NAC PLANNER", ANSI_BOLD))
    print(c(f"  max-incidence: {max_incidence} deg   max-resolution: {max_resolution} m/px", ANSI_DIM))
    print(c("  (No images will be downloaded or processed)", ANSI_DIM))
    print(c("=" * 74, ANSI_CYAN))
    print()

    results: list[dict] = []
    grand_total = 0
    grand_start = time.perf_counter()

    for idx, (min_lat, max_lat) in enumerate(BANDS, 1):
        label = f"{min_lat},{max_lat}"
        print(c(f"  [{idx}/{len(BANDS)}] Band {min_lat} to {max_lat} ... querying QuickMap + running solver ...", ANSI_YELLOW), flush=True)
        t0 = time.perf_counter()
        try:
            pids = select_coverage_nacs(
                roi_coords=roi_polygon(min_lat, max_lat),
                max_resolution=max_resolution,
                max_incidence=max_incidence,
                band_label=label,
            )
            elapsed = time.perf_counter() - t0
            results.append({"idx": idx, "min_lat": min_lat, "max_lat": max_lat,
                             "nac_count": len(pids), "elapsed": elapsed, "status": "ok"})
            grand_total += len(pids)
            print(c(f"         -> {len(pids)} NACs selected  ({fmt_time(elapsed)})", ANSI_GREEN))
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            results.append({"idx": idx, "min_lat": min_lat, "max_lat": max_lat,
                             "nac_count": -1, "elapsed": elapsed, "status": f"ERROR: {exc}"})
            print(c(f"         -> ERROR: {exc}", ANSI_RED))

    grand_elapsed = time.perf_counter() - grand_start

    print()
    print(c("=" * 74, ANSI_CYAN))
    print(c("  RESULTS", ANSI_BOLD))
    print(c("=" * 74, ANSI_CYAN))

    hdr2 = "{:^4}  {:^26}  {:>8}  {:>12}".format("#", "Latitude Band", "NACs", "Solver Time")
    sep2 = "-" * len(hdr2)
    print(c(hdr2, ANSI_BOLD))
    print(c(sep2, ANSI_CYAN))

    for r in results:
        nac_str = str(r["nac_count"]) if r["nac_count"] >= 0 else "ERROR"
        color   = ANSI_GREEN if r["status"] == "ok" else ANSI_RED
        lat_str = f"{r['min_lat']} to {r['max_lat']}"
        row2 = "{:^4}  {:^26}  {:>8}  {:>12}".format(r["idx"], lat_str, nac_str, fmt_time(r["elapsed"]))
        print(c(row2, color))

    print(c(sep2, ANSI_CYAN))
    total_row = "{:^4}  {:^26}  {:>8}  {:>12}".format("", "TOTAL", str(grand_total), fmt_time(grand_elapsed))
    print(c(total_row, ANSI_BOLD))
    print()

    print(c("GRAND TOTAL", ANSI_BOLD))
    print(f"  Total NACs required      :  {c(str(grand_total), ANSI_CYAN)}")
    print(f"  Est. download size       :  {c(f'{grand_total * 529 / 1024:.1f} GB (compressed)', ANSI_CYAN)}")
    print(f"  Planning query duration  :  {c(fmt_time_days(grand_elapsed), ANSI_DIM)}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Luna pipeline - log analyser & dry-run NAC planner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    sp_a = sub.add_parser("analyze", help="Parse the scan log and report numbers.")
    sp_a.add_argument("--log", type=Path, default=Path("full_scan_v2.log"),
                      help="Path to the log file (default: full_scan_v2.log)")

    sp_p = sub.add_parser("plan", help="Dry-run: query QuickMap + solver for all bands.")
    sp_p.add_argument("--max-incidence", type=float, default=70.0)
    sp_p.add_argument("--max-resolution", type=float, default=1.5)

    sp_s = sub.add_parser("serve", help="Start FastAPI metrics server and SSE stream.")
    sp_s.add_argument("--log", type=Path, default=Path("full_scan_v2.log"),
                      help="Path to the log file (default: full_scan_v2.log)")
    sp_s.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    sp_s.add_argument("--port", type=int, default=8000, help="Port to bind")

    args = parser.parse_args()
    if args.mode == "analyze":
        analyze_log(args.log)
    elif args.mode == "plan":
        run_plan(args.max_incidence, args.max_resolution)
    elif args.mode == "serve":
        run_server(args.log, args.host, args.port)


if __name__ == "__main__":
    main()
