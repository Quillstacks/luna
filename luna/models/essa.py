"""ESSA second-stage refiner — uses Le Corre's full ISIS preprocessing contract.

Flow per NAC:
    CDR product ID → EDR product ID
    → fetch EDR from PDS
    → ISIS3 bash pipeline (lronac2isis → spiceinit → lronaccal →
      lronacecho → cam2map → gdal_translate → cubic-spline 1.5 m/px)
    → project CandidateHit lon/lat into GeoTIFF pixel coords
    → slice 2048×2048 tile
    → ESSA Mask R-CNN inference
    → RefinedHit with comparable score
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import rasterio
import torch
import numpy as np

from luna.config import DATA_DIR, WEIGHTS_DIR
from luna.io import PDSIndex
from luna.models.maskrcnn import build_essa_model

log = logging.getLogger(__name__)

ESSA_CLASSES    = {1: "skylight", 2: "pit"}
DEFAULT_WEIGHTS = WEIGHTS_DIR / "essa.pt"


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RefinedHit:
    rank: int
    product_id: str
    votes: int
    dino_score: float
    lon: float
    lat: float
    x_offset: int
    y_offset: int
    # ESSA-specific fields (optional, 0.0 / "" if not using ESSA)
    essa_score: float = 0.0
    essa_class: str = ""
    essa_lon: float = 0.0
    essa_lat: float = 0.0
    # DINO refiner fields (optional)
    dino_similarity: float = 0.0

    def _repr_html_(self) -> str:
        rows = [
            f"<tr><td>Rank</td><td>{self.rank}</td></tr>",
            f"<tr><td>Product ID</td><td>{self.product_id}</td></tr>",
            f"<tr><td>Votes</td><td>{self.votes}</td></tr>",
            f"<tr><td>DINO Score</td><td>{self.dino_score:.2f}</td></tr>",
            f"<tr><td>Position</td><td>({self.lon:.4f}, {self.lat:.4f})</td></tr>",
            f"<tr><td>Offset</td><td>({self.x_offset}, {self.y_offset})</td></tr>",
        ]
        if self.essa_score > 0:
            rows.extend([
                f"<tr><td>ESSA Score</td><td>{self.essa_score:.2f}</td></tr>",
                f"<tr><td>Class</td><td>{self.essa_class}</td></tr>",
                f"<tr><td>ESSA Position</td><td>({self.essa_lon:.4f}, {self.essa_lat:.4f})</td></tr>",
            ])
        if self.dino_similarity > 0:
            rows.append(f"<tr><td>DINO Similarity</td><td>{self.dino_similarity:.2f}</td></tr>")
        return f"<table><tr><th colspan='2' style='text-align:left'>RefinedHit</th></tr>{" ".join(rows)}</table>"


# ---------------------------------------------------------------------------
# Refiner
# ---------------------------------------------------------------------------

class ESSARefiner:
    def __init__(self, model: torch.nn.Module, device: str) -> None:
        self._model  = model.eval()
        self._device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path = DEFAULT_WEIGHTS,
        device: str | None = None,
    ) -> ESSARefiner:
        if device is None:
            device = (
                "mps"  if torch.backends.mps.is_available() else
                "cuda" if torch.cuda.is_available()          else
                "cpu"
            )
        log.info("Loading ESSA from %s on %s ...", checkpoint, device)
        model = build_essa_model(checkpoint=checkpoint, map_location=device)
        return cls(model=model.to(torch.device(device)), device=device)

    @staticmethod
    def _cdr_to_edr_pid(cdr_pid: str) -> str:
        """Convert CDR product ID to EDR: trailing C → E (LC→LE, RC→RE)."""
        return cdr_pid[:-1] + "E" if cdr_pid.endswith("C") else cdr_pid

    def refine(
        self,
        hits: list,                          # list[CandidateHit]
        out_dir: str | Path | None = None,
        score_thr: float  = 0.8,
        essa_min_score: float = 0.0,
        skip_preprocess: bool = False,
        save_debug_plots: bool = False,
        trace: dict = None,
    ) -> list[RefinedHit]:
        # Lazy imports — heavy, only needed at refine time
        from scripts.essa_smoke import (
            preprocess_edr, lonlat_to_rowcol, rowcol_to_lonlat,
            _read_tile, _infer_tile,
            TILE_SIZE, TARGET_RES_M,
        )

        out_dir    = Path(out_dir) if out_dir else DATA_DIR / "essa_out" / "pipeline_refine"
        edr_index  = PDSIndex(archive="EDR")
        candidates = []

        hits_by_nac: dict[str, list] = defaultdict(list)
        for h in hits:
            hits_by_nac[h.product_id].append(h)

        for cdr_pid, nac_hits in hits_by_nac.items():
            edr_pid = self._cdr_to_edr_pid(cdr_pid)
            workdir = out_dir / cdr_pid
            workdir.mkdir(parents=True, exist_ok=True)

            log.info("Processing %d hits in %s (EDR: %s)", len(nac_hits), cdr_pid, edr_pid)

            if skip_preprocess:
                cands = sorted(workdir.glob(f"*_{TARGET_RES_M}.tif"))
                if not cands:
                    log.warning("skip_preprocess=True but no GeoTIFF in %s — skipping.", workdir)
                    continue
                downscaled = cands[-1]
                log.info("Reusing %s", downscaled)
            else:
                edr_url = edr_index.url_for(edr_pid)
                isolated_edr = workdir / f"{edr_pid}.IMG"
                
                old_cdr = workdir / f"{cdr_pid}.IMG"
                if old_cdr.exists():
                    log.info("Removing stale CDR from workdir...")
                    old_cdr.unlink()
                
                if not isolated_edr.exists():
                    log.info("Downloading raw EDR (%s) directly from PDS...", edr_pid)
                    import urllib.request
                    urllib.request.urlretrieve(edr_url, isolated_edr)
                
                downscaled = preprocess_edr(isolated_edr, workdir)

            t_geotiff_start = time.perf_counter()
            with rasterio.open(downscaled) as src:
                H, W = src.height, src.width
                log.info("GeoTIFF: %dx%d  CRS=%s", W, H, src.crs)
                
                t_geotiff_duration = time.perf_counter() - t_geotiff_start
                if trace is not None:
                    trace["p2_geotiff_loading"] = trace.get("p2_geotiff_loading", 0.0) + t_geotiff_duration

                for hit in nac_hits:
                    row, col = lonlat_to_rowcol(src, hit.lon, hit.lat)
                    r0 = max(0, min(H - TILE_SIZE if H > TILE_SIZE else 0,
                                    row - TILE_SIZE // 2))
                    c0 = max(0, min(W - TILE_SIZE if W > TILE_SIZE else 0,
                                    col - TILE_SIZE // 2))

                    tile      = _read_tile(src, r0, c0)
                    t_infer_start = time.perf_counter()
                    tile_dets = _infer_tile(self._model, tile, self._device, score_thr)
                    t_infer_duration = time.perf_counter() - t_infer_start
                    if trace is not None:
                        trace["p2_mask_rcnn_inference"] = trace.get("p2_mask_rcnn_inference", 0.0) + t_infer_duration

                    essa_score, essa_class = 0.0, "none"
                    best_box = None
                    essa_lon, essa_lat = hit.lon, hit.lat   # fallback = candidate centre
                    for cls_id, cls_name in [(2, "pit"), (1, "skylight")]:
                        cls_dets = [d for d in tile_dets if d["class"] == cls_id]
                        if cls_dets:
                            best       = max(cls_dets, key=lambda d: d["score"])
                            essa_score = best["score"]
                            essa_class = cls_name
                            best_box   = best["box"]
                            # Compute accurate centroid lon/lat from ESSA mask centroid
                            cx_t, cy_t = best["centroid_local"]
                            essa_col = c0 + cx_t
                            essa_row = r0 + cy_t
                            essa_lon, essa_lat = rowcol_to_lonlat(src, essa_row, essa_col)
                            break

                    if essa_score < essa_min_score:
                        continue

                    candidates.append({
                        "hit":        hit,
                        "essa_score": essa_score,
                        "essa_class": essa_class,
                        "essa_lon":   essa_lon,
                        "essa_lat":   essa_lat,
                        "tile":       tile if save_debug_plots else None,
                        "box":        best_box,
                    })

        candidates.sort(key=lambda x: (-x["essa_score"], x["hit"].score))

        # --- Post-ESSA spatial NMS -------------------------------------------
        # Multiple Pithos candidate tiles may contain the same pit.  Deduplicate
        # by ESSA centroid proximity: keep only the highest-scoring hit within
        # a 200 m radius (≈ 133 px at 1.5 m/px).
        _NMS_DIST_DEG = 200 / 1_737_400 * (180 / 3.14159265)  # ~0.0066°
        deduped: list[dict] = []
        for cand in candidates:
            lon_c, lat_c = cand["essa_lon"], cand["essa_lat"]
            duplicate = any(
                abs(lon_c - d["essa_lon"]) < _NMS_DIST_DEG
                and abs(lat_c - d["essa_lat"]) < _NMS_DIST_DEG
                for d in deduped
            )
            if not duplicate:
                deduped.append(cand)
        candidates = deduped
        # ---------------------------------------------------------------------

        refined: list[RefinedHit] = []
        for rank, item in enumerate(candidates, start=1):
            h = item["hit"]
            refined.append(RefinedHit(
                rank       = rank,
                product_id = h.product_id,
                votes      = h.votes,
                dino_score = h.score,
                essa_score = item["essa_score"],
                essa_class = item["essa_class"],
                lon        = h.lon,
                lat        = h.lat,
                essa_lon   = item["essa_lon"],
                essa_lat   = item["essa_lat"],
                x_offset   = h.x_offset,
                y_offset   = h.y_offset,
            ))

            if save_debug_plots and item["tile"] is not None:
                _save_debug_plot(
                    out_dir=out_dir,
                    rank=rank,
                    pid=h.product_id,
                    tile=item["tile"],
                    cls_name=item["essa_class"],
                    score=item["essa_score"],
                    box=item["box"],
                )

        log.info("ESSA refined %d → %d hits.", len(hits), len(refined))
        return refined


# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------

def _save_debug_plot(
    out_dir: Path,
    rank: int,
    pid: str,
    tile: "np.ndarray",
    cls_name: str,
    score: float,
    box: "np.ndarray | None",
) -> None:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))

    if box is not None:
        x0, y0, x1, y1 = [int(b) for b in box]
        
        padding = 150
        
        h, w = tile.shape
        y_min = max(0, y0 - padding)
        y_max = min(h, y1 + padding)
        x_min = max(0, x0 - padding)
        x_max = min(w, x1 + padding)
        
        zoomed_tile = tile[y_min:y_max, x_min:x_max]
        ax.imshow(zoomed_tile, cmap="gray", origin="upper")
        
        new_x0 = x0 - x_min
        new_y0 = y0 - y_min
        new_x1 = x1 - x_min
        new_y1 = y1 - y_min
        
        ax.add_patch(patches.Rectangle(
            (new_x0, new_y0), new_x1 - new_x0, new_y1 - new_y0,
            linewidth=2, edgecolor="red", facecolor="none",
        ))
        ax.text(new_x0, max(0, new_y0 - 10), f"{cls_name}: {score:.4f}",
                color="red", fontsize=14, weight="bold")
    else:
        ax.imshow(tile, cmap="gray", origin="upper")

    ax.set_title(f"Rank {rank} | {pid} | ZOOM")
    ax.axis("off")
    
    plt.savefig(out_dir / f"rank_{rank:02d}_{pid}_{cls_name}_zoomed.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)