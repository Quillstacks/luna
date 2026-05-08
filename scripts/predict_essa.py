"""Run ESSA (Le Corre et al. 2025) on a directory of pre-tiled GeoTIFFs.

Mirrors the upstream inference contract:
    - input tiles must be 2048 x 2048, single-band, GeoTIFF, ~1.5 m/px
    - pixel intensities are normalised by /255 before being passed to the model
    - score threshold defaults to 0.8 (the elbow the paper identified)
    - cross-class NMS at IoU 0.5

Output:
    out/detections.geojson   per-detection polygons in the tile's native CRS,
                             attributes: class (1=skylight, 2=pit), score,
                             tile_id, box_xyxy_px
    out/detections.csv       flat row-per-detection summary

Usage:
    python scripts/fetch_essa_weights.py
    python scripts/predict_essa.py \\
        --tiles data/essa_tiles/MTP \\
        --weights data/weights/essa.pt \\
        --out data/essa_out/MTP
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.features import shapes as rio_shapes
from shapely.geometry import mapping, shape
from torchvision.ops import box_iou, nms

from luna.models import build_essa_model

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("predict_essa")

TILE_SIZE = 2048
ESSA_CLASSES = {1: "skylight", 2: "pit"}


def _load_tile(path: Path) -> tuple[np.ndarray, dict, str]:
    """Return (uint8 array, geotransform dict, CRS WKT) for a 2048x2048 tile."""
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"{path.name}: expected 1 band, got {src.count}")
        if src.width != TILE_SIZE or src.height != TILE_SIZE:
            raise ValueError(
                f"{path.name}: expected {TILE_SIZE}x{TILE_SIZE}, got {src.width}x{src.height}"
            )
        arr = src.read(1)
        transform = src.transform
        crs = src.crs.to_wkt() if src.crs else ""
    if arr.dtype != np.uint8:
        # Tiles may come in as float32 [0,1] or int16; normalise to uint8.
        arr = np.clip(arr, 0, np.iinfo(arr.dtype).max if np.issubdtype(arr.dtype, np.integer) else 1.0)
        if np.issubdtype(arr.dtype, np.floating):
            arr = (arr * 255.0 + 0.5).astype(np.uint8)
        else:
            arr = (arr / arr.max() * 255).astype(np.uint8) if arr.max() else arr.astype(np.uint8)
    return arr, transform, crs


def _to_tensor(arr_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    # ESSA's exact preprocessing: float32 in [0,1], shape (1, H, W).
    t = (arr_u8.astype(np.float32) / 255.0)[None, ...]
    return torch.from_numpy(t).to(device)


def _nms(out: dict, score_thr: float, iou_thr: float = 0.5) -> dict:
    keep_score = out["scores"] >= score_thr
    out = {k: v[keep_score] for k, v in out.items()}
    if out["boxes"].numel() == 0:
        return out
    # Cross-class NMS (mirrors ESSA.py: it suppresses regardless of class).
    keep = nms(out["boxes"], out["scores"], iou_thr)
    return {k: v[keep] for k, v in out.items()}


def _mask_to_polygons(mask_u8: np.ndarray, transform) -> list:
    """Polygonise a binary mask in the tile's CRS. Returns list of shapely geoms."""
    geoms = []
    for geom, val in rio_shapes(mask_u8, mask=mask_u8 > 0, transform=transform):
        if val == 0:
            continue
        g = shape(geom)
        if g.is_empty:
            continue
        geoms.append(g)
    return geoms


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tiles", type=Path, required=True,
                   help="Directory of 2048x2048 single-band GeoTIFF tiles at ~1.5 m/px")
    p.add_argument("--weights", type=Path, default=ROOT / "data" / "weights" / "essa.pt")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--score", type=float, default=0.8,
                   help="Score threshold (paper-tuned elbow = 0.8)")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--mask-thr", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=0, help="Process only the first N tiles (debug)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    tiles = sorted(p for p in args.tiles.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    if not tiles:
        log.error("no .tif/.tiff tiles in %s", args.tiles)
        return 2
    if args.limit:
        tiles = tiles[:args.limit]
    log.info("found %d tiles", len(tiles))

    device = torch.device(args.device)
    log.info("loading ESSA from %s on %s", args.weights, device)
    model = build_essa_model(checkpoint=args.weights, map_location=str(device))
    model.eval().to(device)

    features = []
    rows = []
    crs_wkt = ""
    started = time.time()

    with torch.no_grad():
        for i, tile_path in enumerate(tiles):
            try:
                arr, transform, crs_wkt = _load_tile(tile_path)
            except (ValueError, rasterio.RasterioIOError) as e:
                log.warning("skip %s: %s", tile_path.name, e)
                continue
            t = _to_tensor(arr, device)
            out = model([t])[0]
            out = {k: v.detach().cpu() for k, v in out.items()}
            out = _nms(out, args.score, args.iou)
            n = int(out["boxes"].shape[0])
            if (i + 1) % 25 == 0:
                log.info("  %d/%d tiles, %d dets so far", i + 1, len(tiles), len(rows))
            if n == 0:
                continue
            tile_id = tile_path.stem
            for j in range(n):
                cls = int(out["labels"][j])
                score = float(out["scores"][j])
                box = out["boxes"][j].tolist()
                mask = (out["masks"][j, 0].numpy() >= args.mask_thr).astype(np.uint8)
                if mask.sum() == 0:
                    continue
                polys = _mask_to_polygons(mask, transform)
                if not polys:
                    continue
                # Use the largest polygon (mask occasionally fragments at edges).
                poly = max(polys, key=lambda g: g.area)
                features.append({
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": {
                        "tile_id": tile_id,
                        "class": cls,
                        "class_name": ESSA_CLASSES.get(cls, str(cls)),
                        "score": round(score, 4),
                        "box_xyxy_px": [round(v, 1) for v in box],
                        "area_m2": round(poly.area, 2),
                    },
                })
                cx, cy = poly.centroid.x, poly.centroid.y
                rows.append({
                    "tile_id": tile_id,
                    "class": cls,
                    "class_name": ESSA_CLASSES.get(cls, str(cls)),
                    "score": round(score, 4),
                    "centroid_x": round(cx, 3),
                    "centroid_y": round(cy, 3),
                    "area_m2": round(poly.area, 2),
                })

    elapsed = time.time() - started
    geojson = {
        "type": "FeatureCollection",
        "name": "essa_detections",
        "crs": {"type": "name", "properties": {"name": crs_wkt}} if crs_wkt else None,
        "features": features,
    }
    (args.out / "detections.geojson").write_text(json.dumps(geojson, indent=1))

    fieldnames = ["tile_id", "class", "class_name", "score", "centroid_x", "centroid_y", "area_m2"]
    with (args.out / "detections.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    log.info("done: %d detections from %d tiles in %.2f min -> %s",
             len(rows), len(tiles), elapsed / 60.0, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
