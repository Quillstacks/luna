"""Unified ESSA inference driver — Le Corre's pipeline, end to end.

Two modes:
    around-pit  : preprocess one NAC, slice a few 2048-px tiles around a known
                  catalog lon/lat, run ESSA. Fast smoke test on a known target.
    full-strip  : preprocess one NAC, slide 2048-px tiles with 50%% overlap
                  across the whole strip, dedupe across boundaries. Discovery
                  mode. Optional --validate-lon/-lat checks recall on a known
                  pit inside the frame.

Pipeline (always):
    1. Fetch raw EDR via PDSIndex(archive="EDR").
    2. ISIS3 preprocessing via vendored bash scripts (needs `luna-isis` env;
       on Apple Silicon: `CONDA_SUBDIR=osx-64 conda env create -f
       environments/isis.yml`):
           lronac2isis -> spiceinit (web) -> lronaccal -> lronacecho
           -> mosrange + cam2map (equirectangular) -> gdal_translate (uint8)
           -> gdal_translate -tr 1.5 1.5 -r cubicspline (downscale).
    3. Open the projected GeoTIFF; its embedded CRS+transform is the only
       projection — no bilinear, no SPICE call (cam2map already did the
       camera-model inversion that SPICE would have).
    4. Run ESSA inline on each tile.
    5. Save: per-detection adaptive crops (bbox + mask outline + scale bar),
       overview PNG, detections.json with lon/lat in lunar geographic.

Usage:
    # Smoke test: MTP, default
    python scripts/essa_smoke.py

    # Smoke test: any pit
    python scripts/essa_smoke.py --pit-id 1

    # Full-strip discovery on the BAP NAC
    python scripts/essa_smoke.py --mode full-strip \\
        --product-id M1243133690L \\
        --validate-lon 87.599 --validate-lat 58.6979 --label BAP

    # Full-strip on Mare Frigoris (no catalog ground truth)
    python scripts/essa_smoke.py --mode full-strip \\
        --product-id <NAC_ID> --label Frigoris
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw
from rasterio.warp import transform as rio_transform_pts

from luna.io import PDSIndex, fetch_nac
from luna.models import build_essa_model

ROOT = Path(__file__).resolve().parent.parent
LPA_CSV = ROOT / "catalogs" / "lpa.csv"
PIT_NACS = ROOT / "catalogs" / "pit_nacs.json"
PIP = ROOT / "third_party" / "planetary_image_processing"
log = logging.getLogger("essa_smoke")

TILE_SIZE = 2048
TARGET_RES_M = 1.5
ESSA_CLASSES = {1: "skylight", 2: "pit"}
ISIS_ENV = "luna-isis"
MOON_GEOG = "+proj=longlat +a=1737400 +b=1737400 +no_defs"


def _lookup_pit(pit_id: int, nac_idx: int):
    rows = list(csv.DictReader(LPA_CSV.open()))
    pit = next((r for r in rows if int(r["id"]) == pit_id), None)
    if pit is None:
        raise SystemExit(f"pit id {pit_id} not in {LPA_CSV}")
    nacs = json.loads(PIT_NACS.read_text()).get(str(pit_id), [])
    if not nacs:
        raise SystemExit(f"no NACs catalogued for pit {pit_id}")
    if nac_idx >= len(nacs):
        raise SystemExit(f"nac_idx {nac_idx} out of range (have {len(nacs)})")
    return pit, nacs[nac_idx]["product"]


def _isis_available() -> bool:
    return shutil.which("lronac2isis") is not None


def _conda_run_prefix() -> list[str]:
    if _isis_available():
        return []
    if shutil.which("conda") is None:
        raise SystemExit(
            "ISIS3 not on PATH and `conda` not found.\n"
            f"Setup: `CONDA_SUBDIR=osx-64 conda env create -f environments/isis.yml`."
        )
    return ["conda", "run", "-n", ISIS_ENV, "--no-capture-output"]


def _run_in(workdir: Path, argv: list[str]) -> None:
    log.info("[%s] $ %s", workdir.name, " ".join(argv))
    subprocess.run(argv, cwd=workdir, check=True)


def preprocess_edr(edr_img: Path, workdir: Path) -> Path:
    """Run Le Corre's bash pipeline on a single .IMG. Returns final downscaled .tif."""
    workdir.mkdir(parents=True, exist_ok=True)
    staged = workdir / edr_img.name
    if not staged.exists():
        shutil.copy(edr_img, staged)

    process = PIP / "LROC_NAC_process_and_convert.sh"
    if not process.exists():
        process = None
    downscale = PIP / "downscale.sh"

    prefix = _conda_run_prefix()
    if process is not None:
        _run_in(workdir, prefix + ["bash", str(process), "1", "yes", "50"])
    else:
        _run_in(workdir, prefix + ["bash", str(PIP / "LROC_NAC_process.sh"), "1", "yes", "50"])
        _run_in(workdir, prefix + ["bash", str(PIP / "LROC_NAC_convert.sh"), "1"])

    tifs = list(workdir.glob("*.tiff")) + [t for t in workdir.glob("*.tif")
                                            if not t.name.endswith(f"_{TARGET_RES_M}.tif")]
    if not tifs:
        raise RuntimeError(f"preprocessing produced no .tif/.tiff in {workdir}")
    full = tifs[0]
    log.info("preprocessed: %s", full)

    _run_in(workdir, prefix + ["bash", str(downscale), str(workdir) + "/", str(TARGET_RES_M)])
    downscaled = sorted(workdir.glob(f"*_{TARGET_RES_M}.tif"))
    if not downscaled:
        raise RuntimeError(f"downscale produced no *_{TARGET_RES_M}.tif in {workdir}")
    log.info("downscaled: %s", downscaled[-1])
    return downscaled[-1]


def lonlat_to_rowcol(src, lon: float, lat: float) -> tuple[int, int]:
    if src.crs is None:
        raise RuntimeError("source has no CRS — preprocessing did not embed projection")
    xs, ys = rio_transform_pts(MOON_GEOG, src.crs, [lon], [lat])
    col, row = (~src.transform) * (xs[0], ys[0])
    return int(round(row)), int(round(col))


def rowcol_to_lonlat(src, row: float, col: float) -> tuple[float, float]:
    x, y = src.transform * (col, row)
    lons, lats = rio_transform_pts(src.crs, MOON_GEOG, [x], [y])
    return float(lons[0]), float(lats[0])


def _read_tile(src, r0: int, c0: int) -> np.ndarray:
    window = rasterio.windows.Window(c0, r0, TILE_SIZE, TILE_SIZE)
    arr = src.read(1, window=window, boundless=True, fill_value=0)
    if arr.shape != (TILE_SIZE, TILE_SIZE):
        pad = np.zeros((TILE_SIZE, TILE_SIZE), dtype=arr.dtype)
        pad[:arr.shape[0], :arr.shape[1]] = arr
        arr = pad
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _infer_tile(model, tile_u8: np.ndarray, device, score_thr: float) -> list[dict]:
    t = torch.from_numpy(tile_u8.astype(np.float32) / 255.0)[None, ...].to(device)
    with torch.no_grad():
        out = model([t])[0]
    out = {k: v.detach().cpu() for k, v in out.items()}
    keep = out["scores"] >= score_thr
    if not keep.any():
        return []
    boxes = out["boxes"][keep].numpy()
    scores = out["scores"][keep].numpy()
    labels = out["labels"][keep].numpy()
    masks = (out["masks"][keep, 0].numpy() >= 0.5)
    dets = []
    for i in range(len(scores)):
        if masks[i].sum() == 0:
            continue
        ys, xs = np.where(masks[i])
        dets.append({
            "class": int(labels[i]),
            "score": float(scores[i]),
            "box": [float(v) for v in boxes[i]],
            "mask": masks[i],
            "centroid_local": (float(xs.mean()), float(ys.mean())),
            "mask_area_px": int(masks[i].sum()),
        })
    return dets


def _dedupe(dets: list[dict], dist_px: float) -> list[dict]:
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    kept = []
    for d in dets:
        cx, cy = d["centroid_full"]
        if any(abs(cx - k["centroid_full"][0]) + abs(cy - k["centroid_full"][1]) < dist_px for k in kept):
            continue
        kept.append(d)
    return kept


def _render_crop(src, det: dict, out_path: Path, mark_target_xy=None) -> None:
    """Adaptive crop with bbox + mask outline + scale bar."""
    fx0, fy0, fx1, fy1 = det["box_full"]
    bw, bh = fx1 - fx0, fy1 - fy0
    margin = 128
    crop_size = int(max(512, bw + 2 * margin, bh + 2 * margin))
    cx, cy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
    H, W = src.height, src.width
    r0c = int(max(0, min(H - crop_size, cy - crop_size / 2)))
    c0c = int(max(0, min(W - crop_size, cx - crop_size / 2)))
    arr = src.read(
        1,
        window=rasterio.windows.Window(c0c, r0c, crop_size, crop_size),
        boundless=True, fill_value=0,
    )
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    rgb = np.stack([arr] * 3, axis=-1).copy()

    cmap = {1: (60, 220, 120), 2: (235, 90, 90)}
    cc = cmap.get(det["class"], (255, 255, 0))

    # Place mask in crop coords and tint interior.
    m = det["mask"]
    sx = det["tile_origin"][1] - c0c
    sy = det["tile_origin"][0] - r0c
    mh, mw = m.shape
    x_lo, y_lo = max(0, sx), max(0, sy)
    x_hi, y_hi = min(crop_size, sx + mw), min(crop_size, sy + mh)
    if x_hi > x_lo and y_hi > y_lo:
        sub_m = m[
            max(0, -sy):max(0, -sy) + (y_hi - y_lo),
            max(0, -sx):max(0, -sx) + (x_hi - x_lo),
        ]
        inside = sub_m > 0
        if inside.any():
            block = rgb[y_lo:y_hi, x_lo:x_hi]
            block[inside] = (block[inside].astype(np.uint16) * 6 // 10 + np.array(cc, dtype=np.uint16) * 4 // 10).astype(np.uint8)

    cropped = Image.fromarray(rgb)
    cd = ImageDraw.Draw(cropped, "RGBA")

    # Mask outline via 2-px erosion difference (no scipy required).
    if x_hi > x_lo and y_hi > y_lo and inside.any():
        eroded = np.zeros_like(inside)
        eroded[1:-1, 1:-1] = (
            inside[1:-1, 1:-1] & inside[:-2, 1:-1] & inside[2:, 1:-1]
            & inside[1:-1, :-2] & inside[1:-1, 2:]
        )
        outline = inside & ~eroded
        ys, xs = np.where(outline)
        for px, py in zip(xs, ys):
            cd.point((int(x_lo + px), int(y_lo + py)), fill=(*cc, 255))

    cd.rectangle([fx0 - c0c, fy0 - r0c, fx1 - c0c, fy1 - r0c], outline=(*cc, 255), width=3)

    bar_px = int(round(100 / TARGET_RES_M))
    bar_y = crop_size - 30
    cd.line([(20, bar_y), (20 + bar_px, bar_y)], fill=(255, 255, 255), width=3)
    cd.text((20, bar_y - 18), "100 m", fill=(255, 255, 255, 255))

    area_m2 = det["mask_area_px"] * (TARGET_RES_M ** 2)
    cd.text(
        (6, 6),
        f"{ESSA_CLASSES.get(det['class'], det['class'])} {det['score']:.4f}\n"
        f"lon={det['lon']:.4f} lat={det['lat']:.4f}\n"
        f"bbox {bw * TARGET_RES_M:.0f}×{bh * TARGET_RES_M:.0f} m  area {area_m2 / 1e6:.2f} km²",
        fill=(255, 255, 255, 255),
    )

    if mark_target_xy is not None:
        tx, ty = mark_target_xy
        tx_o, ty_o = tx - c0c, ty - r0c
        if 0 <= tx_o < crop_size and 0 <= ty_o < crop_size:
            cd.line([(tx_o - 14, ty_o), (tx_o + 14, ty_o)], fill=(255, 240, 60), width=2)
            cd.line([(tx_o, ty_o - 14), (tx_o, ty_o + 14)], fill=(255, 240, 60), width=2)

    cropped.save(out_path)


def _render_overview(src, dets: list[dict], out_path: Path, mark_target=None) -> None:
    """Full-strip overview, downsampled to fit in memory."""
    target_h = 1200
    H, W = src.height, src.width
    scale = min(1.0, target_h / max(H, 1))
    ov_h = max(1, int(round(H * scale)))
    ov_w = max(1, int(round(W * scale)))
    full = src.read(1, out_shape=(ov_h, ov_w))
    if full.dtype != np.uint8:
        full = np.clip(full, 0, 255).astype(np.uint8)
    img = Image.fromarray(np.stack([full] * 3, axis=-1))
    draw = ImageDraw.Draw(img, "RGBA")
    cmap = {1: (60, 220, 120, 255), 2: (235, 90, 90, 255)}
    for d in dets:
        x0, y0, x1, y1 = d["box_full"]
        cc = cmap.get(d["class"], (255, 255, 0, 255))
        draw.rectangle([x0 * scale, y0 * scale, x1 * scale, y1 * scale], outline=cc, width=2)
        draw.text((x0 * scale + 2, y0 * scale + 2),
                  f"{ESSA_CLASSES.get(d['class'], d['class'])} {d['score']:.2f}",
                  fill=(255, 255, 255, 255))
    if mark_target is not None:
        tx, ty = mark_target
        tx_o, ty_o = tx * scale, ty * scale
        draw.line([(tx_o - 14, ty_o), (tx_o + 14, ty_o)], fill=(255, 240, 60, 255), width=2)
        draw.line([(tx_o, ty_o - 14), (tx_o, ty_o + 14)], fill=(255, 240, 60, 255), width=2)
    img.save(out_path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("around-pit", "full-strip"), default="around-pit")
    p.add_argument("--pit-id", type=int, default=3,
                   help="LPA pit id (used for around-pit mode and as catalog->NAC lookup)")
    p.add_argument("--nac-idx", type=int, default=0)
    p.add_argument("--product-id", default=None,
                   help="Override the catalog-derived NAC product ID (any LROC NAC)")
    p.add_argument("--label", default=None,
                   help="Output subdir tag (default: pitN_PRODUCTID for around-pit, PRODUCTID for full-strip)")
    p.add_argument("--validate-lon", type=float, default=None)
    p.add_argument("--validate-lat", type=float, default=None)
    p.add_argument("--weights", type=Path, default=ROOT / "data" / "weights" / "essa.pt")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "essa_out")
    p.add_argument("--scratch", type=Path, default=ROOT / "data" / "_scratch")
    p.add_argument("--score", type=float, default=0.8,
                   help="Score threshold (paper-tuned 0.8; 0.5 for exploration)")
    p.add_argument("--overlap", type=float, default=0.5,
                   help="Tile overlap fraction in full-strip mode")
    p.add_argument("--n-tiles", type=int, default=4,
                   help="Tiles to slice in around-pit mode")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Reuse existing downscaled GeoTIFF in workdir, skip ISIS pipeline")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.weights.exists():
        raise SystemExit(f"missing weights: {args.weights}\nRun: python scripts/fetch_essa_weights.py")

    # Resolve target NAC + (optional) catalog point
    if args.product_id:
        product_id = args.product_id
        pit = None
        catalog_lon, catalog_lat = args.validate_lon, args.validate_lat
    else:
        pit, product_id = _lookup_pit(args.pit_id, args.nac_idx)
        catalog_lon = float(pit["longitude"])
        catalog_lat = float(pit["latitude"])
        if args.validate_lon is None:
            args.validate_lon = catalog_lon
            args.validate_lat = catalog_lat

    label = args.label or (f"pit{pit['id']}_{product_id}" if pit else product_id)
    out = args.out / label
    out.mkdir(parents=True, exist_ok=True)
    log.info("mode=%s  product=%s  out=%s", args.mode, product_id, out)
    if catalog_lon is not None:
        log.info("known target: lon=%.4f lat=%.4f%s",
                 catalog_lon, catalog_lat, f"  ({pit['name']})" if pit else "")

    if args.skip_preprocess:
        cands = sorted(out.glob(f"*_{TARGET_RES_M}.tif"))
        if not cands:
            raise SystemExit(f"--skip-preprocess set but no *_{TARGET_RES_M}.tif in {out}")
        downscaled = cands[-1]
        log.info("reusing %s", downscaled)
    else:
        edr_url = PDSIndex(archive="EDR").url_for(product_id)
        log.info("EDR URL: %s", edr_url)
        edr_path = fetch_nac(product_id, dest_dir=args.scratch, url=edr_url)
        log.info("staged EDR: %s (%.1f MB)", edr_path, edr_path.stat().st_size / 1e6)
        downscaled = preprocess_edr(edr_path, out)

    log.info("loading ESSA on %s", args.device)
    model = build_essa_model(checkpoint=args.weights, map_location=args.device)
    model.eval().to(args.device)

    raw_dets: list[dict] = []
    started = time.time()
    with rasterio.open(downscaled) as src:
        H, W = src.height, src.width
        log.info("projected GeoTIFF: %dx%d  CRS=%s", W, H, src.crs)

        # Build anchor list per mode.
        if args.mode == "around-pit":
            if catalog_lon is None:
                raise SystemExit("around-pit needs catalog lon/lat (use --pit-id or --validate-lon/-lat)")
            row, col = lonlat_to_rowcol(src, catalog_lon, catalog_lat)
            log.info("target -> (row=%d, col=%d)", row, col)
            base_r, base_c = row - TILE_SIZE // 2, col - TILE_SIZE // 2
            anchors = [(base_r, base_c)]
            for dy, dx in [(0, TILE_SIZE), (TILE_SIZE, 0), (0, -TILE_SIZE)]:
                anchors.append((anchors[0][0] + dy, anchors[0][1] + dx))
            anchors = anchors[:args.n_tiles]
        else:
            stride = max(1, int(round(TILE_SIZE * (1.0 - args.overlap))))
            anchors = []
            r = 0
            while True:
                c = 0
                while True:
                    anchors.append((min(r, max(H - TILE_SIZE, 0)),
                                    min(c, max(W - TILE_SIZE, 0))))
                    if c + TILE_SIZE >= W:
                        break
                    c += stride
                if r + TILE_SIZE >= H:
                    break
                r += stride
            anchors = list(dict.fromkeys(anchors))
        log.info("tiles to process: %d", len(anchors))

        for i, (r0, c0) in enumerate(anchors):
            r0 = max(0, min(H - TILE_SIZE if H > TILE_SIZE else 0, r0))
            c0 = max(0, min(W - TILE_SIZE if W > TILE_SIZE else 0, c0))
            tile = _read_tile(src, r0, c0)
            tile_dets = _infer_tile(model, tile, args.device, args.score)
            for d in tile_dets:
                cx_t, cy_t = d["centroid_local"]
                cx_full = c0 + cx_t
                cy_full = r0 + cy_t
                fx0, fy0, fx1, fy1 = d["box"]
                d["tile_origin"] = (r0, c0)
                d["box_full"] = [c0 + fx0, r0 + fy0, c0 + fx1, r0 + fy1]
                d["centroid_full"] = (cx_full, cy_full)
                lon, lat = rowcol_to_lonlat(src, cy_full, cx_full)
                d["lon"], d["lat"] = lon, lat
                raw_dets.append(d)
            log.info("  [%d/%d] r=%d c=%d  %d dets%s",
                     i + 1, len(anchors), r0, c0, len(tile_dets),
                     f"  best={max(d['score'] for d in tile_dets):.4f}" if tile_dets else "")

        # Dedupe across tile overlaps (only meaningful in full-strip).
        dedup_dist = TILE_SIZE // 4 if args.mode == "full-strip" else 0
        dets = _dedupe(raw_dets, dist_px=dedup_dist) if dedup_dist > 0 else raw_dets
        log.info("inference: %.1f s  raw=%d  unique=%d",
                 time.time() - started, len(raw_dets), len(dets))

        # Validation against known target (if provided)
        validation = None
        target_xy = None
        if args.validate_lon is not None and args.validate_lat is not None:
            try:
                row_t, col_t = lonlat_to_rowcol(src, args.validate_lon, args.validate_lat)
                target_xy = (col_t, row_t)
                if dets:
                    closest = min(dets, key=lambda d: ((d["centroid_full"][0] - col_t) ** 2
                                                       + (d["centroid_full"][1] - row_t) ** 2))
                    dx = closest["centroid_full"][0] - col_t
                    dy = closest["centroid_full"][1] - row_t
                    dist_px = (dx * dx + dy * dy) ** 0.5
                    validation = {
                        "lon": args.validate_lon, "lat": args.validate_lat,
                        "target_pixel": [int(col_t), int(row_t)],
                        "nearest_distance_px": round(dist_px, 1),
                        "nearest_distance_m": round(dist_px * TARGET_RES_M, 1),
                        "nearest_score": round(closest["score"], 4),
                        "nearest_class": ESSA_CLASSES.get(closest["class"]),
                        "nearest_lon_lat": [round(closest["lon"], 4), round(closest["lat"], 4)],
                    }
                    log.info("validation: nearest det dist=%.1f px (%.0f m) score=%.4f class=%s",
                             dist_px, dist_px * TARGET_RES_M, closest["score"],
                             ESSA_CLASSES.get(closest["class"]))
                else:
                    validation = {"lon": args.validate_lon, "lat": args.validate_lat,
                                  "target_pixel": [int(col_t), int(row_t)],
                                  "nearest_distance_px": None}
                    log.info("validation: no detections; target was at pixel (%d,%d)", col_t, row_t)
            except Exception as e:
                log.warning("validation failed: %s", e)

        # Render outputs
        crops_dir = out / "crops"
        crops_dir.mkdir(exist_ok=True)
        for k, d in enumerate(dets):
            _render_crop(src, d, crops_dir / f"det_{k:03d}.png", mark_target_xy=target_xy)
        _render_overview(src, dets, out / "overview.png", mark_target=target_xy)

    json_dets = []
    for d in dets:
        jd = {
            "class": d["class"],
            "class_name": ESSA_CLASSES.get(d["class"]),
            "score": round(d["score"], 4),
            "tile_origin_rc": list(d["tile_origin"]),
            "box_full_xyxy_px": [round(v, 1) for v in d["box_full"]],
            "centroid_full_xy_px": [round(v, 1) for v in d["centroid_full"]],
            "mask_area_px": d["mask_area_px"],
            "lon": round(d["lon"], 6),
            "lat": round(d["lat"], 6),
        }
        json_dets.append(jd)
    (out / "detections.json").write_text(json.dumps({
        "mode": args.mode,
        "product_id": product_id,
        "label": label,
        "score_thr": args.score,
        "overlap": args.overlap if args.mode == "full-strip" else None,
        "n_tiles": len(anchors),
        "downsampled_size": [W, H],
        "device": args.device,
        "elapsed_s": round(time.time() - started, 1),
        "validate": validation,
        "n_detections": len(dets),
        "detections": json_dets,
    }, indent=2))

    log.info("done: %d unique detections, %s", len(dets), out)
    return 0 if dets else 1


if __name__ == "__main__":
    sys.exit(main())
