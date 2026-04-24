"""Run the trained Mask R-CNN on a fresh NAC and emit a review-ready dataset.

Main-loop step 3 (inference) + step 4 input (reviewable candidates).

For a given NAC product id:
    1. Resolve + download the CDR via PDSIndex.
    2. Decode, attach geometry.
    3. Slide a fixed-size window (matching the training crop size) across
       the full frame with ``--overlap`` stride.
    4. Run the detector on every tile, keep pit instances above
       ``--score-thr``.
    5. NMS across tile boundaries by centroid distance.
    6. Project every surviving detection back to lon/lat via the
       NAC's linear projection.
    7. Emit for each candidate: a crop PNG + binary mask PNG, plus a
       ``candidates.csv`` with lon/lat/score/size. This is what the human
       relabels (main-loop step 4) before feeding it back into training.

Usage:
    python scripts/predict_nac.py M1149067652R \\
        --ckpt checkpoints/maskrcnn_pit.pt \\
        --out-dir data/candidates/M1149067652R
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from luna.io import LinearProjection, fetch_nac, pixel_to_lonlat, read_nac
from luna.models import build_maskrcnn

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("predict_nac")


def _to_uint8(tile: np.ndarray) -> np.ndarray:
    t = np.nan_to_num(tile, nan=0.0, posinf=1.0, neginf=0.0)
    t = np.clip(t, 0.0, 1.0)
    return (t * 255.0 + 0.5).astype(np.uint8)


def _tile_to_tensor(tile: np.ndarray, device: str) -> torch.Tensor:
    t = np.nan_to_num(tile, nan=0.0).astype(np.float32)
    t = np.clip(t, 0.0, 1.0)
    t = np.stack([t, t, t], axis=0)  # 1-channel NAC -> 3-channel
    return torch.from_numpy(t).to(device)


def _centroid_dedupe(
    dets: list[dict], min_dist_px: float
) -> list[dict]:
    """Greedy dedupe: sort by score desc, keep a det if no kept one is within min_dist_px."""
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    kept: list[dict] = []
    for d in dets:
        cx, cy = d["center_px"]
        if any(abs(cx - k["center_px"][0]) + abs(cy - k["center_px"][1]) < min_dist_px for k in kept):
            continue
        kept.append(d)
    return kept


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("product_id")
    p.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / "maskrcnn_pit.pt")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--scratch", type=Path, default=ROOT / "data" / "_scratch")
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=128, help="tile stride overlap in pixels")
    p.add_argument("--score-thr", type=float, default=0.5)
    p.add_argument("--mask-thr", type=float, default=0.5)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--keep-img", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "crops").mkdir(exist_ok=True)
    (args.out_dir / "masks").mkdir(exist_ok=True)

    img_path = fetch_nac(args.product_id, dest_dir=args.scratch)
    nac = read_nac(img_path, geometry=True)
    proj = LinearProjection.from_nac_geometry(nac.geometry, samples=nac.samples, lines=nac.lines)
    log.info("decoded %s  %d lines × %d samples  res=%.3f m/px",
             nac.product_id, nac.lines, nac.samples, nac.resolution_m or 0.0)

    model = build_maskrcnn(num_classes=args.num_classes, pretrained=False)
    state = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(state["model"])
    model.eval().to(args.device)

    step = args.crop_size - args.overlap
    dets: list[dict] = []
    H, W = nac.pixels.shape
    n_tiles = ((H - args.crop_size) // step + 1) * ((W - args.crop_size) // step + 1)
    log.info("sliding window: %d tiles", n_tiles)

    done = 0
    for y0 in range(0, H - args.crop_size + 1, step):
        for x0 in range(0, W - args.crop_size + 1, step):
            tile = nac.pixels[y0:y0 + args.crop_size, x0:x0 + args.crop_size]
            if np.isfinite(tile).mean() < 0.5:
                done += 1
                continue
            t = _tile_to_tensor(tile, args.device)
            with torch.no_grad():
                out = model([t])[0]
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy()
            boxes = out["boxes"].cpu().numpy()
            masks = out["masks"].cpu().numpy()  # (N, 1, H, W)
            for i, s in enumerate(scores):
                if s < args.score_thr or labels[i] != 1:
                    continue
                m = masks[i, 0] >= args.mask_thr
                if not m.any():
                    continue
                ys, xs = np.where(m)
                cx = float(xs.mean()) + x0
                cy = float(ys.mean()) + y0
                lon, lat = pixel_to_lonlat(proj, int(round(cx)), int(round(cy)))
                dets.append({
                    "score": float(s),
                    "center_px": (cx, cy),
                    "tile_origin": (x0, y0),
                    "box_in_tile": boxes[i].tolist(),
                    "mask": m,
                    "tile": tile.copy(),
                    "lon": lon, "lat": lat,
                    "area_px": int(m.sum()),
                })
            done += 1
            if done % 50 == 0:
                log.info("  %d/%d tiles, %d raw dets", done, n_tiles, len(dets))

    dets = _centroid_dedupe(dets, min_dist_px=args.crop_size // 2)
    log.info("after dedupe: %d candidates", len(dets))

    rows: list[dict] = []
    for k, d in enumerate(dets):
        name = f"{nac.product_id}_{k:04d}"
        Image.fromarray(_to_uint8(d["tile"])).save(args.out_dir / "crops" / f"{name}.png")
        Image.fromarray((d["mask"].astype(np.uint8) * 255)).save(args.out_dir / "masks" / f"{name}.png")
        rows.append({
            "name": name,
            "product_id": nac.product_id,
            "score": round(d["score"], 4),
            "center_x": int(round(d["center_px"][0])),
            "center_y": int(round(d["center_px"][1])),
            "tile_x0": d["tile_origin"][0],
            "tile_y0": d["tile_origin"][1],
            "lon": round(d["lon"], 6),
            "lat": round(d["lat"], 6),
            "area_px": d["area_px"],
            "resolution_m": nac.resolution_m,
        })
    with open(args.out_dir / "candidates.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["name", "product_id", "score", "center_x", "center_y",
                      "tile_x0", "tile_y0", "lon", "lat", "area_px", "resolution_m"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    if not args.keep_img:
        img_path.unlink(missing_ok=True)
    log.info("wrote %d candidates -> %s", len(rows), args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
