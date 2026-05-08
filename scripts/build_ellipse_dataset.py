"""Build the training set for the pit-segmentation main loop.

For every (pit, NAC) pair in ``catalogs/pit_nacs.json``:
    1. Resolve the CDR URL via PDSIndex.
    2. Stream the ``.IMG`` into a scratch dir.
    3. Decode it, attach geometry from INDEX.TAB.
    4. Project the LPA lon/lat into pixel space.
    5. Crop a fixed window around the pit.
    6. Rasterise an ellipse from ``funnel_max_m``, ``funnel_min_m``,
       ``azimuth_deg`` as the pseudo-label.
    7. Write crop PNG + COCO annotation (bbox + RLE mask).
    8. Delete the ``.IMG`` — we don't keep the full frame.

The ``catalogs/pit_nacs.json`` file is the source of truth for pit -> NAC
mapping (the LPA CSV's ``reference_nac`` column is empty upstream).

Outputs:
    data/ellipse_ds/images/<pit_id>_<product>.png
    data/ellipse_ds/pits.json          (COCO format)
    data/ellipse_ds/build_log.csv      (success / skip / reason)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from luna.io import PDSIndex, fetch_nac, read_nac
from luna.io.spice_project import ensure_kernels_for_label, ground_to_image
from luna.labels import pit_mask_from_lpa, read_lpa_csv

# ALE/scipy compatibility shim: scipy removed Rotation.as_dcm; older ALE versions
# bundled with pip still expect it. Must be applied before ALE/spiceypy paths touch it.
try:
    from scipy.spatial.transform import Rotation as _R
    if not hasattr(_R, "as_dcm"):
        _R.as_dcm = _R.as_matrix
    if not hasattr(_R, "from_dcm"):
        _R.from_dcm = _R.from_matrix
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("build_ellipse")


def _encode_rle(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _to_uint8(tile: np.ndarray) -> np.ndarray:
    """Clip to [0,1], fill NaNs with 0, return H×W uint8."""
    t = np.nan_to_num(tile, nan=0.0, posinf=1.0, neginf=0.0)
    t = np.clip(t, 0.0, 1.0)
    return (t * 255.0 + 0.5).astype(np.uint8)


def _crop_bounds(cx: int, cy: int, size: int, H: int, W: int) -> Optional[tuple[int, int, int, int]]:
    """Return (y0, y1, x0, x1) of a size×size crop centered on (cx, cy), or None if OOB."""
    half = size // 2
    y0, y1 = cy - half, cy - half + size
    x0, x1 = cx - half, cx - half + size
    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
        return None
    return y0, y1, x0, x1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", type=Path, default=ROOT / "catalogs" / "lpa.csv")
    p.add_argument("--pit-nacs", type=Path, default=ROOT / "catalogs" / "pit_nacs.json")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "ellipse_ds")
    p.add_argument("--scratch", type=Path, default=ROOT / "data" / "_scratch")
    p.add_argument("--crop-size", type=int, default=1024)
    p.add_argument("--limit", type=int, default=0, help="stop after N (pit,NAC) pairs; 0 = all")
    p.add_argument("--keep-img", action="store_true", help="do not delete NAC .IMG after use")
    p.add_argument("--resume", action="store_true", help="skip crops that already exist")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir: Path = args.out_dir
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    pit_nacs = json.loads(args.pit_nacs.read_text())
    lpa = {e.id: e for e in read_lpa_csv(args.catalog)}
    index = PDSIndex()

    pairs: list[tuple[str, str]] = []
    for pit_id, nacs in pit_nacs.items():
        for n in nacs:
            pairs.append((pit_id, n["product"]))
    if args.limit:
        pairs = pairs[: args.limit]
    log.info("%d pit-NAC pairs to process (pits=%d)", len(pairs), len(pit_nacs))

    coco = {
        "info": {"description": "LPA pits, ellipse-approximated masks"},
        "licenses": [],
        "categories": [{"id": 1, "name": "pit", "supercategory": "geomorphology"}],
        "images": [],
        "annotations": [],
    }
    log_rows: list[dict] = []
    next_img_id = 1
    next_ann_id = 1

    def _log(pit_id: str, product: str, status: str, reason: str = "") -> None:
        log_rows.append({"pit_id": pit_id, "product": product, "status": status, "reason": reason})

    t0 = time.time()
    for k, (pit_id, product) in enumerate(pairs):
        entry = lpa.get(pit_id)
        if entry is None or entry.funnel_max_m is None:
            _log(pit_id, product, "skip", "no-catalog-or-no-funnel")
            continue

        stem = f"{pit_id}_{product}"
        out_png = img_dir / f"{stem}.png"
        if args.resume and out_png.exists():
            _log(pit_id, product, "skip", "resume-exists")
            continue

        try:
            img_path = fetch_nac(product, dest_dir=args.scratch)
        except Exception as e:  # noqa: BLE001
            _log(pit_id, product, "fail", f"fetch:{e}")
            continue

        try:
            nac = read_nac(img_path, geometry=True)
        except Exception as e:  # noqa: BLE001
            _log(pit_id, product, "fail", f"read:{e}")
            if not args.keep_img:
                img_path.unlink(missing_ok=True)
            continue

        # SPICE is the only projection — pixel-accurate from the camera model.
        # Catalog lon/lat is the only remaining source of error (~50–100 m for
        # minor pits). The hand-labeling tool refines from there.
        ensure_kernels_for_label(img_path)
        s, l = ground_to_image(img_path, entry.longitude, entry.latitude)
        px, py = int(round(s)), int(round(l))
        proj_source = "spice"

        # Skip if the ellipse is too big to fit inside the chosen crop.
        res = nac.resolution_m or 1.0
        pit_diam_px = entry.funnel_max_m / res
        if pit_diam_px > 0.7 * args.crop_size:
            _log(pit_id, product, "skip", f"pit-too-large-px={pit_diam_px:.0f}")
            if not args.keep_img:
                img_path.unlink(missing_ok=True)
            continue

        bounds = _crop_bounds(px, py, args.crop_size, nac.lines, nac.samples)
        if bounds is None:
            _log(pit_id, product, "skip", f"oob-px=({px},{py})")
            if not args.keep_img:
                img_path.unlink(missing_ok=True)
            continue
        y0, y1, x0, x1 = bounds
        tile = nac.pixels[y0:y1, x0:x1]
        # Reject tiles that are mostly invalid (sentinel NaN).
        valid_frac = float(np.isfinite(tile).mean())
        if valid_frac < 0.8:
            _log(pit_id, product, "skip", f"valid-frac={valid_frac:.2f}")
            if not args.keep_img:
                img_path.unlink(missing_ok=True)
            continue

        cx_crop = px - x0
        cy_crop = py - y0
        mask = pit_mask_from_lpa(
            shape=tile.shape,
            center_px=(cx_crop, cy_crop),
            funnel_max_m=entry.funnel_max_m,
            resolution_m=res,
            funnel_min_m=entry.funnel_min_m,
            azimuth_deg=entry.azimuth_deg,
        )
        if not mask.any():
            _log(pit_id, product, "skip", "empty-mask")
            if not args.keep_img:
                img_path.unlink(missing_ok=True)
            continue

        Image.fromarray(_to_uint8(tile)).save(out_png)
        ys, xs = np.where(mask)
        bx0, by0 = int(xs.min()), int(ys.min())
        bw = int(xs.max() - bx0 + 1)
        bh = int(ys.max() - by0 + 1)
        coco["images"].append({
            "id": next_img_id,
            "file_name": out_png.name,
            "width": int(tile.shape[1]),
            "height": int(tile.shape[0]),
            "pit_id": pit_id,
            "product_id": product,
            "resolution_m": float(res),
            "origin_xy": [int(x0), int(y0)],
            "footprint": nac.footprint,
            "nac_samples": int(nac.samples), "nac_lines": int(nac.lines),
            "center_px_hint": [int(px), int(py)],
            "projection": proj_source,
        })
        coco["annotations"].append({
            "id": next_ann_id,
            "image_id": next_img_id,
            "category_id": 1,
            "bbox": [bx0, by0, bw, bh],
            "area": float(mask.sum()),
            "iscrowd": 0,
            "segmentation": _encode_rle(mask),
            "pit_id": pit_id,
        })
        next_img_id += 1
        next_ann_id += 1

        if not args.keep_img:
            img_path.unlink(missing_ok=True)

        _log(pit_id, product, "ok", "")
        if (k + 1) % 10 == 0:
            elapsed = time.time() - t0
            log.info("%d/%d pairs — %d crops emitted — %.1fs (%.1fs/pair)",
                     k + 1, len(pairs), len(coco["images"]), elapsed, elapsed / (k + 1))

    (out_dir / "pits.json").write_text(json.dumps(coco, indent=2))
    with open(out_dir / "build_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pit_id", "product", "status", "reason"])
        w.writeheader()
        w.writerows(log_rows)

    ok = sum(1 for r in log_rows if r["status"] == "ok")
    skipped = sum(1 for r in log_rows if r["status"] == "skip")
    failed = sum(1 for r in log_rows if r["status"] == "fail")
    log.info("done: ok=%d skip=%d fail=%d  -> %s", ok, skipped, failed, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
