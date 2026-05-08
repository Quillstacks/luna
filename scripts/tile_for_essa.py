"""Slice a preprocessed lunar GeoTIFF into 2048x2048 tiles ESSA can consume.

Input: a single map-projected GeoTIFF at ~1.5 m/px (the output of
third_party/planetary_image_processing + downscale.sh).

Output: a flat directory of 2048x2048 single-band uint8 GeoTIFFs, each with
a valid geotransform + CRS so detections can be back-projected.

Usage:
    python scripts/tile_for_essa.py \\
        --input  data/essa_smoke/pit3_M126710873R/M126710873RE_1.5.tif \\
        --output data/essa_tiles/MTP \\
        --overlap 0.0       # 0 for whole-Moon sweep, 0.25 around known targets
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, transform as window_transform

log = logging.getLogger("tile_for_essa")

TILE_SIZE = 2048


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        a = np.nan_to_num(arr, nan=0.0)
        if a.max() <= 1.0:
            return (np.clip(a, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
        return np.clip(a, 0, 255).astype(np.uint8)
    info = np.iinfo(arr.dtype)
    return ((arr.astype(np.int64) - info.min) * 255 // (info.max - info.min)).astype(np.uint8)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Map-projected lunar GeoTIFF")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overlap", type=float, default=0.0,
                   help="Tile overlap fraction in [0, 0.9). 0 for global sweep, 0.25 for known targets.")
    p.add_argument("--min-valid", type=float, default=0.5,
                   help="Drop tiles where < this fraction of pixels are nonzero (no-data filter)")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    if not 0.0 <= args.overlap < 0.9:
        raise SystemExit("--overlap must be in [0, 0.9)")

    stride = max(1, int(round(TILE_SIZE * (1.0 - args.overlap))))
    written = 0
    skipped = 0

    with rasterio.open(args.input) as src:
        if src.count != 1:
            raise SystemExit(f"expected 1 band, got {src.count}")
        H, W = src.height, src.width
        log.info("input %dx%d, stride=%d, overlap=%.2f", W, H, stride, args.overlap)

        for r0 in range(0, H, stride):
            for c0 in range(0, W, stride):
                if r0 + TILE_SIZE > H or c0 + TILE_SIZE > W:
                    continue
                window = Window(c0, r0, TILE_SIZE, TILE_SIZE)
                arr = src.read(1, window=window)
                valid = float(np.count_nonzero(arr)) / arr.size
                if valid < args.min_valid:
                    skipped += 1
                    continue
                arr = _to_uint8(arr)
                tile_path = args.output / f"{args.input.stem}_r{r0:06d}_c{c0:06d}.tif"
                with rasterio.open(
                    tile_path, "w", driver="GTiff",
                    height=TILE_SIZE, width=TILE_SIZE, count=1, dtype="uint8",
                    transform=window_transform(window, src.transform), crs=src.crs,
                ) as dst:
                    dst.write(arr, 1)
                written += 1
                if written % 50 == 0:
                    log.info("  wrote %d tiles", written)
                if args.limit and written >= args.limit:
                    log.info("limit reached")
                    log.info("done: wrote=%d skipped=%d -> %s", written, skipped, args.output)
                    return 0

    log.info("done: wrote=%d skipped=%d -> %s", written, skipped, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
