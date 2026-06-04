"""
scripts/visualize_hits.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Visualisiert ESSA-bestätigte Pit-/Skylight-Kandidaten aus dem letzten Pipeline-Lauf.
Schneidet Tiles über lon/lat → GeoTIFF-Pixel-Projektion (identisch zu ESSA).

Aufruf vom Projekt-Root:
    python scripts/visualize_hits.py --nac M1118880788RC --score 0.10 --skip-preprocess
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["KMP_WARNINGS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
import rasterio.windows
from rasterio.warp import transform as rio_transform_pts

from luna import LunaPipeline

MOON_GEOG = "+proj=longlat +a=1737400 +b=1737400 +no_defs"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("dinov3").setLevel(logging.WARNING)
log = logging.getLogger("luna.visualize")

# Match ESSA TILE_SIZE (2048×2048 px)
ESSA_TILE = 2048
CROP      = ESSA_TILE // 2


def lonlat_to_rowcol(src, lon: float, lat: float) -> tuple[int, int]:
    """Project lon/lat (degrees, Moon) → GeoTIFF pixel (row, col).
    Uses the same CRS transform as essa_smoke.py."""
    if src.crs is None:
        raise RuntimeError("GeoTIFF has no embedded CRS")
    xs, ys = rio_transform_pts(MOON_GEOG, src.crs, [lon], [lat])
    col, row = (~src.transform) * (xs[0], ys[0])
    return int(round(row)), int(round(col))


def normalise_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalise to [0,1]; handles both float and uint8 GeoTIFFs."""
    arr = arr.astype(np.float32)
    lo, hi = np.nanpercentile(arr, [2, 98])
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)


def render_hits(hits, tif_path: Path, out_dir: Path, nac_id: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(hits)
    if n == 0:
        log.warning("Keine Treffer zum Visualisieren.")
        return None, None

    log.info("Öffne GeoTIFF: %s", tif_path)
    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        log.info("GeoTIFF: %dx%d px", W, H)

        tiles = []
        infos = []
        for hit in hits:
            row, col = lonlat_to_rowcol(src, hit.essa_lon, hit.essa_lat)
            # Center crop (CROP × CROP) around the ESSA detection centroid
            half = CROP // 2
            r0 = max(0, min(H - CROP, row - half))
            c0 = max(0, min(W - CROP, col - half))
            r1 = min(H, r0 + CROP)
            c1 = min(W, c0 + CROP)
            tile = src.read(1, window=rasterio.windows.Window(c0, r0, c1-c0, r1-r0))
            tile_norm = normalise_uint8(tile)
            tiles.append(tile_norm)
            # Detection centre within the crop
            det_row = row - r0
            det_col = col - c0
            infos.append((det_row, det_col))
            log.info("  Hit #%02d essa_lon=%.5f essa_lat=%.5f → row=%d col=%d",
                     hit.rank, hit.essa_lon, hit.essa_lat, row, col)

    # -------------------------------------------------------------------------
    # 1. Tile Mosaic
    # -------------------------------------------------------------------------
    cols_per_row = min(n, 3)
    rows_count   = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows_count, cols_per_row,
                             figsize=(5.5 * cols_per_row, 5.5 * rows_count),
                             facecolor="#0d0d0d")
    axes_flat = np.array(axes).flatten()

    for i, (ax, hit, tile, (drow, dcol)) in enumerate(zip(axes_flat, hits, tiles, infos)):
        ax.imshow(tile, cmap="gray", interpolation="bilinear", vmin=0, vmax=1)

        color = "#00e5ff" if hit.essa_class.lower() == "skylight" else "#ff5555"
        # Draw crosshair at detection centre
        ax.axhline(drow, color=color, lw=0.8, alpha=0.5)
        ax.axvline(dcol, color=color, lw=0.8, alpha=0.5)
        # Draw a small box around the expected pit
        box_half = 60
        rect = mpatches.Rectangle(
            (dcol - box_half, drow - box_half), box_half * 2, box_half * 2,
            linewidth=1.8, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.set_title(
            f"#{hit.rank:02d}  {hit.essa_class.upper()}"
            f"  conf={hit.essa_score:.4f}\n"
            f"ESSA  {hit.essa_lat:.5f}°  {hit.essa_lon:.5f}°",
            color=color, fontsize=8.5, pad=4,
        )
        ax.axis("off")

    # Hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"LCVK + ESSA — {nac_id}  ({n} Kandidaten)",
                 color="white", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    tiles_path = out_dir / f"{nac_id}_tiles.png"
    fig.savefig(tiles_path, dpi=160, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    log.info("Tile-Mosaik → %s", tiles_path)

    # -------------------------------------------------------------------------
    # 2. NAC Overview Strip (downsampled)
    # -------------------------------------------------------------------------
    with rasterio.open(tif_path) as src:
        scale = 16
        full  = src.read(1)
    thumb = normalise_uint8(full[::scale, ::scale])
    th, tw = thumb.shape

    fig2, ax2 = plt.subplots(figsize=(4, 12), facecolor="#0d0d0d")
    ax2.imshow(thumb, cmap="gray", aspect="auto", vmin=0, vmax=1)
    ax2.set_facecolor("#0d0d0d")

    with rasterio.open(tif_path) as src:
        for hit in hits:
            row, col = lonlat_to_rowcol(src, hit.essa_lon, hit.essa_lat)
            cx = col / scale
            cy = row / scale
            color = "#00e5ff" if hit.essa_class.lower() == "skylight" else "#ff5555"
            marker = 8
            ax2.plot(cx, cy, "o", ms=marker, mfc="none", mec=color, mew=1.5)
            ax2.text(cx + 2, cy, f"#{hit.rank:02d} {hit.essa_score:.2f}",
                     color=color, fontsize=6, va="center",
                     bbox=dict(facecolor="#0d0d0dcc", edgecolor="none", pad=1))

    ax2.set_title(f"Overview  ·  {nac_id}", color="white", fontsize=9, pad=5)
    ax2.axis("off")
    fig2.tight_layout()
    overview_path = out_dir / f"{nac_id}_overview.png"
    fig2.savefig(overview_path, dpi=160, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig2)
    log.info("Overview → %s", overview_path)

    return tiles_path, overview_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nac",             default="M1118880788RC")
    p.add_argument("--score",           type=float, default=0.10)
    p.add_argument("--query-dir",       default="data/_scratch/pits/")
    p.add_argument("--out-dir",         default="data/_scratch/dumps/")
    p.add_argument("--skip-preprocess", action="store_true")
    p.add_argument("--force-reingest",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    pipeline = LunaPipeline.from_pretrained("F1nnSBK/lunar-dinov3-lora")

    t0 = time.perf_counter()
    hits = pipeline.scan(args.nac, query_dir=args.query_dir,
                         force_reingest=args.force_reingest)
    refined = pipeline.refine(
        hits,
        score_thr        = args.score,
        essa_min_score   = args.score,
        output_dir       = args.out_dir,
        skip_preprocess  = args.skip_preprocess,
    )
    log.info("Pipeline: %.1fs → %d Treffer", time.perf_counter() - t0, len(refined))

    if not refined:
        print("Keine Treffer.")
        return

    dump_root = Path(args.out_dir) / args.nac
    tifs = sorted(dump_root.glob("*_1.5.tif")) or sorted(dump_root.glob("*.tif"))
    if not tifs:
        log.error("Kein GeoTIFF in %s gefunden", dump_root)
        return

    out_vis = dump_root / "vis"
    tiles_p, overview_p = render_hits(refined, tifs[0], out_vis, args.nac)
    print(f"\nGespeichert:")
    if tiles_p:    print(f"  Tile-Mosaik → {tiles_p}")
    if overview_p: print(f"  Overview    → {overview_p}")


if __name__ == "__main__":
    main()
