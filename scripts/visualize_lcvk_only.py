import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["KMP_WARNINGS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

import pickle
from pathlib import Path
import numpy as np
import rasterio
import rasterio.windows
from rasterio.warp import transform as rio_transform_pts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from luna import LunaPipeline
from luna.screening.pithos import PithosMIDB

MOON_GEOG = "+proj=longlat +a=1737400 +b=1737400 +no_defs"
CROP_SIZE = 512 # Size of crop around the hit coordinate

def lonlat_to_rowcol(src, lon: float, lat: float) -> tuple[int, int]:
    if src.crs is None:
        raise RuntimeError("GeoTIFF has no embedded CRS")
    xs, ys = rio_transform_pts(MOON_GEOG, src.crs, [lon], [lat])
    col, row = (~src.transform) * (xs[0], ys[0])
    return int(round(row)), int(round(col))

def normalise_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = np.nanpercentile(arr, [2, 98])
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)

def main():
    pipeline = LunaPipeline.from_pretrained("F1nnSBK/lunar-dinov3-lora")
    nac = "M1116841932RC"
    query_dir = Path("data/_scratch/pits/")
    
    query_paths = sorted(query_dir.glob("*.npy"))
    query_names = [p.stem for p in query_paths]
    
    index_prefix = f"data/_scratch/indices/pithos_{nac}"
    with open(f"{index_prefix}_meta.pkl", "rb") as f:
        metadata = pickle.load(f)
        
    query_vecs = pipeline._encode_queries(query_dir)
    
    vote_map = {}
    best_dist = {}
    best_query_idx = {}
    
    with PithosMIDB() as db:
        db.load_index(nac, f"{index_prefix}.bin")
        ids_mat, dists_mat = db.batch_search(nac, query_vecs, k=1000)
        
    for q_idx in range(ids_mat.shape[0]):
        for idx, dist in zip(ids_mat[q_idx], dists_mat[q_idx]):
            idx = int(idx)
            if idx < 0:
                continue
            vote_map[idx] = vote_map.get(idx, 0) + 1
            if idx not in best_dist or dist < best_dist[idx]:
                best_dist[idx] = float(dist)
                best_query_idx[idx] = q_idx

    # Sort by best distance
    ranked = sorted(best_dist.keys(), key=lambda i: best_dist[i])
    
    # Run NMS (keep top 50)
    nms_hits = pipeline._nms(ranked, vote_map, best_dist, metadata, top_k=50, min_dist_px=512.0)
    
    tif_path = Path(f"data/_scratch/dumps/{nac}/M1116841932RE_1.5.tif")
    if not tif_path.exists():
        tif_path = Path(f"data/_scratch/dumps/{nac}/M1116841932RE.tif")
        
    print(f"Opening GeoTIFF: {tif_path}")
    
    # Matplotlib Grid Setup: 5 rows, 10 columns
    fig, axes = plt.subplots(5, 10, figsize=(32, 16), facecolor="#0d0d0d")
    axes_flat = axes.flatten()
    
    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        
        for rank, (idx, votes, score) in enumerate(nms_hits, start=1):
            meta = metadata[idx]
            q_idx = best_query_idx[idx]
            feature_name = query_names[q_idx].split("_M1")[0].replace("_", " ")
            
            # Project lon/lat to row/col
            row, col = lonlat_to_rowcol(src, meta.lon, meta.lat)
            
            # Crop window around the projected pixel
            half = CROP_SIZE // 2
            r0 = max(0, min(H - CROP_SIZE, row - half))
            c0 = max(0, min(W - CROP_SIZE, col - half))
            r1 = min(H, r0 + CROP_SIZE)
            c1 = min(W, c0 + CROP_SIZE)
            
            tile = src.read(1, window=rasterio.windows.Window(c0, r0, c1-c0, r1-r0))
            tile_norm = normalise_uint8(tile)
            
            ax = axes_flat[rank - 1]
            ax.imshow(tile_norm, cmap="gray", interpolation="bilinear", vmin=0, vmax=1)
            
            # Centroid offsets within the cropped frame
            drow = row - r0
            dcol = col - c0
            
            # Draw the actual 256x256 candidate tile boundary (box_half = 128)
            # centered around the tile center
            box_half = 128
            rect = mpatches.Rectangle(
                (dcol - box_half, drow - box_half), box_half * 2, box_half * 2,
                linewidth=0.7, edgecolor="#00e5ff", facecolor="none", alpha=0.4, linestyle="--"
            )
            ax.add_patch(rect)
            
            # Labels
            lon = meta.lon - 360 if meta.lon > 180 else meta.lon
            ax.set_title(
                f"#{rank:02d} | Dist: {score:.1f}\n"
                f"{feature_name}\n"
                f"({lon:.4f}°, {meta.lat:.4f}°)",
                color="#00e5ff", fontsize=7.5, pad=4, fontweight="bold"
            )
            ax.axis("off")
            
        # Hide unused subplots if NMS returned fewer than 50
        for ax in axes_flat[len(nms_hits):]:
            ax.axis("off")
            
    plt.suptitle(f"Pithos Top 50 Retrieval Matches for {nac} (Distance-based)", color="white", fontsize=18, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    # Save to the local temp directory as SVG
    out_dir = Path("temp")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "pithos_only_top50.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"Visualization saved to: {out_path}")

if __name__ == "__main__":
    main()
