import logging
import os
import pickle
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt

from luna.io.pds_index import PDSIndex
from luna.io.projection import LinearProjection, pixel_to_lonlat
from luna.models.dinov3 import DINOEncoder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("luna.query_top_20")

PRODUCT_ID = "M157906985RC"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INDEX_PATH = DATA_DIR / "_scratch" / "indices" / f"faiss_{PRODUCT_ID}.index"
META_PATH  = DATA_DIR / "_scratch" / "indices" / f"faiss_{PRODUCT_ID}_meta.pkl"
IMG_PATH   = DATA_DIR / "_scratch" / f"{PRODUCT_ID}.IMG"
QUERY_NPY  = DATA_DIR / "_scratch" / "pits" / "Aristarchus_6_M109548636LC.npy"

# --- TOP 20 ---
TOP_K = 20
PDS3_OFFSET = 5064
IMG_WIDTH = 5064
ZOOM_SIZE = 256 
OUTPUT_PLOT = PROJECT_ROOT / "temp" / f"top_20_hits_{PRODUCT_ID}.png"

def load_projection(product_id: str) -> LinearProjection | None:
    log.info("Loading geometry for %s...", product_id)
    pds = PDSIndex()
    try:
        raw = pds.geometry_for(product_id)
        geometry = {str(k).lower(): (v.value if hasattr(v, 'value') else v) for k, v in raw.items()}
        samples = int(geometry.get("line_samples", 5064))
        lines = int(geometry.get("image_lines", 52224))
        return LinearProjection.from_nac_geometry(geometry, samples=samples, lines=lines)
    except Exception as e:
        log.warning("Projection failed: %s", e)
        return None

def spatial_nms(distances, indices, metadata_store, top_k=20, min_dist_px=512.0):
    filtered_dists, filtered_indices, accepted_centers = [], [], []
    for dist, idx in zip(distances[0], indices[0]):
        meta = metadata_store[idx]
        cx, cy = meta.x_offset + meta.width / 2.0, meta.y_offset + meta.height / 2.0
        
        if not any(np.sqrt((cx-acx)**2 + (cy-acy)**2) < min_dist_px for acx, acy in accepted_centers):
            accepted_centers.append((cx, cy))
            filtered_dists.append(dist)
            filtered_indices.append(idx)
            
        if len(filtered_indices) == top_k: 
            break
            
    return np.array([filtered_dists]), np.array([filtered_indices])

def norm_raw_crop(crop: np.ndarray) -> np.ndarray:
    valid = crop[crop > -32752]
    if valid.size == 0: 
        return np.zeros_like(crop, dtype=np.float32)
        
    f_min, f_max = valid.min(), valid.max()
    if f_max > f_min:
        return (crop - f_min) / (f_max - f_min)
        
    return np.zeros_like(crop, dtype=np.float32)

def plot_results(query_img, distances, indices, metadata_store, projection):
    file_size = os.path.getsize(str(IMG_PATH))
    height = (file_size - PDS3_OFFSET) // (IMG_WIDTH * 2)
    raw_img = np.memmap(str(IMG_PATH), dtype=np.int16, mode="r", offset=PDS3_OFFSET, shape=(height, IMG_WIDTH))

    # 5x5 Grid für 1 Query + 20 Results
    fig, axes = plt.subplots(5, 5, figsize=(22, 22))
    axes = axes.flatten()
    
    # Anchor Bild auf Index 0
    axes[0].imshow(query_img, cmap="gray")
    axes[0].set_title("QUERY ANCHOR", fontweight="bold", color="blue")
    axes[0].axis("off")

    for i, (rank, dist, idx) in enumerate(zip(range(1, TOP_K + 1), distances[0], indices[0]), start=1):
        meta = metadata_store[idx]
        cx, cy = int(meta.x_offset + meta.width // 2), int(meta.y_offset + meta.height // 2)
        
        coord_str = "No Meta"
        if projection:
            lon, lat = pixel_to_lonlat(projection, cx, cy)
            coord_str = f"{lat:.5f}N\n{lon:.5f}E"

        x0, y0 = max(0, cx - ZOOM_SIZE // 2), max(0, cy - ZOOM_SIZE // 2)
        x1, y1 = min(IMG_WIDTH, x0 + ZOOM_SIZE), min(height, y0 + ZOOM_SIZE)
        
        crop = raw_img[y0:y1, x0:x1].copy().astype(np.float32)
        
        axes[i].imshow(norm_raw_crop(crop), cmap="gray")
        axes[i].set_title(f"Rank {rank} (D:{dist:.3f})\n{coord_str}", fontsize=9)
        axes[i].axis("off")
        log.info("Rank %02d | Dist: %.4f | Coords: %s", rank, dist, coord_str.replace('\n', ', '))

    for j in range(TOP_K + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    os.makedirs(OUTPUT_PLOT.parent, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=200)
    plt.close()

def main():
    projection = load_projection(PRODUCT_ID)
    encoder = DINOEncoder(lora_dir="F1nnSBK/lunar-dinov3-lora", base_weights_path="F1nnSBK/lunar-dinov3-lora", model_size="vits16", device="mps")
    
    img_array = np.load(QUERY_NPY)
    q_norm = norm_raw_crop(img_array)
    batch = q_norm.astype(np.float32)
    if batch.ndim == 2:
        batch = np.expand_dims(batch, axis=0)
    batch_uint8 = (batch * 255).astype(np.uint8)
    
    q_vec = encoder.encode(batch_uint8)
    
    import faiss
    log.info("Loading Metadata (Pickle)...")
    with open(META_PATH, "rb") as f: 
        metadata = pickle.load(f)
    
    log.info("Loading FAISS Index...")
    index = faiss.read_index(str(INDEX_PATH))
    
    q_vec = np.ascontiguousarray(q_vec, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(q_vec)
    
    dists, ids = index.search(q_vec, 150)
    dists, ids = spatial_nms(dists, ids, metadata, top_k=TOP_K)

    plot_results(q_norm, dists, ids, metadata, projection)
    log.info("HITL Report generated: %s", OUTPUT_PLOT)

if __name__ == "__main__":
    main()