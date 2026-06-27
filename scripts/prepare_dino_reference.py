"""Prepare DINO reference embeddings from known pit images.

Creates a database of DINOv3 embeddings from LPA (Lunar Pit Atlas) pits
for use with DINORefiner as a lightweight alternative to ESSA.

Usage:
    python scripts/prepare_dino_reference.py [--catalog catalogs/lpa.csv] [--output data/dino_reference.npy]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from luna.config import DATA_DIR, WEIGHTS_DIR
from luna.io.pds_fetch import fetch_nac
from luna.models.dinov3 import DINOEncoder

log = logging.getLogger(__name__)


def load_lpa_catalog(catalog_path: Path) -> list[dict]:
    """Load Lunar Pit Atlas catalog."""
    import pandas as pd
    df = pd.read_csv(catalog_path)
    pits = []
    for _, row in df.iterrows():
        pits.append({
            "pit_id": int(row["pit_id"]),
            "lon": float(row["lon"]),
            "lat": float(row["lat"]),
            "diameter": float(row["funnel_max_m"]),
            "product_id": row.get("product_id", ""),
        })
    return pits


def extract_pit_tile(nac_path: Path, lon: float, lat: float, size: int = 256) -> np.ndarray | None:
    """Extract a tile centered at (lon, lat) from a NAC image."""
    from luna.io.spice_project import ground_to_image
    
    # Get image dimensions
    img = np.memmap(nac_path, dtype=np.int16, mode='r')
    height, width = img.shape
    
    try:
        # Project lon/lat to pixel coordinates
        label_path = nac_path.with_suffix('.IMG.lbl')
        if not label_path.exists():
            label_path = nac_path.parent / (nac_path.stem + '.lbl')
        
        x_px, y_px = ground_to_image(str(label_path), lon, lat)
        
        if x_px is None or y_px is None:
            return None
        
        x0 = max(0, int(x_px) - size // 2)
        y0 = max(0, int(y_px) - size // 2)
        
        # Ensure we stay within image bounds
        x0 = max(0, min(x0, width - size))
        y0 = max(0, min(y0, height - size))
        
        tile = img[y0:y0+size, x0:x0+size].copy()
        return tile
    except Exception as e:
        log.warning("Failed to extract tile for (%f, %f): %s", lon, lat, e)
        return None


def normalize_tile(tile: np.ndarray) -> np.ndarray:
    """Normalize a NAC tile to [0, 1] for DINO input."""
    from luna.config import LROC_VALID_MIN
    valid = tile[tile > LROC_VALID_MIN]
    if len(valid) == 0:
        return np.zeros_like(tile, dtype=np.float32)
    lo, hi = float(valid.min()), float(valid.max())
    if hi > lo:
        return (tile - lo) / (hi - lo)
    return np.zeros_like(tile, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DINO reference embeddings from LPA pits")
    parser.add_argument("--catalog", type=Path, default=DATA_DIR / "catalogs" / "lpa.csv",
                        help="Path to LPA catalog CSV")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "dino_reference.npy",
                        help="Output path for reference embeddings")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of pits to process (for testing)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for DINO encoder (auto-detect if None)")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load catalog
    if not args.catalog.exists():
        log.error("Catalog not found: %s", args.catalog)
        log.info("Download from: https://lroc.im-ldi.com/atlases/pits/list")
        return
    
    pits = load_lpa_catalog(args.catalog)
    if args.limit:
        pits = pits[:args.limit]
    
    log.info("Loaded %d pits from catalog", len(pits))
    
    # Initialize DINO encoder
    device = args.device
    if device is None:
        import torch
        device = (
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )
    
    encoder = DINOEncoder(
        lora_dir=str(WEIGHTS_DIR / "lunar-dinov3-lora"),
        base_weights_path=str(WEIGHTS_DIR / "lunar-dinov3-lora"),
        matryoshka_dim=384,
        device=device,
    )
    
    # Collect embeddings
    embeddings: list[np.ndarray] = []
    processed = 0
    skipped = 0
    
    for pit in tqdm(pits, desc="Processing pits"):
        pit_id = pit["pit_id"]
        product_id = pit.get("product_id", "")
        lon, lat = pit["lon"], pit["lat"]
        
        # Try to get NAC for this pit
        # For now, skip pits without product_id (need mapping from catalog)
        if not product_id:
            skipped += 1
            continue
        
        try:
            nac_path = DATA_DIR / "_scratch" / f"{product_id}.IMG"
            if not nac_path.exists():
                log.info("Fetching %s for pit %d...", product_id, pit_id)
                try:
                    nac_path = fetch_nac(product_id, dest_dir=DATA_DIR / "_scratch")
                except Exception as e:
                    log.warning("Failed to fetch NAC %s: %s", product_id, e)
                    skipped += 1
                    continue
            
            # Extract tile at pit location
            tile = extract_pit_tile(nac_path, lon, lat, size=256)
            if tile is None:
                skipped += 1
                continue
            
            # Normalize and convert to 3 channels
            norm_tile = normalize_tile(tile)
            batch = np.expand_dims(norm_tile, 0)
            batch = (batch * 255).astype(np.uint8)
            batch_3ch = np.stack([batch[0]] * 3, axis=0)
            
            # Get DINO embedding
            with torch.no_grad():
                import torch
                torch_batch = torch.from_numpy(batch_3ch).float() / 255.0
                embedding = encoder.encode(torch_batch.to(device)).cpu().numpy()
            
            embeddings.append(embedding[0])  # (384,)
            processed += 1
            
        except Exception as e:
            log.warning("Failed to process pit %d: %s", pit_id, e)
            skipped += 1
    
    # Save embeddings
    if embeddings:
        embeddings_array = np.stack(embeddings)  # (N, 384)
        np.save(args.output, embeddings_array)
        log.info("Saved %d reference embeddings to %s", len(embeddings_array), args.output)
        log.info("Skipped %d pits", skipped)
    else:
        log.error("No embeddings generated. Check catalog and NAC availability.")


if __name__ == "__main__":
    main()
