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
    import csv
    import json
    
    # Load pit_nacs to map pit_id -> product_id
    pit_nacs_path = catalog_path.parent / "pit_nacs.json"
    pit_nacs = {}
    if pit_nacs_path.exists():
        try:
            pit_nacs = json.loads(pit_nacs_path.read_text())
        except Exception:
            pass

    pits = []
    with open(catalog_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pit_id = row.get("id") or row.get("pit_id")
            if not pit_id:
                continue
            
            # Find a product_id from the catalog row or pit_nacs
            product_id = row.get("product_id") or row.get("reference_nac")
            if not product_id and str(pit_id) in pit_nacs:
                nacs = pit_nacs[str(pit_id)]
                if nacs:
                    product_id = nacs[0].get("product")
            
            lon = row.get("longitude") or row.get("lon")
            lat = row.get("latitude") or row.get("lat")
            diameter = row.get("funnel_max_m") or row.get("diameter")

            pits.append({
                "pit_id": int(pit_id),
                "lon": float(lon) if lon else 0.0,
                "lat": float(lat) if lat else 0.0,
                "diameter": float(diameter) if diameter else 0.0,
                "product_id": product_id or "",
            })
    return pits


def extract_pit_tile(nac_path: Path, lon: float, lat: float, size: int = 256) -> np.ndarray | None:
    """Extract a tile centered at (lon, lat) from a NAC image."""
    from luna.io.spice_project import ground_to_image
    from luna.io.nac_reader import read_nac
    
    try:
        # Load NAC image geometry and dimensions
        nac_img = read_nac(nac_path, geometry=True)
        height, width = nac_img.lines, nac_img.samples
        
        # Project lon/lat to pixel coordinates
        label_path = nac_path.with_suffix('.IMG.lbl')
        if not label_path.exists():
            label_path = nac_path.parent / (nac_path.stem + '.lbl')
        if not label_path.exists():
            label_path = nac_path
            
        from luna.io.spice_project import ground_to_image, ensure_kernels_for_label
        ensure_kernels_for_label(str(label_path))
        
        x_px, y_px = ground_to_image(str(label_path), lon, lat)
        
        if x_px is None or y_px is None:
            return None
        
        x0 = max(0, int(x_px) - size // 2)
        y0 = max(0, int(y_px) - size // 2)
        
        # Ensure we stay within image bounds
        x0 = max(0, min(x0, width - size))
        y0 = max(0, min(y0, height - size))
        
        tile = nac_img.pixels[y0:y0+size, x0:x0+size].copy()
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
    root_catalog = Path(__file__).resolve().parent.parent / "catalogs" / "lpa.csv"
    default_catalog = root_catalog if root_catalog.exists() else DATA_DIR / "catalogs" / "lpa.csv"
    parser.add_argument("--catalog", type=Path, default=default_catalog,
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
        lora_dir="F1nnSBK/lunar-dinov3-lora",
        base_weights_path="F1nnSBK/lunar-dinov3-lora",
        matryoshka_dim=384,
        device=device,
    )
    
    # Check if we can build directly from pre-extracted tiles in data/_scratch/pits/
    pits_dir = DATA_DIR / "_scratch" / "pits"
    pre_extracted_files = list(pits_dir.glob("*.npy")) if pits_dir.exists() else []
    # Filter out dino_reference.npy if it is there
    pre_extracted_files = [f for f in pre_extracted_files if f.name != "dino_reference.npy"]
    
    if pre_extracted_files:
        log.info("Found %d pre-extracted pit tiles in %s. Building reference database from them directly...", 
                 len(pre_extracted_files), pits_dir)
        if args.limit:
            pre_extracted_files = pre_extracted_files[:args.limit]
            
        embeddings: list[np.ndarray] = []
        for fpath in tqdm(pre_extracted_files, desc="Encoding pre-extracted tiles"):
            try:
                tile = np.load(fpath)
                if tile.shape != (256, 256):
                    continue
                norm_tile = normalize_tile(tile)
                batch = np.expand_dims(norm_tile, 0)
                batch = (batch * 255).astype(np.uint8)
                embedding = encoder.encode(batch)
                embeddings.append(embedding[0])
            except Exception as e:
                log.warning("Failed to process pre-extracted tile %s: %s", fpath.name, e)
                
        if embeddings:
            embeddings_array = np.stack(embeddings)
            np.save(args.output, embeddings_array)
            log.info("Saved %d reference embeddings to %s", len(embeddings_array), args.output)
            return
            
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
            # Get DINO embedding
            embedding = encoder.encode(batch)
            
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
