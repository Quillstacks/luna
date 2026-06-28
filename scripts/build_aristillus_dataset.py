#!/usr/bin/env python3
"""Build Aristillus dataset with 3 known pits for Stage-2 refiner training.

This script:
1. Loads M1118880788RC NAC image
2. Extracts tiles around the 3 known Aristillus pits (Ground Truth)
3. Creates a parquet-based Hugging Face dataset
4. Pushes to Hugging Face Hub as beta version

Known Pits in M1118880788RC:
- Aristillus 1: Lat 33.6622, Lon 0.7149, Depth 11m
- Aristillus 3: Lat 33.5169, Lon 0.9128, Depth 8m
- Aristillus 4: Lat 33.4981, Lon 0.9447, Depth 14m

Metadata from SPICE:
- Incidence Angle: 35.03°
- Sub-Solar Azimuth: calculated from sub_solar_longitude
- Resolution: 1.047 m/px
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from luna.io.projection import get_image_of_roi, LinearProjection
from luna.io.pds_index import PDSIndex
from luna.io.nac_reader import read_nac

log = logging.getLogger(__name__)


@dataclass
class PitInfo:
    """Information about a known pit."""
    name: str
    pit_id: int
    latitude: float
    longitude: float
    depth_m: float
    x_offset: int  # Pixel X coordinate in NAC image
    y_offset: int  # Pixel Y coordinate in NAC image


@dataclass
class DatasetConfig:
    """Configuration for dataset building."""
    dataset_name: str = "aristillus-pits-beta"
    description: str = "Aristillus crater pits for Stage-2 refiner training (3 known pits)"
    tile_size: int = 256
    num_tiles_per_pit: int = 1  # Number of tiles to extract around each pit
    splits: dict = None
    
    def __post_init__(self):
        if self.splits is None:
            # With 3 pits, use simple splits
            self.splits = {"train": 2, "test": 0, "val": 1}



def get_aristillus_pits() -> List[PitInfo]:
    """Get the 3 known Aristillus pits with their metadata."""
    pits = [
        PitInfo(name="Aristillus 1", pit_id=39, latitude=33.6622, longitude=0.7149, depth_m=11.0, x_offset=0, y_offset=0),
        PitInfo(name="Aristillus 3", pit_id=41, latitude=33.5169, longitude=0.9128, depth_m=8.0, x_offset=0, y_offset=0),
        PitInfo(name="Aristillus 4", pit_id=42, latitude=33.4981, longitude=0.9447, depth_m=14.0, x_offset=0, y_offset=0),
    ]
    return pits


def extract_tiles_around_pit(
    image_data: np.ndarray,
    x_center: int,
    y_center: int,
    tile_size: int = 256,
    num_tiles: int = 1,
) -> List[Tuple[int, int, np.ndarray]]:
    """Extract tiles around a pit center coordinate.
    
    Returns list of (x_offset, y_offset, tile_array) tuples.
    """
    tiles = []
    h, w = image_data.shape
    
    # For now, just extract one centered tile
    y_start = y_center - tile_size // 2
    y_end = y_start + tile_size
    x_start = x_center - tile_size // 2
    x_end = x_start + tile_size
    
    # Clamp to image bounds
    y_start = max(0, y_start)
    y_end = min(h, y_end)
    x_start = max(0, x_start)
    x_end = min(w, x_end)
    
    # Extract tile
    tile = image_data[y_start:y_end, x_start:x_end]
    
    # Pad if necessary
    if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
        pad_h = max(0, tile_size - tile.shape[0])
        pad_w = max(0, tile_size - tile.shape[1])
        tile = np.pad(tile, ((0, pad_h), (0, pad_w)), mode='reflect')
    
    # Calculate actual center offset (adjusted for clamping)
    actual_x_center = x_start + tile_size // 2
    actual_y_center = y_start + tile_size // 2
    
    tiles.append((actual_x_center, actual_y_center, tile))
    
    return tiles


def calculate_sub_solar_azimuth(
    sub_solar_latitude: float,
    sub_solar_longitude: float,
    center_latitude: float,
    center_longitude: float,
) -> float:
    """Calculate sub-solar azimuth for a specific location.
    
    Azimuth is measured clockwise from North (0° = North, 90° = East, 180° = South, 270° = West).
    """
    # Convert to radians
    lat1, lon1 = np.radians(center_latitude), np.radians(center_longitude)
    lat2, lon2 = np.radians(sub_solar_latitude), np.radians(sub_solar_longitude)
    
    # Calculate bearing (azimuth from location to sub-solar point)
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    bearing = np.arctan2(y, x)
    
    # Convert to degrees and normalize to [0, 360]
    azimuth = np.degrees(bearing) % 360
    
    return float(azimuth)


def main():
    parser = argparse.ArgumentParser(description="Build Aristillus pits dataset")
    parser.add_argument(
        "--nac-path",
        type=Path,
        default=project_root / "data/_scratch" / "M1118880788RC.IMG",
        help="Path to M1118880788RC.IMG"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "datasets" / "aristillus-pits",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="aristillus-pits-beta",
        help="Name for Hugging Face dataset (default: aristillus-pits-beta)"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Tile size in pixels (default: 256)"
    )
    parser.add_argument(
        "--num-tiles",
        type=int,
        default=1,
        help="Number of tiles to extract per pit (default: 1)"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,  # Beta version should be public
        help="Make dataset private on HF Hub (default: False for beta)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    log.info("=" * 60)
    log.info("BUILDING ARISTILLUS PITS DATASET (BETA)")
    log.info("=" * 60)
    log.info(f"NAC path: {args.nac_path}")
    log.info(f"Output dir: {args.output_dir}")
    log.info(f"Dataset name: {args.dataset_name}")
    log.info(f"Tile size: {args.tile_size}")
    log.info("")
    
    # Load NAC image and create projection
    log.info("Loading NAC image and projection...")
    nac_img = read_nac(args.nac_path, geometry=True)
    
    # Create LinearProjection from geometry
    proj = LinearProjection.from_nac_geometry(
        nac_img.geometry, samples=nac_img.samples, lines=nac_img.lines
    )
    
    # Get geometry metadata
    geom = nac_img.geometry
    incidence_angle = float(geom.get('incidence_angle', 35.03))
    sub_solar_latitude = float(geom.get('sub_solar_latitude', 1.1))
    sub_solar_longitude = float(geom.get('sub_solar_longitude', 14.21))
    center_latitude = float(geom.get('center_latitude', 33.82))
    center_longitude = float(geom.get('center_longitude', 0.85))
    resolution_m = float(geom.get('resolution', 1.047))
    
    # Calculate sub-solar azimuth for image center
    sub_solar_azimuth = calculate_sub_solar_azimuth(
        sub_solar_latitude, sub_solar_longitude,
        center_latitude, center_longitude
    )
    
    log.info(f"NAC Image: {nac_img.product_id}")
    log.info(f"  Shape: {nac_img.pixels.shape}")
    log.info(f"  Incidence Angle: {incidence_angle}°")
    log.info(f"  Sub-Solar Azimuth: {sub_solar_azimuth:.2f}°")
    log.info(f"  Resolution: {resolution_m:.3f} m/px")
    log.info("")
    
    # Get known Aristillus pits
    pits = get_aristillus_pits()
    log.info(f"Processing {len(pits)} known pits:")
    
    # Extract tiles for each pit
    all_samples = []
    pit_counter = 0
    
    for pit in pits:
        # Calculate pixel coordinates using LinearProjection
        u, v = proj._solve_uv(pit.longitude, pit.latitude)
        x_offset = int(u * (proj.samples - 1))
        y_offset = int(v * (proj.lines - 1))
        pit.x_offset = x_offset
        pit.y_offset = y_offset
        
        log.info(f"  {pit.name} (ID {pit.pit_id}):")
        log.info(f"    Lat/Lon: {pit.latitude}°, {pit.longitude}°")
        log.info(f"    Pixel: ({x_offset}, {y_offset})")
        log.info(f"    Depth: {pit.depth_m}m")
        
        # Extract tiles using get_image_of_roi
        try:
            tile = get_image_of_roi(
                nac_img.product_id,  # product_id
                pit.latitude,
                pit.longitude,
                width=args.tile_size,
                height=args.tile_size
            )
            tiles = [(x_offset, y_offset, tile)]
        except Exception as e:
            log.warning(f"  Failed to extract tile for {pit.name}: {e}")
            continue
        
        for i, (tile_x, tile_y, tile) in enumerate(tiles):
            pit_counter += 1
            
            # Handle NaN values and normalize
            tile = np.nan_to_num(tile, nan=0.0)
            valid = tile[tile > -32752]
            if len(valid) > 0:
                lo, hi = valid.min(), valid.max()
                if hi - lo > 0:
                    tile_normalized = (tile - lo) / (hi - lo)
                else:
                    tile_normalized = tile
            else:
                tile_normalized = tile
            
            # Create sample
            sample = {
                "id": f"{pit.name.replace(' ', '_')}_{pit.pit_id}_{i}",
                "product_id": nac_img.product_id,
                "pit_id": pit.pit_id,
                "pit_name": pit.name,
                "latitude": pit.latitude,
                "longitude": pit.longitude,
                "depth_m": pit.depth_m,
                "x_offset": tile_x,
                "y_offset": tile_y,
                "tile_x": tile_x,
                "tile_y": tile_y,
                "incidence_angle": incidence_angle,
                "sub_solar_azimuth": sub_solar_azimuth,
                "resolution_m": resolution_m,
                "image": tile_normalized.astype(np.float32).tolist(),
                "is_pit": True,  # Ground truth
                "class": "PIT",
            }
            
            all_samples.append(sample)
    
    log.info(f"Total samples: {len(all_samples)}")
    log.info("")
    
    # Create splits
    config = DatasetConfig(
        dataset_name=args.dataset_name,
        tile_size=args.tile_size,
    )
    
    # With 3 pits (3 samples), use simple splits
    # Train: 2 samples, Val: 1 sample
    splits = {
        "train": all_samples[:2],
        "val": all_samples[2:],
    }
    
    if len(all_samples) > 3:
        # If we have more samples, adjust splits
        splits = {
            "train": all_samples[:int(len(all_samples) * 0.8)],
            "val": all_samples[int(len(all_samples) * 0.8):],
        }
    
    # Create and save datasets
    datasets = {}
    for split_name, split_samples in splits.items():
        if len(split_samples) == 0:
            continue
        
        log.info(f"Building {split_name} split with {len(split_samples)} samples")
        
        # Add split to samples
        for sample in split_samples:
            sample["split"] = split_name
        
        # Create DataFrame and Dataset
        df = pd.DataFrame(split_samples)
        dataset = Dataset.from_pandas(df)
        datasets[split_name] = dataset
        
        # Save to disk
        split_dir = args.output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(split_dir)
        
        # Save as Parquet
        parquet_path = split_dir / f"{split_name}.parquet"
        df.to_parquet(parquet_path)
        log.info(f"Saved {split_name} dataset to {parquet_path}")
    
    dataset_dict = DatasetDict(datasets)
    
    log.info("")
    log.info("=" * 60)
    log.info("DATASET BUILD SUMMARY")
    log.info("=" * 60)
    for split, dataset in dataset_dict.items():
        log.info(f"{split}: {len(dataset)} samples")
    log.info(f"Total: {len(dataset_dict['train']) + len(dataset_dict['val'])} samples")
    log.info(f"Output directory: {args.output_dir.absolute()}")
    log.info("")
    
    # Push to HF if requested
    if args.push:
        hf_token = args.hf_token or os.environ.get('HF_TOKEN')
        if not hf_token:
            log.info("HF_TOKEN not provided. Run: export HF_TOKEN=your_token")
            log.info("Or use --hf-token argument")
        else:
            log.info("Pushing to Hugging Face Hub...")
            try:
                login(token=hf_token)
                api = HfApi()
                
                dataset_id = args.dataset_name
                
                # Push each split
                for split, dataset in dataset_dict.items():
                    log.info(f"Pushing {split} split to Hugging Face Hub...")
                    dataset.push_to_hub(
                        repo_id=dataset_id,
                        split=split,
                        private=args.private,
                    )
                
                log.info(f"Dataset pushed to: https://huggingface.co/datasets/{dataset_id}")
                
            except Exception as e:
                log.error(f"Failed to push to HF: {e}")
                import traceback
                traceback.print_exc()
                return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
