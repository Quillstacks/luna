#!/usr/bin/env python3
"""Build refiner dataset from LUNA pipeline top-K hits for Stage-2 dense decoder training.

This script:
1. Parses top-K hits from LUNA scan log files
2. Extracts corresponding tiles from .IMG files
3. Downloads missing .IMG files via PDS batch downloader
4. Creates a Hugging Face dataset in Parquet format
5. Supports train/test/val splits
6. Optionally pushes to Hugging Face Hub

Usage:
    python scripts/build_refiner_dataset.py \
        --log-file giordano_bruno_scan.log \
        --output-dir data/datasets/lunar-pit-refiner \
        --top-k 10 \
        --dataset-name lunar-pit-refiner-v1 \
        --push

Dataset Structure:
    - Each sample contains:
        - product_id: NAC product ID (e.g., M185219795LC)
        - tile_image: 256x256 tile as numpy array
        - lat: Latitude
        - lon: Longitude
        - x_offset: X offset in original image
        - y_offset: Y offset in original image
        - score: Confidence score from Stage-1
        - class: PIT/BOULDER/etc.
        - rank: Rank in top-K
        - split: train/test/val
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from luna.io.pds_fetch import fetch_nac
from luna.io.nac_reader import read_nac

log = logging.getLogger(__name__)


@dataclass
class CandidateHit:
    """Represents a candidate detection from Stage-1."""
    rank: int
    product_id: str
    class_name: str
    score: float
    lat: float
    lon: float
    x_offset: int
    y_offset: int
    tile_x: int
    tile_y: int
    
    @property
    def img_filename(self) -> str:
        """Get the .IMG filename for this product."""
        return f"{self.product_id}.IMG"


@dataclass
class DatasetConfig:
    """Configuration for dataset building."""
    dataset_name: str = "lunar-pit-refiner"
    description: str = "Lunar pit candidates for Stage-2 refiner training"
    tile_size: int = 256
    splits: dict = None  # {"train": 0.8, "test": 0.1, "val": 0.1}
    
    def __post_init__(self):
        if self.splits is None:
            self.splits = {"train": 0.8, "test": 0.1, "val": 0.1}


class LogParser:
    """Parse LUNA pipeline scan logs to extract top-K hits and their source product IDs."""
    
    # Regex pattern to match table rows in log output
    # Example: "│  01  │      PIT       │     0.9606 │  36.562847° │ 103.440480° │    (3776, 18368)    │      [3648, 18240]      │"
    TABLE_ROW_PATTERN = re.compile(
        r'\s*(\d+)\s*[│┃]\s*(\w+)\s*[│┃]\s*([\d.]+)\s*[│┃]\s*([\d.-]+)°?\s*[│┃]\s*([\d.-]+)°?\s*[│┃]\s*\((\d+),\s*(\d+)\)\s*[│┃]\s*\[(\d+),\s*(\d+)\]'
    )
    
    # Pattern to extract product IDs from "Keeping" lines
    KEEPING_PATTERN = re.compile(r'Keeping \(contains top \d+ hit\): (\w+)\.IMG')
    
    # Pattern to extract all product IDs from coverage plan
    COVERAGE_PATTERN = re.compile(r'Product IDs\s*[│┃]\s*([\w, ]+)')
    
    @classmethod
    def parse_log_file(cls, log_path: Path, top_k: int = 10) -> List[CandidateHit]:
        """Parse a LUNA scan log file to extract candidate hits with product IDs."""
        hits = []
        product_ids = set()
        
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Step 1: Extract all product IDs from the log
        # From "Keeping" lines - these contain the actual source images for top hits
        for match in cls.KEEPING_PATTERN.finditer(content):
            product_ids.add(match.group(1))
        
        # From coverage plan
        coverage_match = cls.COVERAGE_PATTERN.search(content)
        if coverage_match:
            product_ids_str = coverage_match.group(1).replace(' ', '')
            product_ids.update([pid.strip() for pid in product_ids_str.split(',') if pid.strip()])
        
        log.info(f"Found product IDs in log: {product_ids}")
        
        if not product_ids:
            log.warning("No product IDs found in log file. Hits will have empty product_id.")
        
        # Step 2: Parse hit table
        hit_data = []
        for match in cls.TABLE_ROW_PATTERN.finditer(content):
            try:
                groups = match.groups()
                if len(groups) != 9:
                    log.warning(f"Unexpected number of groups: {len(groups)} in row: {match.group(0)}")
                    continue
                
                hit_data.append({
                    'rank': int(groups[0]),
                    'class_name': groups[1].strip(),
                    'score': float(groups[2]),
                    'lat': float(groups[3]),
                    'lon': float(groups[4]),
                    'x_offset': int(groups[5]),
                    'y_offset': int(groups[6]),
                    'tile_x': int(groups[7]),
                    'tile_y': int(groups[8]),
                })
            except (ValueError, IndexError) as e:
                log.warning(f"Failed to parse row: {match.group(0)} - {e}")
                continue
        
        # Sort by rank and take top K
        hit_data.sort(key=lambda h: h['rank'])
        hit_data = hit_data[:top_k]
        
        # Step 3: Assign product IDs to hits
        # This is a heuristic: since we don't have exact mapping from log,
        # we'll try to match hits to product IDs based on proximity
        product_id_list = list(product_ids)
        
        # Simple approach: distribute hits across available product IDs
        # This assumes each product ID contains roughly the same number of hits
        if product_id_list and len(hit_data) > 0:
            for i, hit_info in enumerate(hit_data):
                # Assign product ID by round-robin
                product_id = product_id_list[i % len(product_id_list)]
                
                hit = CandidateHit(
                    rank=hit_info['rank'],
                    product_id=product_id,
                    class_name=hit_info['class_name'],
                    score=hit_info['score'],
                    lat=hit_info['lat'],
                    lon=hit_info['lon'],
                    x_offset=hit_info['x_offset'],
                    y_offset=hit_info['y_offset'],
                    tile_x=hit_info['tile_x'],
                    tile_y=hit_info['tile_y'],
                )
                hits.append(hit)
        else:
            # Fallback: create hits without product IDs
            for hit_info in hit_data:
                hit = CandidateHit(
                    rank=hit_info['rank'],
                    product_id="",
                    class_name=hit_info['class_name'],
                    score=hit_info['score'],
                    lat=hit_info['lat'],
                    lon=hit_info['lon'],
                    x_offset=hit_info['x_offset'],
                    y_offset=hit_info['y_offset'],
                    tile_x=hit_info['tile_x'],
                    tile_y=hit_info['tile_y'],
                )
                hits.append(hit)
        
        # Sort by rank
        hits.sort(key=lambda h: h.rank)
        
        return hits


class NACImageManager:
    """Manage NAC image downloading and caching."""
    
    def __init__(self, cache_dir: Path, max_concurrent: int = 4):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        
    def get_nac_path(self, product_id: str) -> Optional[Path]:
        """Get path to NAC image, downloading if necessary."""
        img_filename = f"{product_id}.IMG"
        img_path = self.cache_dir / img_filename
        
        if img_path.exists():
            log.info(f"Found cached NAC: {img_path}")
            return img_path
        
        # Try to download
        log.info(f"Downloading NAC: {product_id}")
        try:
            fetched_path = fetch_nac(product_id, dest_dir=self.cache_dir)
            return Path(fetched_path)
        except Exception as e:
            log.error(f"Failed to download NAC {product_id}: {e}")
            return None


class TileExtractor:
    """Extract tiles from NAC images."""
    
    def __init__(self, tile_size: int = 256):
        self.tile_size = tile_size
    
    def extract_tile(self, nac_path: Path, x_offset: int, y_offset: int, tile_size: Optional[int] = None) -> Optional[np.ndarray]:
        """Extract a tile from a NAC image."""
        tile_size = tile_size or self.tile_size
        
        try:
            nac_img = read_nac(nac_path)
            image_data = nac_img.pixels.astype(np.float32)
            
            # Handle NaN values
            image_data = np.nan_to_num(image_data, nan=0.0)
            
            # Extract tile centered at (x_offset, y_offset)
            y_start = y_offset - tile_size // 2
            y_end = y_start + tile_size
            x_start = x_offset - tile_size // 2
            x_end = x_start + tile_size
            
            # Clamp to image bounds
            h, w = image_data.shape
            y_start = max(0, y_start)
            y_end = min(h, y_end)
            x_start = max(0, x_start)
            x_end = min(w, x_end)
            
            # Extract and pad if necessary
            tile = image_data[y_start:y_end, x_start:x_end]
            
            # Pad to tile_size if needed
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                pad_h = max(0, tile_size - tile.shape[0])
                pad_w = max(0, tile_size - tile.shape[1])
                tile = np.pad(tile, ((0, pad_h), (0, pad_w)), mode='reflect')
            
            # Normalize to [0, 1]
            valid = tile[tile > -32752]  # Filter out invalid values
            if len(valid) > 0:
                lo, hi = valid.min(), valid.max()
                if hi - lo > 0:
                    tile = (tile - lo) / (hi - lo)
            
            return tile.astype(np.float32)
            
        except Exception as e:
            log.error(f"Failed to extract tile from {nac_path}: {e}")
            return None


class DatasetBuilder:
    """Build Hugging Face dataset from candidate hits."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.nac_manager = NACImageManager(cache_dir=project_root / "data" / "_scratch")
        self.tile_extractor = TileExtractor(tile_size=config.tile_size)
    
    def build_from_hits(
        self,
        hits: List[CandidateHit],
        output_dir: Path,
        split: Optional[str] = None,
    ) -> Dataset:
        """Build dataset from candidate hits."""
        data = []
        
        for i, hit in enumerate(hits):
            log.info(f"Processing hit {i+1}/{len(hits)}: {hit.product_id} (rank {hit.rank})")
            
            # Get NAC image
            nac_path = self.nac_manager.get_nac_path(hit.product_id)
            if nac_path is None:
                log.warning(f"Skipping {hit.product_id}: NAC image not available")
                continue
            
            # Extract tile
            tile = self.tile_extractor.extract_tile(
                nac_path, hit.x_offset, hit.y_offset, self.config.tile_size
            )
            
            if tile is None or tile.shape != (self.config.tile_size, self.config.tile_size):
                log.warning(f"Skipping {hit.product_id}: Invalid tile shape {tile.shape if tile is not None else 'None'}")
                continue
            
            # Add to dataset
            sample = {
                "id": f"{hit.product_id}_{hit.rank}",
                "product_id": hit.product_id,
                "rank": hit.rank,
                "class": hit.class_name,
                "score": hit.score,
                "lat": hit.lat,
                "lon": hit.lon,
                "x_offset": hit.x_offset,
                "y_offset": hit.y_offset,
                "tile_x": hit.tile_x,
                "tile_y": hit.tile_y,
                "image": tile.tolist(),  # Convert to list for JSON serialization
            }
            
            if split:
                sample["split"] = split
            
            data.append(sample)
            
            # Save tile as PNG for debugging
            debug_dir = output_dir / "debug_tiles"
            debug_dir.mkdir(parents=True, exist_ok=True)
            try:
                import matplotlib.pyplot as plt
                plt.imsave(debug_dir / f"{hit.product_id}_{hit.rank}.png", tile, cmap='gray')
                plt.close()
            except ImportError:
                # matplotlib might not be available
                pass
        
        # Create dataset
        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df)
        
        return dataset
    
    def build_and_save(
        self,
        hits: List[CandidateHit],
        output_dir: Path,
        top_k: int,
        split_ratios: Optional[dict] = None,
    ) -> DatasetDict:
        """Build and save complete dataset with train/test/val splits."""
        # Take top K
        hits = hits[:top_k]
        
        # Build splits
        split_ratios = split_ratios or self.config.splits
        
        # Calculate split sizes
        total = len(hits)
        split_sizes = {split: int(ratio * total) for split, ratio in split_ratios.items()}
        
        # Adjust to ensure all samples are used
        remaining = total - sum(split_sizes.values())
        if remaining > 0:
            # Add remaining to train
            split_sizes["train"] += remaining
        
        # Assign splits
        current = 0
        split_hits = {}
        for split, size in split_sizes.items():
            end = current + size
            split_hits[split] = hits[current:end]
            current = end
        
        # Build datasets
        datasets = {}
        for split, split_hit_list in split_hits.items():
            if len(split_hit_list) == 0:
                continue
            
            log.info(f"Building {split} split with {len(split_hit_list)} samples")
            dataset = self.build_from_hits(split_hit_list, output_dir, split=split)
            datasets[split] = dataset
            
            # Save to disk
            split_dir = output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(split_dir)
            
            # Save as Parquet
            df = dataset.to_pandas()
            parquet_path = split_dir / f"{split}.parquet"
            df.to_parquet(parquet_path)
            log.info(f"Saved {split} dataset to {parquet_path}")
        
        return DatasetDict(datasets)
    
    def push_to_hf(
        self,
        dataset_dict: DatasetDict,
        hf_dataset_name: str,
        token: Optional[str] = None,
        private: bool = True,
    ):
        """Push dataset to Hugging Face Hub."""
        if token:
            login(token=token)
        
        api = HfApi()
        
        # Create dataset on HF
        dataset_id = f"{hf_dataset_name}"
        
        # Push each split
        for split, dataset in dataset_dict.items():
            log.info(f"Pushing {split} split to Hugging Face Hub...")
            dataset.push_to_hub(
                repo_id=dataset_id,
                split=split,
                private=private,
            )
        
        log.info(f"Dataset pushed to: https://huggingface.co/datasets/{dataset_id}")
        
        return dataset_id


def main():
    parser = argparse.ArgumentParser(description="Build refiner dataset from LUNA Stage-1 hits")
    parser.add_argument(
        "--log-file",
        type=Path,
        required=True,
        help="Path to LUNA scan log file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "datasets" / "lunar-pit-refiner",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top hits to include (default: 10)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="lunar-pit-refiner-v1",
        help="Name for Hugging Face dataset (default: lunar-pit-refiner-v1)"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Tile size in pixels (default: 256)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
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
        default=True,
        help="Make dataset private on HF Hub (default: True)"
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
    log.info("BUILDING REFINER DATASET FOR STAGE-2")
    log.info("=" * 60)
    log.info(f"Log file: {args.log_file}")
    log.info(f"Output dir: {args.output_dir}")
    log.info(f"Top K: {args.top_k}")
    log.info(f"Tile size: {args.tile_size}")
    log.info(f"Dataset name: {args.dataset_name}")
    log.info(f"Splits: train={args.train_ratio}, test={args.test_ratio}, val={args.val_ratio}")
    log.info("")
    
    # Parse log file
    log.info("Parsing log file...")
    hits = LogParser.parse_log_file(args.log_file, top_k=args.top_k)
    log.info(f"Found {len(hits)} candidate hits in log file")
    
    if len(hits) == 0:
        log.error("No hits found in log file!")
        return 1
    
    # Filter to top K
    hits = hits[:args.top_k]
    log.info(f"Using top {len(hits)} hits")
    
    # Show hits summary
    for hit in hits:
        log.info(f"  Rank {hit.rank}: {hit.product_id} {hit.class_name} score={hit.score:.4f} "
                 f"lat={hit.lat:.6f} lon={hit.lon:.6f} offset=({hit.x_offset}, {hit.y_offset})")
    
    # Configure dataset
    config = DatasetConfig(
        dataset_name=args.dataset_name,
        tile_size=args.tile_size,
        splits={
            "train": args.train_ratio,
            "test": args.test_ratio,
            "val": args.val_ratio,
        }
    )
    
    # Build dataset
    log.info("Building dataset...")
    builder = DatasetBuilder(config)
    
    try:
        dataset_dict = builder.build_and_save(
            hits=hits,
            output_dir=args.output_dir,
            top_k=args.top_k,
            split_ratios=config.splits,
        )
        
        log.info("")
        log.info("=" * 60)
        log.info("DATASET BUILD SUMMARY")
        log.info("=" * 60)
        for split, dataset in dataset_dict.items():
            log.info(f"{split}: {len(dataset)} samples")
        log.info(f"Total: {sum(len(d) for d in dataset_dict.values())} samples")
        log.info(f"Output directory: {args.output_dir.absolute()}")
        log.info("")
        
        # Push to HF if requested
        if args.push:
            hf_token = args.hf_token or os.environ.get('HF_TOKEN')
            if not hf_token:
                log.info("HF_TOKEN not provided. Run: export HF_TOKEN=your_token")
                log.info("Or use --hf-token argument")
                return 0
            
            log.info("Pushing to Hugging Face Hub...")
            builder.push_to_hf(
                dataset_dict=dataset_dict,
                hf_dataset_name=args.dataset_name,
                token=hf_token,
                private=args.private,
            )
        
        return 0
        
    except Exception as e:
        log.error(f"Failed to build dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())