"""
End-to-end execution pipeline for Lunar Pit detection.
"""

import os
os.environ["KMP_WARNINGS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import logging
import time
from typing import List, Any

from luna import LunaPipeline

class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("dinov3").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H.O.L.E. Lunar Pit Detection Pipeline")
    parser.add_argument("--nac", type=str, default="M1118880788RC", 
                        help="LROC NAC Product ID to scan")
    parser.add_argument("--score", type=float, default=0.10, 
                        help="ESSA minimum confidence score threshold")
    parser.add_argument("--query-dir", type=str, default="data/_scratch/pits/", 
                        help="Directory containing DINOv3 query anchors")
    parser.add_argument("--out-dir", type=str, default="data/_scratch/dumps/", 
                        help="Directory for output SVGs/PNGs and TIFs")
    parser.add_argument("--skip-preprocess", action="store_true", 
                        help="Skip ISIS preprocessing and reuse existing GeoTIFF")
    return parser.parse_args()


def print_report(refined_hits: List[Any], duration_scan: float, duration_refine: float) -> None:
    total_time = duration_scan + duration_refine
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}" + "=" * 95 + f"{Colors.RESET}")
    print(f"{Colors.BOLD}VERIFIED LUNAR TARGETS ({len(refined_hits)} Skylights/Pits Confirmed){Colors.RESET}")
    print(f"{Colors.CYAN}" + "=" * 95 + f"{Colors.RESET}")

    if not refined_hits:
        print(f"{Colors.YELLOW}No targets met the confidence threshold.{Colors.RESET}")
    
    for h in refined_hits:
        center_x = h.x_offset + 128
        center_y = h.y_offset + 128
        cls_color = Colors.GREEN if h.essa_class.lower() == "pit" else Colors.YELLOW
        
        print(
            f"Rank {h.rank:02d} | "
            f"{cls_color}[{h.essa_class.upper()}]{Colors.RESET} Conf: {h.essa_score:.4f} | "
            f"Lat: {h.lat:10.6f}°, Lon: {h.lon:10.6f}° | "
            f"Center (X: {center_x:5d}, Y: {center_y:5d}) [Top-Left: {h.x_offset}, {h.y_offset}]"
        )

    print(f"{Colors.CYAN}" + "=" * 95 + f"{Colors.RESET}")
    print(f"{Colors.BOLD}PIPELINE PERFORMANCE SUMMARY:{Colors.RESET}")
    print(f"  - DINOv3 Vector Scan Stage : {duration_scan:6.2f} seconds")
    print(f"  - ESSA Refinement Stage    : {duration_refine:6.2f} seconds")
    print(f"  - Total Processing Time    : {total_time:6.2f} seconds")
    print(f"{Colors.CYAN}" + "=" * 95 + f"{Colors.RESET}\n")


def main() -> None:
    args = parse_arguments()
    setup_logging()
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}Initializing Pipeline on target: {args.nac}{Colors.RESET}\n")

    pipeline = LunaPipeline.from_pretrained("F1nnSBK/lunar-dinov3-lora")

    start_scan = time.perf_counter()
    hits = pipeline.scan(args.nac, query_dir=args.query_dir)
    duration_scan = time.perf_counter() - start_scan

    start_refine = time.perf_counter()
    refined = pipeline.refine(
        hits,
        score_thr=args.score,
        essa_min_score=args.score, 
        output_dir=args.out_dir,
        skip_preprocess=args.skip_preprocess
    )
    duration_refine = time.perf_counter() - start_refine

    print_report(refined, duration_scan, duration_refine)


if __name__ == "__main__":
    main()