import os
os.environ["KMP_WARNINGS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import logging
import time
from typing import List, Any

import torch
import psutil
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from luna import LunaPipeline
from luna.config import MAX_BATCH_SIZE, TILE_SIZE, STRIDE

console = Console()


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
    parser.add_argument("--force-reingest", action="store_true",
                        help="Rebuild the Pithos index even if one already exists")
    parser.add_argument("--trace", action="store_true",
                        help="Enable deep execution profiling with step-by-step timestamps")
    return parser.parse_args()


def get_system_info() -> dict:
    device_name = "CPU Only"
    vram_str = "N/A"
    
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        device_name = f"NVIDIA GPU: {torch.cuda.get_device_name(dev)}"
        vram_bytes = torch.cuda.get_device_properties(dev).total_memory
        vram_str = f"{vram_bytes / (1024 ** 3):.2f} GB"
    elif torch.backends.mps.is_available():
        device_name = "Apple Silicon (MPS)"
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        vram_str = f"{ram_gb:.1f} GB (Unified)"
            
    return {
        "device_name": device_name,
        "vram": vram_str,
    }


def format_duration(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.2f} s"
    elif seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds * 1e6:.2f} µs"


STEP_NAMES = {
    "p1_pytorch_dino_inference": "  └─ PyTorch DINO Inference",
    "p1_pithos_index_scan": "  └─ Pithos MIDB Index Scan",
    "p1_cpu_nms_filtering": "  └─ CPU NMS Filtering",
    "p2_geotiff_loading": "  └─ GeoTIFF Loading",
    "p2_mask_rcnn_inference": "  └─ Mask R-CNN Inference"
}


def print_report(refined_hits: List[Any], duration_scan: float, duration_refine: float, trace_data: dict = None) -> None:
    total_time = duration_scan + duration_refine
    
    # Title Panel
    title_text = Text("DETECTIONS REPORT: VERIFIED LUNAR PIT CANDIDATES", style="bold white")
    console.print("\n")
    console.print(Panel(title_text, expand=False, border_style="cyan", subtitle=f"{len(refined_hits)} targets confirmed"))

    # Results Table
    if refined_hits:
        table = Table(border_style="cyan", show_lines=True)
        table.add_column("Rank", justify="center", style="bold yellow")
        table.add_column("Classification", justify="center")
        table.add_column("Confidence", justify="right")
        table.add_column("Latitude", justify="right")
        table.add_column("Longitude", justify="right")
        table.add_column("Center Pixel (X, Y)", justify="center", style="dim")
        table.add_column("Bounding Box (Top-Left)", justify="center", style="dim")
        
        for h in refined_hits:
            center_x = h.x_offset + 128
            center_y = h.y_offset + 128
            
            # Format classification cell with colors
            if h.essa_class.lower() == "pit":
                cls_text = Text("PIT", style="bold red")
            else:
                cls_text = Text("SKYLIGHT", style="bold green")
                
            conf_style = "bold green" if h.essa_score >= 0.8 else "yellow"
            
            table.add_row(
                f"{h.rank:02d}",
                cls_text,
                Text(f"{h.essa_score:.4f}", style=conf_style),
                f"{h.lat:10.6f}°",
                f"{h.lon:10.6f}°",
                f"({center_x}, {center_y})",
                f"[{h.x_offset}, {h.y_offset}]"
            )
        console.print(table)
    else:
        console.print("[bold red]No candidate targets met the ESSA confidence threshold.[/]")
        
    # Performance Summary Panel
    console.print("\n")
    if trace_data:
        perf_table = Table(title="Pipeline Performance Summary", show_header=True, border_style="dim", width=80)
        perf_table.add_column("Pipeline Stage / Operational Step", style="bold cyan")
        perf_table.add_column("Duration", style="green", justify="right")
        perf_table.add_column("Budget %", style="yellow", justify="right")
        
        perf_table.add_row("Phase 1: DINOv3 Vector Scan Stage (Total)", format_duration(duration_scan), f"{(duration_scan/total_time)*100:5.1f}%")
        
        for step, duration in trace_data.items():
            if step.startswith("p1_"):
                clean_step_name = STEP_NAMES.get(step, f"  └─ {step[3:].replace('_', ' ').title()}")
                perf_table.add_row(clean_step_name, format_duration(duration), f"{(duration/total_time)*100:5.1f}%", style="dim")
                
        perf_table.add_row("Phase 2: ESSA Refinement Stage (Total)", format_duration(duration_refine), f"{(duration_refine/total_time)*100:5.1f}%")
        
        for step, duration in trace_data.items():
            if step.startswith("p2_"):
                clean_step_name = STEP_NAMES.get(step, f"  └─ {step[3:].replace('_', ' ').title()}")
                perf_table.add_row(clean_step_name, format_duration(duration), f"{(duration/total_time)*100:5.1f}%", style="dim")
                
        perf_table.add_section()
        perf_table.add_row("Total Pipeline Execution Time", format_duration(total_time), "100.0%", style="bold gold1")
    else:
        perf_table = Table(title="Pipeline Performance Summary", show_header=False, border_style="dim", width=60)
        perf_table.add_column("Stage", style="bold cyan")
        perf_table.add_column("Duration", style="green", justify="right")
        
        perf_table.add_row("DINOv3 Vector Scan Stage", format_duration(duration_scan))
        perf_table.add_row("ESSA Refinement Stage", format_duration(duration_refine))
        perf_table.add_row("Total Processing Time", format_duration(total_time))
    
    console.print(perf_table)
    console.print("\n")


def main() -> None:
    args = parse_arguments()
    setup_logging()
    
    # Print welcome card
    welcome_text = Text("\nLUNA PIPELINE: DEEP LUNAR PIT DETECTION\n", style="bold white", justify="center")
    console.print(Panel(welcome_text, border_style="cyan"))

    # Print Configuration Dashboard
    sys_info = get_system_info()
    info_table = Table(title="Runtime & Hardware Parameters", show_header=False, border_style="dim", width=80)
    info_table.add_column("Parameter", style="bold cyan")
    info_table.add_column("Value", style="green")

    info_table.add_row("Hardware Target", sys_info["device_name"])
    info_table.add_row("GPU Memory (VRAM)", sys_info["vram"])
    info_table.add_row("Resolved Batch Size", f"{MAX_BATCH_SIZE} tiles")
    info_table.add_row("Slicing Dimensions", f"{TILE_SIZE}x{TILE_SIZE} px (Stride: {STRIDE} px)")
    info_table.add_row("Target NAC Product", args.nac)
    info_table.add_row("Preprocessing Mode", "Skip (Reuse GeoTIFF Cache)" if args.skip_preprocess else "Full (ISIS Pipeline)")
    
    console.print(info_table)
    console.print("\n")

    # Load Model Weights
    console.print("[bold yellow]>>> Initializing Model Weights & LoRA Adapters...[/]")
    pipeline = LunaPipeline.from_pretrained("F1nnSBK/lunar-dinov3-lora")

    # Dictionary to collect granular timing metrics
    trace_data = {} if args.trace else None

    # Phase 1: DINOv3 Vector Scan
    console.print("\n[bold cyan]>>> Phase 1: Running DINOv3 Vector Scan & Pithos Index Matching...[/]")
    start_scan = time.perf_counter()
    hits = pipeline.scan(
        args.nac,
        query_dir=args.query_dir,
        top_k=150,
        search_k=200,
        force_reingest=args.force_reingest,
        trace=trace_data
    )
    duration_scan = time.perf_counter() - start_scan

    # Phase 2: ESSA Refinement
    console.print("\n[bold magenta]>>> Phase 2: Running ESSA Mask R-CNN Refinement Stage...[/]")
    start_refine = time.perf_counter()
    refined = pipeline.refine(
        hits,
        score_thr=args.score,
        essa_min_score=args.score, 
        output_dir=args.out_dir,
        skip_preprocess=args.skip_preprocess,
        trace=trace_data
    )
    duration_refine = time.perf_counter() - start_refine

    # Output report
    print_report(refined, duration_scan, duration_refine, trace_data)


if __name__ == "__main__":
    main()