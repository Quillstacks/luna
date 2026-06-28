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

console = Console(width=120)


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
    parser.add_argument("--roi", type=str, default=None,
                        help="Region of Interest coordinates (format: 'lon1,lat1 lon2,lat2 ...')")
    parser.add_argument("-y", "--yes", action="store_true", default=False,
                        help="Skip coverage confirmation prompt and proceed immediately")
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
    parser.add_argument("--metrics", action="store_true",
                        help="Print MetricsReport after the run")
    parser.add_argument("--search-k", type=int, default=200,
                        help="Number of nearest neighbors per query in KNN search")
    parser.add_argument("--refiner", type=str, default="essa", choices=["essa", "dino"],
                        help="Second-stage refiner: 'essa' (Mask R-CNN, accurate) or 'dino' (lightweight)")
    
    # Pithos / Index options
    parser.add_argument("--pithos-use-fp16", action="store_true", default=False,
                        help="Use FP16 precision for Pithos index (faster, less accurate)")
    parser.add_argument("--pithos-use-cuda", action="store_true", default=False,
                        help="Use CUDA-optimized Pithos build (requires CUDA hardware)")
    
    # Output options
    parser.add_argument("--attention-overlay", action="store_true",
                        help="Generate and save attention map overlays for refined candidates")
    parser.add_argument("--cleanup", action="store_true", default=False,
                        help="Delete downloaded raw .IMG files from scratch directory after refinement")
    
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


def print_report(refined_hits: List[Any], duration_scan: float, duration_refine: float, trace_data: dict = None, refiner_type: str = "essa") -> None:
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
            
            # Format classification cell with colors based on refiner type
            if refiner_type == "essa":
                if h.essa_class.lower() == "pit":
                    cls_text = Text("PIT", style="bold red")
                else:
                    cls_text = Text("SKYLIGHT", style="bold green")
                score = h.essa_score
                conf_style = "bold green" if score >= 0.8 else "yellow"
            else:  # dino refiner
                cls_text = Text("PIT", style="bold red") if h.dino_similarity >= 0.85 else Text("CANDIDATE", style="bold yellow")
                score = h.dino_similarity
                conf_style = "bold green" if score >= 0.85 else "yellow"
            
            table.add_row(
                f"{h.rank:02d}",
                cls_text,
                Text(f"{score:.4f}", style=conf_style),
                f"{h.lat:10.6f}°",
                f"{h.lon:10.6f}°",
                f"({center_x}, {center_y})",
                f"[{h.x_offset}, {h.y_offset}]"
            )
        console.print(table)
    else:
        if refiner_type == "essa":
            console.print("[bold red]No candidate targets met the ESSA confidence threshold.[/]")
        else:
            console.print("[bold red]No candidate targets met the DINO similarity threshold.[/]")
        
    # Performance Summary Panel
    console.print("\n")
    refiner_name = "ESSA" if refiner_type == "essa" else "DINO"
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
                
        perf_table.add_row(f"Phase 2: {refiner_name} Refinement Stage (Total)", format_duration(duration_refine), f"{(duration_refine/total_time)*100:5.1f}%")
        
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
        perf_table.add_row(f"{refiner_name} Refinement Stage", format_duration(duration_refine))
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
    refiner_name = "ESSA" if args.refiner == "essa" else "DINO"
    info_table = Table(title="Runtime & Hardware Parameters", show_header=False, border_style="dim", width=80)
    info_table.add_column("Parameter", style="bold cyan")
    info_table.add_column("Value", style="green")

    info_table.add_row("Hardware Target", sys_info["device_name"])
    info_table.add_row("GPU Memory (VRAM)", sys_info["vram"])
    info_table.add_row("Resolved Batch Size", f"{MAX_BATCH_SIZE} tiles")
    info_table.add_row("Slicing Dimensions", f"{TILE_SIZE}x{TILE_SIZE} px (Stride: {STRIDE} px)")
    info_table.add_row("Target NAC Product", args.nac)
    preprocess_text = "Skip (Reuse GeoTIFF Cache)" if args.skip_preprocess else "Full (ISIS Pipeline)"
    info_table.add_row("Preprocessing Mode", preprocess_text)
    info_table.add_row("Refiner", f"{refiner_name} Refiner")
    if args.roi:
        info_table.add_row("Target ROI", args.roi)
    else:
        info_table.add_row("Target NAC Product", args.nac)
    
    console.print(info_table)
    console.print("\n")

    # Plan coverage and verify target_pids
    target_pids = [args.nac]
    if args.roi:
        try:
            vertices = []
            for pair in args.roi.strip().split():
                parts = pair.split(",")
                vertices.append((float(parts[0]), float(parts[1])))
            if len(vertices) < 3:
                raise ValueError("An ROI requires at least 3 vertices.")
            if vertices[0] != vertices[-1]:
                vertices.append(vertices[0])  # Close the polygon
        except Exception as e:
            console.print(f"[bold red]Error parsing --roi: {e}[/]")
            return

        from luna.io.coverage import select_coverage_nacs
        console.print("[bold yellow]>>> Calculating optimal ROI coverage mosaic (along-track consistent)...[/]")
        pids = select_coverage_nacs(roi_coords=vertices)
        
        if not pids:
            console.print("[bold red]No NAC images found covering the specified ROI.[/]")
            return

        # Display coverage plan summary
        plan_table = Table(title="Coverage Plan Summary", border_style="cyan", width=80)
        plan_table.add_column("Property", style="bold cyan")
        plan_table.add_column("Value", style="green")
        
        plan_table.add_row("Total Selected Images", str(len(pids)))
        plan_table.add_row("Estimated Download Size", f"{len(pids) * 529:.1f} MB (compressed)")
        plan_table.add_row("Product IDs", ", ".join(pids))
        
        console.print(plan_table)
        console.print("\n")

        if not args.yes:
            ans = input("Do you want to proceed with downloading and scanning these images? [y/N]: ").strip().lower()
            if ans not in ("y", "yes"):
                console.print("[bold red]Scan aborted by user.[/]")
                return
        target_pids = pids

    # Load Model Weights
    console.print("[bold yellow]>>> Initializing Model Weights & LoRA Adapters...[/]")
    from luna.config import LunaConfig
    config = LunaConfig(
        save_attention_overlay=args.attention_overlay,
        pithos_use_fp16=args.pithos_use_fp16,
        pithos_use_cuda=args.pithos_use_cuda,
    )
    pipeline = LunaPipeline.from_pretrained("F1nnSBK/lunar-dinov3-lora", refiner=args.refiner, config=config)

    # Dictionary to collect granular timing metrics
    trace_data = {} if args.trace else None

    # Phase 1: DINOv3 Vector Scan
    console.print("\n[bold cyan]>>> Phase 1: Running DINOv3 Vector Scan & Pithos Index Matching...[/]")
    start_scan = time.perf_counter()
    
    if args.metrics:
        hits, scan_metrics = pipeline.scan(
            target_pids,
            query_dir=args.query_dir,
            top_k=150,
            search_k=args.search_k,
            force_reingest=args.force_reingest,
            trace=trace_data,
            metrics=True
        )
    else:
        hits = pipeline.scan(
            target_pids,
            query_dir=args.query_dir,
            top_k=150,
            search_k=args.search_k,
            force_reingest=args.force_reingest,
            trace=trace_data
        )
    
    duration_scan = time.perf_counter() - start_scan

    # Phase 2: Refinement
    console.print(f"\n[bold magenta]>>> Phase 2: Running {refiner_name} Refinement Stage...[/]")
    start_refine = time.perf_counter()
    
    if args.metrics:
        refined, metrics = pipeline.refine(
            hits,
            score_thr=args.score,
            essa_min_score=args.score, 
            output_dir=args.out_dir,
            skip_preprocess=args.skip_preprocess,
            trace=trace_data,
            metrics=True
        )
    else:
        refined = pipeline.refine(
            hits,
            score_thr=args.score,
            essa_min_score=args.score, 
            output_dir=args.out_dir,
            skip_preprocess=args.skip_preprocess,
            trace=trace_data
        )
    
    duration_refine = time.perf_counter() - start_refine

    # Print MetricsReport if requested
    if args.metrics:
        console.print("\n")
        console.print(metrics)

    # Output report
    print_report(refined, duration_scan, duration_refine, trace_data, args.refiner)

    # Cleanup temporary LROC raw files if requested (keeping those containing top 10 detections)
    if args.cleanup:
        console.print("\n[bold yellow]>>> Cleaning up temporary LROC NAC .IMG files (sparing top 10 source images)...[/]")
        keep_pids = {h.product_id for h in refined[:10]}
        for pid in target_pids:
            if pid in keep_pids:
                console.print(f"  [green]Keeping (contains top 10 hit):[/] {pid}.IMG")
                continue
            img_path = config.scratch_dir / f"{pid}.IMG"
            if img_path.exists():
                try:
                    img_path.unlink()
                    console.print(f"  Removed: {img_path.name}")
                except Exception as e:
                    console.print(f"  [red]Failed to remove {img_path.name}: {e}[/]")


if __name__ == "__main__":
    main()