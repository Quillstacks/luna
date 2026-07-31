import os
os.environ["KMP_WARNINGS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import logging
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Any, Dict, Tuple

import torch
import psutil
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from luna import LunaPipeline
from luna.config import MAX_BATCH_SIZE, TILE_SIZE, STRIDE, LunaConfig, HF_REPO_ID

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
    parser.add_argument("--nac", type=str, default="M1118880788RC", help="LROC NAC Product ID to scan")
    parser.add_argument("--roi", type=str, default=None, help="Region of Interest coordinates")
    parser.add_argument("-y", "--yes", action="store_true", default=False, help="Skip confirmation prompt")
    parser.add_argument("--score", type=float, default=0.10, help="Minimum confidence score threshold")
    parser.add_argument("--query-dir", type=str, default="data/_scratch/pits/", help="DINOv3 query anchors path")
    parser.add_argument("--out-dir", type=str, default="data/_scratch/dumps/", help="Output directory")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip ISIS preprocessing")
    parser.add_argument("--max-incidence", type=float, default=60.0, help="Maximum solar incidence angle")
    parser.add_argument("--max-resolution", type=float, default=1.5, help="Maximum resolution in m/px")
    parser.add_argument("--force-reingest", action="store_true", help="Force rebuild Pithos index")
    parser.add_argument("--trace", action="store_true", help="Enable deep profiling")
    parser.add_argument("--metrics", action="store_true", help="Print MetricsReport")
    parser.add_argument("--search-k", type=int, default=200, help="KNN neighbors count")
    
    # Supported 'none' to bypass the second-stage refinement completely
    parser.add_argument("--refiner", type=str, default="essa", choices=["essa", "dino", "locate_anything", "stage2", "none"], help="Refiner engine")
    
    parser.add_argument("--pithos-use-fp16", action="store_true", default=False, help="Use FP16 index precision")
    parser.add_argument("--pithos-use-cuda", action="store_true", default=False, help="Use CUDA-optimized Pithos build")
    parser.add_argument("--max-bandwidth", type=float, default=None, help="Download limit in MB/s")
    parser.add_argument("--attention-overlay", action="store_true", help="Generate attention map overlays")
    parser.add_argument("--cleanup", action="store_true", default=False, help="Delete raw files post-run")
    parser.add_argument("--train-svm", action="store_true", default=False, help="Train Stage-2 SVM and exit")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent tracks")
    
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

    iface_info = "N/A"
    try:
        stats = psutil.net_if_stats()
        for name, info in stats.items():
            if name != "lo" and info.isup:
                iface_info = f"{name} ({info.speed} Mbps Link)"
                break
    except Exception:
        pass
            
    return {"device_name": device_name, "vram": vram_str, "iface": iface_info}


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
    title_text = Text("DETECTIONS REPORT: LUNAR MAP SATELLITE SUMMARY", style="bold white")
    console.print("\n")
    console.print(Panel(title_text, expand=False, border_style="cyan", subtitle=f"{len(refined_hits)} database nodes mapped"))

    if refined_hits and refiner_type != "none":
        table = Table(border_style="cyan", show_lines=True)
        table.add_column("Rank", justify="center", style="bold yellow")
        table.add_column("Product ID", justify="center", style="cyan")
        table.add_column("Classification", justify="center")
        table.add_column("Confidence", justify="right")
        table.add_column("Latitude", justify="right")
        table.add_column("Longitude", justify="right")
        table.add_column("Center Pixel (X, Y)", justify="center", style="dim")
        
        for idx, h in enumerate(refined_hits, 1):
            center_x = getattr(h, "x_offset", 0) + 128
            center_y = getattr(h, "y_offset", 0) + 128
            
            if refiner_type == "essa":
                cls_class = getattr(h, "essa_class", "pit")
                cls_text = Text("PIT", style="bold red") if cls_class.lower() == "pit" else Text("SKYLIGHT", style="bold green")
                score = getattr(h, "essa_score", 0.0)
                conf_style = "bold green" if score >= 0.8 else "yellow"
            else:
                sim = getattr(h, "dino_similarity", 0.0)
                cls_text = Text("PIT", style="bold red") if sim >= 0.85 else Text("CANDIDATE", style="bold yellow")
                score = sim
                conf_style = "bold green" if score >= 0.85 else "yellow"
            
            table.add_row(
                f"{idx:02d}",
                str(getattr(h, "product_id", "UNKNOWN")),
                cls_text,
                Text(f"{score:.4f}", style=conf_style),
                f"{getattr(h, 'lat', 0.0):10.6f}°",
                f"{getattr(h, 'lon', 0.0):10.6f}°",
                f"({center_x}, {center_y})"
            )
        console.print(table)
    elif refiner_type == "none":
        console.print(f"[bold green]Database Sync Complete. Mapped {len(refined_hits)} raw vector coordinates into Pithos memory.[/]")
    else:
        console.print("[bold red]No candidate targets stabilized across evaluation thresholds.[/]")
        
    console.print("\n")
    refiner_name = {"essa": "ESSA", "stage2": "Stage-2", "locate_anything": "LocateAnything", "none": "Disabled"}.get(refiner_type, "DINO")
    
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


def execute_parallel_worker(pid: str, args_dict: dict) -> Tuple[str, List[Any], float, float, dict]:
    """Isolated multiprocessing lifecycle execution framework with adaptive recovery."""
    try:
        # Enforce safe hardware routing and kill terminal bar spam inside the worker
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["TQDM_DISABLE"] = "1"  # Completely silences all nested tqdm progress bars
        
        import sys
        import contextlib
        
        config = LunaConfig(
            save_attention_overlay=args_dict["attention_overlay"],
            pithos_use_fp16=args_dict["pithos_use_fp16"],
            pithos_use_cuda=args_dict["pithos_use_cuda"],
            max_bandwidth_mbps=args_dict["max_bandwidth"],
        )
        
        active_refiner = None if args_dict["refiner"] == "none" else args_dict["refiner"]
        trace_data = {} if args_dict["trace"] else None
        
        max_attempts = 3
        all_hits = []
        duration_scan = 0.0
        duration_refine = 0.0

        for attempt in range(max_attempts):
            try:
                start_scan = time.perf_counter()
                
                # Mute sys.stderr during load to swallow the "Using cache found in..." PyTorch Hub spam
                with open(os.devnull, "w") as fnull:
                    with contextlib.redirect_stderr(fnull):
                        pipeline = LunaPipeline.from_pretrained(HF_REPO_ID, refiner=active_refiner, config=config)
                
                # Phase 1 Ingestion & Vector Mapping
                hits = pipeline.scan(
                    [pid],
                    query_dir=args_dict["query_dir"],
                    top_k=150,
                    search_k=args_dict["search_k"],
                    force_reingest=args_dict["force_reingest"],
                    trace=trace_data
                )
                duration_scan = time.perf_counter() - start_scan
                all_hits = hits
                break
                
            except ValueError as e:
                if "reshape" in str(e) and attempt < max_attempts - 1:
                    img_path = config.scratch_dir / f"{pid}.IMG"
                    if img_path.exists():
                        try:
                            img_path.unlink()
                        except Exception:
                            pass
                    time.sleep(2 * (attempt + 1))
                    continue
                raise e

        # Phase 2 Refinement Execution Layer
        if args_dict["refiner"] != "none" and all_hits:
            start_refine = time.perf_counter()
            refined_output = pipeline.refine(
                all_hits,
                score_thr=args_dict["score"],
                essa_min_score=args_dict["score"], 
                output_dir=args_dict["out_dir"],
                skip_preprocess=args_dict["skip_preprocess"],
                trace=trace_data
            )
            duration_refine = time.perf_counter() - start_refine
        else:
            refined_output = all_hits
            duration_refine = 0.0

        # Purge image footprint cache from scratch disk
        if args_dict["cleanup"]:
            keep_pids = {h.product_id for h in refined_output[:10]} if args_dict["refiner"] != "none" else set()
            if pid not in keep_pids:
                img_path = config.scratch_dir / f"{pid}.IMG"
                if img_path.exists():
                    try:
                        img_path.unlink()
                    except Exception:
                        pass
                        
        return pid, refined_output, duration_scan, duration_refine, (trace_data or {})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        # Log to stderr inside worker, which is captured by the main log
        sys.stderr.write(f"\n[WORKER ERROR] Node {pid} failed: {e}\n{tb}\n")
        sys.stderr.flush()
        # Raise standard RuntimeError which can always be pickled cleanly
        raise RuntimeError(f"Worker failed for node {pid}: {e}\n{tb}")


def main() -> None:
    args = parse_arguments()
    setup_logging()
    
    welcome_text = Text("\nLUNA PIPELINE: HIGH-THROUGHPUT PIT DETECTION\n", style="bold white", justify="center")
    console.print(Panel(welcome_text, border_style="cyan"))

    sys_info = get_system_info()
    refiner_name = {"essa": "ESSA", "stage2": "Stage-2", "locate_anything": "LocateAnything", "none": "Disabled"}.get(args.refiner, "DINO")
    
    info_table = Table(title="Runtime & Hardware Parameters", show_header=False, border_style="dim", width=80)
    info_table.add_column("Parameter", style="bold cyan")
    info_table.add_column("Value", style="green")
    info_table.add_row("Hardware Target", sys_info["device_name"])
    info_table.add_row("GPU Memory (VRAM)", sys_info["vram"])
    info_table.add_row("Network Interface", sys_info["iface"])
    info_table.add_row("Resolved Batch Size", f"{MAX_BATCH_SIZE} tiles")
    info_table.add_row("Bandwidth Limit", f"{args.max_bandwidth} MB/s" if args.max_bandwidth else "Unlimited")
    info_table.add_row("Processing Architecture", f"Parallel Threading ({args.workers} workers)" if args.workers > 1 else "Sequential")
    info_table.add_row("Refiner Model", f"{refiner_name}")
    
    if args.roi:
        info_table.add_row("Target ROI", args.roi)
    else:
        info_table.add_row("Target NAC Product", args.nac)
        
    console.print(info_table)
    console.print("\n")

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
                vertices.append(vertices[0])
        except Exception as e:
            console.print(f"[bold red]Error parsing --roi: {e}[/]")
            return

        from luna.io.coverage import select_coverage_nacs
        console.print("[bold yellow]>>> Calculating optimal ROI coverage mosaic (along-track consistent)...[/]")
        pids = select_coverage_nacs(roi_coords=vertices, max_incidence=args.max_incidence, max_resolution=args.max_resolution)
        
        if not pids:
            console.print("[bold red]No NAC images found covering the specified ROI.[/]")
            return

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

    if args.train_svm:
        console.print("[bold green]>>> Phase 2: Training Consolidated SVM Secondary Classifier...[/]")
        return

    args_dict = vars(args)
    all_refined_hits = []
    total_duration_scan = 0.0
    total_duration_refine = 0.0
    combined_trace = {}

    if args.workers > 1 and len(target_pids) > 1:
        console.print(f"[bold gold1]>>> Initiating parallel multi-process engine utilizing {args.workers} concurrent tracks...[/]")
        mp.set_start_method("spawn", force=True)
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(execute_parallel_worker, pid, args_dict): pid 
                for pid in target_pids
            }
            
            for future in as_completed(futures):
                pid = futures[future]
                try:
                    p_pid, p_refined, p_dur_scan, p_dur_refine, p_trace = future.result()
                    all_refined_hits.extend(p_refined)
                    total_duration_scan += p_dur_scan
                    total_duration_refine += p_dur_refine
                    combined_trace.update(p_trace)
                    console.print(f"[bold green][NODE COMPLETE][/bold green] Successfully processed lattice frame {pid}")
                except Exception as ex:
                    console.print(f"[bold red][NODE FAILED][/bold red] Process execution burst on node {pid}: {ex}")
    else:
        console.print("[bold yellow]>>> Initializing Sequential Pipeline Driver...[/]")
        from luna.config import LunaConfig
        config = LunaConfig(
            save_attention_overlay=args.attention_overlay,
            pithos_use_fp16=args.pithos_use_fp16,
            pithos_use_cuda=args.pithos_use_cuda,
            max_bandwidth_mbps=args.max_bandwidth,
        )
        active_refiner = None if args.refiner == "none" else args.refiner
        pipeline = LunaPipeline.from_pretrained(HF_REPO_ID, refiner=active_refiner, config=config)
        
        start_scan = time.perf_counter()
        trace_data = {} if args.trace else None
        
        hits = pipeline.scan(
            target_pids,
            query_dir=args.query_dir,
            top_k=150,
            search_k=args.search_k,
            force_reingest=args.force_reingest,
            trace=trace_data
        )
        total_duration_scan = time.perf_counter() - start_scan
        
        start_refine = time.perf_counter()
        if args.refiner != "none":
            all_refined_hits = pipeline.refine(
                hits,
                score_thr=args.score,
                essa_min_score=args.score, 
                output_dir=args.out_dir,
                skip_preprocess=args.skip_preprocess,
                trace=trace_data
            )
            total_duration_refine = time.perf_counter() - start_refine
        else:
            all_refined_hits = hits
            total_duration_refine = 0.0
            
        combined_trace = trace_data or {}

    print_report(all_refined_hits, total_duration_scan, total_duration_refine, combined_trace, args.refiner)

    try:
        import json
        from pathlib import Path
        json_path = Path("data/_scratch/refined_hits.json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        serialized = []
        for idx, h in enumerate(all_refined_hits, 1):
            serialized.append({
                "rank": idx,
                "product_id": str(getattr(h, "product_id", "UNKNOWN")),
                "lat": float(getattr(h, "lat", 0.0)),
                "lon": float(getattr(h, "lon", 0.0)),
                "x_offset": float(getattr(h, "x_offset", 0.0)),
                "y_offset": float(getattr(h, "y_offset", 0.0)),
                "score": float(getattr(h, "essa_score", getattr(h, "dino_similarity", 0.0))),
                "class": str(getattr(h, "essa_class", "candidate"))
            })
            
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)
        console.print(f"[bold green]Saved consolidated metrics to {json_path}[/]")
    except Exception as e:
        console.print(f"[red]Failed serialization pipeline: {e}[/]")


if __name__ == "__main__":
    main()