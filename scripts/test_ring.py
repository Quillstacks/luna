#!/usr/bin/env python3
"""
Global Lunar Sweep Orchestrator
--------------------------------
Targeted NAC scan with optional auto-download from PDS.
Pipeline: LMAX Disruptor (C++) → DINOv3 (MPS) → FAISS (RAM → disk).
"""

import os

# Must be set before any OpenMP-linked library is imported
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import logging
import queue
import sys
import threading
import time
from pathlib import Path

from luna.io import fetch_nac
from luna.models.dinov3 import DINOEncoder
from luna.screening import FaissLocalStore
from luna.screening.engine import DisruptorEngine, MappedStripe, NACTransformer, push_stripe_to_ring
from luna.screening.protocols import TileMetadata

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PIDS_TO_SCAN: list[str] = [
    "M157906985RC",
]

IMG_WIDTH    = 5064
HEADER_BYTES = 5064
TILE_SIZE    = 256
STRIDE       = 192
BATCH_SIZE   = 128   # sweet spot for M4 unified memory
DINO_DIM     = 384

log = logging.getLogger("luna.scripts.test_ring")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE WORKERS
# ─────────────────────────────────────────────────────────────────────────────
def db_writer(vector_store: FaissLocalStore, q: queue.Queue) -> None:
    """Consumes (embeddings, metadata) pairs and upserts them into the FAISS buffer."""
    while True:
        item = q.get()
        if item is None:          # poison pill
            q.task_done()
            break
        try:
            embeddings, metadata = item
            vector_store.upsert(embeddings, metadata)
        except Exception as exc:
            log.error("DB upsert failed: %s", exc)
        finally:
            q.task_done()


def gpu_worker(
    encoder: DINOEncoder,
    in_queue: queue.Queue,
    out_queue: queue.Queue,
    total_expected: int,
) -> None:
    """Runs DINOv3 forward passes and forwards (embeddings, metadata) to the DB queue."""
    log.debug("GPU worker ready")
    tiles_processed = 0

    while True:
        item = in_queue.get()
        if item is None:          # poison pill — propagate downstream
            out_queue.put(None)
            in_queue.task_done()
            break
        try:
            batch_array, metadata = item
            embeddings = encoder.encode(batch_array)
            out_queue.put((embeddings, metadata))

            tiles_processed += batch_array.shape[0]
            if tiles_processed % 1024 == 0 or tiles_processed == total_expected:
                log.info("GPU forward pass: %d / %d tiles", tiles_processed, total_expected)
        except Exception as exc:
            log.error("GPU inference failed: %s", exc)
        finally:
            in_queue.task_done()


def batch_fetcher(
    transformer: NACTransformer,
    gpu_queue: queue.Queue,
    sink: list,
    stop: threading.Event,
    product_id: str,
) -> None:
    """Polls the C++ ring buffer, assembles tile metadata, and feeds the GPU queue."""
    log.debug("Batch fetcher listening to C++ ring buffer")
    tiles_processed  = 0
    tiles_per_row    = (IMG_WIDTH - TILE_SIZE) // STRIDE + 1

    while not stop.is_set():
        if not transformer.is_batch_ready:
            time.sleep(0.001)
            continue

        batch_array        = transformer.get_current_batch().copy()
        batch_size         = batch_array.shape[0]
        transformer.is_batch_ready = 0
        sink.append(batch_array)

        metadata = [
            TileMetadata(
                product_id=product_id,
                x_offset=((tiles_processed + i) % tiles_per_row) * STRIDE,
                y_offset=((tiles_processed + i) // tiles_per_row) * STRIDE,
                width=TILE_SIZE,
                height=TILE_SIZE,
            )
            for i in range(batch_size)
        ]

        gpu_queue.put((batch_array, metadata))
        tiles_processed += batch_size


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-NAC ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
def process_nac(
    img_path: Path,
    product_id: str,
    encoder: DINOEncoder,
    output_dir: Path,
) -> bool:
    """Run one NAC strip through the full LMAX → GPU → FAISS pipeline."""
    log.info("Processing %s", product_id)

    file_size      = os.path.getsize(img_path)
    height         = (file_size - HEADER_BYTES) // (IMG_WIDTH * 2)
    tiles_x        = (IMG_WIDTH - TILE_SIZE) // STRIDE + 1
    tiles_y        = (height    - TILE_SIZE) // STRIDE + 1
    expected_tiles = tiles_x * tiles_y

    if expected_tiles <= 0:
        log.warning("%s: image too small or corrupt — skipping", product_id)
        return False

    log.info("%s: %dx%d px | %d tiles expected", product_id, IMG_WIDTH, height, expected_tiles)

    # Initialise pipeline components
    vector_store = FaissLocalStore(vector_dim=DINO_DIM)
    gpu_queue    = queue.Queue()
    db_queue     = queue.Queue()
    engine       = DisruptorEngine(size=8, num_consumers=1)
    stripe       = MappedStripe(str(img_path))
    transformer  = NACTransformer(engine, max_batch_size=BATCH_SIZE, consumer_id=0)
    all_tiles: list[object] = []
    stop_event   = threading.Event()

    threads = [
        threading.Thread(target=transformer.run_forever, daemon=True),
        threading.Thread(target=batch_fetcher, args=(transformer, gpu_queue, all_tiles, stop_event, product_id), daemon=True),
        threading.Thread(target=gpu_worker,    args=(encoder, gpu_queue, db_queue, expected_tiles), daemon=True),
        threading.Thread(target=db_writer,     args=(vector_store, db_queue), daemon=True),
    ]
    for t in threads:
        t.start()

    t0 = time.time()
    push_stripe_to_ring(engine, stripe, width=IMG_WIDTH, height=height, stripe_id=1337)

    # Drain — wait until all tiles have been collected
    while True:
        collected = sum(b.shape[0] for b in all_tiles)
        if collected >= expected_tiles:
            break
        if (expected_tiles - collected) < BATCH_SIZE:
            transformer.trigger_final_flush()
            time.sleep(0.05)
        time.sleep(0.01)

    log.info("%s: LMAX feed done in %.4fs — waiting for GPU/DB", product_id, time.time() - t0)

    # Ordered shutdown
    stop_event.set()
    threads[1].join()           # fetcher done
    gpu_queue.put(None)         # poison pill → GPU worker
    gpu_queue.join()
    db_queue.join()

    log.info("%s: total pipeline runtime %.4fs", product_id, time.time() - t0)
    transformer.stop()

    index_prefix = str(output_dir / f"faiss_{product_id}")
    vector_store.save_to_disk(index_prefix)
    log.info("%s: index saved to %s", product_id, output_dir)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-speed sweep for lunar pit detection.")
    parser.add_argument("--data-dir",     type=Path, help="Local directory to scan instead of PIDS_TO_SCAN.")
    parser.add_argument("--scratch-dir",  type=Path, default=Path("data/_scratch"))
    parser.add_argument("--output-dir",   type=Path, default=Path("data/indices"))
    parser.add_argument("--keep",         action="store_true", help="Retain downloaded .IMG files after processing.")
    parser.add_argument("--adapter-file", type=str,  default="luna/models/dinov3/adapter_model.safetensors")
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = _parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading DINOv3 + LoRA onto MPS")
    try:
        encoder = DINOEncoder(
            lora_dir=str(Path(args.adapter_file).parent),
            matryoshka_dim=DINO_DIM,
            device="mps",
        )
    except Exception as exc:
        log.critical("Model initialisation failed: %s", exc)
        sys.exit(1)

    if PIDS_TO_SCAN:
        targets: list = PIDS_TO_SCAN
    elif args.data_dir:
        targets = list(args.data_dir.glob("**/*.IMG"))
    else:
        targets = []

    if not targets:
        log.error("No targets found — set PIDS_TO_SCAN or pass --data-dir")
        sys.exit(1)

    log.info("Starting sweep — %d NAC(s) queued", len(targets))

    for i, target in enumerate(targets, start=1):
        img_path      = None
        is_downloaded = False
        product_id    = str(target)   # fallback for error logging

        try:
            if isinstance(target, str):
                product_id    = target
                log.info("[%d/%d] Downloading %s from PDS", i, len(targets), product_id)
                img_path      = fetch_nac(product_id, dest_dir=args.scratch_dir)
                is_downloaded = True
            else:
                img_path   = target
                product_id = img_path.stem

            process_nac(img_path, product_id, encoder, args.output_dir)

        except Exception as exc:
            log.error("[%d/%d] %s failed — skipping: %s", i, len(targets), product_id, exc)

        finally:
            if is_downloaded and img_path and img_path.exists():
                log.info("%s: file retained in %s/", product_id, args.scratch_dir.name)

    log.info("Sweep complete")


if __name__ == "__main__":
    main()