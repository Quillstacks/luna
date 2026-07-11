from __future__ import annotations

import gc
import logging
import pickle
import time
from pathlib import Path
from typing import Callable

import torch
import numpy as np

from luna.config import (
    INDEX_DIR, SCRATCH_DIR, TILE_SIZE, STRIDE, MAX_BATCH_SIZE, LunaConfig
)
from luna.io.pds_fetch import fetch_nac
from luna.screening.candidate_gen import DataIngestor
from luna.storage.pithos_store import PithosStore
from luna.screening.protocols import TileMetadata

log = logging.getLogger(__name__)


class LunaIngestor:
    def __init__(self, encoder, device: str, config: LunaConfig | None = None) -> None:
        self._encoder = encoder
        self._device = device
        self._config = config or LunaConfig()

    def _ingest(self, nac_path: Path) -> tuple[PithosStore, list[TileMetadata]]:
        tile_size = self._config.tile_size
        stride = self._config.stride
        max_batch_size = self._config.max_batch_size
        log.info(
            "Slicing and embedding %s (Tile: %d, Stride: %d, Batch Size: %d) …",
            nac_path.name, tile_size, stride, max_batch_size,
        )
        pithos_use_fp16 = getattr(self._config, 'pithos_use_fp16', False)
        pithos_use_cuda = getattr(self._config, 'pithos_use_cuda', False)
        store = PithosStore(use_fp16=pithos_use_fp16, use_cuda=pithos_use_cuda)
        ingestor = DataIngestor(model=self._encoder, store=store,
                                max_batch_size=max_batch_size)
        ingestor.ingest_nac(path=nac_path, tile_size=tile_size, stride=stride)

        ingestor.screener.shutdown()
        del ingestor
        if self._device == "mps":
            torch.mps.empty_cache()
        elif self._device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        log.info("Ingestion complete. Generated %d tile embeddings.", len(store._metadata))
        return store, store._metadata

    def _save_index(self, store: PithosStore, nac_path: Path) -> Path:
        index_dir = self._config.index_dir
        index_dir.mkdir(parents=True, exist_ok=True)
        if self._device == "mps":
            torch.mps.empty_cache()
        elif self._device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        prefix = str(index_dir / f"pithos_{nac_path.stem}")
        log.info("Compiling Pithos PLAN index → %s.bin …", prefix)
        store.save_to_disk(prefix)
        return Path(prefix)

    def ingest(
        self,
        product_ids: str | list[str],
        force_reingest: bool = False,
        trace: dict | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> list[str]:
        """Download raw images and compile their Pithos indexes.
        
        Returns a list of successfully ingested product IDs.
        """
        if isinstance(product_ids, str):
            product_ids = [product_ids]

        scratch_dir = self._config.scratch_dir
        index_dir = self._config.index_dir

        # Setup background pre-fetching queue
        import queue
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        download_queue = queue.Queue(maxsize=4)
        
        def download_task(pid):
            nac_path = scratch_dir / f"{pid}.IMG"
            index_prefix = str(index_dir / f"pithos_{pid}")
            index_exists = Path(f"{index_prefix}.bin").exists()
            
            if not force_reingest and index_exists and nac_path.exists():
                download_queue.put((pid, nac_path, None))
                return
            
            if not nac_path.exists():
                try:
                    log.info("Background Downloader: Fetching %s from PDS …", pid)
                    max_bandwidth = getattr(self._config, 'max_bandwidth_mbps', None)
                    fetched_path = fetch_nac(
                        pid, 
                        dest_dir=scratch_dir,
                        max_bandwidth_mbps=max_bandwidth
                    )
                    download_queue.put((pid, fetched_path, None))
                except Exception as e:
                    log.error("Background Downloader: Failed to fetch %s: %s", pid, e)
                    download_queue.put((pid, nac_path, e))
            else:
                download_queue.put((pid, nac_path, None))
        
        def downloader_worker():
            try:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    executor.map(download_task, product_ids)
            except KeyboardInterrupt:
                log.warning("Downloader worker received interrupt signal.")
            except Exception as e:
                log.error("Downloader worker crashed: %s", e)
                    
        downloader_thread = threading.Thread(target=downloader_worker, daemon=True)
        downloader_thread.start()

        ingested_pids = []

        for idx, _ in enumerate(product_ids):
            pid, nac_path, q_err = download_queue.get()
            if q_err is not None:
                log.error("Skipping %s due to background download error: %s", pid, q_err)
                continue

            index_prefix = str(index_dir / f"pithos_{pid}")
            index_exists = Path(f"{index_prefix}.bin").exists()

            if force_reingest or not index_exists:
                log.info("Ingesting %s …", pid)
                if on_progress:
                    on_progress(f"Ingesting {pid}", idx, len(product_ids))

                t_ingest_start = time.perf_counter()
                store, metadata = self._ingest(nac_path)
                t_ingest_elapsed = time.perf_counter() - t_ingest_start
                
                if trace is not None:
                    trace[f"ingest_s_{pid}"] = t_ingest_elapsed
                    trace[f"ingest_tiles_{pid}"] = len(store._metadata)
                    trace[f"ingest_tiles_per_s_{pid}"] = len(store._metadata) / t_ingest_elapsed if t_ingest_elapsed > 0 else 0
                
                t_compile_start = time.perf_counter()
                index_path = self._save_index(store, nac_path)
                t_compile_elapsed = time.perf_counter() - t_compile_start
                
                if trace is not None:
                    trace[f"index_compile_s_{pid}"] = t_compile_elapsed
                    if index_path.exists():
                        trace[f"index_size_bytes_{pid}"] = index_path.stat().st_size
                
                del store
                gc.collect()
            else:
                log.info("Pithos index for %s already exists.", pid)
                if trace is not None:
                    bin_path = f"{index_prefix}.bin"
                    if Path(bin_path).exists():
                        trace[f"index_size_bytes_{pid}"] = Path(bin_path).stat().st_size

            ingested_pids.append(pid)

        return ingested_pids
