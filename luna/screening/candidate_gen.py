"""Sliding-window NAC screener backed by a Cython LMAX-Disruptor ring buffer.

Pipeline per NAC frame:
1. **Tiling** — NACTransformer slices the memory-mapped stripe into fixed-size
   windows inside the ring buffer (zero-copy, Cython speed).
2. **Embedding** — Python consumer drains ready batches through an EmbeddingModel.
3. **Retrieval** — embeddings are queried against a VectorStore; hits above
   score_threshold are unprojected to selenographic coordinates and returned
   as CandidateHit objects.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

from luna.io import LinearProjection, read_nac
from luna.io import pixel_to_lonlat
from luna.screening.engine import DisruptorEngine
from luna.screening.engine import NACTransformer
from luna.screening.engine import MappedStripe, push_stripe_to_ring
from luna.utils.normalization import LunaNormalizer
from .protocols import EmbeddingModel, TileMetadata, VectorStore

log = logging.getLogger("luna.screening.candidate_gen")

_RING_SIZE    = 1024
_POLL_INTERVAL_S = 0.001


# ---------------------------------------------------------------------------
# ScreenerEngine — Cython ring buffer management
# ---------------------------------------------------------------------------

class ScreenerEngine:
    def __init__(self, stats_path: Path, max_batch_size: int = 64) -> None:
        self.engine       = DisruptorEngine(size=_RING_SIZE, num_consumers=1)
        self.transformer  = NACTransformer(self.engine, max_batch_size=max_batch_size)
        self.normalizer   = LunaNormalizer(stats_path)
        self._batch_ready = threading.Event()
        self._alive       = True

        self._thread = threading.Thread(
            target=self.transformer.run_forever,
            daemon=True,
            name="luna.disruptor.transformer",
        )
        self._thread.start()

    def submit_nac(self, nac_path: Path, width: int, height: int) -> MappedStripe:
        stripe = MappedStripe(str(nac_path))
        push_stripe_to_ring(self.engine, stripe, width, height, stripe_id=0)
        return stripe

    def get_batch(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
        while not self.transformer.is_batch_ready:
            time.sleep(0)
        batch, offsets = self.transformer.get_current_batch_with_offsets()
        self.transformer.is_batch_ready = 0
        return batch, offsets

    def shutdown(self) -> None:
        if not self._alive:
            return
        self._alive = False

        self.transformer.stop()

        self.transformer.is_batch_ready = 0
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            log.warning("Transformer thread did not exit within timeout.")

    def __del__(self) -> None:
        self.shutdown()


# ---------------------------------------------------------------------------
# DataIngestor — orchestrates projection, embedding, and store upsert
# ---------------------------------------------------------------------------

class DataIngestor:
    """Slice a NAC frame via the Cython ring buffer, embed, and ingest into a VectorStore."""

    def __init__(
        self,
        model: EmbeddingModel,
        store: VectorStore,
        stats_path: Path,
        max_batch_size: int = 256,
    ) -> None:
        self.model   = model
        self.store   = store
        self.screener = ScreenerEngine(stats_path=stats_path, max_batch_size=max_batch_size)

    # ------------------------------------------------------------------
    # Coordinate projection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_spice_coord_fn(
        label_path: Path,
    ) -> Callable[[float, float], tuple[float, float]]:
        import pvl
        import spiceypy as sp
        from luna.io.spice_project import (
            _NAC_PARAMS, _nac_side_from_pid, _read_times, furnish_kernels,
        )

        furnish_kernels()
        lbl  = pvl.load(str(label_path))
        side = _nac_side_from_pid(str(lbl["PRODUCT_ID"]))
        p    = _NAC_PARAMS[side]
        et_start, _, line_rate, _, _ = _read_times(label_path)

        radii       = sp.bodvrd("MOON", "RADII", 3)[1]
        re_km, rp_km = float(radii[0]), float(radii[2])
        f_body      = (re_km - rp_km) / re_km
        boresight   = p["boresight_sample"]
        px_per_mm   = p["px_per_mm"]
        focal_mm    = p["focal_mm"]
        frame       = p["frame"]

        def _fn(x: float, y: float) -> tuple[float, float]:
            et       = et_start + y * line_rate
            y_focal  = (x - boresight) / px_per_mm
            look_cam = np.array([0.0, y_focal, focal_mm])
            try:
                point, _, _ = sp.sincpt(
                    "Ellipsoid", "MOON", et, "IAU_MOON", "NONE", "LRO", frame, look_cam
                )
                lon, lat, _ = sp.recgeo(point, re_km, f_body)
                return float(np.rad2deg(lon)), float(np.rad2deg(lat))
            except Exception as e:
                log.debug("SPICE sincpt failed at (%.1f, %.1f): %s", x, y, e)
                return 0.0, 0.0

        return _fn

    def _build_coord_fn(
        self,
        path: Path,
        img_geometry,
        lines: int,
        samples: int,
    ) -> Callable[[float, float], tuple[float, float]]:
        try:
            proj = LinearProjection.from_nac_geometry(img_geometry, lines=lines, samples=samples)
            log.info("Using bilinear projection for %s", path.name)
            return lambda x, y: pixel_to_lonlat(proj, x, y)
        except (ValueError, KeyError, TypeError) as e:
            log.warning("Bilinear projection failed (%s), falling back to SPICE ...", e)

        from luna.io.spice_project import ensure_kernels_for_label
        ensure_kernels_for_label(path)
        coord_fn = self._build_spice_coord_fn(path)
        coord_fn(samples / 2.0, lines / 2.0)
        log.info("SPICE kernels active for %s", path.name)
        return coord_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_nac(
        self,
        path: str | Path,
        tile_size: int = 256,
        stride: int = 192,
        batch_size: int = 256,
    ) -> int:
        """Tile, embed, and ingest a single NAC frame via multi-threaded pipeline."""
        path = Path(path)
        img  = read_nac(path, geometry=True)
        lines, samples = img.pixels.shape

        coord_fn = self._build_coord_fn(path, img.geometry, lines, samples)
        stripe = self.screener.submit_nac(path, width=samples, height=lines)

        total_tiles   = (
            ((lines   - tile_size) // stride + 1)
            * ((samples - tile_size) // stride + 1)
        )
        
        gpu_queue = queue.Queue(maxsize=50)
        db_queue  = queue.Queue(maxsize=50)

        def fetch_worker():
            fetched = 0
            is_first_batch = True
            
            while fetched < total_tiles:
                # --- TIMING: Cython Fetch ---
                t_start = time.perf_counter()
                batch, offsets = self.screener.get_batch()
                t_fetch = time.perf_counter() - t_start
                
                # --- TIMING: CPU Math ---
                t_start_math = time.perf_counter()
                meta_batch = [
                    TileMetadata(
                        product_id = img.product_id,
                        x_offset   = x,
                        y_offset   = y,
                        width      = tile_size,
                        height     = tile_size,
                        lon        = coord_fn(x + tile_size / 2.0, y + tile_size / 2.0)[0],
                        lat        = coord_fn(x + tile_size / 2.0, y + tile_size / 2.0)[1],
                    )
                    for x, y in offsets
                ]
                t_math = time.perf_counter() - t_start_math

                if is_first_batch:
                    log.info(f"⏱[FETCHER] Cython I/O: {t_fetch:.4f}s | CPU Math: {t_math:.4f}s (Batch Size: {len(batch)})")
                    is_first_batch = False

                gpu_queue.put((batch, meta_batch))
                fetched += len(batch)
                
            gpu_queue.put(None)

        def gpu_worker():
            is_first_batch = True
            while True:
                item = gpu_queue.get()
                if item is None:
                    db_queue.put(None)
                    gpu_queue.task_done()
                    break
                    
                batch, meta_batch = item
                
                # --- TIMING: GPU Inference ---
                t_start = time.perf_counter()
                embeddings = self.model.encode(batch)
                t_gpu = time.perf_counter() - t_start
                
                if is_first_batch:
                    log.info(f"⏱[GPU] MPS Inference: {t_gpu:.4f}s (Batch Size: {len(batch)})")
                    is_first_batch = False

                db_queue.put((embeddings, meta_batch))
                gpu_queue.task_done()

        threads = [
            threading.Thread(target=fetch_worker, daemon=True, name="Luna-Fetcher"),
            threading.Thread(target=gpu_worker, daemon=True, name="Luna-GPU")
        ]
        for t in threads:
            t.start()

        total_ingested = 0
        is_first_batch = True
        
        with tqdm(total=total_tiles, desc=f"Ingesting {img.product_id}", unit="tile", leave=False) as pbar:
            while True:
                item = db_queue.get()
                if item is None:
                    db_queue.task_done()
                    break
                    
                embeddings, meta_batch = item
                
                # --- TIMING: FAISS DB Upsert ---
                t_start = time.perf_counter()
                self.store.upsert(embeddings, meta_batch)
                t_db = time.perf_counter() - t_start
                
                if is_first_batch:
                    log.info(f"⏱[DB] FAISS Upsert: {t_db:.4f}s (Batch Size: {len(embeddings)})")
                    is_first_batch = False

                batch_len = len(meta_batch)
                total_ingested += batch_len
                pbar.update(batch_len)
                db_queue.task_done()

        for t in threads:
            t.join()
            
        del stripe
        return total_ingested