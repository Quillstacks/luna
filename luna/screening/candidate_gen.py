from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Generator

import numpy as np
from tqdm import tqdm

from luna.io import LinearProjection, read_nac, pixel_to_lonlat
from luna.screening.engine import DisruptorEngine, NACTransformer, MappedStripe, push_stripe_to_ring
from luna.config import RING_SIZE
from .protocols import EmbeddingModel, TileMetadata, VectorStore

log = logging.getLogger(__name__)


class ScreenerEngine:
    def __init__(self, max_batch_size: int = 64) -> None:
        self.engine      = DisruptorEngine(size=RING_SIZE, num_consumers=1)
        self.transformer = NACTransformer(self.engine, max_batch_size=max_batch_size)
        self._alive      = True

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

    def get_batch(self) -> tuple[np.ndarray, np.ndarray]:
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

    def __del__(self) -> None:
        if getattr(self, "_alive", False):
            self.shutdown()


class DataIngestor:
    def __init__(
        self,
        model: EmbeddingModel,
        store: VectorStore,
        max_batch_size: int = 64,
    ) -> None:
        self.model    = model
        self.store    = store
        self.screener = ScreenerEngine(max_batch_size=max_batch_size)

    @staticmethod
    def _build_spice_coord_fn(label_path: Path) -> Callable[[float, float], tuple[float, float]]:
        import pvl
        import spiceypy as sp
        from luna.io.spice_project import _NAC_PARAMS, _nac_side_from_pid, _read_times, furnish_kernels

        furnish_kernels()
        lbl  = pvl.load(str(label_path))
        side = _nac_side_from_pid(str(lbl["PRODUCT_ID"]))
        p    = _NAC_PARAMS[side]
        et_start, _, line_rate, _, _ = _read_times(label_path)

        radii        = sp.bodvrd("MOON", "RADII", 3)[1]
        re_km, rp_km = float(radii[0]), float(radii[2])
        f_body       = (re_km - rp_km) / re_km

        def _fn(x: float, y: float) -> tuple[float, float]:
            et       = et_start + y * line_rate
            y_focal  = (x - p["boresight_sample"]) / p["px_per_mm"]
            look_cam = np.array([0.0, y_focal, p["focal_mm"]])
            try:
                point, _, _ = sp.sincpt(
                    "Ellipsoid", "MOON", et, "IAU_MOON", "NONE", "LRO", p["frame"], look_cam
                )
                lon, lat, _ = sp.recgeo(point, re_km, f_body)
                return float(np.rad2deg(lon)), float(np.rad2deg(lat))
            except Exception as e:
                log.debug("SPICE sincpt failed at (%.1f, %.1f): %s", x, y, e)
                return 0.0, 0.0

        return _fn

    @staticmethod
    def _build_coord_fn(
        path: Path, img_geometry, lines: int, samples: int
    ) -> Callable[[float, float], tuple[float, float]]:
        try:
            from luna.io.spice_project import ensure_kernels_for_label
            ensure_kernels_for_label(path)
            coord_fn = DataIngestor._build_spice_coord_fn(path)
            # Test projection at center
            test_lon, test_lat = coord_fn(samples / 2.0, lines / 2.0)
            if test_lon != 0.0 or test_lat != 0.0:
                log.info("Using precise SPICE projection for %s", path.name)
                return coord_fn
        except Exception as e:
            log.warning("SPICE projection failed (%s), falling back to bilinear ...", e)

        from shapely.geometry import shape
        
        # 1. Bilinear Projection
        if img_geometry:
            try:
                proj = LinearProjection.from_nac_geometry(img_geometry, lines=lines, samples=samples)
                log.info("Using bilinear projection for %s", path.name)
                return lambda x, y: pixel_to_lonlat(proj, x, y)
            except (ValueError, KeyError, TypeError) as e:
                log.warning("Bilinear projection failed (%s), falling back to SPICE ...", e)

        # 2. SPICE Projection
        try:
            from luna.io.spice_project import ensure_kernels_for_label
            ensure_kernels_for_label(path)
            coord_fn = DataIngestor._build_spice_coord_fn(path)
            
            for sx in [0, samples // 2, samples]:
                for sy in [0, lines // 2, lines]:
                    test_lon, test_lat = coord_fn(sx, sy)
                    if test_lon != 0.0 or test_lat != 0.0:
                        log.info("SPICE kernels active for %s", path.name)
                        return coord_fn
        except Exception as spice_err:
            log.warning("SPICE initialization failed: %s", spice_err)

        # 3. Robust Envelope Fallback (Catching None/Empty geometries)
        log.warning("All primary projections failed for %s. Applying robust envelope mapping.", path.name)
        try:
            if not img_geometry:
                min_lon, min_lat, max_lon, max_lat = 0.0, -90.0, 0.0, -75.0
            elif isinstance(img_geometry, dict):
                if img_geometry.get("type") is not None:
                    min_lon, min_lat, max_lon, max_lat = shape(img_geometry).bounds
                elif "bounds" in img_geometry:
                    min_lon, min_lat, max_lon, max_lat = img_geometry["bounds"]
                elif "bbox" in img_geometry:
                    min_lon, min_lat, max_lon, max_lat = img_geometry["bbox"]
                else:
                    min_lon, min_lat, max_lon, max_lat = 0.0, -90.0, 0.0, -75.0
            else:
                min_lon, min_lat, max_lon, max_lat = img_geometry.bounds

            min_lon = 0.0 if np.isnan(min_lon) else min_lon
            max_lon = 0.0 if np.isnan(max_lon) else max_lon
            min_lat = -90.0 if np.isnan(min_lat) else min_lat
            max_lat = -90.0 if np.isnan(max_lat) else max_lat
            
            return lambda x, y: (
                min_lon + (x / max(samples, 1)) * (max_lon - min_lon),
                min_lat + (y / max(lines, 1)) * (max_lat - min_lat)
            )
        except Exception as fatal_err:
            raise ValueError(f"Failed to compile coordinate mappings for {path.name}: {fatal_err}")

    def _safe_encode(self, batch: np.ndarray) -> np.ndarray:
        """Encode a batch safely, falling back and halving the batch on GPU/MPS OOM errors."""
        if len(batch) == 0:
            dim = getattr(self.model, "matryoshka_dim", 384)
            return np.empty((0, dim), dtype=np.float32)

        import torch
        try:
            return self.model.encode(batch)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower() or "oom" in str(e).lower()
            if not is_oom:
                raise e

            if len(batch) <= 1:
                log.error("Out of memory encountered even with batch size of 1. Cannot recover.")
                raise e

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

            half_size = len(batch) // 2
            log.warning("OOM detected during DINO encoding, clearing cache and halving batch of size %d to %d ...", len(batch), half_size)

            sub_batches = [batch[:half_size], batch[half_size:]]
            results = [self._safe_encode(sub) for sub in sub_batches]
            return np.concatenate(results, axis=0)

    def _stream(
        self,
        path: Path,
        tile_size: int,
        stride: int,
    ) -> Generator[tuple[np.ndarray, list[TileMetadata]], None, None]:
        """Yield (embeddings, metadata) batches for a single NAC."""
        img            = read_nac(path, geometry=True)
        lines, samples = img.pixels.shape
        stripe         = self.screener.submit_nac(path, width=samples, height=lines)

        # Build coordinate function once per NAC (bilinear → SPICE → envelope fallback).
        # Returns (lon, lat) for a given pixel (x, y) center.
        coord_fn = DataIngestor._build_coord_fn(
            path, getattr(img, "geometry", None), lines, samples
        )

        total_tiles = (
            ((lines   - tile_size) // stride + 1)
            * ((samples - tile_size) // stride + 1)
        )
        fetched = 0

        half = tile_size / 2.0
        with tqdm(total=total_tiles, desc=f"Ingesting {img.product_id}",
                  unit="tile", dynamic_ncols=True, smoothing=0.0) as pbar:
            while fetched < total_tiles:
                batch, offsets = self.screener.get_batch()
                raw_count = len(batch)

                meta_batch = []
                for x, y in offsets:
                    lon, lat = coord_fn(float(x) + half, float(y) + half)
                    meta_batch.append(TileMetadata(
                        product_id = img.product_id,
                        x_offset   = int(x),
                        y_offset   = int(y),
                        width      = tile_size,
                        height     = tile_size,
                        lon        = lon,
                        lat        = lat,
                    ))

                valid_indices = [i for i in range(raw_count) if batch[i].max() > 0]
                if valid_indices:
                    valid_batch = batch[valid_indices]
                    valid_meta  = [meta_batch[i] for i in valid_indices]
                    embeddings  = self._safe_encode(valid_batch)
                    yield embeddings, valid_meta

                fetched   += raw_count
                pbar.update(raw_count)

        del stripe

    def ingest_nac(
        self,
        path: str | Path,
        tile_size: int = 256,
        stride: int    = 192,
    ) -> int:
        total = 0
        for embeddings, meta_batch in self._stream(Path(path), tile_size, stride):
            self.store.upsert(embeddings, meta_batch)
            total += len(embeddings)
        return total