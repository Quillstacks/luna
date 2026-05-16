# luna/screening/candidate_gen.py
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
from .protocols import EmbeddingModel, TileMetadata, VectorStore

log = logging.getLogger(__name__)

_RING_SIZE = 1024


class ScreenerEngine:
    def __init__(self, max_batch_size: int = 64) -> None:
        self.engine      = DisruptorEngine(size=_RING_SIZE, num_consumers=1)
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
        if self._thread.is_alive():
            log.warning("Transformer thread did not exit within timeout.")

    def __del__(self) -> None:
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

    def _build_coord_fn(
        self, path: Path, img_geometry, lines: int, samples: int
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

    def _stream(
        self,
        path: Path,
        tile_size: int,
        stride: int,
    ) -> Generator[tuple[np.ndarray, list[TileMetadata]], None, None]:
        """Yield (embeddings, metadata) batches for a single NAC."""
        img            = read_nac(path, geometry=True)
        lines, samples = img.pixels.shape
        coord_fn       = self._build_coord_fn(path, img.geometry, lines, samples)
        stripe         = self.screener.submit_nac(path, width=samples, height=lines)

        total_tiles = (
            ((lines   - tile_size) // stride + 1)
            * ((samples - tile_size) // stride + 1)
        )
        fetched = 0

        with tqdm(total=total_tiles, desc=f"Ingesting {img.product_id}",
                  unit="tile", dynamic_ncols=True, smoothing=0.0) as pbar:
            while fetched < total_tiles:
                batch, offsets = self.screener.get_batch()

                meta_batch = [
                    TileMetadata(
                        product_id = img.product_id,
                        x_offset   = int(x),
                        y_offset   = int(y),
                        width      = tile_size,
                        height     = tile_size,
                        lon        = coord_fn(x + tile_size / 2.0, y + tile_size / 2.0)[0],
                        lat        = coord_fn(x + tile_size / 2.0, y + tile_size / 2.0)[1],
                    )
                    for x, y in offsets
                ]

                embeddings = self.model.encode(batch)
                fetched   += len(embeddings)
                pbar.update(len(embeddings))
                yield embeddings, meta_batch

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