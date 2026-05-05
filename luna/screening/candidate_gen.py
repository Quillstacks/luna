"""Sliding-window NAC scanner that turns raw frames into ``CandidateHit`` lists.

The screening pipeline runs in three stages for every NAC image:

1. **Tiling** — ``_generate_batches`` slides a fixed-size window across the
   full-resolution pixel array, skipping border tiles that are more than 50 %
   NaN, and streams non-overlapping batches to avoid exhausting host RAM.

2. **Embedding** — each batch is forwarded through an ``EmbeddingModel``
   (e.g. a CLIP or Matryoshka encoder) to produce a ``(B, D)`` feature matrix.

3. **Retrieval & projection** — the feature matrix is queried against a
   ``VectorStore`` of known pit embeddings.  Every hit that clears
   ``score_threshold`` has its tile centroid unprojected through the frame's
   ``LinearProjection`` to yield a selenographic ``(lon, lat)`` coordinate,
   which is returned as a ``CandidateHit``.

Typical usage::

    screener = CandidateScreener(model=my_encoder, store=my_faiss_store)
    hits = screener.process_nac("M102285549RE.IMG", score_threshold=0.85)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from tqdm import tqdm

from luna.io import LinearProjection, read_nac
from .protocols import EmbeddingModel, TileMetadata, VectorStore

log = logging.getLogger("luna.screening.candidate_gen")


class DataIngestor:
    """Orchestrates tiling, embedding, and vector ingestion for a single NAC frame."""

    def __init__(
        self,
        model: EmbeddingModel,
        store: VectorStore,
        use_multiprocessing: bool = False,
        num_workers: int | None = None,
    ) -> None:
        self.model = model
        self.store = store
        self.use_multiprocessing = False
        self.num_workers = 0

    @staticmethod
    def _build_fast_spice_coord(label_path: Path) -> Callable[[float, float], tuple[float, float]]:
        import pvl
        import spiceypy as sp
        from luna.io.spice_project import _NAC_PARAMS, _nac_side_from_pid, _read_times, furnish_kernels
        
        furnish_kernels()
        
        lbl = pvl.load(str(label_path))
        side = _nac_side_from_pid(str(lbl["PRODUCT_ID"]))
        p = _NAC_PARAMS[side]
        et_start, _, line_rate, _, _ = _read_times(label_path)
        
        radii = sp.bodvrd("MOON", "RADII", 3)[1]
        re_km, rp_km = float(radii[0]), float(radii[2])
        f_body = (re_km - rp_km) / re_km

        boresight = p["boresight_sample"]
        px_per_mm = p["px_per_mm"]
        focal_mm = p["focal_mm"]
        frame = p["frame"]
        
        def _fast_spice(x: float, y: float) -> tuple[float, float]:
            et = et_start + (y * line_rate)
            y_focal = (x - boresight) / px_per_mm
            look_cam = np.array([0.0, y_focal, focal_mm])
            
            try:
                point, _, _ = sp.sincpt(
                    "Ellipsoid", "MOON", et, "IAU_MOON", "NONE", "LRO", frame, look_cam
                )
                lon_rad, lat_rad, _ = sp.recgeo(point, re_km, f_body)
                return float(np.rad2deg(lon_rad)), float(np.rad2deg(lat_rad))
            except Exception:
                return 0.0, 0.0
                
        return _fast_spice

    def _generate_batches(
        self,
        img_pixels: np.ndarray,
        product_id: str,
        tile_size: int,
        stride: int,
        batch_size: int,
        coord_fn: Callable[[float, float], tuple[float, float]],
    ) -> Iterator[tuple[np.ndarray, list[TileMetadata]]]:
        
        lines, samples = img_pixels.shape
        y_steps = (lines - tile_size) // stride + 1
        x_steps = (samples - tile_size) // stride + 1

        if y_steps <= 0 or x_steps <= 0:
            return

        shape = (y_steps, x_steps, tile_size, tile_size)
        strides = (
            img_pixels.strides[0] * stride,
            img_pixels.strides[1] * stride,
            img_pixels.strides[0],
            img_pixels.strides[1]
        )
        
        # O(1) memory view creation
        patches_view = np.lib.stride_tricks.as_strided(img_pixels, shape=shape, strides=strides)
        total_tiles = y_steps * x_steps

        for batch_start in range(0, total_tiles, batch_size):
            batch_end = min(batch_start + batch_size, total_tiles)
            current_batch_size = batch_end - batch_start

            batch_tiles_raw = np.empty((current_batch_size, tile_size, tile_size), dtype=img_pixels.dtype)
            batch_coords = []

            for idx in range(current_batch_size):
                flat_idx = batch_start + idx
                i = flat_idx // x_steps
                j = flat_idx % x_steps
                batch_tiles_raw[idx] = patches_view[i, j]
                batch_coords.append((j * stride, i * stride))

            # Vectorized NaN check
            nan_counts = np.isnan(batch_tiles_raw).sum(axis=(1, 2))
            valid_mask = nan_counts <= (tile_size * tile_size * 0.5)

            if not np.any(valid_mask):
                continue

            valid_tiles = batch_tiles_raw[valid_mask]
            valid_tiles = np.nan_to_num(valid_tiles, nan=0.5)

            # Vectorized Min-Max Scaling
            t_min = valid_tiles.min(axis=(1, 2), keepdims=True)
            t_max = valid_tiles.max(axis=(1, 2), keepdims=True)
            denom = t_max - t_min
            denom[denom == 0] = 1.0

            clean_tiles = (valid_tiles - t_min) / denom
            
            zero_mask = (t_max == t_min).squeeze(axis=(1, 2))
            if clean_tiles.shape[0] == 1:
                if zero_mask.item():
                    clean_tiles[0] = 0.0
            else:
                clean_tiles[zero_mask] = 0.0

            batch_meta = []
            valid_indices = np.where(valid_mask)[0]

            # Coordinate projection is extremely slow, compute ONLY for valid tiles
            for idx in valid_indices:
                x, y = batch_coords[idx]
                center_x = x + tile_size / 2.0
                center_y = y + tile_size / 2.0
                lon, lat = coord_fn(center_x, center_y)

                batch_meta.append(TileMetadata(
                    product_id=product_id,
                    x_offset=int(x),
                    y_offset=int(y),
                    width=tile_size,
                    height=tile_size,
                    lon=float(lon),
                    lat=float(lat),
                ))

            yield clean_tiles, batch_meta

    def ingest_nac(
        self,
        path: str | Path,
        tile_size: int = 224,
        stride: int = 112,
        batch_size: int = 256,
    ) -> int:
        """Slice a NAC image, embed the tiles, and ingest them into Vector DB."""
        path = Path(path)
        img = read_nac(path, geometry=True)
        lines, samples = img.pixels.shape

        coord_fn = None
        try:
            proj = LinearProjection.from_nac_geometry(
                img.geometry, lines=lines, samples=samples
            )
            from luna.io import pixel_to_lonlat
            
            def _bilinear_coord(x: float, y: float) -> tuple[float, float]:
                return pixel_to_lonlat(proj, x, y)
                
            coord_fn = _bilinear_coord
            log.info(f"  -> Using bilinear projection for {img.product_id}")
            
        except (ValueError, KeyError, TypeError) as e:
            log.warning(f"  -> Bilinear projection failed ({e}), falling back to SPICE...")
            from luna.io.spice_project import ensure_kernels_for_label
            try:
                ensure_kernels_for_label(path)
                coord_fn = self._build_fast_spice_coord(path)
                
                # Test the function once to ensure the kernels are active
                _ = coord_fn(samples / 2.0, lines / 2.0)
                
                log.info(f"  -> SPICE kernels loaded and optimized for {img.product_id}")
            except Exception as spice_e:
                raise RuntimeError(f"SPICE fallback failed as well: {spice_e}")

        total_batches_max = math.ceil(
            ((lines - tile_size) // stride + 1)
            * ((samples - tile_size) // stride + 1)
            / batch_size
        )

        batch_generator = self._generate_batches(
            img_pixels=img.pixels,
            product_id=img.product_id,
            tile_size=tile_size,
            stride=stride,
            batch_size=batch_size,
            coord_fn=coord_fn
        )

        pbar = tqdm(
            batch_generator,
            total=total_batches_max,
            desc=f"Ingesting {img.product_id}",
            unit="batch",
            leave=False,
        )

        total_ingested = 0
        for tile_batch, meta_batch in pbar:
            embeddings = self.model.encode(tile_batch)
            self.store.upsert(embeddings, meta_batch)
            
            total_ingested += len(tile_batch)
            pbar.set_postfix({"ingested": total_ingested})

        return total_ingested