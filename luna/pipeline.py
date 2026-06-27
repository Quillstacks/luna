"""High-level Luna pipeline — the single entry point for scanning the Moon.

Typical usage::

    from luna import LunaPipeline

    pipeline = LunaPipeline.from_pretrained("F1nnSBK/lunar-dinov3-lora")
    hits = pipeline.scan("M1343438359LC", query_dir="data/_scratch/pits/")
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, overload, Union

import numpy as np
import torch

from luna.config import (
    DINO_DIM, FINAL_TOP_K, INDEX_DIR, LROC_VALID_MIN,
    MAX_BATCH_SIZE, MIN_DIST_PX, SCRATCH_DIR, SEARCH_K,
    STRIDE, TILE_SIZE, LunaConfig,
)
from luna.io.pds_fetch import fetch_nac
from luna.screening.candidate_gen import DataIngestor
from luna.screening.pithos import PithosMIDB
from luna.screening.protocols import TileMetadata
from luna.storage.pithos_store import PithosStore
from luna.models import RefinedHit
from luna.metrics import MetricsReport

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateHit:
    rank: int
    product_id: str
    votes: int
    score: float
    lon: float
    lat: float
    x_offset: int
    y_offset: int

    def _repr_html_(self) -> str:
        return (
            f"<table><tr><th colspan='2' style='text-align:left'>CandidateHit</th></tr>"
            f"<tr><td>Rank</td><td>{self.rank}</td></tr>"
            f"<tr><td>Product ID</td><td>{self.product_id}</td></tr>"
            f"<tr><td>Votes</td><td>{self.votes}</td></tr>"
            f"<tr><td>Score</td><td>{self.score:.2f}</td></tr>"
            f"<tr><td>Position</td><td>({self.lon:.4f}, {self.lat:.4f})</td></tr>"
            f"<tr><td>Offset</td><td>({self.x_offset}, {self.y_offset})</td></tr>"
            f"</table>"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LunaPipeline:
    def __init__(self, encoder, device: str, config: LunaConfig | None = None) -> None:
        self._encoder = encoder
        self._device  = device
        self._config  = config or LunaConfig()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        matryoshka_dim: int = DINO_DIM,
        device: str | None = None,
        config: LunaConfig | None = None,
    ) -> LunaPipeline:
        from luna.models.dinov3 import DINOEncoder
        if device is None:
            device = (
                "mps"  if torch.backends.mps.is_available() else
                "cuda" if torch.cuda.is_available()          else
                "cpu"
            )
        batch_size = config.max_batch_size if config else MAX_BATCH_SIZE
        log.info("Loading encoder from %s on %s (Batch Size: %d) …", repo_id, device, batch_size)
        encoder = DINOEncoder(
            lora_dir          = repo_id,
            base_weights_path = repo_id,
            matryoshka_dim    = matryoshka_dim,
            device            = device,
        )
        return cls(encoder=encoder, device=device, config=config)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def _ingest(self, nac_path: Path) -> tuple[PithosStore, list[TileMetadata]]:
        tile_size = self._config.tile_size
        stride = self._config.stride
        max_batch_size = self._config.max_batch_size
        log.info(
            "Slicing and embedding %s (Tile: %d, Stride: %d, Batch Size: %d) …",
            nac_path.name, tile_size, stride, max_batch_size,
        )
        store    = PithosStore()
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

    # ------------------------------------------------------------------
    # Query encoding
    # ------------------------------------------------------------------

    def _encode_queries(self, query_dir: str | Path) -> np.ndarray:
        """Return raw float32 embeddings (N_queries, DINO_DIM).

        Binarization is deferred to ``_search`` so the same float32 vectors
        can be reused across multiple NACs without re-encoding.
        """
        query_dir = Path(query_dir)
        paths     = sorted(query_dir.glob("*.npy"))
        if not paths:
            raise FileNotFoundError(f"No .npy files found in {query_dir}")

        vecs = []
        for p in paths:
            arr   = np.load(p).astype(np.float32)
            valid = arr[arr > LROC_VALID_MIN]
            lo, hi = (valid.min(), valid.max()) if valid.size > 0 else (0.0, 1.0)
            norm  = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
            batch = (np.expand_dims(norm, 0) * 255).astype(np.uint8)
            vecs.append(self._encoder.encode(batch))

        if self._device == "mps":
            torch.mps.empty_cache()
        elif self._device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        stacked = np.vstack(vecs).astype(np.float32)
        log.info("Encoded %d query anchors, shape %s.", len(paths), stacked.shape)
        return stacked

    # ------------------------------------------------------------------
    # Search (Pithos Hamming KNN)
    # ------------------------------------------------------------------

    @staticmethod
    def _search(
        index_prefix: str,
        metadata: list[TileMetadata],
        query_vecs_f32: np.ndarray,
        k: int,
        trace: dict = None,
    ) -> tuple[dict, dict]:
        """
        Execute Pithos batch KNN search on raw float32 queries.

        Returns
        -------
        vote_map  : dict[int, int]   — number of query anchors that hit each tile
        best_dist : dict[int, float] — lowest Hamming distance seen for each tile
                                       (lower = better, unlike FAISS inner-product)
        """
        index_bin  = f"{index_prefix}.bin"
        index_name = Path(index_prefix).stem
        log.info("Loading Pithos index %s …", index_bin)

        db = PithosMIDB()
        db.load_index(index_name, index_bin)
        log.info(
            "Executing Hamming KNN (k=%d) for %d query vectors …",
            k, len(query_vecs_f32),
        )
        t_scan_start = time.perf_counter()
        # Pithos accepts raw float32 queries — no pre-binarization needed
        ids_mat, dists_mat = db.batch_search(index_name, query_vecs_f32, k)
        db.drop_index(index_name)
        if trace is not None:
            trace["p1_pithos_index_scan"] = time.perf_counter() - t_scan_start

        vote_map:  dict[int, int]   = {}
        best_dist: dict[int, float] = {}

        for row in range(ids_mat.shape[0]):
            for idx, dist in zip(ids_mat[row], dists_mat[row]):
                idx = int(idx)
                if idx < 0:
                    continue
                vote_map[idx] = vote_map.get(idx, 0) + 1
                # Lower Hamming distance = closer match
                if idx not in best_dist or dist < best_dist[idx]:
                    best_dist[idx] = float(dist)

        log.info(
            "Search retrieved %d unique candidate tiles via voting.", len(vote_map)
        )
        return vote_map, best_dist

    @staticmethod
    def _nms(
        ranked_ids: list[int],
        vote_map: dict,
        best_dist: dict,
        metadata: list[TileMetadata],
        top_k: int,
        min_dist_px: float,
        trace: dict = None,
    ) -> list[tuple[int, int, float]]:
        log.info(
            "Applying NMS (min_dist: %.1f px, max_targets: %d) on %d tiles …",
            min_dist_px, top_k, len(ranked_ids),
        )
        t_nms_start = time.perf_counter()
        hits:     list[tuple[int, int, float]] = []
        accepted: list[tuple[float, float]]    = []

        for idx in ranked_ids:
            meta = metadata[idx]
            cx   = meta.x_offset + meta.width  / 2.0
            cy   = meta.y_offset + meta.height / 2.0

            if any(
                np.sqrt((cx - ax) ** 2 + (cy - ay) ** 2) < min_dist_px
                for ax, ay in accepted
            ):
                continue

            accepted.append((cx, cy))
            hits.append((idx, vote_map[idx], best_dist[idx]))
            if len(hits) == top_k:
                break

        if trace is not None:
            trace["p1_cpu_nms_filtering"] = time.perf_counter() - t_nms_start
        log.info("NMS complete. Retained %d non-overlapping hits.", len(hits))
        return hits

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @overload
    def scan(
        self,
        product_ids: str | list[str],
        query_dir: str | Path,
        top_k: int | None = ...,
        search_k: int | None = ...,
        min_dist_px: float | None = ...,
        force_reingest: bool = ...,
        trace: dict = ...,
        metrics: Literal[False] = ...,
        on_progress: Callable[[str, int, int], None] | None = ...,
    ) -> list[CandidateHit]: ...

    @overload
    def scan(
        self,
        product_ids: str | list[str],
        query_dir: str | Path,
        top_k: int | None = ...,
        search_k: int | None = ...,
        min_dist_px: float | None = ...,
        force_reingest: bool = ...,
        trace: dict = ...,
        metrics: Literal[True] = ...,
        on_progress: Callable[[str, int, int], None] | None = ...,
    ) -> tuple[list[CandidateHit], MetricsReport]: ...

    def scan(
        self,
        product_ids: str | list[str],
        query_dir: str | Path,
        top_k: int | None = None,
        search_k: int | None = None,
        min_dist_px: float | None = None,
        force_reingest: bool = False,
        trace: dict = None,
        metrics: bool = False,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> list[CandidateHit] | tuple[list[CandidateHit], MetricsReport]:
        # Use config defaults if not provided
        if top_k is None:
            top_k = self._config.final_top_k
        if search_k is None:
            search_k = self._config.search_k
        if min_dist_px is None:
            min_dist_px = self._config.min_dist_px
        
        if isinstance(product_ids, str):
            product_ids = [product_ids]

        # Initialize trace dict if metrics is requested
        if metrics and trace is None:
            trace = {}

        metadata_map: dict[str, list[TileMetadata]] = {}
        total_tiles = 0
        total_scan_start = time.perf_counter()
        index_dir = self._config.index_dir
        scratch_dir = self._config.scratch_dir

        for pid in product_ids:
            nac_path     = scratch_dir / f"{pid}.IMG"
            index_prefix = str(index_dir / f"pithos_{pid}")
            index_exists = Path(f"{index_prefix}.bin").exists()

            if not nac_path.exists():
                log.info("Fetching %s from PDS …", pid)
                nac_path = fetch_nac(pid, dest_dir=scratch_dir)

            if force_reingest or not index_exists:
                log.info("Ingesting %s …", pid)
                t_ingest_start = time.perf_counter()
                store, metadata = self._ingest(nac_path)
                t_ingest_elapsed = time.perf_counter() - t_ingest_start
                
                if trace is not None:
                    trace[f"ingest_s_{pid}"] = t_ingest_elapsed
                    trace[f"ingest_tiles_{pid}"] = len(store._metadata)
                    trace[f"ingest_tiles_per_s_{pid}"] = len(store._metadata) / t_ingest_elapsed if t_ingest_elapsed > 0 else 0
                
                total_tiles += len(store._metadata)
                
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
                log.info("Pithos index for %s already exists, loading metadata …", pid)
                meta_path = f"{index_prefix}_meta.pkl"
                with open(meta_path, "rb") as f:
                    metadata_map[pid] = pickle.load(f)
                
                if trace is not None:
                    bin_path = f"{index_prefix}.bin"
                    if Path(bin_path).exists():
                        trace[f"index_size_bytes_{pid}"] = Path(bin_path).stat().st_size

        t_dino_start = time.perf_counter()
        query_vecs = self._encode_queries(query_dir)
        if trace is not None:
            trace["p1_pytorch_dino_inference"] = time.perf_counter() - t_dino_start

        all_hits: list[CandidateHit] = []

        for pid in product_ids:
            index_prefix = str(index_dir / f"pithos_{pid}")
            metadata     = metadata_map[pid]

            vote_map, best_dist = self._search(
                index_prefix, metadata, query_vecs, k=search_k, trace=trace
            )
            # Sort: lowest Hamming distance first
            ranked   = sorted(
                vote_map.keys(),
                key=lambda i: best_dist[i],
            )
            nms_hits = self._nms(
                ranked, vote_map, best_dist, metadata,
                top_k=top_k, min_dist_px=min_dist_px, trace=trace
            )

            for rank, (idx, votes, score) in enumerate(nms_hits, start=len(all_hits) + 1):
                meta = metadata[idx]
                all_hits.append(CandidateHit(
                    rank=rank, product_id=pid, votes=votes, score=score,
                    lon=meta.lon, lat=meta.lat,
                    x_offset=meta.x_offset, y_offset=meta.y_offset,
                ))

        all_hits.sort(key=lambda h: h.score)
        for i, h in enumerate(all_hits):
            object.__setattr__(h, "rank", i + 1)

        total_scan_elapsed = time.perf_counter() - total_scan_start
        
        if trace is not None:
            trace["total_scan_s"] = total_scan_elapsed
            trace["n_queries"] = len(query_vecs)
            trace["n_candidates_raw"] = len(all_hits)
            
            # Aggregate ingest metrics across all product IDs
            trace["ingest_s"] = sum(trace.get(f"ingest_s_{pid}", 0.0) for pid in product_ids)
            trace["ingest_tiles"] = sum(trace.get(f"ingest_tiles_{pid}", 0) for pid in product_ids)
            trace["index_compile_s"] = sum(trace.get(f"index_compile_s_{pid}", 0.0) for pid in product_ids)
            trace["index_size_bytes"] = sum(trace.get(f"index_size_bytes_{pid}", 0) for pid in product_ids)
            
            # Calculate tiles per second
            total_ingest_time = trace["ingest_s"]
            trace["ingest_tiles_per_s"] = trace["ingest_tiles"] / total_ingest_time if total_ingest_time > 0 else 0.0
        
        log.info("Scan complete: %d hits across %d NACs.", len(all_hits), len(product_ids))
        
        if metrics:
            report = MetricsReport.from_trace(trace)
            return all_hits, report
        
        return all_hits


    @overload
    def refine(
        self,
        hits: list[CandidateHit],
        checkpoint: str | Path | None = ...,
        score_thr: float = ...,
        essa_min_score: float = ...,
        output_dir: str | Path | None = ...,
        skip_preprocess: bool = ...,
        trace: dict = ...,
        metrics: Literal[False] = ...,
        on_progress: Callable[[str, int, int], None] | None = ...,
    ) -> list[RefinedHit]: ...

    @overload
    def refine(
        self,
        hits: list[CandidateHit],
        checkpoint: str | Path | None = ...,
        score_thr: float = ...,
        essa_min_score: float = ...,
        output_dir: str | Path | None = ...,
        skip_preprocess: bool = ...,
        trace: dict = ...,
        metrics: Literal[True] = ...,
        on_progress: Callable[[str, int, int], None] | None = ...,
    ) -> tuple[list[RefinedHit], MetricsReport]: ...

    def refine(
        self,
        hits: list[CandidateHit],
        checkpoint: str | Path | None = None,
        score_thr: float = 0.5,
        essa_min_score: float = 0.0,
        output_dir: str | Path | None = None,
        skip_preprocess: bool = False,
        trace: dict = None,
        metrics: bool = False,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> list[RefinedHit] | tuple[list[RefinedHit], MetricsReport]:
        from luna.models import ESSARefiner
        from luna.config import WEIGHTS_DIR

        # Initialize trace dict if metrics is requested
        if metrics and trace is None:
            trace = {}

        # Disable tqdm if progress callback is provided
        if on_progress is not None:
            os.environ["LUNA_DISABLE_TQDM"] = "1"

        log.info("Initializing ESSARefiner stage on %s …", self._device)
        if on_progress:
            on_progress("Initializing ESSARefiner", 0, 1)
        
        checkpoint = checkpoint or (WEIGHTS_DIR / "essa.pt")
        refiner = ESSARefiner.from_checkpoint(checkpoint, device=self._device)

        if on_progress:
            on_progress("Initializing ESSARefiner", 1, 1)
            on_progress("Running ESSA refinement", 0, len(hits))

        total_refine_start = time.perf_counter()
        
        log.info("Passing %d candidates to ESSA (score_thr=%.2f) …", len(hits), score_thr)
        refined = refiner.refine(
            hits=hits,
            out_dir=output_dir,
            score_thr=score_thr,
            essa_min_score=essa_min_score,
            save_debug_plots=output_dir is not None,
            skip_preprocess=skip_preprocess,
            trace=trace,
        )
        
        if on_progress:
            on_progress("Running ESSA refinement", len(hits), len(hits))
            on_progress("ESSA refinement complete", 1, 1)

        total_refine_elapsed = time.perf_counter() - total_refine_start
        
        if trace is not None:
            trace["total_s"] = trace.get("total_scan_s", 0.0) + total_refine_elapsed
            trace["essa_s"] = total_refine_elapsed
            trace["essa_hits_in"] = len(hits)
            trace["essa_hits_out"] = len(refined)
            trace["essa_refinement_ratio"] = len(refined) / len(hits) if len(hits) > 0 else 0.0
        
        if metrics:
            report = MetricsReport.from_trace(trace)
            return refined, report
        
        return refined