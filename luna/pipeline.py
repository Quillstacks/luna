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
# Spatial NMS
# ---------------------------------------------------------------------------

def apply_lunar_spatial_nms(hits: list, distance_threshold_meters: float = 150.0) -> list:
    """Apply Non-Maximum Suppression using spherical (Haversine) distance on the Moon.

    Supports CandidateHit, RefinedHit, and dictionary inputs. Re-ranks returned hits.
    For raw candidates, lower Hamming distance (score) is better. For refined hits,
    higher similarity/score is better.
    """
    if not hits:
        return []

    import dataclasses

    first_hit = hits[0]
    is_refined = (
        hasattr(first_hit, "dino_similarity") or 
        hasattr(first_hit, "essa_score") or 
        (isinstance(first_hit, dict) and ("dino_similarity" in first_hit or "essa_score" in first_hit))
    )

    if is_refined:
        # Refined hits: higher similarity or essa score is better
        def get_score(h):
            if hasattr(h, "dino_similarity"):
                return max(h.dino_similarity, h.essa_score)
            if isinstance(h, dict):
                return max(h.get("dino_similarity", 0.0), h.get("essa_score", 0.0))
            return 0.0
        reverse = True
    else:
        # Candidate hits: lower Hamming distance is better
        def get_score(h):
            if hasattr(h, "score"):
                return h.score
            if isinstance(h, dict):
                return h.get("score", 9999.0)
            return 9999.0
        reverse = False

    sorted_hits = sorted(hits, key=get_score, reverse=reverse)
    keep = []

    def get_lat_lon(h):
        if isinstance(h, dict):
            return np.radians(h["lat"]), np.radians(h["lon"])
        return np.radians(h.lat), np.radians(h.lon)

    lats = np.array([get_lat_lon(h)[0] for h in sorted_hits])
    lons = np.array([get_lat_lon(h)[1] for h in sorted_hits])

    r_moon = 1737400.0
    num_hits = len(sorted_hits)
    suppressed = np.zeros(num_hits, dtype=bool)

    for i in range(num_hits):
        if suppressed[i]:
            continue

        keep.append(sorted_hits[i])

        lat_i, lon_i = lats[i], lons[i]

        dlat = lats[i+1:] - lat_i
        dlon = lons[i+1:] - lon_i

        a = np.sin(dlat/2)**2 + np.cos(lat_i) * np.cos(lats[i+1:]) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distances = r_moon * c

        overlapping_indices = np.where(distances < distance_threshold_meters)[0]
        suppressed[i + 1 + overlapping_indices] = True

    # Re-rank hits
    re_ranked = []
    for rank, h in enumerate(keep, start=1):
        if hasattr(h, "rank"):
            re_ranked.append(dataclasses.replace(h, rank=rank))
        else:
            re_ranked.append(h)
    return re_ranked


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LunaPipeline:
    def __init__(self, encoder, device: str, config: LunaConfig | None = None) -> None:
        self._encoder = encoder
        self._device  = device
        self._config  = config or LunaConfig()
        self._refiner_type = "essa"  # Default refiner

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
        refiner: str = "essa",
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
        pipeline = cls(encoder=encoder, device=device, config=config)
        pipeline._refiner_type = refiner
        return pipeline

    # ------------------------------------------------------------------
    # Private backward-compatibility helpers (delegating to ingestor/classifier)
    # ------------------------------------------------------------------

    def _ingest(self, nac_path: Path) -> tuple[PithosStore, list[TileMetadata]]:
        from luna.ingestor import LunaIngestor
        ingestor = LunaIngestor(encoder=self._encoder, device=self._device, config=self._config)
        return ingestor._ingest(nac_path)

    def _save_index(self, store: PithosStore, nac_path: Path) -> Path:
        from luna.ingestor import LunaIngestor
        ingestor = LunaIngestor(encoder=self._encoder, device=self._device, config=self._config)
        return ingestor._save_index(store, nac_path)

    def _encode_queries(self, query_dir: str | Path) -> np.ndarray:
        from luna.classifier import get_classifier
        classifier = get_classifier(self._refiner_type, self._encoder, self._device, self._config)
        return classifier._encode_queries(query_dir)

    def _load_and_encode_pit_queries(self, pits_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from luna.classifier import get_classifier
        classifier = get_classifier(self._refiner_type, self._encoder, self._device, self._config)
        return classifier._load_and_encode_pit_queries(pits_dir)

    def _search(
        self,
        index_prefix: str,
        metadata: list[TileMetadata],
        query_vecs_f32: np.ndarray,
        families: np.ndarray | None = None,
        thresholds: np.ndarray | None = None,
        k: int = 1000,
        trace: dict = None,
    ) -> tuple[list[int], np.ndarray]:
        from luna.classifier import get_classifier
        classifier = get_classifier(self._refiner_type, self._encoder, self._device, self._config)
        return classifier._search(
            index_prefix=index_prefix,
            metadata=metadata,
            query_vecs_f32=query_vecs_f32,
            families=families,
            thresholds=thresholds,
            k=k,
            trace=trace
        )

    def _nms(
        self,
        candidate_indices: list[int],
        voting_mask: np.ndarray,
        metadata: list[TileMetadata],
        top_k: int,
        min_dist_px: float,
        trace: dict = None,
    ) -> list[tuple[int, int, float]]:
        from luna.classifier import get_classifier
        classifier = get_classifier(self._refiner_type, self._encoder, self._device, self._config)
        return classifier._nms(
            candidate_indices=candidate_indices,
            voting_mask=voting_mask,
            metadata=metadata,
            top_k=top_k,
            min_dist_px=min_dist_px,
            trace=trace
        )

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
        if isinstance(product_ids, str):
            product_ids = [product_ids]

        if metrics and trace is None:
            trace = {}

        total_scan_start = time.perf_counter()

        # 1. Ingest
        from luna.ingestor import LunaIngestor
        ingestor = LunaIngestor(encoder=self._encoder, device=self._device, config=self._config)
        ingested_pids = ingestor.ingest(
            product_ids=product_ids,
            force_reingest=force_reingest,
            trace=trace,
            on_progress=on_progress
        )

        # 2. Candidate Generation (Search + NMS)
        from luna.classifier import get_classifier
        classifier = get_classifier(self._refiner_type, self._encoder, self._device, self._config)
        candidates = classifier.generate_candidates(
            product_ids=ingested_pids,
            query_dir=query_dir,
            top_k=top_k,
            search_k=search_k,
            min_dist_px=min_dist_px,
            trace=trace
        )

        total_scan_elapsed = time.perf_counter() - total_scan_start
        
        if trace is not None:
            trace["total_scan_s"] = total_scan_elapsed
            trace["n_queries"] = len(classifier._query_queries) if classifier._query_queries is not None else 0
            trace["n_candidates_raw"] = len(candidates)
            
            # Aggregate ingest metrics across all product IDs
            trace["ingest_s"] = sum(trace.get(f"ingest_s_{pid}", 0.0) for pid in product_ids)
            trace["ingest_tiles"] = sum(trace.get(f"ingest_tiles_{pid}", 0) for pid in product_ids)
            trace["index_compile_s"] = sum(trace.get(f"index_compile_s_{pid}", 0.0) for pid in product_ids)
            trace["index_size_bytes"] = sum(trace.get(f"index_size_bytes_{pid}", 0) for pid in product_ids)
            
            # Calculate tiles per second
            total_ingest_time = trace["ingest_s"]
            trace["ingest_tiles_per_s"] = trace["ingest_tiles"] / total_ingest_time if total_ingest_time > 0 else 0.0
        
        log.info("Scan complete: %d hits across %d NACs.", len(candidates), len(product_ids))
        
        if metrics:
            report = MetricsReport.from_trace(trace)
            return candidates, report
        
        return candidates

    def scan_roi(
        self,
        roi_coords: list[tuple[float, float]],
        query_dir: str | Path,
        max_resolution: float = 1.5,
        min_incidence: float = 30.0,
        max_incidence: float = 60.0,
        top_k: int | None = None,
        search_k: int | None = None,
        min_dist_px: float | None = None,
        force_reingest: bool = False,
        trace: dict = None,
        metrics: bool = False,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> list[CandidateHit] | tuple[list[CandidateHit], MetricsReport]:
        """Scan a Region of Interest (ROI) defined by lat/lon coordinate vertices.

        Automatically selects a minimal coverage set of lighting-consistent NAC
        images, runs the Pithos scan, and deduplicates candidate hits spatially on the Moon.
        """
        from luna.io.coverage import select_coverage_nacs

        pids = select_coverage_nacs(
            roi_coords=roi_coords,
            max_resolution=max_resolution,
            min_incidence=min_incidence,
            max_incidence=max_incidence,
        )
        if not pids:
            log.warning("No NAC images selected to cover the given ROI.")
            if metrics:
                return [], MetricsReport.from_trace(trace or {})
            return []

        log.info("ROI coverage solver selected %d images: %s", len(pids), pids)

        return self.scan(
            product_ids=pids,
            query_dir=query_dir,
            top_k=top_k,
            search_k=search_k,
            min_dist_px=min_dist_px,
            force_reingest=force_reingest,
            trace=trace,
            metrics=metrics,
            on_progress=on_progress,
        )

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
        if metrics and trace is None:
            trace = {}

        # Disable tqdm if progress callback is provided
        if on_progress is not None:
            os.environ["LUNA_DISABLE_TQDM"] = "1"

        total_refine_start = time.perf_counter()

        from luna.classifier import get_classifier
        classifier = get_classifier(self._refiner_type, self._encoder, self._device, self._config)
        refined = classifier._refine_candidates(
            hits=hits,
            checkpoint=checkpoint,
            score_thr=score_thr,
            essa_min_score=essa_min_score,
            output_dir=output_dir,
            skip_preprocess=skip_preprocess,
            trace=trace,
            on_progress=on_progress,
        )

        # Apply spatial NMS to remove duplicate confirmed detections in overlap areas
        refined = apply_lunar_spatial_nms(refined, distance_threshold_meters=150.0)

        total_refine_elapsed = time.perf_counter() - total_refine_start
        
        if trace is not None:
            trace["total_s"] = trace.get("total_scan_s", 0.0) + total_refine_elapsed
            refiner_key = "dino" if self._refiner_type == "dino" else "essa"
            trace[f"{refiner_key}_s"] = total_refine_elapsed
            trace[f"{refiner_key}_hits_in"] = len(hits)
            trace[f"{refiner_key}_hits_out"] = len(refined)
            trace[f"{refiner_key}_refinement_ratio"] = len(refined) / len(hits) if len(hits) > 0 else 0.0
        
        if metrics:
            report = MetricsReport.from_trace(trace)
            return refined, report
        
        return refined