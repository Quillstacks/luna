from __future__ import annotations

import gc
import logging
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch

from luna.config import (
    DINO_DIM, FINAL_TOP_K, INDEX_DIR, LROC_VALID_MIN,
    MAX_BATCH_SIZE, MIN_DIST_PX, SCRATCH_DIR, SEARCH_K,
    STRIDE, TILE_SIZE, LunaConfig
)
from luna.screening.pithos import PithosMIDB
from luna.screening.protocols import TileMetadata
from luna.models.essa import RefinedHit
from luna.pipeline import CandidateHit, apply_lunar_spatial_nms

log = logging.getLogger(__name__)


class LunaClassifier(ABC):
    def __init__(self, encoder, device: str, config: LunaConfig | None = None) -> None:
        self._encoder = encoder
        self._device = device
        self._config = config or LunaConfig()
        self._query_queries = None
        self._query_families = None
        self._query_thresholds = None

    def _encode_queries(self, query_dir: str | Path) -> np.ndarray:
        query_path = Path(query_dir)
        
        # 1. If query_dir points directly to a precompiled file
        if query_path.is_file():
            arr = np.load(query_path).astype(np.float32)
            if arr.ndim == 2 and arr.shape[1] == 384:
                log.info("Loaded query database from file %s, shape %s.", query_path.name, arr.shape)
                return arr
                
        # 2. If query_dir is a directory, check for precompiled files first
        if query_path.is_dir():
            ref_file = query_path / "dino_reference.npy"
            if not ref_file.exists():
                from luna.config import DATA_DIR
                ref_file = DATA_DIR / "dino_reference.npy"
                
            if ref_file.exists():
                arr = np.load(ref_file).astype(np.float32)
                if arr.ndim == 2 and arr.shape[1] == 384:
                    log.info("Loaded precompiled query database from %s, shape %s.", ref_file, arr.shape)
                    return arr

        # 3. Fallback: load and encode individual query npy tiles
        paths = sorted(query_path.glob("*.npy")) if query_path.is_dir() else [query_path]
        if not paths:
            raise FileNotFoundError(f"No .npy files found in {query_dir}")

        vecs = []
        for p in paths:
            arr = np.load(p).astype(np.float32)
            if arr.ndim == 2 and arr.shape[1] == 384:
                vecs.append(arr)
            elif arr.ndim == 1 and arr.shape[0] == 384:
                vecs.append(np.expand_dims(arr, 0))
            else:
                valid = arr[arr > LROC_VALID_MIN]
                lo, hi = (valid.min(), valid.max()) if valid.size > 0 else (0.0, 1.0)
                norm = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
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

    def _load_and_encode_pit_queries(self, pits_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pits_path = Path(pits_dir)
        pit_files = sorted(pits_path.glob("*.npy"))
        
        if not pit_files:
            raise FileNotFoundError(f"No pit .npy files found in {pits_dir}")
        
        log.info("Loading %d pit patches from %s...", len(pit_files), pits_dir)
        
        pit_images = []
        from PIL import Image
        for f in pit_files:
            img = np.load(f)
            if img.ndim == 2:
                if img.shape != (256, 256):
                    img = np.array(Image.fromarray(img).resize((256, 256), Image.Resampling.BILINEAR))
                pit_images.append(img)
        
        if not pit_images:
            raise ValueError("No valid pit images found")
        
        normalized = []
        for img in pit_images:
            valid = img[img > LROC_VALID_MIN]
            lo, hi = (valid.min(), valid.max()) if valid.size > 0 else (0.0, 1.0)
            norm = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
            normalized.append(norm)
        
        batch_size = self._config.max_batch_size if self._config else 64
        all_embeddings = []
        
        for i in range(0, len(normalized), batch_size):
            batch_imgs = normalized[i:i + batch_size]
            batch_input = np.stack([(img * 255).astype(np.uint8) for img in batch_imgs])
            embeddings = self._encoder.encode(batch_input)
            all_embeddings.append(embeddings)
        
        queries = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        num_queries = len(queries)
        
        import hashlib
        families = np.zeros(num_queries, dtype=np.int32)
        for i, f in enumerate(pit_files):
            h = int(hashlib.md5(f.stem.encode('utf-8')).hexdigest(), 16)
            family_id = h % 8
            families[i] = family_id
        
        base_threshold = 42
        thresholds = np.full(num_queries, base_threshold, dtype=np.int32)
        
        family_counts = np.bincount(families, minlength=8)
        for i in range(num_queries):
            family_id = families[i]
            count_factor = max(1, 10 - family_counts[family_id] // 4)
            thresholds[i] = min(63, max(16, base_threshold - count_factor))
        
        if self._device == "mps":
            torch.mps.empty_cache()
        elif self._device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        log.info(
            "Encoded %d pit queries: shape %s, families: %s, threshold range: [%d, %d]",
            num_queries, queries.shape, np.unique(families, return_counts=True),
            thresholds.min(), thresholds.max()
        )
        
        return queries, families, thresholds

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
        index_bin = f"{index_prefix}.bin"
        index_name = Path(index_prefix).stem
        total_records = len(metadata)
        
        log.info("Loading Pithos index %s …", index_bin)
        
        db = PithosMIDB()
        db.load_index(index_name, index_bin)
        
        if families is not None and thresholds is not None:
            log.info(
                "Executing Multi-Family Resonant Voting for %d queries across %d families…",
                len(query_vecs_f32), len(np.unique(families))
            )
            t_scan_start = time.perf_counter()
            
            voting_mask = np.zeros(total_records, dtype=np.uint8)
            
            resonant_count = db.query_planetary_grid(
                index_name=index_name,
                queries=query_vecs_f32,
                families=families.astype(np.int32),
                thresholds=thresholds.astype(np.int32),
                voting_mask=voting_mask,
            )
            
            if trace is not None:
                trace["p1_pithos_resonant_voting"] = time.perf_counter() - t_scan_start
            
            db.drop_index(index_name)
            candidate_indices = np.where(voting_mask != 0)[0].tolist()
            
            log.info("Resonant voting retrieved %d candidate tiles.", resonant_count)
            return candidate_indices, voting_mask
        else:
            log.info("Executing Hamming KNN (k=%d) for %d query vectors …", k, len(query_vecs_f32))
            t_scan_start = time.perf_counter()
            ids_mat, dists_mat = db.batch_search(index_name, query_vecs_f32, k)
            db.drop_index(index_name)
            
            if trace is not None:
                trace["p1_pithos_index_scan"] = time.perf_counter() - t_scan_start

            vote_map: dict[int, int] = {}
            best_dist: dict[int, float] = {}

            for row in range(ids_mat.shape[0]):
                for idx, dist in zip(ids_mat[row], dists_mat[row]):
                    idx = int(idx)
                    if idx < 0:
                        continue
                    vote_map[idx] = vote_map.get(idx, 0) + 1
                    if idx not in best_dist or dist < best_dist[idx]:
                        best_dist[idx] = float(dist)

            log.info("Search retrieved %d unique candidate tiles via voting.", len(vote_map))
            voting_mask = np.zeros(total_records, dtype=np.uint8)
            for idx in vote_map:
                voting_mask[idx] = 1
            return list(vote_map.keys()), voting_mask

    @staticmethod
    def _nms(
        candidate_indices: list[int],
        voting_mask: np.ndarray,
        metadata: list[TileMetadata],
        top_k: int,
        min_dist_px: float,
        trace: dict = None,
    ) -> list[tuple[int, int, float]]:
        log.info("Applying NMS (min_dist: %.1f px, max_targets: %d) on %d candidate tiles …",
                 min_dist_px, top_k, len(candidate_indices))
        t_nms_start = time.perf_counter()
        hits: list[tuple[int, int, float]] = []
        accepted: list[tuple[float, float]] = []

        sorted_indices = sorted(candidate_indices, key=lambda i: -voting_mask[i])

        for idx in sorted_indices:
            meta = metadata[idx]
            cx = meta.x_offset + meta.width / 2.0
            cy = meta.y_offset + meta.height / 2.0

            if any(
                np.sqrt((cx - ax) ** 2 + (cy - ay) ** 2) < min_dist_px
                for ax, ay in accepted
            ):
                continue

            accepted.append((cx, cy))
            hits.append((idx, int(voting_mask[idx]), float(voting_mask[idx])))
            if len(hits) == top_k:
                break

        if trace is not None:
            trace["p1_cpu_nms_filtering"] = time.perf_counter() - t_nms_start
        log.info("NMS complete. Retained %d non-overlapping hits.", len(hits))
        return hits

    @abstractmethod
    def _refine_candidates(
        self,
        hits: list[CandidateHit],
        checkpoint: str | Path | None = None,
        score_thr: float = 0.5,
        essa_min_score: float = 0.0,
        output_dir: str | Path | None = None,
        skip_preprocess: bool = False,
        trace: dict = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> list[RefinedHit]:
        """Concrete refinement implementation implemented by subclasses."""
        pass

    def generate_candidates(
        self,
        product_ids: list[str],
        query_dir: str | Path,
        top_k: int | None = None,
        search_k: int | None = None,
        min_dist_px: float | None = None,
        trace: dict = None,
    ) -> list[CandidateHit]:
        """Perform vector search on ingested Pithos indexes and return CandidateHits."""
        if top_k is None:
            top_k = self._config.final_top_k
        if search_k is None:
            search_k = self._config.search_k
        if min_dist_px is None:
            min_dist_px = self._config.min_dist_px

        index_dir = self._config.index_dir
        scratch_dir = self._config.scratch_dir

        t_dino_start = time.perf_counter()
        
        query_path = Path(query_dir)
        use_resonant_voting = False
        
        if query_path.is_dir():
            pit_files = list(query_path.glob("*.npy"))
            if len(pit_files) > 10:
                use_resonant_voting = True
                log.info("Detected pit database with %d entries. Activating Multi-Family Resonant Voting.", len(pit_files))
        
        if use_resonant_voting and self._query_queries is None:
            self._query_queries, self._query_families, self._query_thresholds = self._load_and_encode_pit_queries(query_dir)
            query_vecs = self._query_queries
        else:
            if self._query_queries is not None and use_resonant_voting:
                query_vecs = self._query_queries
            else:
                query_vecs = self._encode_queries(query_dir)
        
        if trace is not None:
            trace["p1_pytorch_dino_inference"] = time.perf_counter() - t_dino_start

        all_candidate_hits: list[CandidateHit] = []

        for pid in product_ids:
            index_prefix = str(index_dir / f"pithos_{pid}")
            meta_path = f"{index_prefix}_meta.pkl"
            
            if not Path(meta_path).exists() or not Path(f"{index_prefix}.bin").exists():
                raise FileNotFoundError(f"Database index files not found for {pid}. Please run ingest first.")

            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)

            # 1. Vector Search
            if use_resonant_voting and self._query_families is not None:
                candidate_indices, voting_mask = self._search(
                    index_prefix, metadata, query_vecs,
                    families=self._query_families,
                    thresholds=self._query_thresholds,
                    k=search_k, trace=trace
                )
            else:
                candidate_indices, voting_mask = self._search(
                    index_prefix, metadata, query_vecs, k=search_k, trace=trace
                )

            # 2. NMS
            nms_hits = self._nms(
                candidate_indices, voting_mask, metadata,
                top_k=top_k, min_dist_px=min_dist_px, trace=trace
            )

            # 3. Coordinate Resolution
            coord_fn = None
            nac_path = scratch_dir / f"{pid}.IMG"
            for rank, (idx, votes, score) in enumerate(nms_hits, start=len(all_candidate_hits) + 1):
                meta = metadata[idx]
                lat, lon = meta.lat, meta.lon
                if lat == 0.0 and lon == 0.0:
                    if coord_fn is None:
                        from luna.io.nac_reader import read_nac
                        from luna.screening.candidate_gen import DataIngestor
                        img = read_nac(nac_path, geometry=True)
                        lines, samples = img.pixels.shape
                        coord_fn = DataIngestor._build_coord_fn(nac_path, img.geometry, lines, samples)
                    
                    x_center = meta.x_offset + meta.width / 2.0
                    y_center = meta.y_offset + meta.height / 2.0
                    lon, lat = coord_fn(x_center, y_center)
                
                all_candidate_hits.append(CandidateHit(
                    rank=rank, product_id=pid, votes=votes, score=score,
                    lon=lon, lat=lat,
                    x_offset=meta.x_offset, y_offset=meta.y_offset,
                ))

        # Deduplicate candidates across overlap areas
        return apply_lunar_spatial_nms(all_candidate_hits, distance_threshold_meters=150.0)

    def classify(
        self,
        product_ids: list[str],
        query_dir: str | Path,
        top_k: int | None = None,
        search_k: int | None = None,
        min_dist_px: float | None = None,
        checkpoint: str | Path | None = None,
        score_thr: float = 0.5,
        essa_min_score: float = 0.0,
        output_dir: str | Path | None = None,
        skip_preprocess: bool = False,
        trace: dict = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> list[RefinedHit]:
        """Perform vector search on Pithos indexes and run secondary refiners on candidate hits."""
        # 1. Generate Candidates
        candidates = self.generate_candidates(
            product_ids=product_ids,
            query_dir=query_dir,
            top_k=top_k,
            search_k=search_k,
            min_dist_px=min_dist_px,
            trace=trace,
        )

        # 2. Refine Candidates
        refined = self._refine_candidates(
            hits=candidates,
            checkpoint=checkpoint,
            score_thr=score_thr,
            essa_min_score=essa_min_score,
            output_dir=output_dir,
            skip_preprocess=skip_preprocess,
            trace=trace,
            on_progress=on_progress,
        )

        return refined


class ESSAClassifier(LunaClassifier):
    def _refine_candidates(
        self,
        hits: list[CandidateHit],
        checkpoint: str | Path | None = None,
        score_thr: float = 0.5,
        essa_min_score: float = 0.0,
        output_dir: str | Path | None = None,
        skip_preprocess: bool = False,
        trace: dict = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> list[RefinedHit]:
        from luna.models import ESSARefiner
        from luna.config import WEIGHTS_DIR

        log.info("Initializing ESSARefiner stage on %s …", self._device)
        if on_progress:
            on_progress("Initializing ESSARefiner", 0, 1)
        
        checkpoint = checkpoint or (WEIGHTS_DIR / "essa.pt")
        refiner = ESSARefiner.from_checkpoint(checkpoint, device=self._device)

        if on_progress:
            on_progress("Initializing ESSARefiner", 1, 1)
            on_progress("Running ESSA refinement", 0, len(hits))

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
            
        return refined


class DINOClassifier(LunaClassifier):
    def _refine_candidates(
        self,
        hits: list[CandidateHit],
        checkpoint: str | Path | None = None,
        score_thr: float = 0.5,
        essa_min_score: float = 0.0,
        output_dir: str | Path | None = None,
        skip_preprocess: bool = False,
        trace: dict = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> list[RefinedHit]:
        from luna.models.dino_refiner import DINORefiner
        
        log.info("Initializing DINORefiner stage on %s …", self._device)
        if on_progress:
            on_progress("Initializing DINORefiner", 0, 1)
        
        refiner = DINORefiner(device=self._device)
        
        if on_progress:
            on_progress("Initializing DINORefiner", 1, 1)
            on_progress("Running DINO refinement", 0, len(hits))
        
        log.info("Passing %d candidates to DINO refiner (score_thr=%.2f) …", len(hits), score_thr)
        refined = refiner.refine(
            hits=hits,
            out_dir=output_dir,
            score_thr=score_thr,
            esa_min_score=essa_min_score,
            save_debug_plots=False,
            skip_preprocess=True,
            trace=trace,
            save_attention_overlay=self._config.save_attention_overlay if hasattr(self._config, 'save_attention_overlay') else False,
        )
        
        if on_progress:
            on_progress("Running DINO refinement", len(hits), len(hits))
            on_progress("DINO refinement complete", 1, 1)
            
        return refined


class Stage2Classifier(LunaClassifier):
    def _refine_candidates(
        self,
        hits: list[CandidateHit],
        checkpoint: str | Path | None = None,
        score_thr: float = 0.5,
        essa_min_score: float = 0.0,
        output_dir: str | Path | None = None,
        skip_preprocess: bool = False,
        trace: dict = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> list[RefinedHit]:
        from luna.models.stage2_decoder import Stage2Refiner
        from luna.config import PROJECT_ROOT
        
        log.info("Initializing Stage2Refiner stage on %s …", self._device)
        if on_progress:
            on_progress("Initializing Stage2Refiner", 0, 1)
            
        checkpoint = checkpoint or (PROJECT_ROOT / "data" / "weights" / "stage2_decoder_best.pt")
        refiner = Stage2Refiner.from_checkpoint(
            checkpoint_path=checkpoint,
            dino_encoder=self._encoder,
            device=self._device
        )
        
        if on_progress:
            on_progress("Initializing Stage2Refiner", 1, 1)
            on_progress("Running Stage2 refinement", 0, len(hits))
            
        log.info("Passing %d candidates to Stage2 refiner (score_thr=%.2f) …", len(hits), score_thr)
        refined = refiner.refine(
            hits=hits,
            out_dir=output_dir,
            score_thr=score_thr,
            save_debug_plots=output_dir is not None,
            trace=trace,
        )
        
        if on_progress:
            on_progress("Running Stage2 refinement", len(hits), len(hits))
            on_progress("Stage2 refinement complete", 1, 1)
            
        return refined


def get_classifier(
    refiner_type: str,
    encoder,
    device: str,
    config: LunaConfig | None = None
) -> LunaClassifier:
    if refiner_type == "essa":
        return ESSAClassifier(encoder, device, config)
    elif refiner_type == "dino":
        return DINOClassifier(encoder, device, config)
    elif refiner_type == "stage2":
        return Stage2Classifier(encoder, device, config)
    else:
        raise ValueError(f"Unknown refiner_type: {refiner_type}")
