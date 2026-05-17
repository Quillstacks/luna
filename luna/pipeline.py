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
import faiss

import gc
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from luna.config import (
    DINO_DIM, FINAL_TOP_K, INDEX_DIR, LROC_VALID_MIN,
    MAX_BATCH_SIZE, MIN_DIST_PX, SCRATCH_DIR, SEARCH_K,
    STRIDE, TILE_SIZE,
)
from luna.io.pds_fetch import fetch_nac
from luna.screening.candidate_gen import DataIngestor
from luna.screening.protocols import TileMetadata
from luna.storage.faiss_store import FaissLocalStore
from luna.models import RefinedHit

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateHit:
    rank: int
    product_id: str
    votes: int
    score: float          # best inner-product across all anchor queries
    lon: float
    lat: float
    x_offset: int
    y_offset: int


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LunaPipeline:
    def __init__(self, encoder, device: str) -> None:
        self._encoder = encoder
        self._device  = device

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        matryoshka_dim: int = DINO_DIM,
        device: str | None = None,
    ) -> LunaPipeline:
        from luna.models.dinov3 import DINOEncoder
        if device is None:
            device = (
                "mps"  if torch.backends.mps.is_available() else
                "cuda" if torch.cuda.is_available()          else
                "cpu"
            )
        log.info("Loading encoder from %s on %s ...", repo_id, device)
        encoder = DINOEncoder(
            lora_dir          = repo_id,
            base_weights_path = repo_id,
            matryoshka_dim    = matryoshka_dim,
            device            = device,
        )
        return cls(encoder=encoder, device=device)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def _ingest(self, nac_path: Path) -> tuple[FaissLocalStore, list[TileMetadata]]:
        log.info("Slicing and embedding %s (Tile: %d, Stride: %d) ...", nac_path.name, TILE_SIZE, STRIDE)
        store    = FaissLocalStore(vector_dim=DINO_DIM)
        ingestor = DataIngestor(model=self._encoder, store=store,
                                max_batch_size=MAX_BATCH_SIZE)
        ingestor.ingest_nac(path=nac_path, tile_size=TILE_SIZE, stride=STRIDE)

        ingestor.screener.shutdown()
        del ingestor
        if self._device == "mps":
            torch.mps.empty_cache()
        gc.collect()
        log.info("Ingestion complete. Generated %d tile embeddings.", len(store._metadata))
        return store, store._metadata

    def _save_index(self, store: FaissLocalStore, nac_path: Path) -> Path:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        if self._device == "mps":
            torch.mps.empty_cache()
        gc.collect()
        prefix = str(INDEX_DIR / f"faiss_{nac_path.stem}")
        log.info("Writing FAISS index and metadata to %s ...", prefix)
        store.save_to_disk(prefix)
        return Path(prefix)

    # ------------------------------------------------------------------
    # Query encoding
    # ------------------------------------------------------------------

    def _encode_queries(self, query_dir: str | Path) -> np.ndarray:
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
        gc.collect()

        stacked = np.vstack(vecs).astype(np.float32)
        faiss.normalize_L2(stacked)
        log.info("Encoded %d query anchors into matrix shape %s.", len(paths), stacked.shape)
        return stacked

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @staticmethod
    def _search(
        index_prefix: str,
        metadata: list[TileMetadata],
        query_vecs: np.ndarray,
        k: int,
    ) -> tuple[dict, dict]:
        from collections import Counter

        log.info("Loading FAISS index %s.index for matching ...", index_prefix)
        index = faiss.read_index(f"{index_prefix}.index")
        log.info("Executing k-NN search (k=%d) for %d query vectors ...", k, len(query_vecs))
        dists, ids = index.search(query_vecs, k)

        vote_map: dict[int, int]   = Counter()
        best_dist: dict[int, float] = {}

        for row in range(ids.shape[0]):
            for dist, idx in zip(dists[row], ids[row]):
                if idx < 0:
                    continue
                vote_map[idx] = vote_map.get(idx, 0) + 1
                if idx not in best_dist or dist > best_dist[idx]:
                    best_dist[idx] = float(dist)
        log.info("Search retrieved %d unique candidate tiles via voting.", len(vote_map))
        return vote_map, best_dist

    @staticmethod
    def _nms(
        ranked_ids: list[int],
        vote_map: dict,
        best_dist: dict,
        metadata: list[TileMetadata],
        top_k: int,
        min_dist_px: float,
    ) -> list[tuple[int, int, float]]:
        log.info("Applying NMS (min_dist: %.1f px, max_targets: %d) on %d tiles ...", min_dist_px, top_k, len(ranked_ids))
        hits: list[tuple[int, int, float]] = []
        accepted: list[tuple[float, float]] = []

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
        log.info("NMS complete. Retained %d non-overlapping hits.", len(hits))
        return hits

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(
        self,
        product_ids: str | list[str],
        query_dir: str | Path,
        top_k: int       = FINAL_TOP_K,
        search_k: int    = SEARCH_K,
        min_dist_px: float = MIN_DIST_PX,
        force_reingest: bool = False,
    ) -> list[CandidateHit]:
        if isinstance(product_ids, str):
            product_ids = [product_ids]

        metadata_map: dict[str, list[TileMetadata]] = {}

        for pid in product_ids:
            nac_path     = SCRATCH_DIR / f"{pid}.IMG"
            index_prefix = str(INDEX_DIR / f"faiss_{pid}")
            index_exists = Path(f"{index_prefix}.index").exists()

            if not nac_path.exists():
                log.info("Fetching %s from PDS ...", pid)
                nac_path = fetch_nac(pid, dest_dir=SCRATCH_DIR)

            if force_reingest or not index_exists:
                log.info("Ingesting %s ...", pid)
                store, metadata = self._ingest(nac_path)
                metadata_map[pid] = metadata
                self._save_index(store, nac_path)
                del store
                gc.collect()
            else:
                log.info("Index for %s already exists, loading metadata ...", pid)
                with open(f"{index_prefix}_meta.pkl", "rb") as f:
                    metadata_map[pid] = pickle.load(f)

        query_vecs = self._encode_queries(query_dir)        # MPS → FAISS normalize

        all_hits: list[CandidateHit] = []

        for pid in product_ids:
            index_prefix = str(INDEX_DIR / f"faiss_{pid}")
            metadata     = metadata_map[pid]

            vote_map, best_dist = self._search(index_prefix, metadata, query_vecs, k=search_k)
            ranked   = sorted(vote_map.keys(), key=lambda i: (-vote_map[i], -best_dist[i]))
            nms_hits = self._nms(ranked, vote_map, best_dist, metadata,
                                top_k=top_k, min_dist_px=min_dist_px)

            for rank, (idx, votes, score) in enumerate(nms_hits, start=len(all_hits) + 1):
                meta = metadata[idx]
                all_hits.append(CandidateHit(
                    rank=rank, product_id=pid, votes=votes, score=score,
                    lon=meta.lon, lat=meta.lat,
                    x_offset=meta.x_offset, y_offset=meta.y_offset,
                ))

        all_hits.sort(key=lambda h: (-h.votes, -h.score))
        for i, h in enumerate(all_hits):
            object.__setattr__(h, "rank", i + 1)

        log.info("Scan complete: %d hits across %d NACs.", len(all_hits), len(product_ids))
        return all_hits
        

    def refine(
        self,
        hits: list[CandidateHit],
        checkpoint: str | Path | None = None,
        score_thr: float = 0.5,
        essa_min_score: float = 0.0,
        output_dir: str | Path | None = None,
    ) -> list[RefinedHit]:
        from luna.models import ESSARefiner
        from luna.config import WEIGHTS_DIR

        log.info("Initializing ESSARefiner stage on %s ...", self._device)
        checkpoint = checkpoint or (WEIGHTS_DIR / "essa.pt")
        refiner    = ESSARefiner.from_checkpoint(checkpoint, device=self._device)

        # Build NAC context dicts from hits
        pids = {h.product_id for h in hits}
        nac_paths, nac_offsets, nac_dims = {}, {}, {}

        log.info("Resolving image geometry and PVL labels for %d unique NAC frames ...", len(pids))
        for pid in pids:
            img_path = SCRATCH_DIR / f"{pid}.IMG"
            if not img_path.exists():
                log.warning("NAC %s not found locally — skipping ESSA for this frame.", pid)
                continue
            import pvl
            with open(img_path, "rb") as f:
                label = pvl.load(f)
            img_block        = label["IMAGE"]
            nac_paths[pid]   = img_path
            nac_offsets[pid] = int(label["RECORD_BYTES"]) * int(label.get("LABEL_RECORDS", 1))
            nac_dims[pid]    = (int(img_block["LINE_SAMPLES"]), int(img_block["LINES"]))

        log.info("Passing %d candidates to ESSA network (score_threshold: %.2f) ...", len(hits), score_thr)
        refined_hits = refiner.refine(
            hits          = hits,
            nac_img_paths = nac_paths,
            nac_offsets   = nac_offsets,
            nac_dims      = nac_dims,
            score_thr     = score_thr,
            essa_min_score = essa_min_score,
            output_dir     = output_dir
        )
        
        log.info("Refinement complete. %d / %d hits passed verification.", len(refined_hits), len(hits))
        return refined_hits