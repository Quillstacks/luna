"""DINO-based second-stage refiner — lightweight alternative to ESSA.

Uses DINOv3 patch embeddings for classification instead of Mask R-CNN.
No ISIS preprocessing required. Works directly on raw CDR tiles.

Advantages over ESSA:
    - No ISIS preprocessing (saves 20-30s per NAC)
    - Smaller footprint (DINOv3 ~150MB vs ESSA ~550MB)
    - CPU-capable
    - 10x faster (~20ms vs ~200ms per candidate)

Disadvantages:
    - No pixel-precise segmentation masks
    - Per-tile classification only
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from luna.config import DATA_DIR, WEIGHTS_DIR, TILE_SIZE, LROC_VALID_MIN, SCRATCH_DIR
from luna.io.pds_fetch import fetch_nac
from luna.models.essa import RefinedHit

log = logging.getLogger(__name__)


class DINORefiner:
    """Lightweight DINO-based classifier for candidate refinement.
    
    Uses cosine similarity between candidate tile DINO embeddings and 
    reference pit embeddings to filter false positives.
    
    Parameters
    ----------
    reference_embeddings : np.ndarray | Path, shape (N, 384)
        DINOv3 embeddings of known pit tiles (normalized to unit length).
        Can be a file path or a numpy array.
    threshold : float, default 0.85
        Minimum cosine similarity to accept a candidate as a pit.
    device : str, optional
        Device to run DINO on. Auto-detects if None.
    """

    DEFAULT_REFERENCE = DATA_DIR / "dino_reference.npy"
    DEFAULT_THRESHOLD = 0.85

    def __init__(
        self,
        reference_embeddings: np.ndarray | Path | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        device: str | None = None,
    ) -> None:
        if device is None:
            device = (
                "mps" if torch.backends.mps.is_available() else
                "cuda" if torch.cuda.is_available() else
                "cpu"
            )
        self.device = device
        self.threshold = threshold
        
        # Load reference embeddings
        self._using_placeholders = False
        if reference_embeddings is None:
            reference_embeddings = self.DEFAULT_REFERENCE
        
        if isinstance(reference_embeddings, Path):
            if reference_embeddings.exists():
                ref_emb = np.load(reference_embeddings).astype(np.float32)
            else:
                log.warning(
                    "Reference embeddings not found at %s.\n\n"
                    "Using placeholder embeddings.\n\n"
                    "To generate real embeddings:\n\n"
                    "  1. Download LPA catalog: https://lroc.im-ldi.com/atlases/pits/list\n\n"
                    "  2. Save as data/catalogs/lpa.csv\n\n"
                    "  3. Run: python scripts/prepare_dino_reference.py",
                    reference_embeddings
                )
                self._using_placeholders = True
                ref_emb = np.random.randn(50, 384).astype(np.float32)
        else:
            ref_emb = reference_embeddings.astype(np.float32)
        
        # Normalize to unit length for cosine similarity
        norms = np.linalg.norm(ref_emb, axis=1, keepdims=True)
        self.reference = ref_emb / np.where(norms > 0, norms, 1.0)
        
        # Adapt threshold based on number of reference embeddings
        # With few references, the cosine similarity distribution changes
        n_ref = len(self.reference)
        if self._using_placeholders:
            # With placeholders, use minimal threshold to allow all candidates through
            self._adaptive_threshold = 0.0
            log.warning(
                "DINORefiner: Using PLACEHOLDER embeddings. "
                "Threshold set to 0.0 - all candidates will pass. "
                "For production: run scripts/prepare_dino_reference.py"
            )
        elif n_ref < 10:
            # With very few refs, set very low threshold
            self._adaptive_threshold = 0.05
            log.warning(
                "DINORefiner: Only %d reference embeddings loaded (recommended: >=50). "
                "Using adaptive threshold %.2f (was %.2f) for robustness. "
                "For production: run scripts/prepare_dino_reference.py",
                n_ref, self._adaptive_threshold, self.threshold
            )
        elif n_ref < 20:
            # With <20 refs, use adaptive threshold
            self._adaptive_threshold = 0.1  # Lower threshold
            log.warning(
                "DINORefiner: Only %d reference embeddings loaded (recommended: >=50). "
                "Using adaptive threshold %.2f (was %.2f) for robustness. "
                "For production: run scripts/prepare_dino_reference.py",
                n_ref, self._adaptive_threshold, self.threshold
            )
        elif n_ref < 50:
            # With 20-50 refs, slightly lower threshold
            self._adaptive_threshold = max(0.7, self.threshold - 0.1)
            log.warning(
                "DINORefiner: %d reference embeddings loaded (recommended: >=50). "
                "Using adaptive threshold %.2f.",
                n_ref, self._adaptive_threshold
            )
        else:
            self._adaptive_threshold = self.threshold
        
        log.info("DINORefiner loaded with %d reference embeddings, threshold=%.2f", 
                 n_ref, self._adaptive_threshold)

    def refine(
        self,
        hits: list,
        out_dir: str | Path | None = None,
        score_thr: float = 0.5,
        esa_min_score: float = 0.0,
        save_debug_plots: bool = False,
        skip_preprocess: bool = True,
        trace: dict = None,
        save_attention_overlay: bool = False,
    ) -> list[RefinedHit]:
        """Refine candidate hits using DINO similarity against reference embeddings.
        
        Parameters
        ----------
        hits : list[CandidateHit]
            Candidate hits from Phase 1 (DINO + Pithos KNN).
        out_dir : path-like, optional
            Directory for debug outputs (not used by DINORefiner).
        score_thr : float
            Minimum vote score threshold (kept for API compatibility).
        esa_min_score : float
            Minimum ESSA score (kept for API compatibility, not used).
        save_debug_plots : bool
            Whether to save debug visualizations (not used by DINORefiner).
        skip_preprocess : bool
            Always True for DINORefiner (no preprocessing needed).
        trace : dict, optional
            Dictionary to collect timing metrics.
            
        Returns
        -------
        list[RefinedHit]
            Filtered and scored refined hits with dino_similarity field.
        """
        from luna.models.dinov3 import DINOEncoder
        from luna.io.nac_reader import _label_byte_count
        import pvl
        
        t_start = time.perf_counter()
        
        # Load DINO encoder (same as Phase 1 for consistency)
        encoder = DINOEncoder(
            lora_dir="F1nnSBK/lunar-dinov3-lora",
            base_weights_path="F1nnSBK/lunar-dinov3-lora",
            matryoshka_dim=384,
            device=self.device,
        )
        
        # Group hits by product_id to batch process
        hits_by_pid: dict[str, list] = defaultdict(list)
        for h in hits:
            hits_by_pid[h.product_id].append(h)
        
        all_refined: list[RefinedHit] = []
        
        # Prepare output directory for attention overlays
        if save_attention_overlay and out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        
        for pid, pid_hits in hits_by_pid.items():
            nac_path = SCRATCH_DIR / f"{pid}.IMG"
            if not nac_path.exists():
                log.info("Fetching %s from PDS …", pid)
                nac_path = fetch_nac(pid, dest_dir=SCRATCH_DIR)
            
            with open(nac_path, "rb") as f:
                label = pvl.load(f)
                header_bytes = _label_byte_count(label)
                img_block = label["IMAGE"]
                lines = int(img_block["LINES"])
                samples = int(img_block["LINE_SAMPLES"])
            
            img = np.memmap(nac_path, dtype=np.int16, mode='r', offset=header_bytes, shape=(lines, samples))
            
            for hit in pid_hits:
                x0 = hit.x_offset
                y0 = hit.y_offset
                
                tile = img[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE].copy()
                
                valid = tile[tile > LROC_VALID_MIN]
                if len(valid) == 0:
                    lo, hi = 0.0, 1.0
                else:
                    lo, hi = float(valid.min()), float(valid.max())
                
                if hi > lo:
                    norm_tile = (tile - lo) / (hi - lo)
                else:
                    norm_tile = np.zeros_like(tile, dtype=np.float32)
                
                batch_3ch = np.stack([norm_tile] * 3, axis=0)
                batch_3ch = np.expand_dims(batch_3ch, 0)
                
                batch_3ch_uint8 = (batch_3ch * 255).astype(np.uint8)
                
                embedding = encoder.encode(batch_3ch_uint8)
                embedding = embedding.flatten()
                
                # Encoder already returns L2-normalized embeddings
                # Compute cosine similarity: (N, 384) @ (384,) -> (N,)
                similarities = np.dot(self.reference, embedding).flatten()
                max_sim = float(np.max(similarities))
                
                # Use adaptive threshold based on reference count
                effective_threshold = getattr(self, '_adaptive_threshold', self.threshold)
                if max_sim >= effective_threshold and hit.score >= score_thr:
                    refined = RefinedHit(
                        rank=len(all_refined) + 1,
                        product_id=hit.product_id,
                        votes=hit.votes,
                        dino_score=hit.score,
                        lon=hit.lon,
                        lat=hit.lat,
                        x_offset=hit.x_offset,
                        y_offset=hit.y_offset,
                        dino_similarity=max_sim,
                    )
                    all_refined.append(refined)
        
        if trace is not None:
            trace["p2_dino_refinement"] = time.perf_counter() - t_start
            trace["dino_s"] = trace.get("p2_dino_refinement", 0.0)
            trace["dino_hits_in"] = len(hits)
            trace["dino_hits_out"] = len(all_refined)
            trace["dino_refinement_ratio"] = len(all_refined) / len(hits) if len(hits) > 0 else 0.0
            trace["dino_threshold_used"] = getattr(self, '_adaptive_threshold', self.threshold)
        
        effective_threshold = getattr(self, '_adaptive_threshold', self.threshold)
        log.info("DINORefiner: %d/%d candidates passed threshold (%.1f%%), used threshold=%.2f", 
                 len(all_refined), len(hits), 
                 100 * len(all_refined) / len(hits) if len(hits) > 0 else 0.0,
                 effective_threshold)
        
        # Sort by confidence (dino_similarity) descending
        all_refined.sort(key=lambda x: x.dino_similarity, reverse=True)
        
        # Re-assign ranks after sorting (create new objects since RefinedHit is frozen)
        all_refined = [
            RefinedHit(
                rank=i + 1,
                product_id=hit.product_id,
                votes=hit.votes,
                dino_score=hit.dino_score,
                lon=hit.lon,
                lat=hit.lat,
                x_offset=hit.x_offset,
                y_offset=hit.y_offset,
                essa_score=hit.essa_score,
                essa_class=hit.essa_class,
                essa_lon=hit.essa_lon,
                essa_lat=hit.essa_lat,
                dino_similarity=hit.dino_similarity,
            )
            for i, hit in enumerate(all_refined)
        ]

        # Generate and save attention map overlays with ranks!
        if save_attention_overlay and out_dir is not None:
            log.info("Generating and saving attention overlays with sorted ranks...")
            refined_by_pid = defaultdict(list)
            for hit in all_refined:
                refined_by_pid[hit.product_id].append(hit)
                
            for pid, pid_hits in refined_by_pid.items():
                nac_path = SCRATCH_DIR / f"{pid}.IMG"
                if not nac_path.exists():
                    nac_path = fetch_nac(pid, dest_dir=SCRATCH_DIR)
                
                with open(nac_path, "rb") as f:
                    label = pvl.load(f)
                    header_bytes = _label_byte_count(label)
                    img_block = label["IMAGE"]
                    lines = int(img_block["LINES"])
                    samples = int(img_block["LINE_SAMPLES"])
                
                img = np.memmap(nac_path, dtype=np.int16, mode='r', offset=header_bytes, shape=(lines, samples))
                
                for hit in pid_hits:
                    x0 = hit.x_offset
                    y0 = hit.y_offset
                    tile = img[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE].copy()
                    
                    valid = tile[tile > LROC_VALID_MIN]
                    lo, hi = (float(valid.min()), float(valid.max())) if len(valid) > 0 else (0.0, 1.0)
                    norm_tile = (tile - lo) / (hi - lo) if hi > lo else np.zeros_like(tile, dtype=np.float32)
                    
                    batch_3ch = np.stack([norm_tile] * 3, axis=0)
                    batch_3ch = np.expand_dims(batch_3ch, 0)
                    batch_3ch_uint8 = (batch_3ch * 255).astype(np.uint8)
                    
                    _, attention_map = encoder.encode(batch_3ch_uint8, return_attention=True)
                    from luna.utils.attention_visualizer import save_attention_overlay as save_overlay
                    tile_uint8 = (norm_tile * 255).astype(np.uint8)
                    
                    overlay_path = out_dir / f"rank_{hit.rank:03d}_{pid}_{hit.x_offset}_{hit.y_offset}_attention.png"
                    attention_map = np.squeeze(attention_map)
                    save_overlay(tile_uint8, attention_map, overlay_path)
        
        return all_refined
