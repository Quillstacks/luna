"""
luna.latent_map.topology
~~~~~~~~~~~~~~~~~~~~~~~~
Analyzes pairwise relationships and topological structure among known pit anchors
in the 384-dimensional DINOv3 latent space.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from PIL import Image

from luna.config import LROC_VALID_MIN
from luna.models.dinov3 import DINOEncoder
from luna.screening.pithos import PithosMIDB

log = logging.getLogger(__name__)


class PitTopologyAnalyzer:
    """
    Encodes reference pit patches and calculates pair-wise topological distance matrices.
    """

    def __init__(self, encoder: DINOEncoder | None = None) -> None:
        self.encoder = encoder

    def _ensure_encoder(self) -> DINOEncoder:
        if self.encoder is None:
            from luna.config import HF_REPO_ID
            log.info("Initializing DINOv3 encoder for topology analysis...")
            self.encoder = DINOEncoder(
                lora_dir=HF_REPO_ID,
                base_weights_path=HF_REPO_ID,
                matryoshka_dim=384,
            )
        return self.encoder

    def load_and_encode_anchors(
        self, pits_dir: str | Path
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Load all pit patch .npy files in pits_dir, normalize, and encode via DINOv3.

        Returns:
            anchor_info: list of dicts with keys (name, file_path, category, family_id)
            embeddings: (N, 384) float32 array of latent vectors
        """
        encoder = self._ensure_encoder()
        pits_path = Path(pits_dir)
        pit_files = sorted(pits_path.glob("*.npy"))

        if not pit_files:
            raise FileNotFoundError(f"No pit .npy files found in {pits_dir}")

        log.info("Loading and encoding %d pit anchors from %s...", len(pit_files), pits_dir)

        anchor_info: List[Dict[str, Any]] = []
        pit_images: List[np.ndarray] = []

        for f in pit_files:
            img = np.load(f)
            if img.ndim == 2:
                if img.shape != (256, 256):
                    img = np.array(Image.fromarray(img).resize((256, 256), Image.Resampling.BILINEAR))
                pit_images.append(img)
            else:
                continue

            # Classify category based on filename
            fname = f.stem
            category = "Pit Candidate"
            if "Marius_Hills" in fname or "Tranquillitatis" in fname or "Ingenii" in fname or "Moscoviense" in fname:
                category = "Mare Skylight"
            elif "Copernicus" in fname or "Aristarchus" in fname or "Aristillus" in fname or "Crookes" in fname:
                category = "Impact Melt Pit"
            elif "Highland" in fname:
                category = "Highland Feature"

            # Assign family ID
            h = int(hashlib.md5(fname.encode("utf-8")).hexdigest(), 16)
            family_id = h % 8

            anchor_info.append({
                "name": fname,
                "file_path": str(f),
                "category": category,
                "family_id": family_id,
            })

        # Normalize images
        normalized = []
        for img in pit_images:
            valid = img[img > LROC_VALID_MIN]
            lo, hi = (valid.min(), valid.max()) if valid.size > 0 else (0.0, 1.0)
            norm = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
            normalized.append(norm)

        # Encode in batches
        batch_input = np.stack([(img * 255).astype(np.uint8) for img in normalized])
        embeddings = encoder.encode(batch_input).astype(np.float32)

        return anchor_info, embeddings

    def compute_distance_matrices(
        self, embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Cosine Distance matrix and Pithos PolarQuant Hamming Distance matrix.

        Returns:
            cosine_dists: (N, N) float32, cosine distance (1 - cosine_similarity)
            hamming_dists: (N, N) int32, Pithos 384-bit Hamming distances
        """
        n = len(embeddings)
        # Cosine distance
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized_emb = embeddings / norms
        cosine_sim = normalized_emb @ normalized_emb.T
        cosine_dists = np.clip(1.0 - cosine_sim, 0.0, 2.0).astype(np.float32)

        # PolarQuant-Hadamard binarization for Pithos Hamming distance
        binary_vecs = PithosMIDB.binarize(embeddings)  # (N, 6) int64 packed
        
        # Unpack bits for exact bitwise Hamming calculation
        bits = np.unpackbits(binary_vecs.view(np.uint8), axis=1)[:, :384]  # (N, 384)
        
        # Pairwise Hamming distance = sum(bit_i != bit_j)
        # Using XOR over bits
        hamming_dists = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            hamming_dists[i] = np.sum(bits != bits[i], axis=1)

        return cosine_dists, hamming_dists
