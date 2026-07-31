"""
luna.latent_map.pit_filter
~~~~~~~~~~~~~~~~~~~~~~~~~~
Candidate Pit Isolation & Geometry Filtering Module.

Implements a 3-Stage Filter Cascade to isolate genuine lunar collapse pits:
1. Sammon Strain & Outlier Gate (Rejects sensor noise & corrupted frames).
2. Contrastive Pithos Scoring (Hard negative rille/fracture mining).
3. Shadow Geometry & Aspect Ratio Filter (Rejects linear fracture shadows).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np

from luna.screening.pithos import PithosMIDB

log = logging.getLogger(__name__)


class PitCandidateFilter:
    """
    Filters raw vector search results into pure verified pit candidates.
    """

    def __init__(
        self,
        max_aspect_ratio: float = 2.2,
        max_sammon_strain: float = 10.0,
    ) -> None:
        self.max_aspect_ratio = max_aspect_ratio
        self.max_sammon_strain = max_sammon_strain

    def analyze_shadow_aspect_ratio(self, image_patch: np.ndarray) -> Tuple[float, float]:
        """
        Analyzes shadow geometry in a 256x256 image patch using OpenCV contour analysis.

        Returns:
            aspect_ratio: Ratio of bounding box max_dimension / min_dimension.
            shadow_area_ratio: Fraction of patch covered by deep shadow (< 15th percentile).
        """
        if image_patch.ndim == 3:
            gray = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_patch.copy()

        # Handle NaNs or zeros
        gray = np.nan_to_num(gray, nan=0.0)
        if gray.max() > 1.0:
            gray = gray / 255.0

        valid_mask = gray > 0
        if not valid_mask.any():
            return 99.0, 0.0

        # Threshold deep shadow region (below 15th percentile intensity)
        thresh_val = np.percentile(gray[valid_mask], 15)
        binary_shadow = ((gray < thresh_val) & valid_mask).astype(np.uint8) * 255

        shadow_area_ratio = float(np.sum(binary_shadow > 0) / (gray.size + 1e-6))

        # Find contours of shadow regions
        contours, _ = cv2.findContours(binary_shadow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 1.0, shadow_area_ratio

        # Find largest shadow contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < 16:  # Too small to be a pit
            return 1.0, shadow_area_ratio

        # Fit minimum area bounding rectangle
        rect = cv2.minAreaRect(largest_contour)
        (x, y), (w, h), angle = rect

        min_dim = min(w, h)
        max_dim = max(w, h)

        aspect_ratio = float(max_dim / (min_dim + 1e-5))
        return aspect_ratio, shadow_area_ratio

    def compute_contrastive_score(
        self,
        tile_embedding: np.ndarray,
        pit_anchors_binary: np.ndarray,
        rille_anchors_binary: np.ndarray,
    ) -> Tuple[int, int, int]:
        """
        Computes Contrastive Pit Score = d_rille - d_pit.

        Returns:
            pit_score: d_rille - d_pit (Higher is better/more pit-like).
            min_d_pit: Min Hamming distance to any pit anchor.
            min_d_rille: Min Hamming distance to any rille negative anchor.
        """
        bin_tile = PithosMIDB.binarize(tile_embedding.reshape(1, 384))
        bits_tile = np.unpackbits(bin_tile.view(np.uint8), axis=1)[:, :384]

        bits_pits = np.unpackbits(pit_anchors_binary.view(np.uint8), axis=1)[:, :384]
        d_pit = int(np.min(np.sum(bits_pits != bits_tile, axis=1)))

        if rille_anchors_binary is not None and len(rille_anchors_binary) > 0:
            bits_rilles = np.unpackbits(rille_anchors_binary.view(np.uint8), axis=1)[:, :384]
            d_rille = int(np.min(np.sum(bits_rilles != bits_tile, axis=1)))
        else:
            d_rille = 384

        pit_score = d_rille - d_pit
        return pit_score, d_pit, d_rille

    def is_valid_pit(
        self,
        sammon_strain: float,
        aspect_ratio: float,
        d_pit: int,
        contrastive_score: int,
    ) -> Tuple[bool, str]:
        """
        Evaluates whether a candidate passes all 3 filter gates.
        """
        if sammon_strain > self.max_sammon_strain:
            return False, f"Rejected: High Strain/Noise ({sammon_strain:.2f} > {self.max_sammon_strain})"

        if aspect_ratio > self.max_aspect_ratio:
            return False, f"Rejected: Linear Fracture/Rille Shadow (Aspect Ratio {aspect_ratio:.2f} > {self.max_aspect_ratio})"

        if d_pit > 120:
            return False, f"Rejected: Too Distant from Pit Anchors (Hamming Dist {d_pit} > 120)"

        if contrastive_score < 10:
            return False, f"Rejected: High Rille Overlap (Contrastive Score {contrastive_score} < 10)"

        return True, "VERIFIED PIT CANDIDATE"
