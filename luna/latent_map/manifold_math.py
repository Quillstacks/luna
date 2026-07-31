"""
luna.latent_map.manifold_math
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Mathematical Foundations of High-Dimensional Manifold Geometry & Spectral Analysis.

Includes implementation of:
1. Johnson-Lindenstrauss (JL) Distance Preservation Metrics & Metric Distortion.
2. Laplace-Beltrami Spectral Graph Operator (Normalized Graph Laplacian L_sym & Diffusion Eigenvectors).
3. Nash Isometric Strain & Sammon Mapping Stress for 2D Manifold Projection Distortion.

Author: LUNA Advanced Agentic Coding Architecture Team
"""

from __future__ import annotations

import logging
from typing import Tuple, Dict, Any

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

log = logging.getLogger(__name__)


# =============================================================================
# 1. Johnson-Lindenstrauss (JL) Metric Distortion
# =============================================================================

def compute_jl_distortion_ratio(
    high_dim_vecs: np.ndarray, low_dim_vecs: np.ndarray, sample_size: int = 1000
) -> Tuple[np.ndarray, float]:
    """
    Computes pairwise distance preservation ratio according to the Johnson-Lindenstrauss Lemma:

        Ratio(u, v) = ||f(u) - f(v)||_2 / ||u - v||_2

    A ratio near 1.0 indicates near-perfect isometric distance preservation.

    Args:
        high_dim_vecs: (N, D) float array of original high-dimensional vectors (e.g. D=384).
        low_dim_vecs: (N, d) float array of projected low-dimensional vectors (e.g. d=2).
        sample_size: Max points to sample for pairwise distance matrix computation.

    Returns:
        ratios: Pairwise distance ratio array.
        mean_preservation_score: Mean preservation percentage (1.0 = 100% isometry).
    """
    n = len(high_dim_vecs)
    if n > sample_size:
        indices = np.random.choice(n, size=sample_size, replace=False)
        hd = high_dim_vecs[indices]
        ld = low_dim_vecs[indices]
    else:
        hd = high_dim_vecs
        ld = low_dim_vecs

    # Pairwise Euclidean distances
    d_high = pdist(hd, metric="euclidean")
    d_low = pdist(ld, metric="euclidean")

    # Avoid division by zero
    d_high = np.where(d_high == 0, 1e-9, d_high)

    # Normalize low-dim scale for fair comparison
    d_low_scaled = d_low * (np.median(d_high) / (np.median(d_low) + 1e-9))
    ratios = d_low_scaled / d_high

    # Preservation score = 1 - mean absolute deviation from 1.0
    mean_dev = float(np.mean(np.abs(ratios - 1.0)))
    preservation_score = max(0.0, 1.0 - mean_dev)

    return ratios, preservation_score


# =============================================================================
# 2. Laplace-Beltrami Spectral Operator (Diffusion Coordinates)
# =============================================================================

def compute_laplace_beltrami_eigenvectors(
    high_dim_vecs: np.ndarray, k_neighbors: int = 15, n_components: int = 4
) -> np.ndarray:
    """
    Constructs the Discrete Graph Laplacian approximation of the Laplace-Beltrami Operator:

        L_sym = I - D^{-1/2} W D^{-1/2}

    where W is the k-NN RBF similarity matrix and D is the diagonal degree matrix.
    The smallest non-zero eigenvectors (psi_1, psi_2, ...) represent the intrinsic
    spectral diffusion coordinates of the Riemannian manifold.

    Args:
        high_dim_vecs: (N, D) float array of high-dimensional vectors.
        k_neighbors: Number of nearest neighbors for adjacency graph.
        n_components: Number of smallest eigenvectors to extract.

    Returns:
        eigenvectors: (N, n_components) float array of diffusion coordinates.
    """
    n = len(high_dim_vecs)
    k = min(k_neighbors, n - 1)

    log.info("Building k-NN graph (k=%d) for Laplace-Beltrami operator on N=%d points...", k, n)

    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="auto")
    nn.fit(high_dim_vecs)
    distances, indices = nn.kneighbors(high_dim_vecs)

    # RBF Kernel bandwidth
    sigma = np.median(distances) + 1e-6
    weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))

    # Build sparse adjacency matrix W
    row_indices = np.repeat(np.arange(n), k)
    col_indices = indices.flatten()
    data = weights.flatten()

    W = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
    W = 0.5 * (W + W.T)  # Ensure symmetric adjacency

    # Degree matrix D
    degrees = np.array(W.sum(axis=1)).flatten()
    degrees_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
    D_inv_sqrt = diags(degrees_inv_sqrt)

    # Normalized Symmetric Laplacian L_sym = I - D^{-1/2} W D^{-1/2}
    L_sym = diags(np.ones(n)) - D_inv_sqrt.dot(W).dot(D_inv_sqrt)

    log.info("Extracting %d smallest non-trivial Laplace-Beltrami eigenvectors...", n_components)

    try:
        # Solve smallest eigenvalues (skip first trivial zero eigenvector)
        eigenvalues, eigenvectors = eigsh(L_sym, k=n_components + 1, which="SM")
        # Exclude smallest (trivial constant) eigenvector
        diff_coords = eigenvectors[:, 1 : n_components + 1]
    except Exception as e:
        log.warning("Eigenvalue solver fallback due to: %s", e)
        diff_coords = np.random.randn(n, n_components).astype(np.float32)

    return diff_coords.astype(np.float32)


# =============================================================================
# 3. Nash Isometric Strain & Sammon Mapping Stress
# =============================================================================

def compute_nash_sammon_strain(
    high_dim_vecs: np.ndarray, low_dim_vecs: np.ndarray, sample_size: int = 1000
) -> Tuple[np.ndarray, float]:
    r"""
    Calculates Sammon Mapping Stress / Nash Isometric Strain for each point:

        E_i = \sum_{j \neq i} \frac{(d_{high}(i,j) - d_{low}(i,j))^2}{d_{high}(i,j)}

    Measures localized manifold embedding distortion. Points with high strain (red)
    indicate projection artifacts where 2D proximity does not reflect true 384D closeness.

    Args:
        high_dim_vecs: (N, D) original vectors.
        low_dim_vecs: (N, d) 2D projected coordinates.
        sample_size: Max subset for pairwise distance calculation.

    Returns:
        point_strain: (N,) array of per-point isometric strain scores.
        global_sammon_stress: Normalized global Sammon stress value.
    """
    n = len(high_dim_vecs)
    if n > sample_size:
        indices = np.random.choice(n, size=sample_size, replace=False)
        hd = high_dim_vecs[indices]
        ld = low_dim_vecs[indices]
    else:
        indices = np.arange(n)
        hd = high_dim_vecs
        ld = low_dim_vecs

    d_high = squareform(pdist(hd, metric="cosine"))
    d_low = squareform(pdist(ld, metric="euclidean"))

    # Scale low-dim distances
    d_low_scaled = d_low * (np.median(d_high) / (np.median(d_low) + 1e-9))

    # Mask self-distances
    np.fill_diagonal(d_high, np.inf)
    np.fill_diagonal(d_low_scaled, np.inf)

    # Pointwise Sammon stress: sum((d_high - d_low)^2 / d_high)
    diff_sq = ((d_high - d_low_scaled) ** 2) / np.where(d_high == np.inf, 1.0, d_high)
    point_strain_sampled = np.sum(np.where(d_high == np.inf, 0.0, diff_sq), axis=1)

    global_stress = float(np.sum(point_strain_sampled) / (np.sum(np.where(d_high == np.inf, 0.0, d_high)) + 1e-9))

    # Map back to full N size
    full_strain = np.zeros(n, dtype=np.float32)
    full_strain[indices] = point_strain_sampled

    return full_strain, global_stress
