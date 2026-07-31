"""
luna.latent_map
~~~~~~~~~~~~~~~
Latent space exploration, topology, and parametric manifold analysis modules
for DINOv3 lunar tile embeddings.
"""

from __future__ import annotations

from luna.latent_map.topology import PitTopologyAnalyzer
from luna.latent_map.prober import LatentSpaceProber
from luna.latent_map.parametric_umap import ParametricUMAPEncoder, ParametricUMAPNet
from luna.latent_map.pit_filter import PitCandidateFilter

# Aliases for backwards compatibility
ParametricUMAP = ParametricUMAPEncoder
PitFilter = PitCandidateFilter

__all__ = [
    "PitTopologyAnalyzer",
    "LatentSpaceProber",
    "ParametricUMAPEncoder",
    "ParametricUMAPNet",
    "ParametricUMAP",
    "PitCandidateFilter",
    "PitFilter",
]
