# `luna.latent_map` Reference

`luna.latent_map` provides tools for non-linear manifold projection, spectral graph analysis, metric preservation evaluation, and interactive visual exploration of lunar DINOv3 feature embeddings.

---

## Manifold Mathematics & Metric Preservation (`luna.latent_map.manifold_math`)

::: luna.latent_map.manifold_math
    options:
      members:
        - compute_jl_distortion_ratio
        - compute_laplace_beltrami_spectrum
        - compute_nash_isometric_distortion
        - compute_sammon_stress

---

## Topology Analyzer (`luna.latent_map.topology`)

::: luna.latent_map.topology
    options:
      members:
        - PitTopologyAnalyzer

---

## Parametric UMAP Neural Network (`luna.latent_map.parametric_umap`)

::: luna.latent_map.parametric_umap
    options:
      members:
        - ParametricUMAPEncoder
        - ParametricUMAPNet

---

## Latent Space Prober (`luna.latent_map.prober`)

::: luna.latent_map.prober
    options:
      members:
        - LatentSpaceProber

---

## Pit Candidate Filter (`luna.latent_map.pit_filter`)

::: luna.latent_map.pit_filter
    options:
      members:
        - PitCandidateFilter
