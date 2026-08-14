# LUNA

> **L.U.N.A.** — *Lots of Unexplained NAC Anomalies*  
> Python library for lunar pit detection, SPICE photogrammetry, and orbital imagery analysis on NASA LROC NAC products.

---

## Key Features

- **SPICE Photogrammetry (`luna.io`)**: Pushbroom camera model implementing ground-to-image and image-to-ground ray-tracing directly in Python via `spiceypy`.
- **Targeted SPICE Pool Management**: On-demand kernel loading (`furnish_kernels_for_date`) to prevent CSPICE DAF-handle table exhaustion across multi-year missions.
- **Topography & Elevation Correction**: Parallax compensation using LOLA lunar elevations (`alt_m` via planetocentric `sp.latrec`).
- **Ascending & Descending Orbit Support**: Automatic flight-direction detection and corner-mapping in `LinearProjection`.
- **Vision Models (`luna.models`)**: Feature extraction with DINOv3 Vision Transformers, Stage-2 decoders, and boundary refiners.
- **Screening Pipeline (`luna.screening`)**: Disruptor ring-buffer paired with native C++ `Pithos` binary embedding search for candidate retrieval.
- **Latent Manifold Analysis (`luna.latent_map`)**: UMAP projections, metric estimations, and topology inspection of lunar terrain.
- **PDS Resolution**: Binary search index lookup over fixed-width PDS3 `INDEX.TAB` archives via HTTP range requests.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Data["Data Layer"]
        PDS["NASA PDS Archive"] -->|HTTP Range Reads| PDSIndex["luna.io.PDSIndex"]
        PDSIndex -->|On-Demand CDR/EDR| Reader["luna.io.read_nac"]
        NAIF["NASA NAIF SPICE"] -->|Targeted Fetch| Kernels["luna.io.kernel_fetch"]
    end

    subgraph Photogrammetry["Photogrammetry & Projection"]
        Kernels --> SPICE["luna.io.spice_project"]
        Reader --> Proj["luna.io.LinearProjection"]
        SPICE -->|Ray-surface intersection with alt_m| CoordMap["(Sample, Line) ⟷ (Lon, Lat)"]
        Proj -->|Bilinear Newton Solver| CoordMap
    end

    subgraph Pipeline["ML & Screening Pipeline"]
        Reader --> RingBuffer["Lock-Free Disruptor Ring"]
        RingBuffer --> DINO["DINOv3 Feature Encoder"]
        DINO --> Pithos["pithosdb Vector Index"]
        Pithos --> NMS["Spatial Lunar NMS Filtering"]
        NMS --> Candidates["Lunar Pit Candidates"]
    end

    subgraph Evaluation["Evaluation & Latent Analysis"]
        Candidates --> Stage2["Stage-2 Dense Prediction Decoder"]
        Candidates --> LatentMap["UMAP Latent Space & Topology"]
        Stage2 --> VerifiedPits["Verified Morphological Detections"]
    end
```

---

## Quick Example

```python
from luna.io.pds_fetch import fetch_nac
from luna.io.spice_project import ground_to_image

# 1. Fetch NAC product on-demand from PDS
img_path = fetch_nac("M1118880788RC", dest_dir="data/_scratch")

# 2. Project lunar ground coordinates (lon, lat, elevation) to exact pixel coordinates
# Example: Marius Hills Skylight (alt = -2010 m relative to lunar sphere)
sample_px, line_px = ground_to_image(
    img_path,
    lon_deg=303.228,
    lat_deg=14.091,
    alt_m=-2010.0
)

print(f"Target pixel coordinate: Sample = {sample_px:.2f}, Line = {line_px:.2f}")
```

---

## Documentation Roadmap

- [**Installation Guide**](getting-started/installation.md): Set up virtual environment, GPU dependencies, and SPICE kernels.
- [**SPICE & Photogrammetry Guide**](guide/spice-and-photogrammetry.md): Deep dive into CSPICE kernel pool limits, pushbroom camera geometry, topography parallax, and ascending orbit math.
- [**Vision & Models Guide**](guide/vision-and-models.md): DINOv3 feature backbones and Stage-2 decoders.
- [**Screening & Streaming Guide**](guide/screening-and-pithos.md): Multi-scale binary indexing and ring-buffer streaming.
- [**API Reference**](reference/io.md): Comprehensive module and class documentation.
