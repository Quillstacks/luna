# Screening & Streaming Engine

`luna.screening` implements a high-throughput streaming architecture to parallelize tile extraction, memory-mapped I/O, and candidate screening across large LROC NAC images ($\sim 5064 \times 52224$ pixels).

---

## 1. Lock-Free Disruptor Streaming Engine (`luna.screening.engine`)

To prevent I/O bottlenecks when processing 500 MB to 1 GB raw raster files, `luna.screening.engine` uses a lock-free Disruptor ring buffer architecture implemented in Cython/C++:

```text
[ Disk / Memory-Mapped File (MappedStripe) ]
                    |
                    v (Zero-copy raw pointers)
[ Disruptor Ring Buffer (DisruptorEngine) ]
                    |
                    v (SIMD Normalization & Slicing)
[ Fast Tensor Slicer (NACTransformer) ]
                    |
                    v (Batch GPU Producer)
[ PyTorch DINOv3 Feature Encoder ]
```

### Core Components:
- **`DisruptorEngine`**: Lock-free single-producer multi-consumer ring buffer with 64-byte cache-line padding to eliminate CPU false sharing.
- **`MappedStripe` & `push_stripe_to_ring`**: Memory-maps raw PDS3 raster segments without copying data into Python memory.
- **`NACTransformer`**: Cython/C++ tensor extractor performing on-the-fly invalid-pixel masking (filtering $DN \le -32752$), scaling, and batch normalization directly into contiguous tensor memory.

---

## 2. Pithos Vector Index Integration (`pithosdb`)

For fast similarity screening over millions of image tiles, `luna` integrates with the external **[pithosdb](https://pypi.org/project/pithosdb/)** vector database:

```bash
pip install pithosdb
```

### How LUNA Uses Pithos:
- **Vector Storage (`luna.storage.PithosStore`)**: Collects 384-dimensional DINOv3 float32 embeddings during tile ingestion and compiles an off-heap `.bin` index.
- **Matryoshka Cascade Tiers**: Evaluates binary Hamming distances across progressive tiers ($64 \to 128 \to 256 \to 384$ bits) to discard background terrain early.
- **Multi-Family Resonant Voting**: Queries reference pit template families and accumulates consensus scores to highlight morphological anomalies.

---

## 3. Spatial Non-Maximum Suppression (NMS)

Raw screening candidates are clustered and filtered using spherical Haversine surface distance on the Moon:

```python
from luna.pipeline import apply_lunar_spatial_nms

# Suppress overlapping hits within 150 meters on the lunar surface
filtered_hits = apply_lunar_spatial_nms(raw_hits, distance_threshold_meters=150.0)
```


