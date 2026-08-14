# Screening & Pithos Engine

Processing global LROC NAC coverage requires evaluating $> 80,000$ images (each $\sim 5000 \times 52000$ pixels, $> 500\text{ MB}$). `luna.screening` implements a zero-copy, lock-free streaming architecture capable of screening an entire NAC in **$< 15\text{ seconds}$**.

---

## 1. Lock-Free Disruptor Ring Buffer

The `DisruptorEngine` decouples disk I/O and tile extraction from GPU batch inference:

```
[ Disk / Memory-Mapped File ]
          |
          v (Zero-copy stripe slice)
[ Disruptor Ring Buffer (Lock-Free) ]
          |
          v (Batch producer)
[ GPU Tensor Batch Engine ]
```

- **Thread-safe**: Lock-free atomic sequence pointers.
- **Batched I/O**: Slices image rows directly into pinned memory buffers.

---

## 2. Native Pithos Binary Retrieval Index

`Pithos` is a C++ SIMD-accelerated binary embedding index:

1. **Embedding Binarization**:
   Continuous 1024-dimensional DINOv3 vectors $\vec{v}$ are mapped to 1024-bit binary hashes:

   $$b_i = \begin{cases} 1 & \text{if } v_i \ge 0 \\ 0 & \text{if } v_i < 0 \end{cases}$$

2. **SIMD POPCOUNT Hamming Distance**:
   Calculates bitwise distance via hardware `_mm_popcnt_u64`:

   $$d_H(\vec{a}, \vec{b}) = \text{popcount}(\vec{a} \oplus \vec{b})$$

3. **Multi-Query Resonant Voting**:
   Queries a set of known pit template anchors and accumulates votes across morphological variants.

```python
from luna.screening.pithos import PithosMIDB

with PithosMIDB() as db:
    db.load_index("M1118880788RC", "data/indices/pithos_M1118880788RC.bin")
    ids, dists = db.batch_search("M1118880788RC", query_vectors, k=1000)
```

---

## 3. Spatial Non-Maximum Suppression (NMS)

Candidates are clustered using spatial Euclidean distance in detector pixel space ($512\text{ px}$ radius) to eliminate redundant detections across sliding window steps:

```python
nms_hits = pipeline._nms(
    ranked_indices,
    vote_map,
    best_dist,
    metadata,
    top_k=100,
    min_dist_px=512.0
)
```
