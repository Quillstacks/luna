# Screening & Pithos Engine

`luna.screening` implements a high-throughput streaming architecture to parallelize tile extraction and search across large LROC NAC images ($\sim 5064 \times 52224$ pixels).

---

## 1. Tile Slicing & Ingestion Pipeline

The `DataIngestor` slices full-resolution LROC NAC strips into overlapping tiles ($256 \times 256\text{ px}$ with stride $192\text{ px}$):

```
[ LROC NAC PDS3 Raster (~500 MB) ]
          │
          ▼ (Strided 256x256 window)
[ Dynamic Batched Tensor Generator ]
          │
          ▼ (PyTorch DINOv3 ViT-S/16)
[ 384-dimensional Float32 Embeddings ]
          │
          ▼ (Sequential ID mapping)
[ Pithos PLAN Binary Index (.bin + _meta.pkl) ]
```

---

## 2. Native Pithos Model-Isomorphic Database (MIDB)

`Pithos` is an AOT-compiled native C/C++ library for SIMD-accelerated binary embedding indexing and search:

1. **PolarQuant-Hadamard Transform**:
   Continuous 384-dimensional DINOv3 vectors $\vec{v}$ are transformed via a randomized sign vector $\vec{d}$ and the $384 \times 384$ Kronecker-Hadamard matrix $H_{384} = H_{12} \otimes H_{32}$:

$$\vec{u} = \frac{1}{\sqrt{384}} H_{384} (\vec{v} \odot \vec{d})$$

$$b_i = \begin{cases} 1 & \text{if } u_i \ge 0 \\ 0 & \text{if } u_i < 0 \end{cases}$$

2. **SIMD POPCOUNT Hamming Distance**:
   Calculates bitwise Hamming distance across 384 bits (6 64-bit words) via hardware `_mm_popcnt_u64`:

$$d_H(\vec{a}, \vec{b}) = \sum_{k=0}^{5} \text{popcount}(a_k \oplus b_k)$$

3. **Matryoshka Cascade Tiers**:
   Evaluates distances in progressive tier stages ($64 \to 128 \to 256 \to 384$ bits) to early-exit non-matching background terrain.

4. **Multi-Family Resonant Voting**:
   Queries multiple morphological pit template families simultaneously and accumulates votes across matching candidates.

### Python API Example

```python
from luna.screening.pithos import PithosMIDB

# Access singleton Pithos instance
db = PithosMIDB()

# Search nearest neighbors against an index
db.open_index("data/_scratch/indices/pithos_M1118880788RC.bin")
results = db.search_knn(query_vectors_float32, k=1000)
```

---

## 3. Spatial Non-Maximum Suppression (NMS)

Candidate hits are clustered using lunar surface distance to eliminate redundant overlapping detections:

```python
from luna.pipeline import apply_lunar_spatial_nms

# Apply spherical Haversine NMS on the Moon (radius = 150m)
nms_hits = apply_lunar_spatial_nms(raw_hits, distance_threshold_meters=150.0)
```

