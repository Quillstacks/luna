# Tutorial: End-to-End Pit Detection

In this tutorial, we will execute a complete scan over an LROC NAC image covering the Marius Hills region to identify collapse skylights.

---

## 1. Setup & Imports

```python
from pathlib import Path
from luna import LunaPipeline
from luna.io.pds_fetch import fetch_nac
from luna.io.nac_reader import read_nac
from luna.io.spice_project import ground_to_image
from luna.config import HF_REPO_ID

# Initialize detection pipeline with fine-tuned weights
pipeline = LunaPipeline.from_pretrained(HF_REPO_ID)
```

---

## 2. Ingesting the Product

```python
product_id = "M1118880788RC"

# Download and parse image data
img_path = fetch_nac(product_id, dest_dir="data/_scratch")
nac_img = read_nac(img_path, geometry=True)
print(f"Ingested {product_id}: {nac_img.samples}x{nac_img.lines} px, resolution={nac_img.resolution_m} m/px.")
```

---

## 3. Screening for Pit Candidates

```python
# Execute candidate screening against reference pit templates
hits = pipeline.scan(
    product_ids=product_id,
    query_dir="data/_scratch/pits/",
    top_k=20,
    min_dist_px=512.0
)

print(f"Identified {len(hits)} high-probability candidates.")
for hit in hits:
    print(f"Rank {hit.rank}: ({hit.lon:.4f}°, {hit.lat:.4f}°) | Offset: ({hit.x_offset}, {hit.y_offset}) px | Score: {hit.score:.2f}")
```

---

## 4. Validating with SPICE Ground Coordinates

For any candidate, verify ground-truth alignment with SPICE pushbroom geometry:

```python
top_hit = hits[0]

# Compute detector position with localized terrain elevation alt_m
sample, line = ground_to_image(img_path, top_hit.lon, top_hit.lat, alt_m=-2010.0)

print(f"Detected Pixel Offset : ({top_hit.x_offset}, {top_hit.y_offset})")
print(f"SPICE Ray Intersection: ({sample:.1f}, {line:.1f})")
```

