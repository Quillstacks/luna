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

# Initialize detection pipeline
pipeline = LunaPipeline.from_pretrained(HF_REPO_ID)
```

---

## 2. Ingesting the Product

```python
product_id = "M1118880788RC"

# Download and parse image data
lbl_path, img_path = fetch_nac(product_id)
nac_img = read_nac(img_path, geometry=True)
print(f"Ingested {product_id} with {nac_img.lines} lines.")
```

---

## 3. Screening for Pit Candidates

```python
# Execute streaming candidate screening
candidates = pipeline.scan_product(
    product_id=product_id,
    top_k=20,
    min_dist_px=512.0
)

print(f"Identified {len(candidates)} high-probability candidates.")
for c in candidates:
    print(f"Candidate at sample {c.sample}, line {c.line} -> Confidence: {c.score:.4f}")
```

---

## 4. Validating with Ground Coordinates

For any candidate, verify ground-truth alignment with SPICE geometry:

```python
top_hit = candidates[0]
sample, line = ground_to_image(lbl_path, top_hit.lon, top_hit.lat, alt_m=-2010.0)

print(f"Predicted Pixel: ({top_hit.sample}, {top_hit.line})")
print(f"SPICE Projected: ({sample:.1f}, {line:.1f})")
```
