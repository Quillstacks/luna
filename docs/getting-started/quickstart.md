# Quickstart Guide

This tutorial covers the primary workflows in `luna`: fetching orbital imagery, projecting coordinates with SPICE photogrammetry, running multi-scale feature screening, and inspecting detections.

---

## 1. On-Demand Image Fetching & Projection

NASA PDS LROC NAC images are downloaded and cached automatically, and `ground_to_image` / `image_to_ground` operate in the `MOON_ME` frame with optional scalar elevation (`alt_m`) or `LunarDEM` terrain models:

```python
from luna.io.dem import LunarDEM, fetch_dem_for_region
from luna.io.nac_reader import read_nac
from luna.io.pds_fetch import fetch_nac
from luna.io.spice_project import ground_to_image, image_to_ground

# 1. Fetch NAC CDR (.IMG)
img_path = fetch_nac("M1118880788RC", dest_dir="data/_scratch")

# 2. Read into memory with geometry metadata
nac_img = read_nac(img_path, geometry=True)
print(f"Loaded {nac_img.product_id}: shape = {nac_img.lines} x {nac_img.samples} px")

# 3a. Forward Ground-to-Image Projection on the 1737.4 km reference sphere (default)
s_sphere, l_sphere = ground_to_image(img_path, lon_deg=303.228, lat_deg=14.091)

# 3b. Forward Projection with explicit scalar elevation (Marius Hills Pit: alt = -2010 m)
sample, line = ground_to_image(
    img_path,
    lon_deg=303.228,
    lat_deg=14.091,
    alt_m=-2010.0,
)
print(f"Ground point mapped to detector: sample={sample:.1f}, line={line:.1f}")

# 3c. Forward & Reverse Projection with optional SLDEM2015 LunarDEM
tiles = fetch_dem_for_region(min_lat=13.5, max_lat=14.5, min_lon=302.5, max_lon=304.0)
with LunarDEM(tiles) as dem:
    s_dem, l_dem = ground_to_image(img_path, lon_deg=303.228, lat_deg=14.091, dem=dem)
    lon, lat = image_to_ground(img_path, sample=s_dem, line=l_dem, dem=dem)
    print(f"Reverse projected with DEM: lon={lon:.4f} deg, lat={lat:.4f} deg")
```

---

## 2. SPICE-First ROI Extraction

Extract calibrated regions of interest (ROI) centered on lunar coordinates. `get_image_of_roi` prioritizes exact SPICE `ground_to_image` projection (with optional `alt_m` or `dem`) and falls back to bilinear corner interpolation (`LinearProjection`) only when SPICE kernels are unavailable:

```python
from luna.io.projection import get_image_of_roi

# Crop a 512x512 tile around target pit coordinates using SPICE projection
tile = get_image_of_roi(
    product_id="M1118880788RC",
    lat=14.091,
    lon=303.228,
    width=512,
    height=512,
    use_spice=True,
    alt_m=-2010.0,
)

# Output is a float32 I/F reflectance raster (lines, samples)
print(f"Tile shape: {tile.shape}, mean I/F: {tile.mean():.4f}")
```

---

## 3. Candidate Screening Pipeline

Screen entire orbital strips using the `LunaPipeline`:

```python
from pathlib import Path
from luna import LunaPipeline
from luna.config import HF_REPO_ID

# Initialize pre-trained multi-stage detection pipeline
pipeline = LunaPipeline.from_pretrained(HF_REPO_ID)

# Run full scan over a NAC strip
hits = pipeline.scan(
    product_ids="M1118880788RC",
    query_dir="data/_scratch/pits/",
    top_k=50,
    min_dist_px=512.0
)

for hit in hits[:5]:
    print(f"Rank {hit.rank}: Score = {hit.score:.3f} | Offset=({hit.x_offset}, {hit.y_offset}) px | Coords=({hit.lon:.3f} deg, {hit.lat:.3f} deg)")
```

---

## 4. Local Documentation Server

Launch the documentation server locally to inspect live API docs and interactive guides:

```bash
mkdocs serve
```
Visit `http://localhost:8000` in your browser.
