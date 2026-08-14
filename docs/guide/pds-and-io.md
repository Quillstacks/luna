# PDS Data Ingestion & I/O

`luna.io` provides tools for indexing, downloading, and decoding LROC NAC products directly from the NASA Planetary Data System (PDS) archive.

---

## 1. PDS Binary-Search Index Resolver (`PDSIndex`)

The LROC CDR archive is distributed across $\sim 35$ volumes (`LROLRC_1001` through `LROLRC_1035`), containing millions of entries in fixed-width `INDEX.TAB` files.

`PDSIndex` resolves any product ID to its full PDS URL in **$< 50\text{ ms}$** via HTTP Range Requests:

1. **Volume Characterization**: Fetches the first and last `PRODUCT_ID` of each volume index to determine bounding intervals.
2. **Binary Search**: Performs logarithmic binary search over byte ranges in the matching volume's `INDEX.TAB`.
3. **Local Cache**: Caches volume boundaries in `data/pds_volume_index_cdr.json`.

```python
from luna.io.pds_index import PDSIndex

idx = PDSIndex()

# Instant lookup
url = idx.url_for("M1118880788RC")
print(f"PDS URL: {url}")

# Fetch full geometry metadata row
geom = idx.geometry_for("M1118880788RC")
print(f"Emission angle: {geom['emission_angle']}°, Resolution: {geom['resolution']} m/px")
```

---

## 2. On-Demand CDR/EDR Fetching (`fetch_nac`)

`fetch_nac` downloads the image raster (`.IMG`) and accompanying label (`.LBL` or detached headers) to local disk with checksum caching:

```python
from luna.io.pds_fetch import fetch_nac

# Downloads both .IMG and .LBL to data/_scratch/ if not already present
lbl_path, img_path = fetch_nac("M1118880788RC", dest_dir="data/_scratch")
```

---

## 3. Fast NAC Decoding (`read_nac`)

`read_nac` parses PDS3 labels and loads raw 16-bit signed integer DN values into float32 calibrated $I/F$ reflectance:

$$I/F = \text{DN} \cdot \text{SCALING\_FACTOR} + \text{OFFSET}$$

Null pixels and instrument saturations ($\le -32768$) are converted to `np.nan`:

```python
from luna.io.nac_reader import read_nac

nac = read_nac("data/_scratch/M1118880788RC.IMG", geometry=True)

print(f"Lines: {nac.lines}, Samples: {nac.samples}")
print(f"Center: ({nac.center_lon:.3f}°, {nac.center_lat:.3f}°)")
print(f"Valid pixel ratio: {np.isfinite(nac.pixels).mean() * 100:.1f}%")
```
