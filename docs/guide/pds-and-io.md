# PDS Data Ingestion & I/O

`luna.io` provides high-throughput tools for indexing, downloading, and decoding LROC NAC products directly from the NASA Planetary Data System (PDS) archive.

---

## 1. PDS Binary-Search Index Resolver (`PDSIndex`)

The LROC CDR archive is distributed across $\sim 65+$ volumes (`LROLRC_1001` through `LROLRC_1065+` on the NASA MCP PDS4 mirror and JPL/ASU mirrors), containing millions of entries in fixed-width `INDEX.TAB` files.

`PDSIndex` resolves any product ID to its full PDS URL in **$< 50\text{ ms}$** via HTTP Range Requests:

1. **Dynamic Volume Discovery & Characterization**: Discovers available volumes (including new releases via JPL directory listing) and fetches the first and last `PRODUCT_ID` of each volume index to determine bounding intervals.
2. **Binary Search**: Performs logarithmic binary search over byte ranges in the matching volume's `INDEX.TAB`.
3. **CSV-First Geometry Parsing**: Parses 21 observation geometry fields via CSV column positions first, falling back to fixed-width byte slices (`GEOMETRY_SLICES`) if needed.
4. **HTTP 429 Resiliency**: Range requests include automated exponential backoff (`1.0 * 2^attempt`) and `urllib3.util.Retry` adapters configured for status codes `429`, `500`, `502`, `503`, and `504`.
5. **Local Cache**: Caches volume boundaries in `data/pds_volume_index_cdr.json`.

```python
from luna.io.pds_index import PDSIndex

idx = PDSIndex()

# Instant lookup
url = idx.url_for("M1118880788RC")
print(f"PDS URL: {url}")

# Fetch full geometry metadata row
geom = idx.geometry_for("M1118880788RC")
print(f"Emission angle: {geom['emission_angle']} deg, Resolution: {geom['resolution']} m/px")
```

---

## 2. Resilient Product Fetching (`fetch_nac`)

`fetch_nac` downloads the calibrated image raster (`.IMG`) to local disk with streaming chunk validation, atomic renaming, and network retry handling:

* **Atomic Writes**: Downloads stream into a temporary `.IMG.tmp` file and atomically replace it via `Path.replace()` once fully written, preventing corrupted partial files on disk.
* **Direct URL Bypass**: When a direct archive URL is already known (e.g., from QuickMap metadata queries), pass `url=...` to bypass `INDEX.TAB` index lookups entirely.
* **Session Pooling & User-Agents**: Uses `get_pds_session()` to maintain keep-alive connection pools and standard browser headers (overridable via `LUNA_USER_AGENT` environment variable or parameter).
* **Bandwidth Limiter**: Optional `max_bandwidth_mbps` parameter enforces transfer speed limits via token-bucket algorithm to prevent saturating shared connections.
* **Transient Error Retry**: Retries automatically on `ConnectionError`, `Timeout`, `ChunkedEncodingError`, and `HTTPError` (including `429 Too Many Requests` and `503 Service Unavailable`).

```python
from luna.io.pds_fetch import fetch_nac, get_pds_session

# Standard fetch with binary index lookup
img_path = fetch_nac("M1118880788RC", dest_dir="data/_scratch")

# High-throughput fetch with explicit direct URL and 15 MB/s bandwidth limit
direct_url = "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0/LROLRC_1005/DATA/SCI/2010295/NAC/M142394466RC.IMG"
img_path = fetch_nac(
    "M142394466RC",
    dest_dir="data/_scratch",
    url=direct_url,
    max_bandwidth_mbps=15.0,
)
```

---

## 3. Fast NAC Decoding (`read_nac`)

`read_nac` parses PDS3 labels and loads raw 16-bit signed integer DN values into float32 calibrated $I/F$ reflectance:

$$I/F = \text{DN} \cdot \text{SCALING\_FACTOR} + \text{OFFSET}$$

Null pixels and instrument saturations ($\le -32768$) are converted to `np.nan`:

```python
import numpy as np
from luna.io.nac_reader import read_nac

nac = read_nac("data/_scratch/M1118880788RC.IMG", geometry=True)

print(f"Lines: {nac.lines}, Samples: {nac.samples}")
print(f"Center: ({nac.center_lon:.3f} deg, {nac.center_lat:.3f} deg)")
print(f"Valid pixel ratio: {np.isfinite(nac.pixels).mean() * 100:.1f}%")
```

---

## 4. Optional SLDEM2015 Terrain Fetching (`luna.io.dem`)

`luna.io.dem` provides optional on-demand downloading and window-based querying of NASA LOLA + Kaguya SLDEM2015 tiles (`512 ppd`, `~59 m/px`):

* **Tile Resolution (`get_sldem2015_tile_name`)**: Maps planetocentric `(lat_deg, lon_deg)` in `[-60, +60]` latitude to 30x45 degree SLDEM2015 tile stems.
* **GDAL Sidecar Scale Validation (`fetch_dem_for_region`, `LunarDEM`)**: Downloads both the `.JP2` raster and its `.JP2.aux.xml` GDAL sidecar (`<Scale>0.5</Scale>`), verifying the `0.5 m/DN` vertical scaling factor on open.
* **Singleton Convenience (`default_dem`)**: Opens all tiles under `data/dem` (or `LUNA_DEM_DIR`) as a shared `LunarDEM` instance.
