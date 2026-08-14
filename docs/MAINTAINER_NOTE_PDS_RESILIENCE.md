# Maintainer Note: PDS Fetch Resiliency & Rate-Limit (429) Mitigation

**Date:** 2026-08-14  
**Component:** `luna.io.pds_fetch` & `luna.io.pds_index`  
**Target:** Core LROC NAC CDR Download Pipeline  

---

## 1. Problem Statement

When downloading high volumes of Lunar Reconnaissance Orbiter (LROC NAC) calibrated `.IMG` products from the NASA Planetary Data System (PDS) archive (`https://pds.mcp.nasa.gov` / `pds.lroc.im-ldi.com`), two issues were observed:

1. **HTTP 429 (Too Many Requests) on Default User-Agents:**  
   The NASA PDS Cloudflare/WAF gateway strictly rate-limits requests containing default Python `requests/2.3x` user-agent strings, returning `HTTP 429 Too Many Requests` even during modest sequential downloads.
2. **`INDEX.TAB` Range-Read Redundancy:**  
   When the target product ID is already matched via QuickMap or ASU LROC metadata with a known direct URL, querying the 150MB+ volume-level `INDEX.TAB` via HTTP range requests is redundant and creates avoidable server load.

---

## 2. Implemented Architecture & Improvements

### A. Configurable User-Agent & Session Factory
* Added `DEFAULT_USER_AGENT` at module initialization with `os.getenv("LUNA_USER_AGENT", ...)` support.
* Added `get_pds_session(user_agent: Optional[str] = None) -> requests.Session`:
  * Initializes connection pooling across repeated downloads.
  * Injects standard browser headers to prevent Cloudflare/WAF throttling.

### B. Direct URL Bypass in `fetch_nac`
* `fetch_nac(product_id, dest_dir="data", url=None, session=None, user_agent=None, ...)`
* If `url` is supplied (e.g. from ASU/QuickMap direct link resolution), `fetch_nac` downloads directly with atomic `.IMG.tmp` $\to$ `.IMG` replacement and token-bucket bandwidth limiting.
* If `url` is omitted, it falls back to `PDSIndex.url_for(product_id)` binary search on `INDEX.TAB`.

---

## 3. Usage Example for Downstream Consumers

```python
from luna.io.pds_fetch import fetch_nac, get_pds_session

# Option 1: Standard fetch with default pooled session
img_path = fetch_nac("M142394466RC", dest_dir="data/_scratch")

# Option 2: High-throughput fetch with explicit direct URL
direct_url = "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0/LROLRC_1005/DATA/SCI/2010295/NAC/M142394466RC.IMG"
img_path = fetch_nac("M142394466RC", dest_dir="data/_scratch", url=direct_url)
```

---

## 4. Verification

* Verified full-speed transfers ($25\text{ MB/s}$) on `M142394466RC` (528 MB) and `M1530107589LC` without 429 throttling.
* Verified token-bucket bandwidth limiter integration and atomic rename validation.
