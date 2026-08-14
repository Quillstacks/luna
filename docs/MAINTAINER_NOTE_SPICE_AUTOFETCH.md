# Maintainer Note: SPICE Kernel Auto-Resolution in `ground_to_image`

**Date:** 2026-08-14  
**Component:** `luna.io.spice_project` & `luna.io.kernel_fetch`  
**Target:** NAIF SPICE Ground-to-Image Ray-Tracing Pipeline  

---

## 1. Problem Statement

When performing multi-temporal ground-to-image projections across multiple LRO mission years (e.g. 2009–2026):

1. **`SPKINSUFFDATA` on Cache Misses:**  
   `luna.io.spice_project.ground_to_image` previously called `furnish_kernels_for_date(start_dt)` directly. If the SPK (`.bsp`) or CK (`.bc`) kernels for that specific observation year were not already downloaded locally to `data/spice/lro`, CSPICE threw `SPICE(SPKINSUFFDATA) -- Insufficient ephemeris data has been loaded`.
2. **Asymmetric Kernel Loading:**  
   While `ensure_kernels_for_label` performed on-demand downloading via `ensure_kernels_for_date`, direct calls to `ground_to_image` assumed all kernels were already present on disk.

---

## 2. Implemented Improvement

In `luna/io/spice_project.py`, `ground_to_image` now invokes `ensure_kernels_for_date` prior to furnishing kernels:

```python
label_path = Path(label_path)
lbl = pvl.load(str(label_path))
start_dt = lbl["START_TIME"].replace(tzinfo=None)

from .kernel_fetch import ensure_kernels_for_date
ensure_kernels_for_date(start_dt, root=kernel_root)
furnish_kernels_for_date(start_dt, kernel_root)
```

---

## 3. Key Benefits

1. **Zero-Configuration Ray-Tracing:**  
   Downstream callers can invoke `ground_to_image("path/to/product.IMG", lon, lat, alt)` on arbitrary NAC images from any mission phase without manually managing kernel downloads.
2. **Thread-Safe & Idempotent:**  
   `ensure_kernels_for_date` skips already-downloaded kernels in $O(1)$ disk stat checks and only contacts NASA NAIF when missing files are detected.
3. **Handle Table Protection:**  
   Pairs seamlessly with `furnish_kernels_for_date`, which clears the SPICE pool (`sp.kclear()`) when switching mission years to prevent CSPICE DAF handle table overflow.
