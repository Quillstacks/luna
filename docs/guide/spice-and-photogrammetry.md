# SPICE Photogrammetry & Orbital Geometry

`luna.io.spice_project` implements a pure-Python pushbroom camera model for LROC NAC instruments. It replaces USGS CSM and ISIS `campt` ray-tracing with direct `spiceypy` routines, running at sub-pixel accuracy with zero external runtime dependencies.

---

## 1. LROC Pushbroom Camera Model

The LROC Narrow Angle Camera (NAC-L and NAC-R) is a **line-scan pushbroom sensor**:

- **Focal length**: $f \approx 700\text{ mm}$ ($699.62\text{ mm}$ for NAC-L, $701.57\text{ mm}$ for NAC-R).
- **Linear Detector**: 5064 samples wide $\times$ 1 line.
- **Pixel Pitch**: $0.007\text{ mm}$ ($142.857\text{ px/mm}$).
- **Detector Line Condition**: The detector is located along $x_{\text{focal}} = 0$ in the camera frame. Each image row corresponds to a distinct Ephemeris Time (ET) snapshot as the spacecraft sweeps along its orbit.

```
       Orbit Track (Flight Direction)
              |
              v   ET_0 (Line 0)
       [============== Linear Detector (5064 samples) ===============]
              |
              v   ET_k (Line k)
       [==============================================================]
              |
              v   ET_N (Line H-1)
```

---

## 2. Ground-to-Image & Image-to-Ground Algorithms

To map a ground coordinate $(\lambda, \phi, alt)$ to detector pixel $(s, l)$ in the Mean-Earth/Polar-Axis body-fixed frame (`MOON_ME`, configurable via `LUNA_BODY_FRAME`):

1. **Body-Fixed Coordinate Conversion**:
   Convert longitude $\lambda$ (rad), latitude $\phi$ (rad), and optional elevation $alt$ (km) into the body-fixed `MOON_ME` rectangular vector $\vec{G}_{bf}$ via planetocentric spherical conversion:

$$\vec{G}_{bf} = \text{sp.latrec}(R_{\text{moon}} + alt, \lambda, \phi)$$

2. **Look Vector Formulation**:
   For any trial Ephemeris Time $t \in [t_{\text{start}}, t_{\text{stop}}]$, rotate $\vec{G}_{bf}$ into J2000 and compute the spacecraft look vector:

$$\vec{r}_{\text{look, J2000}}(t) = \mathbf{R}_{\text{MOON\_ME} \to \text{J2000}}(t) \cdot \vec{G}_{bf} - \vec{p}_{\text{sc, J2000}}(t)$$

3. **Camera Frame Projection**:
   Rotate the look vector into the camera frame:

$$\vec{u}_{\text{cam}}(t) = \mathbf{R}_{\text{J2000} \to \text{CAM}}(t) \cdot \vec{r}_{\text{look, J2000}}(t)$$

4. **1D Root Finding (Bisection)**:
   The detector condition requires the camera $x$-component to vanish:

$$\text{Residual}(t) = \frac{u_{\text{cam}, x}(t)}{\|\vec{u}_{\text{cam}}(t)\|} = 0$$

   Because the orbit progresses monotonically over the exposure window ($\sim 10\text{--}30\text{ s}$), bisection converges to $< 10^{-9}\text{ s}$ in $\sim 30$ iterations.

5. **Pixel Coordinates (`ground_to_image`)**:
   At the solved time $t^*$:

$$y_{\text{focal}} = f \cdot \frac{u_{\text{cam}, y}(t^*)}{u_{\text{cam}, z}(t^*)}$$

$$\text{Sample} = \text{BoresightSample} + \text{px\_per\_mm} \cdot y_{\text{focal}}$$

$$\text{Line} = \frac{t^* - t_{\text{start}}}{\text{LineRate}}$$

6. **Analytical Ray-Sphere & Iterative DEM Intersection (`image_to_ground`)**:
   Given pixel $(s, l)$ at $t = t_{\text{start}} + l \cdot \text{LineRate}$, the camera ray $\vec{d}_{\text{bf}}$ in `MOON_ME` is intersected analytically with the sphere of radius $R = R_{\text{moon}} + h$:

$$\|\vec{P}_{\text{sc}} + \tau \vec{d}_{\text{bf}}\|^2 = (R_{\text{moon}} + h)^2$$

   When an optional `LunarDEM` is supplied, $h$ is iteratively updated from the DEM at the current intersection until $|\Delta h| < 1\text{ m}$.

---

## 3. SPICE Kernel Pool Management & DAF Handle Limit

!!! danger "CSPICE DAF Handle Limit (`SpiceDAFNOSUCHHANDLE: handle = 0`)"
    NAIF CSPICE maintains an internal Double Precision Array File (DAF) handle table limited to $\sim 100$ active file segments. When processing multi-year missions, calling a global `furnish_kernels()` loads dozens of `.bc` and `.bsp` files, quickly exceeding this table and causing hard crashes.

### The Solution: Targeted On-Demand Loading

`luna` provides two kernel loading strategies:

1. **`furnish_kernels_for_date(dt, kernel_root)`**:
   - Clears the SPICE pool (`sp.kclear()`) if the current pool does not cover `dt`.
   - Furnishes untimed base kernels (`.tls`, latest `.tsc`, `.tpc`, `.bpc`, `.tf`, `.ti`, `de421.bsp`).
   - Selectively loads **only the exact `.bsp` (orbit) and spacecraft/LROC `.bc` (attitude) kernels** covering the timestamp `dt`.
   - Caches the active time window $[t_{\text{start}}, t_{\text{stop}}]$, making consecutive queries on the same image zero-cost.

2. **`furnish_kernels(kernel_root)`**:
   - Preserved for high-throughput batch workloads within a single mission phase.

3. **Automated On-Demand Fetching**:
   - `ground_to_image` and `image_to_ground` automatically invoke `ensure_kernels_for_date` (when `ensure_kernels=True`) before furnishing, querying the NAIF archive only for missing kernels and raising `MissingKernelError` if SPICE lacks coverage for the epoch.

```python
from datetime import datetime, timezone
from luna.io.spice_project import furnish_kernels_for_date

# Loads base kernels + only the April 2012 LRO attitude/ephemeris
dt = datetime(2012, 4, 15, 14, 30, tzinfo=timezone.utc)
loaded_count = furnish_kernels_for_date(dt)
```

---

## 4. Optional Topography & DEM Integration (`alt_m` & `LunarDEM`)

Elevation and DEM inputs are **completely optional** — omitting both `alt_m` and `dem` projects onto the $1737.4\text{ km}$ lunar reference sphere ($alt = 0\text{ m}$).

For off-nadir NAC observations (emission angle $> 5^\circ$), passing an explicit `alt_m` or a `LunarDEM` instance eliminates horizontal parallax displacement:

```
 Spacecraft Look Ray
      \
       \   (Assumed sphere alt=0) ----> Parallax error delta_x ~ 500 px
        \
=========\============================= Datum (alt = 0 m)
          \
           * (Actual terrain alt = -2010 m)
```

```python
from luna.io import LunarDEM, fetch_dem_for_region, ground_to_image, image_to_ground

# 1. Default reference sphere (no DEM or elevation required)
s_sphere, l_sphere = ground_to_image(label_path, lon, lat)

# 2. Explicit scalar elevation (meters relative to 1737.4 km sphere)
s_true, l_true = ground_to_image(label_path, lon, lat, alt_m=-2010.0)

# 3. Automated SLDEM2015 tile lookup via LunarDEM
tiles = fetch_dem_for_region(min_lat=13.5, max_lat=14.5, min_lon=302.5, max_lon=304.0)
with LunarDEM(tiles) as dem:
    s_dem, l_dem = ground_to_image(label_path, lon, lat, dem=dem)
    lon_back, lat_back = image_to_ground(label_path, s_dem, l_dem, dem=dem)
```

---

## 5. Ascending vs. Descending Orbit Handling

LRO operates in a near-circular polar orbit, taking both descending (North-to-South) and ascending (South-to-North) passes:

- **Descending pass**: Image Line 0 is at the Northern edge ($\phi_{\text{start}} > \phi_{\text{stop}}$).
- **Ascending pass**: Image Line 0 is at the Southern edge ($\phi_{\text{start}} < \phi_{\text{stop}}$).

`LinearProjection.from_nac_geometry` inspects `start_lat` vs. `stop_lat` and corner latitudes to determine orbit direction automatically. If GIS bounding-box corners (North-first) are provided for an ascending pass, `LinearProjection` flips the line mapping so that Line 0 is consistently attached to the start of the sensor exposure.
