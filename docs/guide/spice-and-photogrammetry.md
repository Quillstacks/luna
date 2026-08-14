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

## 2. Ground-to-Image Algorithm (`ground_to_image`)

To map a ground coordinate $(\lambda, \phi, alt)$ to detector pixel $(s, l)$:

1. **Body-Fixed Coordinate Conversion**:
   Convert longitude $\lambda$ (rad), latitude $\phi$ (rad), and elevation $alt$ (km) into the body-fixed `IAU_MOON` rectangular vector $\vec{G}_{bf}$ via planetocentric spherical conversion:

   $$\vec{G}_{bf} = \text{sp.latrec}(R_{\text{moon}} + alt, \lambda, \phi)$$

2. **Look Vector Formulation**:
   For any trial Ephemeris Time $t \in [t_{\text{start}}, t_{\text{stop}}]$, rotate $\vec{G}_{bf}$ into J2000 and compute the spacecraft look vector:

   $$\vec{r}_{\text{look, J2000}}(t) = \mathbf{R}_{\text{IAU\_MOON} \to \text{J2000}}(t) \cdot \vec{G}_{bf} - \vec{p}_{\text{sc, J2000}}(t)$$

3. **Camera Frame Projection**:
   Rotate the look vector into the camera frame:

   $$\vec{u}_{\text{cam}}(t) = \mathbf{R}_{\text{J2000} \to \text{CAM}}(t) \cdot \vec{r}_{\text{look, J2000}}(t)$$

4. **1D Root Finding (Bisection)**:
   The detector condition requires the camera $x$-component to vanish:

   $$\text{Residual}(t) = \frac{u_{\text{cam}, x}(t)}{\|\vec{u}_{\text{cam}}(t)\|} = 0$$

   Because the orbit progresses monotonically over the exposure window ($\sim 10\text{--}30\text{ s}$), bisection converges to $< 10^{-9}\text{ s}$ in $\sim 30$ iterations.

5. **Pixel Coordinates**:
   At the solved time $t^*$:

   $$y_{\text{focal}} = f \cdot \frac{u_{\text{cam}, y}(t^*)}{u_{\text{cam}, z}(t^*)}$$
   $$\text{Sample} = \text{BoresightSample} + \text{px\_per\_mm} \cdot y_{\text{focal}}$$
   $$\text{Line} = \frac{t^* - t_{\text{start}}}{\text{LineRate}}$$

---

## 3. SPICE Kernel Pool Management & DAF Handle Limit

!!! danger "CSPICE DAF Handle Limit (`SpiceDAFNOSUCHHANDLE: handle = 0`)"
    NAIF CSPICE maintains an internal Double Precision Array File (DAF) handle table limited to $\sim 100$ active file segments. When processing multi-year missions, calling a global `furnish_kernels()` loads dozens of `.bc` and `.bsp` files, quickly exceeding this table and causing hard crashes.

### The Solution: Targeted On-Demand Loading

`luna` provides two kernel loading strategies:

1. **`furnish_kernels_for_date(dt, kernel_root)`**:
   - Clears the SPICE pool (`sp.kclear()`) if the current pool does not cover `dt`.
   - Furnishes untimed base kernels (`.tls`, `.tsc`, `.tpc`, `.bpc`, `.tf`, `.ti`, `de421.bsp`).
   - Selectively loads **only the exact `.bsp` (orbit) and `.bc` (attitude) kernels** covering the timestamp `dt`.
   - Caches the active time window $[t_{\text{start}}, t_{\text{stop}}]$, making consecutive queries on the same image zero-cost.

2. **`furnish_kernels(kernel_root)`**:
   - Preserved for high-throughput batch workloads within a single mission phase.

```python
from datetime import datetime, timezone
from luna.io.spice_project import furnish_kernels_for_date

# Loads base kernels + only the April 2012 LRO attitude/ephemeris
dt = datetime(2012, 4, 15, 14, 30, tzinfo=timezone.utc)
loaded_count = furnish_kernels_for_date(dt)
```

---

## 4. Topography & Elevation Parallax (`alt_m`)

For off-nadir NAC observations (emission angle $> 5^\circ$), assuming a spherical Moon ($alt = 0\text{ m}$) introduces significant horizontal parallax displacement (up to **500 pixels** in deep mare basins with $-2\text{ km}$ elevation).

```
 Spacecraft Look Ray
      \
       \   (Assumed sphere alt=0) ----> Parallax error delta_x ~ 500 px
        \
=========\============================= Datum (alt = 0 m)
          \
           * (Actual terrain alt = -2010 m)
```

Pass the LOLA ground elevation in meters to `ground_to_image`:

```python
# With LOLA terrain elevation
s_true, l_true = ground_to_image(label_path, lon, lat, alt_m=-2010.0)

# Without terrain elevation (pure sphere assumption)
s_sphere, l_sphere = ground_to_image(label_path, lon, lat, alt_m=0.0)

print(f"Parallax displacement: {abs(s_true - s_sphere):.1f} px across-track")
```

---

## 5. Ascending vs. Descending Orbit Handling

LRO operates in a near-circular polar orbit, taking both descending (North-to-South) and ascending (South-to-North) passes:

- **Descending pass**: Image Line 0 is at the Northern edge ($\phi_{\text{start}} > \phi_{\text{stop}}$).
- **Ascending pass**: Image Line 0 is at the Southern edge ($\phi_{\text{start}} < \phi_{\text{stop}}$).

`LinearProjection.from_nac_geometry` inspects `start_lat` vs. `stop_lat` and corner latitudes to determine orbit direction automatically. If GIS bounding-box corners (North-first) are provided for an ascending pass, `LinearProjection` flips the line mapping so that Line 0 is consistently attached to the start of the sensor exposure.
