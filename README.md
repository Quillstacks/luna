# luna

Lunar pit segmentation. A Mask R-CNN is trained on ellipse-approximated pit
masks derived from the ASU / Wagner Lunar Pit Atlas (`funnel_max_m`,
`funnel_min_m`, `azimuth_deg`), refined by hand, then run on fresh LROC NAC
frames to surface candidates that a human relabels and feeds back into the
next training round.

```
  LPA catalog + NAC --ellipse--> pseudo-masks --hand-label--> training set
                                                                    |
                                                                    v
                                                              Mask R-CNN train
                                                                    |
                        lon/lat candidates <---- inference on fresh NAC
                                 |
                                 v
                          human relabel -> next training round
```

## ESSA baseline (Le Corre et al. 2025)

Before training anything, validate the inference path against the published
model. ESSA is a `maskrcnn_resnet50_fpn_v2` with 3 classes (bg / skylight /
pit). We do **not** reimplement preprocessing — we run Le Corre's vendored
ISIS3 bash pipeline under [`third_party/`](third_party/) so the bytes ESSA
sees match what it was trained on (radiometric calibration, echo correction,
equirectangular cam2map, cubic-spline downsample to 1.5 m/px).

### One-time setup

```bash
# Project deps (torch, rasterio, ...). Inside your usual venv/conda env.
pip install -e .

# ISIS3 + GDAL — separate conda env, ~3 GB.
# Apple Silicon (arm64) needs the osx-64 subdir; ISIS has no native arm64 build:
CONDA_SUBDIR=osx-64 conda env create -f environments/isis.yml
# Intel Mac / Linux: just `conda env create -f environments/isis.yml`.

# ESSA weights from Zenodo (~552 MB, md5-verified).
python scripts/fetch_essa_weights.py
```

### One pipeline, two modes

```bash
# Smoke test: reproduce MTP detection (LPA id=3 is the gold-standard
# target — paper hits >0.999 on every NAC). 4 tiles around the catalog
# point, ~5 min on CPU.
python scripts/essa_smoke.py

# Full-strip discovery sweep. Slides 2048-px tiles with 50% overlap
# across the whole NAC, dedupes across boundaries. Optional --validate
# checks recall on a known target inside the frame.
python scripts/essa_smoke.py --mode full-strip \
    --product-id M1243133690L \
    --validate-lon 87.599 --validate-lat 58.6979 --label BAP
```

Both modes produce the same artefacts under `data/essa_out/<label>/`:
- `overview.png` — full strip with all detection bboxes drawn
- `crops/det_NNN.png` — adaptive crop per detection with bbox + mask outline + scale bar
- `detections.json` — class, score, bbox, centroid, lon/lat (lunar geographic), validation summary

The pipeline: `PDSIndex(archive="EDR")` resolves URL → fetch `.IMG` → ISIS3 bash
chain (`lronac2isis → spiceinit → lronaccal → lronacecho → cam2map →
gdal_translate → cubic-spline downscale`) via `conda run -n luna-isis` →
projected GeoTIFF (equirectangular CRS embedded) → tile → ESSA inline.
Pixel ↔ lon/lat uses the GeoTIFF's CRS only (no bilinear shortcut).

### Vendored upstream

[`third_party/`](third_party/) is a read-only snapshot of four Le Corre
repos: ESSA inference, PITS depth-from-shadow (Pass-2 candidate
verification), IMFMapper (impact-melt-fracture FP filter), and the bash
preprocessing pipeline. See [`third_party/README.md`](third_party/README.md)
for licenses + attribution. Weights live at
<https://doi.org/10.5281/zenodo.15438463> (CC BY 4.0). Cite Le Corre et al.
(2025) *Icarus* 441:116675 if you publish anything downstream.

## Main loop

1. **Build pseudo-label crops.** `scripts/build_ellipse_dataset.py` streams
   every (pit, NAC) pair in `catalogs/pit_nacs.json`, decodes the CDR, projects
   the LPA lon/lat to pixels via SPICE (`luna.io.spice_project.ground_to_image`),
   rasterises an ellipse from the catalog dimensions, saves a 1024-px crop +
   COCO annotation, and deletes the `.IMG`. Output goes to `data/ellipse_ds/`.
2. **Hand-label.** `scripts/label_pits.py` opens each crop in a tkinter tool
   with the projected center marked as a subtle cross (it's a hint — see
   *Catalog accuracy* below). Draw a polygon, press Enter for next, `A` to
   add another pit on the same crop. Writes
   `data/ellipse_ds/pits_handlabeled.json` alongside the original ellipse file
   so nothing is overwritten.

   > **Labeling note — don't chase the shadow.** A pit typically contains a
   > deep, almost-black shadow that is visually dominant. The shadow is *not*
   > the pit boundary — it's just the sunless portion of the floor/wall. The
   > label should trace the full **rim opening** (the topographic edge where
   > the surface breaks into the pit), which extends well beyond the shadow
   > into the sunlit wall on the opposite side. If you only outline the
   > shadow, the model will learn "dark blob" and fire on every crater floor.
   >
   > **Use neighboring craters as a sun-direction cheat-sheet.** The shadows
   > cast by small craters around the pit all point the same way — that's the
   > illumination direction. For the pit itself, the shadow sits on the
   > anti-sun side of the opening, and the sunlit wall is on the sun side.
   > Read off the direction from any nearby crater, then mirror it across the
   > pit to find where the rim continues beyond the shadow.

   Examples from the current hand-labeled set (red = mask, zoomed ~250 px
   around the pit; NAC native resolution is 0.5–1 m/px):

   | Pit 22, NAC 1 | Pit 22, NAC 2 | Pit 22, NAC 3 |
   |---|---|---|
   | ![](docs/examples/22_M1149067652R_zoom.png) | ![](docs/examples/22_M126499815R_zoom.png) | ![](docs/examples/22_M187806127R_zoom.png) |

   Same pit under three different illumination geometries — notice how the
   shadow-to-sunlit-wall ratio flips. The mask stays on the rim opening in
   all three. On the right, an example of a crop that contains two adjacent
   pits — both get annotated under the same image record:

   ![](docs/examples/23_M126499815L_zoom.png)
3. **Train.** `scripts/train_maskrcnn.py` trains torchvision Mask R-CNN
   (2 classes: background + pit) on the hand-labeled COCO. *Not yet run
   end-to-end — the hand-labeled set is still being built. Expect rough edges;
   treat the first training run as shakedown.*
4. **Sweep fresh NACs.** `scripts/predict_nac.py` runs tile-sliding inference,
   dedupes by centroid, projects each detection back to lon/lat, and emits
   `candidates.csv` + crop/mask PNGs for review. *Also not yet battle-tested;
   depends on step 3 producing a checkpoint first.*
5. **Monthly refresh.** `scripts/monthly_pds_sweep.py` diffs the current LROC
   CDR archive against a stored snapshot; new volumes trigger step 4 on the
   added frames. Output candidates flow back into step 2.

## Install

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; .venv/bin/activate on Unix
pip install -e .
```

Python ≥ 3.10. Torch + CUDA are not pinned — install whatever wheel matches
your GPU before `pip install -e .`.

## Quickstart

```bash
# build a few pseudo-label crops
python scripts/build_ellipse_dataset.py --limit 5

# hand-label them
python scripts/label_pits.py

# see what's in the archive right now
python scripts/monthly_pds_sweep.py --dry-run
```

## Projection

LROC NAC is a pushbroom camera — the across-track coord of a ground point is a
function of spacecraft attitude at the time that pixel was exposed, not a
plane mapping. There is **one** projection path in this repo: full SPICE
inversion via `luna.io.spice_project.ground_to_image` (pure Python,
`spiceypy`). The earlier 4-corner bilinear shortcut was deleted because it
breaks at high latitude (you can lose tens of km of accuracy at 60°N).

Two ways the SPICE projection gets used:
- **Raw EDR / CDR pixels** — call `ground_to_image(label_path, lon, lat)`
  directly with the `.IMG` PVL label. Used by `scripts/build_ellipse_dataset.py`,
  `scripts/predict_nac.py`, and the `build_projection_oracle.py` ground-truth
  generator.
- **After ISIS `cam2map`** — the resulting GeoTIFF carries an equirectangular
  CRS in its header. `essa_smoke.py` uses that CRS via `rasterio.warp.transform`
  for everything pixel↔lonlat — no extra SPICE call needed.

Kernels are fetched on demand from the NAIF archive by
`luna.io.kernel_fetch.ensure_kernels_for_date`. **Cold fetch is ~1–2 GB per
observation year**, cached under `data/spice/lro/`.

## Catalog accuracy caveat

The LPA lat/lon is accurate to ~50–100 m for minor pits — roughly 100 px at
full NAC resolution. SPICE is pixel-exact from the camera model, so the
projected pixel inherits the catalog's positional uncertainty. **The cross
that `label_pits.py` draws is a "start looking here" hint, not truth.**
Always trust the visible pit shadow over the marker.

## Layout

- `luna/io/` — PDS resolver (INDEX.TAB range reads), NAC CDR/EDR reader, SPICE
  projection, kernel fetcher
- `luna/labels/` — LPA CSV loader, ellipse mask rasteriser
- `luna/models/` — torchvision Mask R-CNN wiring + COCO dataset
- `catalogs/lpa.csv` — ASU Lunar Pit Atlas snapshot (278 pits)
- `catalogs/pit_nacs.json` — pit-id → list of referenced NAC products (811 pairs)
- `scripts/` — the 5-step main loop, one script per step

## Data sources

| Dataset | What it is | Access | License |
|---|---|---|---|
| **LROC NAC CDRs** | Radiometrically calibrated narrow-angle camera frames, 0.5–2 m/px, 52k × 5k px | [PDS LROC node](https://pds.lroc.im-ldi.com/data/) | Public domain (NASA) |
| **LRO SPICE** | Spacecraft ephemeris + attitude kernels | [NAIF archive](https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/) | Public domain (NASA) |
| **Lunar Pit Atlas (LPA)** | 278 cataloged pits with lat/lon, funnel + inner diameters, azimuth, depth | [ASU atlas](https://lroc.im-ldi.com/atlases/pits/list) / Wagner & Robinson 2022 | Public domain |

## Contributing

The roadmap (steps 1–5 above) is the backlog. Pick one and run it end-to-end
before touching another — the real work is labeling pits and getting step 3
training, not scaffolding. Local state lives under `data/` — see `.gitignore`.
Don't commit checkpoints, NAC frames, or SPICE kernels; they're regenerable.
