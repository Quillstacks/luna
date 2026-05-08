# third_party — vendored snapshots from Daniel Le Corre

These are unmodified snapshots of code from the four GitHub repositories that
back Le Corre et al. (2025), *New candidate cave entrances on the Moon found
using deep learning*, Icarus 441:116675 — pulled so we can run the validated
preprocessing + inference pipeline as-is, instead of reimplementing it and
hoping our reimplementation matches.

| Subdir | Upstream | Purpose | License |
|---|---|---|---|
| [`essa/`](essa/) | [Entrances-to-Sub-Surface-Areas](https://github.com/dlecorre387/Entrances-to-Sub-Surface-Areas) | Reference inference script for the published Mask R-CNN | Apache 2.0 |
| [`pits/`](pits/) | [Pit-Topography-from-Shadows](https://github.com/dlecorre387/Pit-Topography-from-Shadows) | Apparent-depth profiles from a single image — used for **Pass 2** verification of any candidate ESSA flags | Apache 2.0 |
| [`imfmapper/`](imfmapper/) | [IMFMapper](https://github.com/dlecorre387/IMFMapper) | Detects impact-melt fractures — useful as an **FP filter**: drop ESSA detections that overlap an IMF detection | Apache 2.0 |
| [`planetary_image_processing/`](planetary_image_processing/) | [Planetary-Image-Processing](https://github.com/dlecorre387/Planetary-Image-Processing) | Bash scripts for ISIS3 + GDAL: raw `.IMG` → calibrated, echo-corrected, map-projected, downsampled GeoTIFF | No upstream LICENSE — used here under fair-use for academic reproduction; do not redistribute without contacting the author |

These files are **read-only** in this repo. We do not edit them. If we need a
modified version of any of these, fork it into `luna/` proper and reference
the upstream commit you forked from in a docstring.

To refresh a snapshot, re-run the download lines documented at the top of
[scripts/essa_smoke.py](../scripts/essa_smoke.py) (or `git log` this directory
for the original `curl` commands).
