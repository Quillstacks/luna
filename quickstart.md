# LUNA Pipeline: Quickstart Guide

This document summarizes the core commands and configurations to run LROC NAC scans, launch the visual ROI selector, and monitor large-scale scans on the system.

---

## 1. Environment & Authentication

Before running any pipeline commands, export your Hugging Face token to enable downloading the DINOv3 LoRA adapters:
```bash
export HF_TOKEN=123
```

---

## 2. Visual ROI Selection (Moon Map)

To select regions of interest (ROIs) visually without needing coordinate lookups, use the browser-based interactive map:
```bash
firefox luna_roi_selector.html &
```

### Operation:
1. Click **"Start Drawing"** in the sidebar menu on the right.
2. Click vertices on the lunar map to draw a bounding polygon (connected via pink lines).
3. Click **"Finish & Close"** to close the polygon.
4. Click **"Copy --roi Parameter"** to copy the formatted coordinate string to your clipboard.

---

## 3. Running Pipeline Scans

All scans are executed through the virtual environment (`.venv`).

### A. Scanning a Single NAC Image
To run the pipeline on a single, known LROC NAC product ID:
```bash
./.venv/bin/python fun_with_luna.py --nac M1103559186LC --refiner dino
```

### B. Interactive ROI Mosaic Scan (Recommended)
This calculates coverage, estimates download sizes, and prompts for confirmation before starting:
```bash
./.venv/bin/python fun_with_luna.py --roi "35.2,9.2 35.8,9.2 35.8,8.8 35.2,8.8 35.2,9.2" --refiner dino
```

### C. Persistent Background Scan (via tmux)
For large-scale regions (e.g. running overnight). This continues running even if the SSH/IDE connection drops and utilizes automatic file cleanup (saving disk space while preserving raw imagery for the top 10 detections):
```bash
tmux new-session -d -s luna_scan "export HF_TOKEN=your_token && export PYTHONUNBUFFERED=1 && ./.venv/bin/python -u fun_with_luna.py --roi '<PASTED_COORDINATES>' -y --cleanup --refiner dino >> scan.log 2>&1"
```

---

## 4. Key CLI Arguments

* **`--refiner dino`** *(Critical)*: Performs candidate refinement directly using DINOv3 on the GPU, bypassing Conda/ISIS3 system dependencies required by the standard ESSA refiner.
* **`--cleanup`**: Deletes raw `.IMG` rasters (~1 GB uncompressed per file) after candidate refinement finishes to conserve space. The compact Pithos index binaries (`.bin`) are preserved.
* **`-y` / `--yes`**: Bypasses the interactive size prompts and starts processing/downloading immediately.
* **`--force-reingest`**: Forces raw data indexing even if a cached Pithos index binary is already present.
* **`--attention-overlay`**: Generates and saves DINO model attention overlay heatmaps for verified hits.

---

## 5. Monitoring & Evaluation

### A. Real-Time Log Monitoring
With output buffering disabled, monitor raw logs in real time:
```bash
tail -f hilbert_scan.log
```

### B. Managing the tmux Session
If the scan is running in the background:
* **List active sessions:** `tmux ls`
* **Attach to the scan session:** `tmux attach -t luna_scan`
* **Detach from the session (leave running):** Press `Ctrl` + `B`, then release and press `D`.

### C. Visualizing Detections & Overlays
To generate visual boxes or attention overlays for a target NAC ID manually:
```bash
./.venv/bin/python scripts/visualize_hits.py --nac M1118880788RC --score 0.85 --skip-preprocess
```
Visualizations will be written to the `./plots/` directory.
