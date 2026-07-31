#!/usr/bin/env python3
"""
run_dashboard.py
~~~~~~~~~~~~~~~~
Launches the LUNA Latent Space Interactive Dash Web Dashboard on port 8050.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from luna.latent_map.dash_app import app

if __name__ == "__main__":
    print("🚀 Launching LUNA Latent Space Modern Dash Dashboard on http://0.0.0.0:8050 ...")
    app.run(host="0.0.0.0", port=8050, debug=False)
