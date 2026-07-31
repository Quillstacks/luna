#!/usr/bin/env python3
"""
upload_first_scan_to_hf.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Uploads the First Full Scan archive (indices_old + full_scan.log) 
to the private Hugging Face dataset repository 'F1nnSBK/luna-paper-backup-private'.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKUP_FILE = Path.home() / "hertsch" / "backup" / "luna_first_full_scan_20260731.tar.gz"
REPO_ID = "F1nnSBK/luna-paper-backup-private"
HF_TOKEN = os.environ.get("HF_TOKEN")


def main():
    if not BACKUP_FILE.exists():
        print(f"❌ Error: Backup file not found at {BACKUP_FILE}")
        sys.exit(1)

    file_size_gb = BACKUP_FILE.stat().st_size / (1024**3)
    print(f"🔐 Initializing private Hugging Face upload for account 'F1nnSBK'...")
    print(f"🚀 Uploading First Full Scan Archive '{BACKUP_FILE.name}' ({file_size_gb:.2f} GB) to {REPO_ID}...")

    api = HfApi(token=HF_TOKEN)

    # 1. Create/Ensure PRIVATE Dataset Repository
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=True,
        exist_ok=True
    )

    # 2. Upload file to private repo
    api.upload_file(
        path_or_fileobj=str(BACKUP_FILE),
        path_in_repo=BACKUP_FILE.name,
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    print("\n========================================================")
    print("🎉 SUCCESSFUL PRIVATE UPLOAD OF FIRST FULL SCAN TO HUGGING FACE!")
    print("========================================================")
    print(f"🔒 Visibility:   100% PRIVATE (Only F1nnSBK has access)")
    print(f"📌 Repository:   {REPO_ID}")
    print(f"📁 Archived:     indices_old (26,056 NACs FP16) + full_scan.log")
    print(f"🔗 Private Link: https://huggingface.co/datasets/{REPO_ID}")
    print("========================================================")


if __name__ == "__main__":
    main()
