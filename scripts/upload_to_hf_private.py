#!/usr/bin/env python3
"""
upload_to_hf_private.py
~~~~~~~~~~~~~~~~~~~~~~~
Uploads the paper-ready backup archive to a 100% PRIVATE Hugging Face dataset repository.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKUP_FILE = Path.home() / "hertsch" / "backup" / "luna_paper_ready_backup_20260731.tar.gz"
REPO_ID = "F1nnSBK/luna-paper-backup-private"
HF_TOKEN = os.environ.get("HF_TOKEN")


def main():
    if not BACKUP_FILE.exists():
        print(f"❌ Error: Backup file not found at {BACKUP_FILE}")
        sys.exit(1)

    print(f"🔐 Initializing private Hugging Face upload for account 'F1nnSBK'...")
    api = HfApi(token=HF_TOKEN)

    # 1. Create/Ensure PRIVATE Dataset Repository
    print(f"📦 Ensuring private repository '{REPO_ID}' exists...")
    repo_url = api.create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=True,
        exist_ok=True
    )
    print(f"🔒 Repository URL (PRIVATE): {repo_url}")

    # 2. Upload file to private repo
    file_size_mb = BACKUP_FILE.stat().st_size / (1024**2)
    print(f"🚀 Uploading '{BACKUP_FILE.name}' ({file_size_mb:.2f} MB) to private Hugging Face repo...")
    
    api.upload_file(
        path_or_fileobj=str(BACKUP_FILE),
        path_in_repo="luna_paper_ready_backup_v2_20260731.tar.gz",
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    print("\n========================================================")
    print("🎉 SUCCESSFUL PRIVATE UPLOAD TO HUGGING FACE!")
    print("========================================================")
    print(f"🔒 Visibility:   100% PRIVATE (Only F1nnSBK has access)")
    print(f"📌 Repository:   {REPO_ID}")
    print(f"🔗 Private Link: https://huggingface.co/datasets/{REPO_ID}")
    print("========================================================")


if __name__ == "__main__":
    main()
