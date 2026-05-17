#!/usr/bin/env python3
"""Download a HuggingFace model to the local cache.

Runs as a standalone process so downloads survive server restarts.
Usage: python3 download_model.py <repo_id> [--status-file path]
"""

import argparse
import json
import sys
import time
from pathlib import Path


def download(repo_id: str, status_file: Path = None):
    from huggingface_hub import snapshot_download, HfApi

    def write_status(status: dict):
        if status_file:
            status_file.write_text(json.dumps(status))

    write_status({"repo_id": repo_id, "status": "starting"})

    try:
        api = HfApi()
        info = api.model_info(repo_id, files_metadata=True)
        total_size = sum(
            s.lfs["size"] for s in info.siblings
            if hasattr(s, "lfs") and s.lfs and s.lfs.get("size")
        )

        write_status({
            "repo_id": repo_id,
            "status": "downloading",
            "total_size": total_size,
            "started": time.time(),
        })

        snapshot_download(repo_id)

        write_status({
            "repo_id": repo_id,
            "status": "complete",
            "total_size": total_size,
            "completed": time.time(),
        })

    except Exception as e:
        write_status({
            "repo_id": repo_id,
            "status": "failed",
            "error": str(e),
        })
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--status-file", type=Path, default=None)
    args = parser.parse_args()
    download(args.repo_id, args.status_file)
