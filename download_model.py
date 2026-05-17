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


def is_cached(repo_id: str) -> bool:
    """Check if a model is fully downloaded in the HF cache."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    blob_dir = cache_dir / f"models--{repo_id.replace('/', '--')}" / "blobs"
    if not blob_dir.exists():
        return False
    has_blobs = any(
        f.is_file() and not f.name.endswith(".incomplete")
        for f in blob_dir.iterdir()
    )
    has_incomplete = any(f.name.endswith(".incomplete") for f in blob_dir.iterdir())
    return has_blobs and not has_incomplete


def ensure_downloaded(repo_id: str, log=None):
    """If model isn't cached, spawn a background download and wait for it.

    The download runs as a detached process so it survives server restarts.
    This function blocks until the download completes, calling log() with
    progress updates.
    """
    if is_cached(repo_id):
        if log:
            log(f"Model weights cached")
        return True

    import os
    import subprocess

    project_dir = Path(__file__).parent
    status_dir = project_dir / ".download_status"
    status_dir.mkdir(exist_ok=True)
    status_file = status_dir / f"{repo_id.replace('/', '--')}.json"

    existing = None
    if status_file.exists():
        try:
            existing = json.loads(status_file.read_text())
        except Exception:
            pass

    already_running = False
    if existing and existing.get("status") == "downloading":
        pid = existing.get("pid")
        if pid:
            try:
                os.kill(pid, 0)
                already_running = True
            except (OSError, ProcessLookupError):
                pass

    if not already_running:
        if log:
            log(f"Starting download of {repo_id}...")
        proc = subprocess.Popen(
            [sys.executable, str(project_dir / "download_model.py"),
             repo_id, "--status-file", str(status_file)],
            start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        status_file.write_text(json.dumps({
            "repo_id": repo_id, "status": "downloading", "pid": proc.pid,
        }))
    elif log:
        log(f"Download already in progress for {repo_id}")

    from download_progress import track_downloads
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    blob_dir = cache_dir / f"models--{repo_id.replace('/', '--')}" / "blobs"

    if log:
        from download_progress import _fetch_file_info, _fmt
        totals = {}
        names = {}
        _fetch_file_info(repo_id, totals, names, log)

    while not is_cached(repo_id):
        time.sleep(3)
        if log:
            incompletes = list(blob_dir.glob("*.incomplete")) if blob_dir.exists() else []
            complete = sum(
                f.stat().st_size for f in blob_dir.iterdir()
                if f.is_file() and not f.name.endswith(".incomplete")
            ) if blob_dir.exists() else 0
            incomplete = sum(
                f.stat().st_size for f in incompletes
            )
            total_downloaded = complete + incomplete
            if totals:
                total_expected = sum(totals.values())
                pct = total_downloaded / total_expected * 100 if total_expected else 0
                log(f"Overall: {pct:.0f}% ({_fmt(total_downloaded)} / {_fmt(total_expected)}) — {len(incompletes)} files remaining")
            else:
                log(f"Downloaded {_fmt(total_downloaded)} — {len(incompletes)} files remaining")

        if status_file.exists():
            try:
                st = json.loads(status_file.read_text())
                if st.get("status") == "failed":
                    if log:
                        log(f"Download failed: {st.get('error', 'unknown')}")
                    return False
            except Exception:
                pass

    if log:
        log("Download complete")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--status-file", type=Path, default=None)
    args = parser.parse_args()
    download(args.repo_id, args.status_file)
