"""Monitor HuggingFace model downloads and report progress via a log callback.

Instead of patching tqdm (which is silently disabled in non-TTY server
processes), this monitors the actual .incomplete files in the HF cache
directory to track download progress reliably.
"""

import contextlib
import threading
import time
from pathlib import Path


@contextlib.contextmanager
def track_downloads(log, repo_id=None):
    """Monitor HF cache for download progress during from_pretrained().

    Usage:
        with track_downloads(log, repo_id="org/model"):
            pipe = Pipeline.from_pretrained(repo_id)
    """
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    stop = threading.Event()
    thread = threading.Thread(
        target=_monitor, args=(cache_dir, repo_id, log, stop), daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=3)


def _monitor(cache_dir, repo_id, log, stop):
    """Watch .incomplete files and report download progress every 2s."""
    tracked = {}   # path -> {start, prev, t0}
    totals = {}    # sha256 hash -> expected total bytes
    names = {}     # sha256 hash -> friendly filename

    if repo_id:
        _fetch_file_info(repo_id, totals, names, log)

    repo_dir = None
    if repo_id:
        repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}" / "blobs"

    while not stop.wait(2.0):
        if repo_dir:
            incompletes = list(repo_dir.glob("*.incomplete")) if repo_dir.exists() else []
        else:
            incompletes = list(cache_dir.rglob("*.incomplete"))

        for fpath in incompletes:
            try:
                size = fpath.stat().st_size
            except (FileNotFoundError, OSError):
                continue

            blob_hash = fpath.stem
            total = totals.get(blob_hash)
            name = names.get(blob_hash, blob_hash[:12])

            if fpath not in tracked:
                tracked[fpath] = {"start": size, "prev": size, "t0": time.time()}
                if size > 0 and total:
                    pct = size / total * 100
                    log(f"Resuming {name}: {pct:.0f}% ({_fmt(size)} / {_fmt(total)})")
                elif size > 0:
                    log(f"Resuming {name}: {_fmt(size)} cached")
                continue

            info = tracked[fpath]
            delta = size - info["prev"]
            if delta <= 0:
                continue

            info["prev"] = size
            elapsed = time.time() - info["t0"]
            downloaded = size - info["start"]
            rate = downloaded / elapsed if elapsed > 0 else 0

            if total:
                pct = size / total * 100
                remaining = (total - size) / rate if rate > 0 else 0
                msg = f"{name}: {pct:.0f}% ({_fmt(size)} / {_fmt(total)})"
            else:
                msg = f"{name}: {_fmt(size)}"
                remaining = 0

            if rate > 0:
                msg += f" @ {_fmt(rate)}/s"
            if remaining > 1 and (not total or size < total):
                msg += f", ~{_fmt_time(remaining)} left"

            log(msg)

        finished = [p for p in tracked if not p.exists()]
        for p in finished:
            info = tracked.pop(p)
            blob_hash = p.stem
            total = totals.get(blob_hash)
            name = names.get(blob_hash, blob_hash[:12])
            elapsed = time.time() - info["t0"]
            size_str = _fmt(total) if total else _fmt(info["prev"])
            log(f"{name}: done ({size_str} in {_fmt_time(elapsed)})")


def _fetch_file_info(repo_id, totals, names, log):
    """Get expected blob sizes and filenames from HF API.

    HF cache blobs are named by their LFS sha256 hash, so we can map
    directly from the API's siblings[].lfs.sha256 to blob filenames.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(repo_id, files_metadata=True)

        total_size = 0
        file_count = 0
        for s in info.siblings:
            if hasattr(s, "lfs") and s.lfs:
                sha = s.lfs.get("sha256")
                size = s.lfs.get("size")
                if sha and size:
                    totals[sha] = size
                    friendly = s.rfilename.rsplit("/", 1)[-1]
                    names[sha] = friendly
                    total_size += size
                    file_count += 1

        if total_size > 0:
            log(f"Model has {file_count} files, {_fmt(total_size)} total")
    except Exception:
        pass


def _fmt(b):
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.0f} MB"
    return f"{b / 1e3:.0f} KB"


def _fmt_time(s):
    if s >= 3600:
        return f"{int(s // 3600)}h{int((s % 3600) // 60):02d}m"
    if s >= 60:
        return f"{int(s // 60)}m{int(s % 60):02d}s"
    return f"{s:.0f}s"
