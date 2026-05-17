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
    """Watch .incomplete files and report download progress."""
    tracked = {}  # path -> {start_size, prev_size, total}
    totals = {}   # path -> expected total bytes (if known)

    if repo_id:
        _fetch_expected_sizes(repo_id, cache_dir, totals, log)

    while not stop.wait(2.0):
        if repo_id:
            repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}" / "blobs"
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

            if fpath not in tracked:
                tracked[fpath] = {"start": size, "prev": size, "t0": time.time()}
                if size > 0 and total:
                    pct = size / total * 100
                    log(f"Resuming {blob_hash[:10]}... {pct:.0f}% ({_fmt(size)}/{_fmt(total)})")
                elif size > 0:
                    log(f"Resuming {blob_hash[:10]}... ({_fmt(size)} cached)")
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
                msg = f"Downloading {blob_hash[:10]}... {pct:.0f}% ({_fmt(size)}/{_fmt(total)})"
                if rate > 0:
                    msg += f" | {_fmt(rate)}/s"
                if remaining > 1 and size < total:
                    msg += f" | ~{_fmt_time(remaining)} left"
            else:
                msg = f"Downloading {blob_hash[:10]}... {_fmt(size)}"
                if rate > 0:
                    msg += f" | {_fmt(rate)}/s"

            log(msg)

        # Check for files that finished (incomplete removed)
        finished = [p for p in tracked if not p.exists()]
        for p in finished:
            info = tracked.pop(p)
            total = totals.get(p.stem)
            elapsed = time.time() - info["t0"]
            if total:
                log(f"Downloaded {p.stem[:10]}... {_fmt(total)} in {_fmt_time(elapsed)}")


def _fetch_expected_sizes(repo_id, cache_dir, totals, log):
    """Get expected blob sizes from HF API so we can show percentages."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(repo_id, files_metadata=True)

        repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
        snapshot_dir = repo_dir / "snapshots"
        refs_dir = repo_dir / "refs"

        commit = None
        ref_file = refs_dir / "main"
        if ref_file.exists():
            commit = ref_file.read_text().strip()

        snap = snapshot_dir / commit if commit else None

        size_by_name = {}
        for s in info.siblings:
            if s.size and s.size > 1_000_000:
                size_by_name[s.rfilename] = s.size

        if snap and snap.exists():
            for child in snap.rglob("*"):
                if child.is_symlink():
                    target = child.resolve()
                    rel = str(child.relative_to(snap))
                    if rel in size_by_name:
                        blob_hash = target.name
                        totals[blob_hash] = size_by_name[rel]
    except Exception:
        pass


def _fmt(b):
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.0f} MB"
    return f"{b / 1e3:.0f} KB"


def _fmt_time(s):
    if s >= 60:
        return f"{int(s // 60)}m{int(s % 60):02d}s"
    return f"{s:.0f}s"
