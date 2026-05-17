"""Intercept huggingface_hub download progress and route to a log callback."""

import contextlib
import time


@contextlib.contextmanager
def track_downloads(log):
    """Patch huggingface_hub's tqdm to emit download progress via log(msg).

    Usage:
        with track_downloads(log):
            pipe = Pipeline.from_pretrained(repo)
    """
    try:
        import huggingface_hub.file_download as hf_dl
        import huggingface_hub.utils.tqdm as hf_tqdm
    except ImportError:
        yield
        return

    OrigTqdm = hf_tqdm.tqdm

    class ProgressTqdm(OrigTqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._last_pct = -10
            self._t0 = time.time()
            self._start_n = self.n
            if self.total and self.total > 0 and self._start_n > 0:
                pct = self._start_n / self.total * 100
                log(f"{(self.desc or 'Downloading').rstrip(': ')}: resuming at {pct:.0f}% ({_fmt(self._start_n)}/{_fmt(self.total)})")
                self._last_pct = pct

        def update(self, n=1):
            result = super().update(n)
            if not self.total or self.total <= 0:
                return result

            pct = self.n / self.total * 100
            if pct - self._last_pct >= 5 or self.n >= self.total:
                self._last_pct = pct
                desc = (self.desc or "Downloading").rstrip(": ")

                elapsed = time.time() - self._t0
                downloaded = self.n - self._start_n
                rate = downloaded / elapsed if elapsed > 0 else 0
                remaining_bytes = self.total - self.n
                remaining = remaining_bytes / rate if rate > 0 else 0

                msg = f"{desc}: {pct:.0f}% ({_fmt(self.n)}/{_fmt(self.total)})"
                if elapsed > 1:
                    msg += f" | {_fmt_time(elapsed)} elapsed"
                if remaining > 1 and self.n < self.total:
                    msg += f" | ~{_fmt_time(remaining)} left"
                if rate > 0:
                    msg += f" | {_fmt(rate)}/s"

                log(msg)
            return result

    hf_tqdm.tqdm = ProgressTqdm
    hf_dl.tqdm = ProgressTqdm
    try:
        yield
    finally:
        hf_tqdm.tqdm = OrigTqdm
        hf_dl.tqdm = OrigTqdm


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
