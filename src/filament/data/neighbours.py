"""Neighbouring GONG frames of the training images: extra real training images that share the training labels.

GONG takes an H-alpha frame every minute at each site. Frames a few minutes before and after a training image show the
same filaments at the same place (the Sun turns ~0.15 px per minute at disk centre), but with different seeing, haze
and noise. Their masks are the training image's masks after aligning the solar disk. The competition host allows such
temporally neighbouring observations for training; at inference only the supplied test image is used.

Archive layout (same as the ``url`` field of the training annotations):
    https://gong2.nso.edu/HA/hag/<YYYYMM>/<YYYYMMDD>/<YYYYMMDDhhmmss><site>h.jpg
"""

from __future__ import annotations

import re
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

ARCHIVE = "https://gong2.nso.edu/HA/hag"
FRAME = re.compile(r"(\d{14})([A-Z])h\.jpg")
USER_AGENT = "solar-filament-research (Kaggle filament-segmentation-2026; training data)"


def stem_time(stem: str) -> datetime:
    return datetime.strptime(stem[:14], "%Y%m%d%H%M%S")


def stem_site(stem: str) -> str:
    return stem[14]


def frame_url(stem: str) -> str:
    return f"{ARCHIVE}/{stem[:6]}/{stem[:8]}/{stem}.jpg"


def _get(url: str, timeout: float = 60, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def day_frames(day: str, cache_dir: Path) -> list[str]:
    """Stems of all frames (every site) archived on ``day`` (YYYYMMDD); the listing is cached on disk."""
    cache = Path(cache_dir) / f"{day}.txt"
    if cache.exists():
        return cache.read_text().split()
    html = _get(f"{ARCHIVE}/{day[:6]}/{day}/").decode("utf-8", "replace")
    stems = sorted({m.group(1) + m.group(2) + "h" for m in FRAME.finditer(html)})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(stems))
    return stems


@dataclass(frozen=True)
class Neighbour:
    train_stem: str
    stem: str
    offset_min: float  # minutes relative to the training image (negative = before)


def pick_neighbours(train_stem: str, frames: list[str], offsets=(-5, -2, 2, 5), fallback=(8, -8, 10, -10),
                    tolerance: float = 1.0, exclude_times: list[datetime] = (),
                    exclude_window: float = 30.0) -> list[Neighbour]:
    """The same-site frame nearest to each wanted offset (minutes), within ``tolerance`` minutes of it.

    Annotated images are often a site's first frame of the day, so earlier frames may not exist; ``fallback`` offsets
    fill up to ``len(offsets)`` neighbours. Frames within ``exclude_window`` minutes of any time in
    ``exclude_times`` (the test images) are never used.
    """
    t0, site = stem_time(train_stem), stem_site(train_stem)
    candidates = [(s, (stem_time(s) - t0).total_seconds() / 60) for s in frames
                  if stem_site(s) == site and s != train_stem]
    window = timedelta(minutes=exclude_window)
    out, used = [], set()
    for off in (*offsets, *fallback):
        if len(out) == len(offsets):
            break
        best = min(((s, dt) for s, dt in candidates if abs(dt - off) <= tolerance and s not in used),
                   key=lambda c: abs(c[1] - off), default=None)
        if best is None:
            continue
        t = stem_time(best[0])
        if any(abs(t - x) <= window for x in exclude_times):
            continue
        used.add(best[0])
        out.append(Neighbour(train_stem, best[0], round(best[1], 2)))
    return out


def download_frame(stem: str, out_dir: Path, pause: float = 1.0) -> Path:
    """Download one archive frame to ``out_dir/<stem>.jpg`` unless it is already there (resumable)."""
    path = Path(out_dir) / f"{stem}.jpg"
    if path.exists() and path.stat().st_size > 0:
        return path
    data = _get(frame_url(stem))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    time.sleep(pause)  # be polite to the archive
    return path
