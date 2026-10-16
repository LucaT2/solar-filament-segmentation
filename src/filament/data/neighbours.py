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

import cv2
import numpy as np

from filament.data.images import Disk, disk_mask

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


def align_to(frame: np.ndarray, train: np.ndarray, frame_disk: Disk, train_disk: Disk,
             max_scale_error: float = 0.002) -> np.ndarray:
    """``frame`` moved so its solar disk coincides with ``train``'s; outside the disk, ``train``'s pixels (the
    competition files are post-processed off the disk, the archive frames are not).

    Any interpolation blurs the pixel-scale texture (a half-pixel bilinear shift keeps ~20% of the Laplacian energy),
    so frames are shifted by whole pixels, an exact copy, and only rescaled (Lanczos) if the disk size differs by more
    than ``max_scale_error``. The remaining misalignment is < 1 px.
    """
    s = train_disk.radius / frame_disk.radius
    if abs(s - 1) <= max_scale_error:
        dx, dy = round(train_disk.cx - frame_disk.cx), round(train_disk.cy - frame_disk.cy)
        m, flags = np.float32([[1, 0, dx], [0, 1, dy]]), cv2.INTER_NEAREST
    else:
        m = np.float32([[s, 0, train_disk.cx - s * frame_disk.cx], [0, s, train_disk.cy - s * frame_disk.cy]])
        flags = cv2.INTER_LANCZOS4
    out = cv2.warpAffine(frame, m, (train.shape[1], train.shape[0]), flags=flags,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    on = disk_mask(train.shape, train_disk, 1.0)
    out[~on] = train[~on]
    return out


def frame_quality(aligned: np.ndarray, train: np.ndarray, train_disk: Disk, smooth: float = 3.0) -> dict:
    """How closely an aligned neighbour matches its training image on the disk (inner 95% of the radius).

    ``corr`` compares the images smoothed with a ``smooth``-px Gaussian: it checks that filament-scale structures
    coincide and ignores the pixel-scale granulation that seeing changes from frame to frame (exactly the variation
    the neighbours are meant to add). ``corr_raw`` is the unsmoothed correlation, for reference.
    """
    on = disk_mask(train.shape, train_disk, 0.95)
    a, b = aligned.astype(np.float32), train.astype(np.float32)
    sa, sb = cv2.GaussianBlur(a, (0, 0), smooth), cv2.GaussianBlur(b, (0, 0), smooth)
    sharp = lambda img: float(cv2.Laplacian(img, cv2.CV_32F)[on].var())
    return {"corr": float(np.corrcoef(sa[on], sb[on])[0, 1]), "corr_raw": float(np.corrcoef(a[on], b[on])[0, 1]),
            "mean_ratio": float(a[on].mean() / max(b[on].mean(), 1.0)), "sharp_ratio": sharp(a) / max(sharp(b), 1e-6)}


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
