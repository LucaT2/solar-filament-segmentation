"""Unlabelled GONG H-alpha frames from new days, for semi-supervised training (pseudo-labels).

The competition host allows unlabelled GONG observations for training (inference uses only the supplied test
images). To add *new* scenes — filaments the 707 training images do not show — frames come from days at least
``margin_days`` away from every training and test image (filaments live for days, so frames close to a test date
would show the test scenes). One frame per site and day: the site's middle frame of that day, usually observed
well above the horizon.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta

from filament.data.neighbours import day_frames, stem_site, stem_time


def candidate_days(start: date, end: date, avoid: list[datetime], margin_days: int = 2) -> list[date]:
    """All days in [start, end] at least ``margin_days`` days away from every time in ``avoid``."""
    blocked = set()
    for t in avoid:
        for k in range(-margin_days, margin_days + 1):
            blocked.add((t + timedelta(days=k)).date())
    days, d = [], start
    while d <= end:
        if d not in blocked:
            days.append(d)
        d += timedelta(days=1)
    return days


SITES = "BCLMTU"  # the six GONG sites of the competition data; the archive also holds other observatories


def middle_frames(frames: list[str], sites: str = SITES) -> list[str]:
    """For each of ``sites`` present in a day listing, its middle frame (in time)."""
    by_site: dict[str, list[str]] = {}
    for s in sorted(frames, key=stem_time):
        if stem_site(s) in sites:
            by_site.setdefault(stem_site(s), []).append(s)
    return [v[len(v) // 2] for v in by_site.values()]


def sample_days(days: list[date], n: int, seed: int = 0) -> list[date]:
    """``n`` days spread evenly over the range (one random day per equal-width bin), reproducibly."""
    rng = random.Random(seed)
    if n >= len(days):
        return days
    step = len(days) / n
    return [days[int(i * step) + rng.randrange(max(1, int(step)))] for i in range(n)]


def day_listing(day: date, cache_dir) -> list[str]:
    return day_frames(day.strftime("%Y%m%d"), cache_dir)
