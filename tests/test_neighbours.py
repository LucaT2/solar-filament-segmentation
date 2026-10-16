"""Selection of neighbouring GONG frames (offline: synthetic day listings)."""

from datetime import datetime, timedelta

from filament.data.neighbours import frame_url, pick_neighbours, stem_time


def _frames(start: str, minutes: range, site: str = "C") -> list[str]:
    t0 = datetime.strptime(start, "%Y%m%d%H%M%S")
    return [(t0 + timedelta(minutes=m)).strftime("%Y%m%d%H%M%S") + f"{site}h" for m in minutes]


def test_picks_nearest_same_site_frames():
    frames = _frames("20140609195854", range(-20, 21)) + _frames("20140609195854", range(-20, 21), site="B")
    got = pick_neighbours("20140609195854Ch", frames)
    assert [n.offset_min for n in got] == [-5, -2, 2, 5]
    assert all(n.stem.endswith("Ch") for n in got)  # never another site


def test_first_frame_of_the_day_uses_later_fallbacks():
    frames = _frames("20110109104734", range(0, 30))  # the training frame is the site's first frame
    got = pick_neighbours("20110109104734Ch", frames)
    assert [n.offset_min for n in got] == [2, 5, 8, 10]


def test_frames_near_test_images_are_excluded():
    frames = _frames("20140609195854", range(-20, 21))
    test_time = stem_time("20140609200854Lh")  # 10 minutes after the training image
    got = pick_neighbours("20140609195854Ch", frames, exclude_times=[test_time], exclude_window=6)
    assert all(abs((stem_time(n.stem) - test_time).total_seconds()) > 6 * 60 for n in got)
    # +5 and the +8 fallback lie within 6 min of the test image; +2 (8 min away) is fine; -8 fills the fourth slot
    assert [n.offset_min for n in got] == [-5, -2, 2, -8]


def test_alignment_is_an_exact_whole_pixel_shift():
    import numpy as np

    from filament.data.images import Disk
    from filament.data.neighbours import align_to, frame_quality

    rng = np.random.default_rng(0)
    train = rng.integers(60, 200, (256, 256)).astype(np.uint8)
    frame = np.roll(train, (3, -2), axis=(0, 1))  # the same scene, disk moved by (+3, -2) px
    tdisk, fdisk = Disk(128.0, 128.0, 100.0), Disk(126.0, 131.0, 100.0)
    aligned = align_to(frame, train, fdisk, tdisk)
    inner = np.s_[40:216, 40:216]
    assert np.array_equal(aligned[inner], train[inner])  # pixels copied, not interpolated
    q = frame_quality(aligned, train, tdisk)
    assert q["corr"] > 0.999 and abs(q["sharp_ratio"] - 1) < 0.05


def test_archive_url():
    assert frame_url("20140609195854Bh") == "https://gong2.nso.edu/HA/hag/201406/20140609/20140609195854Bh.jpg"
