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


def _scene(seed: int = 0):
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.normal(128, 40, (256, 256)).astype(np.float32), (0, 0), 3)  # structure, not just noise
    return np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8)


def test_clear_offsets_are_corrected_by_an_exact_whole_pixel_shift():
    import numpy as np

    from filament.data.images import Disk
    from filament.data.neighbours import align_to, frame_quality

    train = _scene()
    frame = np.roll(train, (4, -3), axis=(0, 1))  # the same scene, moved by (+4, -3) px
    tdisk = Disk(128.0, 128.0, 100.0)
    aligned, applied = align_to(frame, train, tdisk)
    assert applied == (-4, 3)
    inner = np.s_[40:216, 40:216]
    assert np.array_equal(aligned[inner], train[inner])  # pixels copied, not interpolated
    q = frame_quality(aligned, train, tdisk)
    assert q["corr"] > 0.999 and abs(q["sharp_ratio"] - 1) < 0.05


def test_small_offsets_are_left_alone():
    import numpy as np

    from filament.data.images import Disk
    from filament.data.neighbours import align_to

    train = _scene()
    frame = np.roll(train, (1, 0), axis=(0, 1))  # within the co-registration accuracy: used as it is
    aligned, applied = align_to(frame, train, Disk(128.0, 128.0, 100.0))
    assert applied == (0, 0)
    on_disk = np.s_[60:196, 60:196]  # inside the radius-100 disk (outside it, the training image's pixels are used)
    assert np.array_equal(aligned[on_disk], frame[on_disk])


def test_neighbour_sampler_swaps_images_and_keeps_the_disk(tmp_path):
    import cv2
    import numpy as np
    import pandas as pd

    from filament.data.images import Disk
    from filament.data.neighbours import load_neighbours
    from filament.segmentation.dataset import ImageInfo, NeighbourSampler

    img = np.full((256, 256), 100, np.uint8)
    for stem, value in (("n1", 150), ("n2", 50)):
        cv2.imwrite(str(tmp_path / f"{stem}.jpeg"), np.full((256, 256), value, np.uint8))
    pd.DataFrame({"train_stem": ["t", "t"], "stem": ["n1", "n2"], "keep": [True, False]}).to_csv(
        tmp_path / "manifest.csv", index=False)
    nb = load_neighbours(tmp_path)
    assert list(nb) == ["t"] and [p.name for p in nb["t"]] == ["n1.jpeg"]  # rejected frames are not used

    info = ImageInfo("t", tmp_path / "t.jpeg", Disk(128.0, 128.0, 100.0), 100.0, 5.0)
    rng = np.random.default_rng(0)
    same, same_info = NeighbourSampler(nb, p=0.0)("t", img, info, rng)
    assert same is img and same_info is info
    swapped, swapped_info = NeighbourSampler(nb, p=1.0)("t", img, info, rng)
    assert abs(float(swapped.mean()) - 150) < 2  # the neighbour's pixels
    assert swapped_info.disk == info.disk and abs(swapped_info.mean - 150) < 2  # its own statistics, same disk
    assert NeighbourSampler(nb, p=1.0)("other", img, info, rng)[0] is img  # images without neighbours unchanged


def test_archive_url():
    assert frame_url("20140609195854Bh") == "https://gong2.nso.edu/HA/hag/201406/20140609/20140609195854Bh.jpg"
