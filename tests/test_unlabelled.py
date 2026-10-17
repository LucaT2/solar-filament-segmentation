"""Day and frame selection for unlabelled GONG frames (offline)."""

from datetime import date, datetime

from filament.data.unlabelled import candidate_days, middle_frames, sample_days


def test_days_near_training_or_test_images_are_excluded():
    avoid = [datetime(2014, 6, 10, 12)]
    days = candidate_days(date(2014, 6, 1), date(2014, 6, 20), avoid, margin_days=2)
    assert all(abs((d - date(2014, 6, 10)).days) > 2 for d in days)
    assert len(days) == 20 - 5


def test_one_middle_frame_per_competition_site():
    frames = ["20140609100000Bh", "20140609110000Bh", "20140609120000Bh", "20140609110000Sh", "20140609090000Lh"]
    assert sorted(middle_frames(frames)) == ["20140609090000Lh", "20140609110000Bh"]  # no site S


def test_sampled_days_are_spread_and_reproducible():
    days = candidate_days(date(2011, 1, 1), date(2011, 12, 31), [])
    a, b = sample_days(days, 12, seed=1), sample_days(days, 12, seed=1)
    assert a == b and len(set(a)) == 12
    assert {d.month for d in a} == set(range(1, 13))  # one per month-sized bin
