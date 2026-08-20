from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neu_oldenlabs.circadian import (circadian_profile, cosinor, hourly_activity,
                                     interdaily_stability, intradaily_variability,
                                     relative_amplitude)


def sine_series(days=10, amplitude=5.0, mesor=10.0, acrophase_hour=3.0):
    index = pd.date_range("2026-07-12", periods=24 * days, freq="h")
    radians = 2 * np.pi * (index.hour - acrophase_hour) / 24
    return pd.Series(mesor + amplitude * np.cos(radians), index=index)


def test_cosinor_recovers_a_known_amplitude_mesor_and_acrophase():
    fit = cosinor(sine_series())
    assert round(fit["mesor"], 6) == 10.0
    assert round(fit["amplitude"], 6) == 5.0
    assert round(fit["acrophase_hour"], 6) == 3.0


def test_interdaily_stability_is_one_for_a_perfectly_repeating_day():
    assert round(interdaily_stability(sine_series()), 6) == 1.0


def test_interdaily_stability_falls_when_days_differ():
    series = sine_series()
    rng = np.random.default_rng(0)
    noisy = series + rng.normal(0, 5, len(series))
    assert interdaily_stability(noisy) < interdaily_stability(series)


def test_intradaily_variability_is_higher_for_a_fragmented_rhythm():
    smooth = sine_series()
    fragmented = pd.Series(np.tile([0.0, 20.0], len(smooth) // 2), index=smooth.index)
    assert intradaily_variability(fragmented) > intradaily_variability(smooth)


def test_relative_amplitude_matches_a_hand_computed_block_profile():
    # Ten hours at 20, then fourteen at 0, repeated: M10 = 20, L5 = 0, RA = 1.
    day = [20.0] * 10 + [0.0] * 14
    index = pd.date_range("2026-07-12", periods=24 * 5, freq="h")
    series = pd.Series(day * 5, index=index)
    result = relative_amplitude(series)
    assert round(result["M10"], 6) == 20.0
    assert round(result["L5"], 6) == 0.0
    assert round(result["RA"], 6) == 1.0


def test_intradaily_variability_does_not_bridge_a_recording_gap():
    # Regular hourly grid 00:00..06:00 (7 slots); hours 03:00 and 04:00 are missing,
    # so on the grid the values are [0, 10, 0, NaN, NaN, 10, 0].
    index = pd.to_datetime([
        "2026-07-12 00:00", "2026-07-12 01:00", "2026-07-12 02:00",
        "2026-07-12 05:00", "2026-07-12 06:00",
    ])
    series = pd.Series([0.0, 10.0, 0.0, 10.0, 0.0], index=index)
    # Gap-aware: only consecutive-hour pairs count. Valid diffs (both endpoints finite):
    # 00->01 = 10, 01->02 = -10, 05->06 = -10 -> 3 diffs, n = 3 + 1 = 4.
    # numerator = 4 * (10^2 + 10^2 + 10^2) = 1200.
    # finite values = [0, 10, 0, 10, 0], mean = 4, sum((x-mean)^2) = 120.
    # denominator = (5 - 1) * 120 = 480. IV = 1200 / 480 = 2.5.
    # (A bridging implementation would instead treat 02:00 -> 05:00 as one consecutive
    # step, changing the diffs used and giving a different, wrong, answer.)
    assert round(intradaily_variability(series), 6) == 2.5


def test_relative_amplitude_is_stable_across_recording_length():
    # Fixed diurnal pattern (10 hours on at 20, 14 hours off at 0) with small,
    # independently seeded per-hour noise. The mean-24-hour-profile definition should
    # give nearly the same RA whether the recording is short or long; a whole-series
    # rolling-window definition drifts with recording length instead (it lets a longer
    # recording pick a more extreme window).
    day_pattern = np.array([20.0] * 10 + [0.0] * 14)

    def make_series(days, seed):
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 0.5, size=24 * days)
        index = pd.date_range("2026-07-12", periods=24 * days, freq="h")
        return pd.Series(np.tile(day_pattern, days) + noise, index=index)

    short = relative_amplitude(make_series(days=3, seed=7))
    long = relative_amplitude(make_series(days=60, seed=8))
    assert abs(short["RA"] - long["RA"]) < 0.05


def test_hourly_activity_aggregates_bins_into_a_per_hour_rate():
    bin_starts = pd.to_datetime([
        "2026-07-12 00:00", "2026-07-12 00:10", "2026-07-12 00:20",
        "2026-07-12 00:30", "2026-07-12 00:40", "2026-07-12 00:50",
    ])
    long = pd.DataFrame({
        "cage": [1] * 6,
        "tag": ["A"] * 6,
        "bin_start": bin_starts,
        "bin_end": bin_starts + pd.Timedelta(minutes=10),
        "duration_s": [600] * 6,
        "measure": ["distance_cm"] * 6,
        "value": [10.0] * 6,
    })
    result = hourly_activity(long)
    assert list(result.columns) == ["cage", "tag", "hour_start", "distance_per_hour"]
    assert len(result) == 1
    row = result.iloc[0]
    assert row["cage"] == 1
    assert row["tag"] == "A"
    assert row["hour_start"] == pd.Timestamp("2026-07-12 00:00")
    # Six 10-minute bins of 10.0 cm each = 60 cm over 3600 s -> 60 cm/hour.
    assert round(row["distance_per_hour"], 6) == 60.0


def test_hourly_activity_raises_when_distance_measure_is_absent():
    long = pd.DataFrame({
        "cage": [1],
        "tag": ["A"],
        "bin_start": [pd.Timestamp("2026-07-12 00:00")],
        "bin_end": [pd.Timestamp("2026-07-12 00:10")],
        "duration_s": [600],
        "measure": ["some_other_measure"],
        "value": [1.0],
    })
    with pytest.raises(ValueError):
        hourly_activity(long)


def _long_frame_for_profile(cage, tag, days=3):
    index = pd.date_range("2026-07-12", periods=24 * days, freq="h")
    radians = 2 * np.pi * (index.hour - 3.0) / 24
    values = 10.0 + 5.0 * np.cos(radians)
    return pd.DataFrame({
        "cage": cage,
        "tag": tag,
        "bin_start": index,
        "bin_end": index + pd.Timedelta(hours=1),
        "duration_s": 3600,
        "measure": "distance_cm",
        "value": values,
    })


def test_circadian_profile_has_one_row_per_animal_and_all_expected_columns():
    long = pd.concat(
        [_long_frame_for_profile(1, "A"), _long_frame_for_profile(2, "B")], ignore_index=True)
    profile = circadian_profile(long)
    assert len(profile) == 2
    expected_columns = {"cage", "tag", "mesor", "amplitude", "acrophase_hour",
                         "IS", "IV", "M10", "L5", "RA"}
    assert expected_columns.issubset(set(profile.columns))
    assert set(zip(profile["cage"], profile["tag"])) == {(1, "A"), (2, "B")}
