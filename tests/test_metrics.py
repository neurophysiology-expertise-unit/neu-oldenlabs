from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from neu_oldenlabs.metrics import (add_time_fields, daily, drop_habituation, resample_bins,
                                   restrict_window)


def frame(rows):
    return pd.DataFrame(rows, columns=["cage", "tag", "bin_start", "bin_end",
                                       "duration_s", "measure", "value"])


def test_phase_split_uses_a_seven_to_nineteen_light_schedule():
    long = frame([
        (1, 1, datetime(2026, 7, 11, 6, 50), datetime(2026, 7, 11, 7, 0), 600.0, "distance_cm", 1),
        (1, 1, datetime(2026, 7, 11, 7, 0), datetime(2026, 7, 11, 7, 10), 600.0, "distance_cm", 1),
        (1, 1, datetime(2026, 7, 11, 19, 0), datetime(2026, 7, 11, 19, 10), 600.0, "distance_cm", 1),
    ])
    phases = add_time_fields(long)["phase"].tolist()
    assert phases == ["dark", "light", "dark"]


def test_rate_measures_are_weighted_by_bin_duration_not_averaged_flat():
    # 60 for nine minutes and 0 for eleven: the flat mean is 30.0, the correct
    # duration-weighted mean is 27.0.
    long = frame([
        (1, 1, datetime(2026, 7, 11, 20, 0), datetime(2026, 7, 11, 20, 9), 540.0,
         "inactive_pct", 60.0),
        (1, 1, datetime(2026, 7, 11, 20, 9), datetime(2026, 7, 11, 20, 20), 660.0,
         "inactive_pct", 0.0),
    ])
    table = daily(long)
    value = table[(table["phase"] == "all") & (table["measure"] == "inactive_pct")]["value"]
    assert round(float(value.iloc[0]), 6) == 27.0


def test_counts_become_a_rate_per_valid_hour():
    long = frame([
        (1, 1, datetime(2026, 7, 11, 20, 0), datetime(2026, 7, 11, 20, 30), 1800.0,
         "rearing_events", 6.0),
    ])
    table = daily(long)
    row = table[(table["phase"] == "all") & (table["measure"] == "rearing_events")].iloc[0]
    assert round(float(row["value"]), 6) == 12.0  # 6 events in half an hour
    assert round(float(row["valid_hours"]), 6) == 0.5


def test_count_rate_is_unchanged_when_half_the_bins_are_masked_away():
    both = frame([
        (1, 1, datetime(2026, 7, 11, 20, 0), datetime(2026, 7, 11, 20, 10), 600.0,
         "rearing_events", 2.0),
        (1, 1, datetime(2026, 7, 11, 20, 10), datetime(2026, 7, 11, 20, 20), 600.0,
         "rearing_events", 2.0),
    ])
    one = both.iloc[:1]
    value_both = daily(both).query("phase == 'all'")["value"].iloc[0]
    value_one = daily(one).query("phase == 'all'")["value"].iloc[0]
    assert round(float(value_both), 6) == round(float(value_one), 6) == 12.0


def test_max_measures_take_the_maximum():
    long = frame([
        (1, 1, datetime(2026, 7, 11, 20, 0), datetime(2026, 7, 11, 20, 10), 600.0,
         "speed_max_cm_s", 8.0),
        (1, 1, datetime(2026, 7, 11, 20, 10), datetime(2026, 7, 11, 20, 20), 600.0,
         "speed_max_cm_s", 19.0),
    ])
    table = daily(long).query("phase == 'all' and measure == 'speed_max_cm_s'")
    assert float(table["value"].iloc[0]) == 19.0


def test_habituation_drops_the_first_calendar_days_per_cage():
    # Two cages with disjoint date ranges, cage 2 starting well after cage 1 ends.
    # A regression that replaced per-cage grouping with a single sorted-date slice
    # across the whole table would keep all of cage 2's dates (they all sort after
    # cage 1's), so this only passes if the drop is computed per cage.
    table = pd.DataFrame({
        "cage": [1, 1, 1, 1, 2, 2, 2, 2],
        "tag": [1, 1, 1, 1, 1, 1, 1, 1],
        "date": [
            pd.Timestamp("2026-07-11").date(), pd.Timestamp("2026-07-12").date(),
            pd.Timestamp("2026-07-13").date(), pd.Timestamp("2026-07-14").date(),
            pd.Timestamp("2026-07-20").date(), pd.Timestamp("2026-07-21").date(),
            pd.Timestamp("2026-07-22").date(), pd.Timestamp("2026-07-23").date(),
        ],
        "phase": ["all"] * 8, "measure": ["distance_cm"] * 8,
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "valid_hours": [1.0, 24.0, 24.0, 24.0, 1.0, 24.0, 24.0, 24.0],
    })
    kept = drop_habituation(table, days=1)
    cage1_dates = [str(d) for d in kept[kept["cage"] == 1]["date"]]
    cage2_dates = [str(d) for d in kept[kept["cage"] == 2]["date"]]
    assert cage1_dates == ["2026-07-13", "2026-07-14"]
    assert cage2_dates == ["2026-07-22", "2026-07-23"]


def test_restrict_window_keeps_only_dates_inside_the_bounds():
    table = pd.DataFrame({
        "cage": [1, 1, 1, 1],
        "date": [pd.Timestamp("2026-07-10").date(), pd.Timestamp("2026-07-11").date(),
                 pd.Timestamp("2026-07-12").date(), pd.Timestamp("2026-07-13").date()],
        "value": [1.0, 2.0, 3.0, 4.0],
    })
    windowed = restrict_window(table, start="2026-07-11", end="2026-07-12")
    assert [str(d) for d in windowed["date"]] == ["2026-07-11", "2026-07-12"]

    only_start = restrict_window(table, start="2026-07-12", end=None)
    assert [str(d) for d in only_start["date"]] == ["2026-07-12", "2026-07-13"]

    only_end = restrict_window(table, start=None, end="2026-07-11")
    assert [str(d) for d in only_end["date"]] == ["2026-07-10", "2026-07-11"]

    unrestricted = restrict_window(table, start=None, end=None)
    assert len(unrestricted) == 4


def test_a_group_missing_every_value_aggregates_to_nan_not_zero():
    # cage 58627's real tag 4 is NaN in 1075 of 1076 hours; pandas sums an
    # all-NaN group to 0.0, a hard, plausible number that would silently pass
    # min_days and enter the group comparison as a real observation.
    long = frame([
        (1, 4, datetime(2026, 7, 11, 20, 0), datetime(2026, 7, 11, 20, 10), 600.0,
         "distance_cm", np.nan),
        (1, 4, datetime(2026, 7, 11, 20, 10), datetime(2026, 7, 11, 20, 20), 600.0,
         "distance_cm", np.nan),
    ])
    table = daily(long)
    value = table[(table["phase"] == "all") & (table["measure"] == "distance_cm")]["value"]
    assert len(value) == 1
    assert np.isnan(value.iloc[0])


def test_missing_values_are_excluded_from_the_duration_weight_not_just_the_numerator():
    # 60 for one nine-minute bin, missing for an eleven-minute bin. The old
    # code summed BOTH bins' duration into the denominator while only the
    # present bin's value entered the numerator, deflating the mean: naive
    # (60*540 + 0*660) / (540+660) = 27.0. The honest duration-weighted mean
    # over only the present bin is 60.0 (its own duration cancels).
    long = frame([
        (1, 1, datetime(2026, 7, 11, 20, 0), datetime(2026, 7, 11, 20, 9), 540.0,
         "inactive_pct", 60.0),
        (1, 1, datetime(2026, 7, 11, 20, 9), datetime(2026, 7, 11, 20, 20), 660.0,
         "inactive_pct", np.nan),
    ])
    table = daily(long)
    value = table[(table["phase"] == "all") & (table["measure"] == "inactive_pct")]["value"]
    assert round(float(value.iloc[0]), 6) == 60.0


def test_resample_bins_reproduces_hand_computed_values_for_each_measure_kind():
    # Two 10-minute-ish source bins (540 s and 660 s), for one rate, one count,
    # and one max measure, both falling inside the same target hour.
    long = frame([
        (1, 1, datetime(2026, 7, 11, 20, 0), datetime(2026, 7, 11, 20, 9), 540.0,
         "inactive_pct", 60.0),
        (1, 1, datetime(2026, 7, 11, 20, 9), datetime(2026, 7, 11, 20, 20), 660.0,
         "inactive_pct", 0.0),
        (1, 1, datetime(2026, 7, 11, 20, 0), datetime(2026, 7, 11, 20, 9), 540.0,
         "distance_cm", 5.0),
        (1, 1, datetime(2026, 7, 11, 20, 9), datetime(2026, 7, 11, 20, 20), 660.0,
         "distance_cm", 3.0),
        (1, 1, datetime(2026, 7, 11, 20, 0), datetime(2026, 7, 11, 20, 9), 540.0,
         "speed_max_cm_s", 5.0),
        (1, 1, datetime(2026, 7, 11, 20, 9), datetime(2026, 7, 11, 20, 20), 660.0,
         "speed_max_cm_s", 9.0),
    ])
    resampled = resample_bins(long, 3600)

    assert len(resampled) == 3
    assert resampled["bin_start"].nunique() == 1
    assert resampled["bin_start"].iloc[0] == datetime(2026, 7, 11, 20, 0)
    assert resampled["bin_end"].iloc[0] == datetime(2026, 7, 11, 21, 0)

    by_measure = resampled.set_index("measure")
    # rate/mean: duration-weighted, (60*540 + 0*660) / (540+660) = 27.0.
    assert round(float(by_measure.loc["inactive_pct", "value"]), 6) == 27.0
    # count: summed, not converted to a rate here (that happens one level up,
    # in `_aggregate_measure`/`daily`): 5.0 + 3.0 = 8.0.
    assert round(float(by_measure.loc["distance_cm", "value"]), 6) == 8.0
    # max: the maximum of the source bins.
    assert float(by_measure.loc["speed_max_cm_s", "value"]) == 9.0
    # duration_s on the output is the summed PRESENT source duration.
    assert float(by_measure.loc["inactive_pct", "duration_s"]) == 1200.0
    assert float(by_measure.loc["distance_cm", "duration_s"]) == 1200.0
    assert float(by_measure.loc["speed_max_cm_s", "duration_s"]) == 1200.0


def test_resample_bins_on_an_already_hourly_frame_is_a_value_level_no_op():
    long = frame([
        (1, 1, datetime(2026, 7, 11, 20, 0), datetime(2026, 7, 11, 21, 0), 3600.0,
         "distance_cm", 42.0),
        (1, 1, datetime(2026, 7, 11, 21, 0), datetime(2026, 7, 11, 22, 0), 3600.0,
         "distance_cm", 17.0),
    ])
    resampled = resample_bins(long, 3600).sort_values("bin_start").reset_index(drop=True)
    original = long.sort_values("bin_start").reset_index(drop=True)
    assert resampled["value"].tolist() == original["value"].tolist()
    assert resampled["duration_s"].tolist() == original["duration_s"].tolist()
    assert resampled["bin_start"].tolist() == original["bin_start"].tolist()


def test_resample_bins_yields_nan_not_zero_for_a_target_bin_with_no_present_source_bin():
    # Every source bin in this target hour is missing, across a rate, a count,
    # and a max measure -- the resampled value must be NaN for every one of
    # them, never the 0.0 an unguarded sum/mean would silently produce.
    long = frame([
        (1, 4, datetime(2026, 7, 11, 20, 0), datetime(2026, 7, 11, 20, 10), 600.0,
         "distance_cm", np.nan),
        (1, 4, datetime(2026, 7, 11, 20, 10), datetime(2026, 7, 11, 20, 20), 600.0,
         "distance_cm", np.nan),
        (1, 4, datetime(2026, 7, 11, 20, 0), datetime(2026, 7, 11, 20, 10), 600.0,
         "inactive_pct", np.nan),
        (1, 4, datetime(2026, 7, 11, 20, 0), datetime(2026, 7, 11, 20, 10), 600.0,
         "speed_max_cm_s", np.nan),
    ])
    resampled = resample_bins(long, 3600)
    assert resampled["value"].isna().all()
    assert (resampled["duration_s"] == 0.0).all()
