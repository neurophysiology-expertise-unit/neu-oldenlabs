from __future__ import annotations

from datetime import datetime

import pandas as pd

from neu_oldenlabs.metrics import add_time_fields, daily, drop_habituation


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
    table = pd.DataFrame({
        "cage": [1, 1, 1], "tag": [1, 1, 1],
        "date": [pd.Timestamp("2026-07-11").date(), pd.Timestamp("2026-07-12").date(),
                 pd.Timestamp("2026-07-13").date()],
        "phase": ["all"] * 3, "measure": ["distance_cm"] * 3,
        "value": [1.0, 2.0, 3.0], "valid_hours": [1.0, 24.0, 24.0],
    })
    kept = drop_habituation(table, days=1)
    assert [str(d) for d in kept["date"]] == ["2026-07-13"]
