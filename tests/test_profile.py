from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neu_oldenlabs.profile import PRIMARY_MEASURES, build_profile


def daily_table(values_by_phase, days=10, cage=1, tag=1):
    rows = []
    for offset in range(days):
        date = (pd.Timestamp("2026-07-13") + pd.Timedelta(days=offset)).date()
        for phase, measures in values_by_phase.items():
            for measure, value in measures.items():
                hours = 24.0 if phase == "all" else 12.0
                rows.append({"cage": cage, "tag": tag, "date": date, "phase": phase,
                             "measure": measure, "value": value, "valid_hours": hours})
    return pd.DataFrame(rows)


def animal_table(cage=1, tag=1, genotype="mut"):
    return pd.DataFrame([{"cage": cage, "animal_number": 999, "ear_mark": "L",
                          "genotype_raw": "Mut/W", "genotype": genotype, "tag": tag}])


def circadian_table(cage=1, tag=1):
    return pd.DataFrame([{"cage": cage, "tag": tag, "mesor": 1.0, "amplitude": 2.0,
                          "acrophase_hour": 3.0, "IS": 0.5, "IV": 0.8,
                          "M10": 9.0, "L5": 1.0, "RA": 0.8}])


def test_profile_has_one_row_per_animal_with_identity_and_measures():
    daily = daily_table({"all": {"distance_cm": 100.0, "inactive_pct": 40.0},
                         "dark": {"inactive_pct": 20.0},
                         "light": {"inactive_pct": 60.0}})
    table = build_profile(daily, circadian_table(), animal_table())
    assert len(table) == 1
    row = table.iloc[0]
    assert row["animal_id"] == "1-1"
    assert row["genotype"] == "mut"
    assert row["distance_cm_all"] == 100.0
    assert row["inactive_pct_dark"] == 20.0
    assert row["IS"] == 0.5
    assert row["n_days"] == 10


def test_dark_light_ratio_is_derived_from_the_phase_columns():
    daily = daily_table({"all": {"distance_cm": 100.0},
                         "dark": {"distance_cm": 150.0},
                         "light": {"distance_cm": 50.0}})
    row = build_profile(daily, circadian_table(), animal_table()).iloc[0]
    assert round(row["distance_dark_light_ratio"], 6) == 3.0


def test_sleep_bout_duration_is_inactive_time_divided_by_bouts():
    # 50% inactive over an hour with 2 bouts per hour -> 15 minutes per bout.
    daily = daily_table({"all": {"inactive_pct": 50.0, "sleep_bouts": 2.0}})
    row = build_profile(daily, circadian_table(), animal_table()).iloc[0]
    assert round(row["sleep_bout_minutes_all"], 6) == 15.0


def test_day_to_day_variability_of_distance_is_reported():
    rows = []
    for offset, value in enumerate([100.0, 200.0, 100.0, 200.0, 100.0, 200.0]):
        date = (pd.Timestamp("2026-07-13") + pd.Timedelta(days=offset)).date()
        rows.append({"cage": 1, "tag": 1, "date": date, "phase": "all",
                     "measure": "distance_cm", "value": value, "valid_hours": 24.0})
    table = build_profile(pd.DataFrame(rows), circadian_table(), animal_table())
    expected = np.std([100, 200, 100, 200, 100, 200], ddof=1) / 150.0
    assert round(float(table.iloc[0]["distance_cv_daily"]), 6) == round(expected, 6)


def test_an_animal_with_too_few_days_gets_nan_not_a_number():
    daily = daily_table({"all": {"distance_cm": 100.0}}, days=3)
    row = build_profile(daily, circadian_table(), animal_table(), min_days=5).iloc[0]
    assert np.isnan(row["distance_cm_all"])
    assert row["n_days"] == 3


def test_primary_measures_are_all_present_as_columns():
    daily = daily_table({"all": {"distance_cm": 1.0, "social_distance_cm": 2.0,
                                 "aggression_events": 3.0, "inactive_pct": 4.0},
                         "dark": {"inactive_pct": 5.0}})
    table = build_profile(daily, circadian_table(), animal_table())
    for measure in PRIMARY_MEASURES:
        assert measure in table.columns


def test_animal_in_daily_data_but_absent_from_animal_table_raises():
    daily = daily_table({"all": {"distance_cm": 100.0}}, cage=1, tag=1)
    other_animal = animal_table(cage=1, tag=2)
    with pytest.raises(ValueError, match="1-1"):
        build_profile(daily, circadian_table(cage=1, tag=1), other_animal)


def test_animal_in_animal_table_but_absent_from_daily_data_raises():
    daily = daily_table({"all": {"distance_cm": 100.0}}, cage=1, tag=1)
    animals = pd.concat([animal_table(cage=1, tag=1), animal_table(cage=1, tag=2)],
                        ignore_index=True)
    with pytest.raises(ValueError, match="1-2"):
        build_profile(daily, circadian_table(cage=1, tag=1), animals)


def test_animal_missing_from_circadian_table_raises():
    daily = daily_table({"all": {"distance_cm": 100.0}}, cage=1, tag=1)
    circadian = circadian_table(cage=1, tag=2)
    with pytest.raises(ValueError, match="1-1"):
        build_profile(daily, circadian, animal_table(cage=1, tag=1))


def test_duplicate_cage_tag_row_still_raises():
    daily = daily_table({"all": {"distance_cm": 100.0}}, cage=1, tag=1)
    duplicated_animals = pd.concat([animal_table(cage=1, tag=1), animal_table(cage=1, tag=1)],
                                   ignore_index=True)
    with pytest.raises(ValueError):
        build_profile(daily, circadian_table(cage=1, tag=1), duplicated_animals)


def test_missing_n_days_counts_as_zero_and_still_triggers_blanking():
    # This animal has only "light"/"dark" rows, no "all"-phase rows, so day_counts
    # has no entry for it at all -- n_days must land as 0, not NaN, so the
    # min-days blanking below still fires for the animal with the least data.
    daily = daily_table({"dark": {"distance_cm": 50.0}, "light": {"distance_cm": 60.0}},
                        days=10)
    row = build_profile(daily, circadian_table(), animal_table(), min_days=5).iloc[0]
    assert row["n_days"] == 0
    assert np.isnan(row["distance_cm_dark"])
