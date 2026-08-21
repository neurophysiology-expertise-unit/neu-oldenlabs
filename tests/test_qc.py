from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from neu_oldenlabs.io import load_cage
from neu_oldenlabs.qc import (apply_rack_mask, bin_regularity, coverage, excluded_tag_table,
                              missing_values, range_violations, recording_breaks, rack_summary,
                              valid_bins)


def out_of_rack_in_bin_two(measure, tag, index):
    if measure == "Cage in rack (1/0):":
        return 0.0 if index == 1 else 1.0
    return 5.0


def test_rack_mask_drops_out_of_rack_bins_and_the_mask_measure(make_workbook):
    path = make_workbook(tags=(1, 2), bins=4, values=out_of_rack_in_bin_two)
    long = load_cage(path, 58616)
    masked = apply_rack_mask(long)
    assert "in_rack" not in set(masked["measure"])
    assert masked["bin_start"].nunique() == 3
    assert len(masked) == 3 * 2 * 21


def test_rack_summary_reports_the_excluded_fraction(make_workbook):
    path = make_workbook(tags=(1, 2), bins=4, values=out_of_rack_in_bin_two)
    summary = rack_summary(load_cage(path, 58616))
    assert summary["bins_out_of_rack"].tolist() == [1, 1]
    assert summary["fraction_out"].round(2).tolist() == [0.25, 0.25]


def test_bin_regularity_counts_each_observed_duration(make_workbook):
    path = make_workbook(tags=(1,), bins=4, minutes=(10, 10, 9, 11))
    table = bin_regularity(load_cage(path, 58616)).set_index("duration_s")
    assert table.loc[600.0, "bins"] == 2
    assert table.loc[540.0, "bins"] == 1
    assert table.loc[660.0, "bins"] == 1


def test_recording_break_is_flagged_when_a_gap_exceeds_the_threshold(make_workbook):
    path = make_workbook(tags=(1,), bins=3, minutes=(10, 25, 10))
    breaks = recording_breaks(load_cage(path, 58616), threshold_min=15)
    assert len(breaks) == 1
    assert breaks.iloc[0]["gap_min"] == 25.0


def test_coverage_counts_valid_bins_and_hours_per_animal_day(make_workbook):
    path = make_workbook(tags=(1, 2), bins=6, start=datetime(2026, 7, 11, 23, 40))
    table = coverage(apply_rack_mask(load_cage(path, 58616)))
    assert set(table["date"].astype(str)) == {"2026-07-11", "2026-07-12"}
    assert table["valid_bins"].sum() == 12
    assert round(table["valid_hours"].sum(), 3) == 2.0


def test_percentages_outside_zero_to_hundred_are_reported_not_corrected(make_workbook):
    def over_range(measure, tag, index):
        return 150.0 if measure == "Time spent eating (%):" else 1.0

    path = make_workbook(tags=(1,), bins=2, values=over_range)
    long = load_cage(path, 58616)
    violations = range_violations(long)
    row = violations[violations["measure"] == "time_eating_pct"].iloc[0]
    assert row["n_violations"] == 2
    assert long.loc[long["measure"] == "time_eating_pct", "value"].max() == 150.0


def test_valid_bins_raises_on_non_finite_in_rack_value(make_workbook):
    """Test that non-finite in_rack values (NaN, inf) cause a crash."""
    def nan_in_rack(measure, tag, index):
        if measure == "Cage in rack (1/0):":
            return float('nan')
        return 5.0

    path = make_workbook(tags=(1,), bins=2, values=nan_in_rack)
    long = load_cage(path, 58616)

    with pytest.raises(ValueError, match="non-finite in_rack values"):
        valid_bins(long)


def test_valid_bins_raises_when_in_rack_measure_missing(make_workbook):
    """Test that absence of in_rack measure causes a crash."""
    path = make_workbook(tags=(1,), bins=2)
    long = load_cage(path, 58616)

    # Remove all in_rack measures
    long_without_rack = long[long["measure"] != "in_rack"]

    with pytest.raises(ValueError, match="no 'in_rack' measure"):
        valid_bins(long_without_rack)


def test_missing_values_reports_the_fraction_missing_per_cage_tag_measure(make_workbook):
    def one_missing(measure, tag, index):
        if measure == "Total distance traveled (cm):" and index == 1:
            return "NaN"
        return 5.0

    path = make_workbook(tags=(1,), bins=4, values=one_missing)
    long = load_cage(path, 58616)
    missing = missing_values(long)
    row = missing[missing["measure"] == "distance_cm"].iloc[0]
    assert row["bins"] == 4
    assert row["missing"] == 1
    assert round(row["fraction_missing"], 6) == 0.25
    # A measure with no missing values still gets a row, at fraction 0.
    other = missing[missing["measure"] == "accel_cm_s"].iloc[0]
    assert other["missing"] == 0
    assert other["fraction_missing"] == 0.0


def test_missing_values_reports_a_fully_missing_measure(make_workbook):
    def all_missing_one_measure(measure, tag, index):
        if measure == "Total distance traveled (cm):":
            return "NaN"
        return 5.0

    path = make_workbook(tags=(1,), bins=3, values=all_missing_one_measure)
    long = load_cage(path, 58616)
    row = missing_values(long).set_index("measure").loc["distance_cm"]
    assert row["missing"] == row["bins"] == 3
    assert row["fraction_missing"] == 1.0


def test_recording_breaks_threshold_scales_with_the_cages_own_bin_width(make_workbook):
    # An hourly-binned cage's normal 60-minute cadence must not itself be
    # flagged as a break just because it exceeds a fixed 15-minute default --
    # the threshold must be relative to the cage's own bin width.
    hourly = make_workbook(tags=(1,), bins=4, minutes=(60,))
    breaks = recording_breaks(load_cage(hourly, 58616))
    assert breaks.empty

    # A gap that is a large multiple of the cage's own cadence is still flagged.
    hourly_with_break = make_workbook(tags=(1,), bins=3, minutes=(60, 300, 60))
    breaks = recording_breaks(load_cage(hourly_with_break, 58616))
    assert len(breaks) == 1
    assert breaks.iloc[0]["gap_min"] == 300.0


def test_apply_rack_mask_raises_on_structurally_missing_rack_status(make_workbook):
    """Test that a bin with no rack status row causes a crash."""
    path = make_workbook(tags=(1,), bins=2)
    long = load_cage(path, 58616)

    # Add a fake row with a bin_start that has no corresponding in_rack measure
    first_bin = long["bin_start"].iloc[0]
    fake_bin_start = first_bin + pd.Timedelta(hours=100)

    fake_row = pd.DataFrame({
        "cage": [58616],
        "tag": [1],
        "bin_start": [fake_bin_start],
        "bin_end": [fake_bin_start + pd.Timedelta(minutes=10)],
        "duration_s": [600.0],
        "measure": ["activity_counts"],
        "value": [50.0]
    })

    long_with_orphan = pd.concat([long, fake_row], ignore_index=True)

    with pytest.raises(ValueError, match="bins with no rack status"):
        apply_rack_mask(long_with_orphan)


def test_excluded_tag_table_reports_bins_present_and_missing_fraction(make_workbook):
    def phantom(measure, tag, index):
        if measure == "Cage in rack (1/0):":
            return 1.0
        if tag == 9:
            return 5.0 if index == 0 else "NaN"
        return 5.0

    path = make_workbook(tags=(1, 9), bins=4, values=phantom)
    long = load_cage(path, 58616)
    table = excluded_tag_table(long, [(58616, 9)])
    assert list(table.columns) == ["cage", "tag", "bins", "present_bins", "missing_fraction"]
    row = table.iloc[0]
    assert (row["cage"], row["tag"]) == (58616, 9)
    assert row["bins"] == 4
    assert row["present_bins"] == 1
    assert round(row["missing_fraction"], 6) == 0.75


def test_excluded_tag_table_ignores_the_in_rack_mask_for_presence(make_workbook):
    # in_rack is a cage-state signal, not animal data -- an excluded tag whose
    # in_rack reading is always present but every real measure is missing must
    # still be reported as fully missing, not as 0% missing.
    def phantom_but_in_rack_present(measure, tag, index):
        if measure == "Cage in rack (1/0):":
            return 1.0
        if tag == 9:
            return "NaN"
        return 5.0

    path = make_workbook(tags=(1, 9), bins=3, values=phantom_but_in_rack_present)
    long = load_cage(path, 58616)
    table = excluded_tag_table(long, [(58616, 9)])
    row = table.iloc[0]
    assert row["bins"] == 3
    assert row["present_bins"] == 0
    assert row["missing_fraction"] == 1.0


def test_excluded_tag_table_is_empty_when_nothing_is_excluded(make_workbook):
    path = make_workbook(tags=(1,), bins=2)
    long = load_cage(path, 58616)
    table = excluded_tag_table(long, [])
    assert table.empty
    assert list(table.columns) == ["cage", "tag", "bins", "present_bins", "missing_fraction"]
