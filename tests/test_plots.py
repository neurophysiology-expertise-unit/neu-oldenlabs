from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from neu_oldenlabs.io import load_cage
from neu_oldenlabs.plots import _cluster_spans, actogram, effect_forest, hour_clusters, phase_profiles


def _cage_with_bin_width(make_workbook, name, cage, minutes, bins, per_bin_value, start):
    def values(measure, tag, index):
        return per_bin_value if measure == "Total distance traveled (cm):" else 1.0

    path = make_workbook(name=name, tags=(1,), bins=bins, minutes=(minutes,), start=start,
                         values=values)
    return load_cage(path, cage=cage)


def test_actogram_normalizes_by_valid_hours_not_bin_count(make_workbook, tmp_path):
    # Two synthetic cages carrying the SAME underlying per-hour activity (60
    # cm/hour) but different bin widths: a 600-second-bin cage with 10 cm per
    # bin (six bins/hour) and a 3600-second-bin cage with 60 cm in its single
    # hourly bin. A duration-weighted (per valid hour) reduction must show
    # both as 60.0. A flat per-bin mean -- the pre-fix behaviour -- would
    # instead show 10.0 for the fine cage and 60.0 for the coarse one: a 6x
    # "difference" that is pure bin-width artefact. This test fails against
    # that old per-bin code.
    start = datetime(2026, 7, 11, 17, 0)
    fine = _cage_with_bin_width(make_workbook, "1001_x_cage.xlsx", 1001,
                                minutes=10, bins=6, per_bin_value=10.0, start=start)
    coarse = _cage_with_bin_width(make_workbook, "1002_x_cage.xlsx", 1002,
                                  minutes=60, bins=1, per_bin_value=60.0, start=start)
    combined = pd.concat([fine, coarse], ignore_index=True)

    output = tmp_path / "figs"
    actogram(combined, output)
    table = pd.read_csv(output / "actogram.csv")

    fine_value = table.loc[table["cage"] == 1001, "value"].iloc[0]
    coarse_value = table.loc[table["cage"] == 1002, "value"].iloc[0]
    assert round(fine_value, 6) == 60.0
    assert round(coarse_value, 6) == 60.0


def test_phase_profiles_normalizes_by_valid_hours_not_bin_count(make_workbook, tmp_path):
    # Same bin-width setup as the actogram test above, but for phase_profiles:
    # a count-kind measure (distance_cm) must reduce to the same per-hour
    # value for both a fine- and a coarse-binned animal.
    start = datetime(2026, 7, 11, 17, 0)
    fine = _cage_with_bin_width(make_workbook, "1001_x_cage.xlsx", 1001,
                                minutes=10, bins=6, per_bin_value=10.0, start=start)
    coarse = _cage_with_bin_width(make_workbook, "1002_x_cage.xlsx", 1002,
                                  minutes=60, bins=1, per_bin_value=60.0, start=start)
    fine["genotype"] = "wt"
    coarse["genotype"] = "wt"
    combined = pd.concat([fine, coarse], ignore_index=True)

    output = tmp_path / "figs"
    phase_profiles(combined, output, ["distance_cm"])
    table = pd.read_csv(output / "phase_profiles.csv")

    row = table[(table["measure"] == "distance_cm") & (table["hour"] == 17)]
    assert len(row) == 1  # both animals pooled into one genotype/hour cell
    assert round(float(row["mean"].iloc[0]), 6) == 60.0


def test_phase_profiles_sem_is_taken_across_animals_not_pooled_bins(make_workbook, tmp_path):
    # Two wt animals in the same fine-binned cage, each with a DIFFERENT but
    # internally-constant per-hour value (40 and 60 cm/hour). If bins were
    # pooled directly (the pre-fix behaviour), the SEM would be computed over
    # 12 near-identical-per-animal bin values with almost no animal-to-animal
    # spread reflected properly; reducing to one value per animal first and
    # then taking the SEM across just the two animals gives a specific,
    # checkable number: sem of [40, 60] = 10.0 (ddof=1 by default).
    start = datetime(2026, 7, 11, 17, 0)

    def values_for(target):
        def values(measure, tag, index):
            return target if measure == "Total distance traveled (cm):" else 1.0
        return values

    path1 = make_workbook(name="1001_x_cage.xlsx", tags=(1,), bins=6, minutes=(10,),
                          start=start, values=values_for(40.0 / 6))
    path2 = make_workbook(name="1001_x_cage_b.xlsx", tags=(2,), bins=6, minutes=(10,),
                          start=start, values=values_for(60.0 / 6))
    animal1 = load_cage(path1, cage=1001)
    animal2 = load_cage(path2, cage=1001)
    combined = pd.concat([animal1, animal2], ignore_index=True)
    combined["genotype"] = "wt"

    output = tmp_path / "figs"
    phase_profiles(combined, output, ["distance_cm"])
    table = pd.read_csv(output / "phase_profiles.csv")

    row = table[(table["measure"] == "distance_cm") & (table["hour"] == 17)].iloc[0]
    assert round(float(row["mean"]), 6) == 50.0
    assert round(float(row["sem"]), 6) == 10.0
    assert int(row["n"]) == 2


def test_actogram_all_missing_cell_is_nan_not_zero(tmp_path):
    # An hour cell whose only bins are NaN (e.g. a literal 'NaN' string in the
    # export, coerced by io.load_cage) must show NaN in the actogram cell.
    # `pandas.Series.sum()` sums an all-NaN group to 0.0, and the old code's
    # duration weight (`duration_s.sum()` over EVERY bin, not just present
    # ones) stayed nonzero, so `total / (seconds / 3600.0)` landed on a hard,
    # plausible 0.0 -- indistinguishable in the figure from a real zero-
    # distance hour -- instead of the honest NaN.
    start = datetime(2026, 7, 11, 20, 0)
    table = pd.DataFrame({
        "cage": [1001, 1001],
        "tag": [1, 1],
        "measure": ["distance_cm", "distance_cm"],
        "bin_start": [start, start + pd.Timedelta(minutes=10)],
        "duration_s": [600.0, 600.0],
        "value": [np.nan, np.nan],
    })
    output = tmp_path / "figs"
    actogram(table, output)
    written = pd.read_csv(output / "actogram.csv")
    row = written[(written["cage"] == 1001) & (written["hour"] == 20)].iloc[0]
    assert np.isnan(row["value"])


def test_actogram_partial_missing_cell_is_not_deflated(tmp_path):
    # One present bin (60.0 cm over 540 s) and one missing bin (660 s) in the
    # SAME hour cell. The honest duration-weighted-per-valid-hour value
    # counts only the present bin's duration: 60.0 / (540 / 3600) = 400.0.
    # The old code's denominator included the missing bin's duration too
    # (540 + 660 = 1200 s), deflating the result to 60.0 / (1200 / 3600) =
    # 180.0 -- a real, plausible-looking number, not an error.
    start = datetime(2026, 7, 11, 20, 0)
    table = pd.DataFrame({
        "cage": [1001, 1001],
        "tag": [1, 1],
        "measure": ["distance_cm", "distance_cm"],
        "bin_start": [start, start + pd.Timedelta(minutes=9)],
        "duration_s": [540.0, 660.0],
        "value": [60.0, np.nan],
    })
    output = tmp_path / "figs"
    actogram(table, output)
    written = pd.read_csv(output / "actogram.csv")
    row = written[(written["cage"] == 1001) & (written["hour"] == 20)].iloc[0]
    assert round(float(row["value"]), 6) == 400.0


def test_phase_profiles_all_missing_animal_is_excluded_not_scored_as_zero(tmp_path):
    # Two wt animals at the same hour: tag 1 has a real value (60.0 cm over a
    # 600 s bin, i.e. 360.0 cm/valid-hour), tag 2 is entirely NaN that hour.
    # `_hourly_value_per_animal`'s per-animal value for tag 2 must be NaN and
    # get dropped from the mean (mean=360.0, n=1) -- not the old code's hard
    # 0.0 (its `seconds` denominator stayed nonzero even with zero present
    # duration), which would wrongly pull the animal into the average as a
    # real zero-activity observation (mean=180.0, n=2).
    start = datetime(2026, 7, 11, 20, 0)
    hourly = pd.DataFrame({
        "cage": [1001, 1001],
        "tag": [1, 2],
        "genotype": ["wt", "wt"],
        "measure": ["distance_cm", "distance_cm"],
        "bin_start": [start, start],
        "duration_s": [600.0, 600.0],
        "value": [60.0, np.nan],
    })
    output = tmp_path / "figs"
    phase_profiles(hourly, output, ["distance_cm"])
    table = pd.read_csv(output / "phase_profiles.csv")
    row = table[(table["measure"] == "distance_cm") & (table["hour"] == 20)].iloc[0]
    assert round(float(row["mean"]), 6) == 360.0
    assert int(row["n"]) == 1


def test_phase_profiles_partial_missing_bin_is_not_deflated(tmp_path):
    # One animal, one hour, two bins: a present one (60.0 cm over 540 s) and
    # a missing one (660 s). The honest per-valid-hour value counts only the
    # present bin's duration: 60.0 / (540 / 3600) = 400.0. The old code's
    # denominator included the missing bin's duration too (1200 s total),
    # deflating the result to 60.0 / (1200 / 3600) = 180.0.
    start = datetime(2026, 7, 11, 20, 0)
    hourly = pd.DataFrame({
        "cage": [1001, 1001],
        "tag": [1, 1],
        "genotype": ["wt", "wt"],
        "measure": ["distance_cm", "distance_cm"],
        "bin_start": [start, start + pd.Timedelta(minutes=9)],
        "duration_s": [540.0, 660.0],
        "value": [60.0, np.nan],
    })
    output = tmp_path / "figs"
    phase_profiles(hourly, output, ["distance_cm"])
    table = pd.read_csv(output / "phase_profiles.csv")
    row = table[(table["measure"] == "distance_cm") & (table["hour"] == 20)].iloc[0]
    assert round(float(row["mean"]), 6) == 400.0


def test_effect_forest_passes_through_gs_own_bootstrap_ci_even_when_difference_is_zero(
    tmp_path,
):
    # `g_ci_low`/`g_ci_high` are now computed upstream by
    # `groups.cluster_bootstrap_g_ci` (bootstrapping g directly) and simply
    # passed through here -- effect_forest must no longer derive them by
    # rescaling `ci_low`/`ci_high` by `hedges_g / difference`, which would
    # divide by zero and collapse the interval to zero width whenever
    # `difference` is exactly 0.0. A measure with difference == 0.0 but a
    # real, nonzero-width g interval (as `groups.compare` can legitimately
    # produce, since g's bootstrap is independent of the raw-difference
    # bootstrap) must keep that nonzero interval intact.
    scan = pd.DataFrame([{
        "measure": "distance_cm_all", "difference": 0.0, "ci_low": -1.0, "ci_high": 1.0,
        "hedges_g": 0.0, "g_ci_low": -0.4, "g_ci_high": 0.6,
        "p_value": 0.5, "p_adjusted_bh": 0.5,
    }])
    output = tmp_path / "figs"
    effect_forest(scan, output)
    table = pd.read_csv(output / "effect_forest.csv")
    row = table.iloc[0]
    assert round(float(row["g_ci_low"]), 6) == -0.4
    assert round(float(row["g_ci_high"]), 6) == 0.6


def test_cluster_spans_simple_contiguous_block():
    assert _cluster_spans([2, 3, 4, 5, 6]) == [(2.0, 7.0)]


def test_cluster_spans_wraps_midnight_into_two_spans():
    assert _cluster_spans([22, 23, 0, 1]) == [(22.0, 24.0), (0.0, 2.0)]


def test_cluster_spans_all_hours():
    assert _cluster_spans(list(range(24))) == [(0.0, 24.0)]


def test_hour_clusters_writes_a_figure_and_csv_and_shades_the_observed_cluster(tmp_path):
    start = datetime(2026, 7, 11, 0, 0)
    rows = []
    for hour in range(24):
        for genotype in ("wt", "mut"):
            for tag in (1, 2):
                bump = 10.0 if (genotype == "mut" and 2 <= hour <= 6) else 0.0
                rows.append({
                    "cage": 1, "tag": tag if genotype == "wt" else tag + 10,
                    "genotype": genotype, "measure": "distance_cm",
                    "bin_start": start + pd.Timedelta(hours=hour), "duration_s": 3600.0,
                    "value": 5.0 + bump,
                })
    hourly = pd.DataFrame(rows)
    cluster_scan = pd.DataFrame([{
        "measure": "distance_cm", "hours": "2-6", "hour_list": [2, 3, 4, 5, 6],
        "n_hours": 5, "sign": 1, "mass": 12.0, "p_value": 0.001,
        "n_permutations": 4800, "exact": True, "threshold": 2.0, "p_adjusted_bh": 0.002,
    }])
    output = tmp_path / "figs"
    hour_clusters(hourly, cluster_scan, output, ["distance_cm"])
    assert (output / "hour_clusters.png").exists()
    table = pd.read_csv(output / "hour_clusters.csv")
    assert set(table["measure"]) == {"distance_cm"}


def test_hour_clusters_with_no_clusters_still_writes_the_profile(tmp_path):
    start = datetime(2026, 7, 11, 0, 0)
    rows = [{
        "cage": 1, "tag": 1, "genotype": "wt", "measure": "distance_cm",
        "bin_start": start + pd.Timedelta(hours=hour), "duration_s": 3600.0, "value": 5.0,
    } for hour in range(24)]
    hourly = pd.DataFrame(rows)
    empty_scan = pd.DataFrame(columns=["measure", "hours", "hour_list", "p_value",
                                       "p_adjusted_bh"])
    output = tmp_path / "figs"
    hour_clusters(hourly, empty_scan, output, ["distance_cm"])
    assert (output / "hour_clusters.png").exists()
