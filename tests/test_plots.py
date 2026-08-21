from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from neu_oldenlabs.io import load_cage
from neu_oldenlabs.plots import actogram, phase_profiles


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
