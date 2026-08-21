from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import pytest

from neu_oldenlabs.config import load_study
from neu_oldenlabs.pipeline import _default_scan_measures, run_study
from neu_oldenlabs.profile import IDENTITY


def test_run_study_produces_profile_and_comparison_tables(synthetic_study, tmp_path):
    study = synthetic_study
    output = tmp_path / "out"
    result = run_study(study, output, cache_dir=None)

    assert len(result["profile"]) == 6
    assert set(result["profile"]["genotype"]) == {"mut", "wt"}
    assert (output / "animal_profile.csv").exists()
    assert (output / "daily_measures.csv").exists()
    assert (output / "group_comparison.csv").exists()
    assert (output / "provenance.json").exists()


def test_missing_cages_are_recorded_not_silently_dropped(synthetic_study, tmp_path):
    result = run_study(synthetic_study, tmp_path / "out", cache_dir=None)
    assert result["missing_cages"] == [1003]


def test_every_figure_writes_its_values_as_csv(synthetic_study, tmp_path):
    study = synthetic_study
    output = tmp_path / "out"
    run_study(study, output, cache_dir=None)
    figures = sorted(p.stem for p in (output / "figures").glob("*.png"))
    tables = sorted(p.stem for p in (output / "figures").glob("*.csv"))
    assert figures and figures == tables


def test_circadian_table_honors_habituation_days(make_workbook, tmp_path):
    # Day 1 (habituation) is a wild anomaly (1000 cm/bin); days 2-5 are a
    # normal, steady 10 cm/bin. If the circadian family ignored
    # drop_habituation (the pre-fix behaviour used the raw masked frame
    # directly), day 1's anomaly would pull the cosinor mesor far above 100;
    # with habituation correctly dropped, the mesor reflects only the
    # steady days and stays close to 10.
    bins_per_day = 6 * 24  # 10-minute bins

    def anomalous_day_one(measure, tag, index):
        if measure != "Total distance traveled (cm):":
            return 1.0
        return 1000.0 if index < bins_per_day else 10.0

    export = make_workbook(name="2001_x_cage.xlsx", tags=(1,), bins=bins_per_day * 5,
                           start=datetime(2026, 7, 11, 0, 0), minutes=(10,),
                           values=anomalous_day_one)
    animals = tmp_path / "animals.csv"
    animals.write_text(
        "cage #,animal #,,genotype,tag #,Tag date,Monitoring start\n2001,1,L,W/w,1,,\n")
    config = tmp_path / "study.json"
    config.write_text(json.dumps({
        "name": "habituation_test", "animals_csv": str(animals),
        "cages": [{"cage": 2001, "export": str(export)}],
        "habituation_days": 1, "lights_on": 7, "lights_off": 19, "min_days": 1,
        "primary_measures": ["distance_cm_all"], "scan_measures": ["distance_cm_all"],
    }))
    result = run_study(load_study(config), tmp_path / "out", cache_dir=None)
    mesor = result["profile"]["mesor"].iloc[0]
    assert mesor < 100.0


def test_circadian_table_honors_the_analysis_window(make_workbook, tmp_path):
    # Same anomaly shape as above, but excluded via window_start instead of
    # habituation_days, to cover restrict_window's effect on the circadian
    # family independently.
    bins_per_day = 6 * 24

    def anomalous_day_one(measure, tag, index):
        if measure != "Total distance traveled (cm):":
            return 1.0
        return 1000.0 if index < bins_per_day else 10.0

    export = make_workbook(name="2002_x_cage.xlsx", tags=(1,), bins=bins_per_day * 5,
                           start=datetime(2026, 7, 11, 0, 0), minutes=(10,),
                           values=anomalous_day_one)
    animals = tmp_path / "animals.csv"
    animals.write_text(
        "cage #,animal #,,genotype,tag #,Tag date,Monitoring start\n2002,1,L,W/w,1,,\n")
    config = tmp_path / "study.json"
    config.write_text(json.dumps({
        "name": "window_test", "animals_csv": str(animals),
        "cages": [{"cage": 2002, "export": str(export)}],
        "habituation_days": 0, "lights_on": 7, "lights_off": 19, "min_days": 1,
        "window_start": "2026-07-12", "window_end": None,
        "primary_measures": ["distance_cm_all"], "scan_measures": ["distance_cm_all"],
    }))
    result = run_study(load_study(config), tmp_path / "out", cache_dir=None)
    mesor = result["profile"]["mesor"].iloc[0]
    assert mesor < 100.0


def test_default_scan_measures_excludes_the_circular_acrophase_hour():
    # acrophase_hour wraps at 24h == 0h and must not be differenced linearly
    # by the group comparison; the empty-scan_measures fallback must exclude
    # it while still leaving every other measure column (and it still lives
    # in the profile table itself, unaffected by this scan-selection helper).
    # This tests the selection logic directly rather than through a full
    # `run_study` (which would run groups.scan's ~4.7s-per-measure compare()
    # over all ~73 default measures -- correct, but far too slow for a unit
    # test) -- see the fast integration coverage in the acrophase_hour-free
    # `synthetic_study`-based tests above for the end-to-end wiring.
    columns = IDENTITY + ["distance_cm_all", "mesor", "acrophase_hour", "IS"]
    profile_table = pd.DataFrame(columns=columns)
    measures = _default_scan_measures(profile_table)
    assert "acrophase_hour" not in measures
    assert set(measures) == {"distance_cm_all", "mesor", "IS"}


def test_default_scan_measures_is_empty_when_only_identity_columns_are_present():
    profile_table = pd.DataFrame(columns=IDENTITY)
    assert _default_scan_measures(profile_table) == []


def test_all_wildtype_study_writes_a_note_and_skips_effect_forest(all_wildtype_study, tmp_path):
    # This mirrors the real study's current live shape: cage 58616 is the only
    # export that exists, and it is five wildtype animals with no mutants. The
    # pipeline must not raise on this -- it must produce a complete descriptive
    # output, since that is what the study owner gets today.
    output = tmp_path / "out"
    result = run_study(all_wildtype_study, output, cache_dir=None)

    assert result["scan"].empty

    comparison_path = output / "group_comparison.csv"
    assert comparison_path.exists()
    comparison_lines = comparison_path.read_text().splitlines()
    assert comparison_lines[0].split(",") == ["note"]
    assert "no loaded cage contains both genotypes" in comparison_lines[1]

    figures = output / "figures"
    assert not (figures / "effect_forest.png").exists()
    assert not (figures / "effect_forest.csv").exists()

    png_stems = sorted(p.stem for p in figures.glob("*.png"))
    csv_stems = sorted(p.stem for p in figures.glob("*.csv"))
    assert png_stems and png_stems == csv_stems


def test_export_tag_not_in_csv_is_excluded_from_profile_and_reported_in_qc(
    make_workbook, tmp_path,
):
    # Study owner's ruling: the animal CSV is the sole authority on which
    # animals exist. Tag 9 appears in the export but is not listed in the
    # animal CSV for cage 3001, so it must never reach the profile, but must
    # be visible in the excluded-tags QC table with its own missing fraction.
    def values(measure, tag, index):
        if measure == "Cage in rack (1/0):":
            return 1.0
        if tag == 9:
            return 5.0 if index < 3 else "NaN"
        return 1.0 + tag + (index % 7)

    total_bins = 6 * 24 * 2
    export = make_workbook(name="3001_x_cage.xlsx", tags=(1, 9), bins=total_bins, values=values)
    animals = tmp_path / "animals.csv"
    animals.write_text(
        "cage #,animal #,,genotype,tag #,Tag date,Monitoring start\n3001,1,L,W/w,1,,\n")
    config = tmp_path / "study.json"
    config.write_text(json.dumps({
        "name": "phantom_tag_test", "animals_csv": str(animals),
        "cages": [{"cage": 3001, "export": str(export)}],
        "habituation_days": 0, "lights_on": 7, "lights_off": 19, "min_days": 1,
        "primary_measures": ["distance_cm_all"], "scan_measures": ["distance_cm_all"],
    }))
    output = tmp_path / "out"
    result = run_study(load_study(config), output, cache_dir=None)

    # The phantom tag never reaches the profile.
    assert set(zip(result["profile"]["cage"], result["profile"]["tag"])) == {(3001, 1)}

    excluded = result["qc"]["excluded_tags"]
    assert list(zip(excluded["cage"], excluded["tag"])) == [(3001, 9)]
    row = excluded.iloc[0]
    assert row["bins"] == total_bins
    assert row["present_bins"] == 3
    assert round(row["missing_fraction"], 6) == round(1 - 3 / total_bins, 6)

    # Written as its own CSV, like every other QC table.
    assert (output / "qc_excluded_tags.csv").exists()


def test_csv_listed_animal_absent_from_export_still_raises(make_workbook, tmp_path):
    # The reverse direction of the join is still a genuine discrepancy: the
    # CSV lists tag 2 for this cage, but the export only carries tag 1.
    def values(measure, tag, index):
        if measure == "Cage in rack (1/0):":
            return 1.0
        return 1.0 + tag + (index % 7)

    export = make_workbook(name="3002_x_cage.xlsx", tags=(1,), bins=6 * 24, values=values)
    animals = tmp_path / "animals.csv"
    animals.write_text(
        "cage #,animal #,,genotype,tag #,Tag date,Monitoring start\n"
        "3002,1,L,W/w,1,,\n,2,R,W/w,2,,\n")
    config = tmp_path / "study.json"
    config.write_text(json.dumps({
        "name": "missing_export_animal_test", "animals_csv": str(animals),
        "cages": [{"cage": 3002, "export": str(export)}],
        "habituation_days": 0, "lights_on": 7, "lights_off": 19, "min_days": 1,
        "primary_measures": ["distance_cm_all"], "scan_measures": ["distance_cm_all"],
    }))
    with pytest.raises(ValueError, match="only in animal table"):
        run_study(load_study(config), tmp_path / "out", cache_dir=None)


def test_resample_bin_s_rebins_every_cage_before_metrics_are_computed(make_workbook, tmp_path):
    # One 10-minute-cadence cage (mirrors 58616) and one already-hourly cage
    # (mirrors the study's other five), reconciled to a common 3600 s width.
    def values(measure, tag, index):
        if measure == "Cage in rack (1/0):":
            return 1.0
        return 1.0 + tag + (index % 7)

    fine = make_workbook(name="58616_x_cage.xlsx", tags=(1,), bins=6 * 24 * 2,
                         minutes=(10,), values=values)
    coarse = make_workbook(name="1002_x_cage.xlsx", tags=(1,), bins=24 * 2,
                           minutes=(60,), values=values)
    animals = tmp_path / "animals.csv"
    animals.write_text(
        "cage #,animal #,,genotype,tag #,Tag date,Monitoring start\n"
        "58616,1,L,W/w,1,,\n1002,2,L,W/w,1,,\n")
    config = tmp_path / "study.json"
    config.write_text(json.dumps({
        "name": "resample_test", "animals_csv": str(animals),
        "cages": [{"cage": 58616, "export": str(fine)}, {"cage": 1002, "export": str(coarse)}],
        "habituation_days": 0, "lights_on": 7, "lights_off": 19, "min_days": 1,
        "primary_measures": ["distance_cm_all"], "scan_measures": ["distance_cm_all"],
        "resample_bin_s": 3600,
    }))
    result = run_study(load_study(config), tmp_path / "out", cache_dir=None)

    assert result["resample_bin_s"] == 3600
    # Only the cage whose native width differed from the target counts as
    # "actually re-binned" -- cage 1002 was already hourly.
    assert result["resampled_cages"] == [58616]

    long = result["long"]
    for cage, block in long.drop_duplicates(subset=["cage", "bin_start"]).groupby("cage"):
        starts = sorted(block["bin_start"])
        for a, b in zip(starts, starts[1:]):
            assert (b - a).total_seconds() == 3600.0

    # `native_long` still reflects the raw, un-resampled bin widths.
    native = result["native_long"]
    assert set(native.loc[native["cage"] == 58616, "duration_s"]) == {600.0}
    assert set(native.loc[native["cage"] == 1002, "duration_s"]) == {3600.0}
