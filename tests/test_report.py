from __future__ import annotations

import dataclasses

from neu_oldenlabs import groups
from neu_oldenlabs.pipeline import run_study
from neu_oldenlabs.report import build_study_report


def test_report_names_missing_cages_and_the_permutation_count(synthetic_study, tmp_path):
    output = tmp_path / "out"
    path = build_study_report(synthetic_study, output)
    text = path.read_text()
    assert "1003" in text                      # the missing cage is named
    assert "within-cage" in text.lower()       # the test is described
    assert "absence of evidence" in text.lower()
    assert "min_attainable_p" in text or "smallest attainable" in text.lower()


def test_report_states_the_dyadic_caveat_for_social_measures(synthetic_study, tmp_path):
    text = build_study_report(synthetic_study, tmp_path / "out").read_text().lower()
    assert "social distance" in text and "cage-mate" in text


def test_report_works_on_an_all_wildtype_study(all_wildtype_study, tmp_path):
    """Mirrors the real study's current live shape: one all-wildtype cage.

    `build_study_report` must not raise when no loaded cage contains both
    genotypes -- it must still produce a complete descriptive report, naming
    the single-genotype cage as contributing no within-cage contrast rather
    than silently omitting the comparison section.
    """
    output = tmp_path / "out"
    path = build_study_report(all_wildtype_study, output)
    text = path.read_text()
    assert path.exists()
    assert "58616" in text
    assert "no loaded cage contains both genotypes" in text.lower()
    assert "hypothesis-generating" in text.lower()


def test_report_prints_the_min_attainable_p_that_compare_actually_returns(
    synthetic_study, tmp_path,
):
    """Bind the printed p-value floor to the number the statistics code returns.

    A hardcoded template could pass every other test in this file while
    printing a wrong or stale floor; this test fails unless the report's text
    contains the exact `min_attainable_p` `groups.compare` computes for this
    fixture's data, formatted the way the report formats it.
    """
    result = run_study(synthetic_study, tmp_path / "reference", cache_dir=None)
    expected = groups.compare(result["profile"], "distance_cm_all")["min_attainable_p"]

    text = build_study_report(synthetic_study, tmp_path / "out").read_text()
    assert f"{expected:.6g}" in text


def test_zero_weight_cage_sentence_names_exactly_the_single_genotype_cage(
    all_wildtype_study, tmp_path,
):
    """The zero-weight-cage sentence must name this fixture's one cage and its
    actual mut/wt counts, not a generic or hardcoded phrase."""
    text = build_study_report(all_wildtype_study, tmp_path / "out").read_text()
    assert "cage 58616 (0 mut / 5 wt)" in text


def test_skipped_scan_prints_the_real_reason_not_the_no_both_genotypes_text(
    synthetic_study, tmp_path,
):
    """When cages ARE genotype-mixed but every scanned measure is individually
    skipped (constant or absent), the report must print that real reason --
    never the "no loaded cage contains both genotypes" text, which would be
    false here: both synthetic cages are genotype-mixed.
    """
    study = dataclasses.replace(synthetic_study, scan_measures=("nonexistent_measure_all",))
    text = build_study_report(study, tmp_path / "out").read_text().lower()
    assert "nonexistent_measure_all (absent)" in text
    assert "no loaded cage contains both genotypes" not in text


def _single_cage_study(make_workbook, tmp_path, minutes):
    """One all-wildtype cage (mirrors `all_wildtype_study`) with a chosen bin width
    pattern, so the aggregation sentence's uniform vs. varying-duration branch can
    be exercised without hardcoding either observed distribution into the test."""
    import json

    from neu_oldenlabs.config import load_study

    def varying(measure, tag, index):
        if measure == "Cage in rack (1/0):":
            return 1.0
        return 1.0 + tag + (index % 7)

    export = make_workbook(name="58616_x_cage.xlsx", tags=(1, 2, 3, 4, 5),
                           bins=6 * 24 * 2, minutes=minutes, values=varying)
    animals = tmp_path / "animals.csv"
    rows = ["cage #,animal #,,genotype,tag #,Tag date,Monitoring start"]
    for index, tag in enumerate((1, 2, 3, 4, 5), start=1):
        prefix = "58616" if tag == 1 else ""
        rows.append(f"{prefix},{index},L,W/w,{tag},,")
    animals.write_text("\n".join(rows) + "\n")

    config = tmp_path / "study.json"
    config.write_text(json.dumps({
        "name": "bin_timing_check", "animals_csv": str(animals),
        "cages": [{"cage": 58616, "export": str(export)}],
        "habituation_days": 0, "lights_on": 7, "lights_off": 19, "min_days": 1,
        "primary_measures": ["distance_cm_all"], "scan_measures": ["distance_cm_all"],
    }))
    return load_study(config)


def test_methods_aggregation_sentence_states_uniform_bin_duration_from_the_fixture(
    make_workbook, tmp_path,
):
    """When every fixture bin has the same duration (the real cage-58616 shape),
    the Methods sentence must say so, derived from `qc.bin_regularity` on this
    run's own data -- not assert a jitter range this fixture does not have."""
    study = _single_cage_study(make_workbook, tmp_path, minutes=(10,))
    text = build_study_report(study, tmp_path / "out").read_text()
    assert "spans exactly 10 minutes (600 s)" in text
    assert "no bin-duration jitter in this export" in text.lower()
    assert "not uniform" not in text.lower()


def test_methods_aggregation_sentence_reports_the_range_when_bins_actually_vary(
    make_workbook, tmp_path,
):
    """When the fixture's bins genuinely vary (9/10/11 minutes), the Methods
    sentence must report that observed range instead of the uniform-duration
    wording -- proving the sentence tracks THIS fixture's data, not a fixed
    string reused from the uniform case."""
    study = _single_cage_study(make_workbook, tmp_path, minutes=(10, 9, 11))
    text = build_study_report(study, tmp_path / "out").read_text()
    assert "bin duration is not uniform" in text.lower()
    assert "ranges from 9 to 11 minutes" in text.lower()
    assert "spans exactly" not in text.lower()


def _two_cage_bin_width_study(make_workbook, tmp_path):
    """One 10-minute-cadence cage (mirrors 58616) and one 60-minute-cadence
    cage (mirrors the study's other five), each internally uniform but
    different from each other -- exactly the shape that makes cadence
    conflation across cages observable."""
    import json

    from neu_oldenlabs.config import load_study

    def varying(measure, tag, index):
        if measure == "Cage in rack (1/0):":
            return 1.0
        return 1.0 + tag + (index % 7)

    fine = make_workbook(name="58616_x_cage.xlsx", tags=(1,), bins=6 * 24,
                         minutes=(10,), values=varying)
    coarse = make_workbook(name="1002_x_cage.xlsx", tags=(1,), bins=24,
                           minutes=(60,), values=varying)
    animals = tmp_path / "animals.csv"
    animals.write_text(
        "cage #,animal #,,genotype,tag #,Tag date,Monitoring start\n"
        "58616,1,L,W/w,1,,\n"
        "1002,2,L,W/w,1,,\n"
    )
    config = tmp_path / "study.json"
    config.write_text(json.dumps({
        "name": "bin_timing_two_cage", "animals_csv": str(animals),
        "cages": [{"cage": 58616, "export": str(fine)},
                  {"cage": 1002, "export": str(coarse)}],
        "habituation_days": 0, "lights_on": 7, "lights_off": 19, "min_days": 1,
        "primary_measures": ["distance_cm_all"], "scan_measures": ["distance_cm_all"],
    }))
    return load_study(config)


def test_methods_bin_timing_judges_each_cages_cadence_against_its_own_bin_width(
    make_workbook, tmp_path,
):
    """A study mixing a 10-minute-cadence cage with a 60-minute-cadence one must
    not pool their gaps into one global modal cadence -- doing so scores the
    60-minute cage's own normal, gapless cadence as a large "coverage hole"
    relative to the 10-minute cage's mode, which is false: cage 1002's cadence
    is perfectly uniform on its own terms. The sentence must say so per cage."""
    study = _two_cage_bin_width_study(make_workbook, tmp_path)
    text = build_study_report(study, tmp_path / "out").read_text().lower()
    # Cage 1002's own cadence must be reported as uniform, not as a coverage
    # hole scored against cage 58616's 10-minute mode.
    assert "cage 1002: uniform at 60 minutes" in text
    assert "50-min coverage hole" not in text
    assert "cage 58616: uniform at 10 minutes" in text
