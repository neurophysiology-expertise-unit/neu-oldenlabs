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
