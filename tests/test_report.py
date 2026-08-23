from __future__ import annotations

import dataclasses

from neu_oldenlabs import groups
from neu_oldenlabs.pipeline import run_study
from neu_oldenlabs.report import (_EFFECT_DISPLAY_COLUMNS, _INFERENCE_DISPLAY_COLUMNS,
                                  _POOLED_COMPARISON_EFFECT_COLUMNS,
                                  _POOLED_COMPARISON_INFERENCE_COLUMNS,
                                  _POOLED_SCAN_EFFECT_COLUMNS,
                                  _POOLED_SCAN_INFERENCE_COLUMNS, _format_p, _p_stars,
                                  build_study_report)


def test_report_names_missing_cages_and_the_permutation_count(synthetic_study, tmp_path):
    output = tmp_path / "out"
    path = build_study_report(synthetic_study, output)
    text = path.read_text()
    assert "1003" in text                      # the missing cage is named
    assert "within-cage" in text.lower()       # the test is described
    assert "absence of evidence" in text.lower()
    assert "min_attainable_p" in text or "smallest attainable" in text.lower()


def test_report_carries_a_published_measure_glossary(synthetic_study, tmp_path):
    # Every measure column actually reported in the profile table earns a
    # published-source definition in the Methods section, citing delali2025 --
    # not the vendor's export header, which is wrong for two of them.
    text = build_study_report(synthetic_study, tmp_path / "out").read_text()
    assert "delali2025" in text
    assert "doi:10.3389/fnbeh.2025.1696654" in text
    assert "distance_cm" in text
    # The two unit-error measures' corrected units appear, not the vendor's.
    assert "cm/s^2" in text or "cm/s²" in text
    assert "deg/s" in text
    # Interpretation-critical qualifiers from the paper survive into the report.
    assert "engagement" in text.lower() and "intake" in text.lower()
    assert "top 1%" in text.lower()
    assert "sleep proxy" in text.lower() or "sleep, i.e." in text.lower()


def test_report_describes_the_cluster_permutation_test(synthetic_study, tmp_path):
    text = build_study_report(synthetic_study, tmp_path / "out").read_text()
    lower = text.lower()
    assert "hour-of-day cluster permutation test" in lower
    assert "distance_cm" in text                 # the tested primary measure, named
    assert "cluster-forming threshold" in lower
    assert "not tuned" in lower
    assert "within cage" in lower
    assert "circular" in lower and "wrap" in lower
    assert "animals, not animal-hours" in lower
    assert "not themselves a significance test" in lower
    assert "p ~= 0.3" in text or "p ≈ 0.3" in text
    assert "permutations." in lower              # states how many were used


def test_report_cluster_section_works_on_an_all_wildtype_study(all_wildtype_study, tmp_path):
    # No cage houses both genotypes, so every measure is skipped inside
    # `groups.cluster_permutation_scan` -- the section must still render, not raise.
    text = build_study_report(all_wildtype_study, tmp_path / "out").read_text()
    assert "hour-of-day cluster permutation test" in text.lower()


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


# ============================================================================
# Table readability: split effect/inference blocks and significance stars
# (owner request: wide tables overlap in the rendered PDF; use stars instead
# of long decimals for p-values, and split each wide table so it fits).
# ============================================================================


def test_p_stars_thresholds_are_pinned_exactly():
    # p == 0.05 gets NO star (the threshold is strict "<", not "<="); 0.049
    # gets one; 0.004 gets two; 0.0004 gets three. These four values pin the
    # exact boundaries the owner asked for ("0<0.05 one star 0<0.005 2
    # start[sic] etc").
    assert _p_stars(0.05) == ""
    assert _p_stars(0.049) == "*"
    assert _p_stars(0.004) == "**"
    assert _p_stars(0.0004) == "***"


def test_p_stars_handles_missing_values():
    assert _p_stars(None) == ""
    assert _p_stars(float("nan")) == ""


def test_format_p_is_compact_and_uses_scientific_notation_for_small_values():
    assert _format_p(0.049) == "0.049"
    assert _format_p(0.0000123) == "1.23e-05"
    assert _format_p(float("nan")) == "n/a"


def test_effect_and_inference_display_columns_both_carry_the_measure_column():
    # Every split block must repeat `measure` so either half reads on its
    # own, including across a page break -- per the owner's suggestion
    # ("repeat the first column and continue tables in the following page").
    for columns in (_EFFECT_DISPLAY_COLUMNS, _INFERENCE_DISPLAY_COLUMNS,
                    _POOLED_COMPARISON_EFFECT_COLUMNS, _POOLED_COMPARISON_INFERENCE_COLUMNS,
                    _POOLED_SCAN_EFFECT_COLUMNS, _POOLED_SCAN_INFERENCE_COLUMNS):
        assert columns[0] == "measure"


def test_inference_blocks_carry_star_columns_next_to_their_p_values():
    assert "p_stars" in _INFERENCE_DISPLAY_COLUMNS
    assert "fdr_stars" in _INFERENCE_DISPLAY_COLUMNS
    assert "pooled_p_stars" in _POOLED_SCAN_INFERENCE_COLUMNS
    assert "pooled_fdr_stars" in _POOLED_SCAN_INFERENCE_COLUMNS
    assert "pooled_p_stars" in _POOLED_COMPARISON_INFERENCE_COLUMNS
    assert "pooled_fdr_stars" in _POOLED_COMPARISON_INFERENCE_COLUMNS


def test_report_prints_significance_stars_and_the_legend(synthetic_study, tmp_path):
    text = build_study_report(synthetic_study, tmp_path / "out").read_text()
    assert "Significance stars" in text
    assert "p < 0.05" in text and "p < 0.005" in text and "p < 0.0005" in text


def test_report_tables_split_into_effect_and_inference_blocks(synthetic_study, tmp_path):
    text = build_study_report(synthetic_study, tmp_path / "out").read_text()
    assert "**Effect sizes:**" in text
    assert "**Inference:**" in text
