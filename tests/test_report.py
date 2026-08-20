from __future__ import annotations

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
