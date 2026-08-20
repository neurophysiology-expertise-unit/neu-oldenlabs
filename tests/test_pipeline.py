from __future__ import annotations

from neu_oldenlabs.pipeline import run_study


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
