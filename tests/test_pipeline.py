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
