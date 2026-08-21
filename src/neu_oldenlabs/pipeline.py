"""Wire the nine analysis modules into one command: load, QC, aggregate, compare, plot.

Ordering matters and is deliberate, not incidental:

1. Animals are loaded once and checked against every cage's tags as it loads,
   so a mismatched export fails before anything downstream is computed.
2. `qc.qc_tables` runs on the RAW (unmasked) long frame -- the rack mask is
   itself one of the things QC reports on, so QC must see the bins it excludes.
3. Everything past QC (daily aggregation, the circadian profile) runs on the
   rack-masked frame, because an out-of-rack bin is not an observation of the
   animal's behaviour.
4. The animal table is filtered to the cages actually loaded before
   `build_profile` is called. `build_profile` enforces a TOTAL join across the
   daily table, the circadian table, and the animal table; the study's animal
   CSV can list cages whose export has not arrived, and reconciling that is
   this function's job, not `build_profile`'s.
5. `groups.scan` is skipped -- with a recorded reason, not silently -- when no
   loaded cage contains both genotypes. A single-genotype study is an expected
   RESULT (an early cohort, a missing export), not an error.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import animals as animals_mod
from . import circadian, groups, io, metrics, plots, provenance, qc
from .config import StudyConfig
from .profile import IDENTITY, build_profile

_PHASE_SUFFIXES = ("_all", "_light", "_dark")


def _base_measure(name: str) -> str:
    """Strip a profile column's phase suffix to recover the raw export measure name."""
    for suffix in _PHASE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _any_cage_has_both_genotypes(profile_table: pd.DataFrame) -> bool:
    counts = profile_table.groupby("cage")["genotype"].nunique()
    return bool((counts >= 2).any())


def _default_scan_measures(profile_table: pd.DataFrame) -> list[str]:
    """Every profile measure column, minus identity fields and `acrophase_hour`.

    Used when `study.scan_measures` is empty (no explicit list configured).
    `acrophase_hour` is a CIRCULAR quantity (it wraps at 24h == 0h): real
    values in this study sit at 0.6-1.3h, just above the wrap, so an animal
    at 23.7h and one at 0.3h would be computed as 23.4h apart with the wrong
    sign under an ordinary linear difference. A circular quantity must not be
    differenced linearly, so it is excluded from the default scan (controller
    ruling). It is still reported in the per-animal profile table for anyone
    who wants to analyse it with a circular-appropriate statistic.
    """
    return [c for c in profile_table.columns if c not in IDENTITY and c != "acrophase_hour"]


def run_study(study: StudyConfig, output: Path, cache_dir: Path | None = None) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    animal_table = animals_mod.load_animals(study.animals_csv)

    long_frames = []
    export_paths = []
    for spec in study.available_cages:
        cage_long = io.load_cage_cached(spec.export, spec.cage, cache_dir)
        animals_mod.check_join(animal_table, spec.cage, set(cage_long["tag"].tolist()))
        long_frames.append(cage_long)
        export_paths.append(spec.export)

    raw_long = pd.concat(long_frames, ignore_index=True)

    qc_tables = qc.qc_tables(raw_long)
    masked_long = qc.apply_rack_mask(raw_long)

    daily_table = metrics.daily(masked_long, study.lights_on, study.lights_off)
    daily_table = metrics.drop_habituation(daily_table, study.habituation_days)
    daily_table = metrics.restrict_window(daily_table, study.window_start, study.window_end)

    # The circadian family (cosinor, IS/IV/RA -- IS is a pre-specified primary
    # measure) must honour the same habituation exclusion and analysis window
    # as the daily measures, or it silently describes a different slice of the
    # recording. `drop_habituation`/`restrict_window` are generic over any
    # table with "cage" and "date" columns, so they are reused here directly
    # on the bin-level frame rather than duplicating the date logic.
    circadian_long = masked_long.copy()
    circadian_long["date"] = circadian_long["bin_start"].dt.date
    circadian_long = metrics.drop_habituation(circadian_long, study.habituation_days)
    circadian_long = metrics.restrict_window(circadian_long, study.window_start, study.window_end)
    circadian_table = circadian.circadian_profile(circadian_long.drop(columns="date"))

    loaded_cages = {spec.cage for spec in study.available_cages}
    animals_loaded = animal_table[animal_table["cage"].isin(loaded_cages)].reset_index(drop=True)

    profile_table = build_profile(daily_table, circadian_table, animals_loaded, study.min_days)

    measures = list(study.scan_measures) or _default_scan_measures(profile_table)

    if _any_cage_has_both_genotypes(profile_table):
        scan_table = groups.scan(profile_table, measures)
        scan_table.to_csv(output / "group_comparison.csv", index=False)
    else:
        # No loaded cage contributes a within-cage contrast (the all-wildtype
        # case -- the real study's only currently-loaded cage, 58616, is
        # exactly this). Mirror `groups.scan`'s own idiom for "nothing to
        # compare": the returned table is genuinely empty (no `measure` rows,
        # so it carries no comparison data a caller could mistake for a real
        # result) with the reason attached as metadata, the same way `scan`
        # attaches `skipped_measures` to its own empty frame. The written
        # CSV is a separate one-row rendering of that same reason, because a
        # human reading `group_comparison.csv` needs the explanation on the
        # page, not only in an in-memory attribute.
        reason = "no loaded cage contains both genotypes; group comparison skipped"
        scan_table = pd.DataFrame(columns=["note"])
        scan_table.attrs["note"] = reason
        pd.DataFrame({"note": [reason]}).to_csv(output / "group_comparison.csv", index=False)

    daily_table.to_csv(output / "daily_measures.csv", index=False)
    profile_table.to_csv(output / "animal_profile.csv", index=False)
    for name, table in qc_tables.items():
        table.to_csv(output / f"qc_{name}.csv", index=False)

    figures = output / "figures"
    animal_genotypes = animals_loaded[["cage", "tag", "genotype"]]
    hourly_with_genotype = masked_long.merge(animal_genotypes, on=["cage", "tag"], how="left")
    daily_with_genotype = daily_table.merge(animal_genotypes, on=["cage", "tag"], how="left")
    phase_measures = list(dict.fromkeys(_base_measure(m) for m in study.primary_measures))
    trajectory_measure = phase_measures[0] if phase_measures else "distance_cm"

    plots.actogram(masked_long, figures, study.lights_on, study.lights_off)
    plots.phase_profiles(hourly_with_genotype, figures, phase_measures,
                         study.lights_on, study.lights_off)
    plots.cage_paired(profile_table, figures, list(study.primary_measures))
    plots.effect_forest(scan_table, figures)
    plots.daily_trajectories(daily_with_genotype, figures, trajectory_measure)
    plots.qc_panel(qc_tables, figures)

    provenance.write_provenance(
        output,
        inputs=[study.path, study.animals_csv, *export_paths],
        parameters={
            "study": study.name, "habituation_days": study.habituation_days,
            "lights_on": study.lights_on, "lights_off": study.lights_off,
            "min_days": study.min_days, "window_start": study.window_start,
            "window_end": study.window_end, "primary_measures": list(study.primary_measures),
            "scan_measures": list(study.scan_measures),
        },
    )

    return {
        "long": masked_long, "daily": daily_table, "profile": profile_table,
        "qc": qc_tables, "scan": scan_table, "missing_cages": study.missing_cages,
    }
