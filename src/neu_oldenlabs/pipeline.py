"""Wire the nine analysis modules into one command: load, QC, aggregate, compare, plot.

Ordering matters and is deliberate, not incidental:

1. Animals are loaded once and checked against every cage's tags as it loads.
   `animals.check_join` is ASYMMETRIC (study owner's ruling: the animal CSV is
   the sole authority on which animals exist): a tag the CSV lists but the
   export lacks is a genuine discrepancy and raises; a tag the export carries
   but the CSV does not list is an untracked/phantom slot, returned rather
   than raised on, and is EXCLUDED from the frame before anything else in this
   function runs -- it never reaches QC, the rack mask, resampling, or
   `build_profile`. It is still reported: `qc.excluded_tag_table` -- computed
   on the unfiltered export data -- becomes the `excluded_tags` entry in
   `qc_tables`, written as a CSV like every other QC table.
2. `qc.qc_tables` runs on the RAW (unmasked, un-resampled, phantom-tag-free)
   long frame -- the rack mask is itself one of the things QC reports on, so QC
   must see the bins it excludes, and bin-width reporting / the missing-value
   audit must describe what actually arrived, before any resampling changes it.
3. The rack mask is applied next, on NATIVE bins, because an out-of-rack bin is
   not an observation of the animal's behaviour and must not contribute to a
   resampled value either.
4. If the study configures `resample_bin_s`, every cage's rack-masked frame is
   then re-binned to that common clock-aligned width (`metrics.resample_bins`)
   -- so a mixed-resolution study (the real study: cage 58616 at 600 s, the
   other five at 3600 s) becomes one resolution before ANY metric is computed.
   The `in_rack` mask itself is never resampled -- it no longer exists in the
   frame by this point, having already done its job in step 3.
5. Everything past that (daily aggregation, the circadian profile) runs on the
   masked, possibly-resampled frame.
6. The animal table is filtered to the cages actually loaded before
   `build_profile` is called. `build_profile` enforces a TOTAL join across the
   daily table, the circadian table, and the animal table; the study's animal
   CSV can list cages whose export has not arrived, and reconciling that is
   this function's job, not `build_profile`'s.
7. `groups.scan` is skipped -- with a recorded reason, not silently -- when no
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


def _cages_actually_resampled(regularity: pd.DataFrame, bin_s: int) -> list[int]:
    """Cages whose native bin width(s) differ from the resampling target.

    A cage already natively at ``bin_s`` is technically passed through
    `metrics.resample_bins` too, but that is a no-op in value terms, so it is
    not "actually re-binned" and should not be reported as such.
    """
    changed = []
    for cage, block in regularity.groupby("cage"):
        if set(block["duration_s"]) != {float(bin_s)}:
            changed.append(cage)
    return sorted(changed)


def run_study(study: StudyConfig, output: Path, cache_dir: Path | None = None) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    animal_table = animals_mod.load_animals(study.animals_csv)

    long_frames = []
    export_paths = []
    excluded_pairs: list[tuple[int, int]] = []
    for spec in study.available_cages:
        cage_long = io.load_cage_cached(spec.export, spec.cage, cache_dir)
        excluded_tags = animals_mod.check_join(
            animal_table, spec.cage, set(cage_long["tag"].tolist()))
        excluded_pairs.extend((spec.cage, tag) for tag in excluded_tags)
        long_frames.append(cage_long)
        export_paths.append(spec.export)

    full_long = pd.concat(long_frames, ignore_index=True)

    # The excluded-tags QC table is built from the UNFILTERED export data --
    # this is the one place a phantom tag's data is still read -- before it is
    # dropped from every frame everything else in this function sees.
    excluded_table = qc.excluded_tag_table(full_long, excluded_pairs)
    if excluded_pairs:
        excluded_keys = pd.DataFrame(sorted(set(excluded_pairs)), columns=["cage", "tag"])
        raw_long = full_long.merge(
            excluded_keys.assign(_excluded=True), on=["cage", "tag"], how="left")
        raw_long = raw_long[raw_long["_excluded"].isna()].drop(columns="_excluded")
        raw_long = raw_long.reset_index(drop=True)
    else:
        raw_long = full_long

    qc_tables = qc.qc_tables(raw_long)
    qc_tables["excluded_tags"] = excluded_table
    native_masked_long = qc.apply_rack_mask(raw_long)

    masked_long = native_masked_long
    resampled_cages: list[int] = []
    if study.resample_bin_s is not None:
        resampled_cages = _cages_actually_resampled(qc_tables["regularity"], study.resample_bin_s)
        masked_long = metrics.resample_bins(masked_long, study.resample_bin_s)

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

    # SECONDARY, cage-ignoring comparison, run at the study owner's explicit
    # request ("run analysis not within but all wildtypes and all mutants
    # irrespectful of where they are coming from") -- see the banner comment
    # in `groups.py` for what pooling away cage costs. Computed over the same
    # measure set as the primary scan, unconditionally: unlike the stratified
    # scan, this does not need any single cage to house both genotypes (it
    # ignores cage entirely), only the study as a whole to contain both
    # genotypes -- `groups.scan_pooled` skips, per-measure, with a recorded
    # reason, exactly like `groups.scan` does when that is not the case (e.g.
    # `all_wildtype_study`, which has zero mutants at all).
    scan_table_pooled = groups.scan_pooled(profile_table, measures)
    scan_table_pooled.to_csv(output / "group_comparison_pooled.csv", index=False)

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
            "resample_bin_s": study.resample_bin_s,
        },
    )

    return {
        "long": masked_long, "native_long": native_masked_long, "daily": daily_table,
        "profile": profile_table, "qc": qc_tables, "scan": scan_table,
        "scan_pooled": scan_table_pooled,
        "missing_cages": study.missing_cages, "excluded_tags": excluded_table,
        "resample_bin_s": study.resample_bin_s, "resampled_cages": resampled_cages,
    }
