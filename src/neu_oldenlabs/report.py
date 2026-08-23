"""Assemble `report.md` -- the deliverable a scientist actually reads.

Every sentence that names a number here -- a missing cage, the exact assignment
count and its p-value floor, a cage that contributes no within-cage contrast --
is generated from the frames `pipeline.run_study` returns, never typed by hand,
so the report cannot drift from the analysis it describes.

`DataFrame.to_markdown()` needs the optional `tabulate` package, which is not
installed in this environment. Rather than add the dependency, tables are
rendered by hand with `_markdown_table` below.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import groups, qc
from .config import StudyConfig
from .io import MEASURE_DEFINITIONS, MEASURE_KIND
from .pipeline import _base_measure, run_study
from .profile import IDENTITY

_DELALIC_CITATION = (
    "Delalic, Florea et al. (2025), \"Smart Lids for deep multi-animal "
    "phenotyping in standard home cages\", Front. Behav. Neurosci., "
    "doi:10.3389/fnbeh.2025.1696654"
)

# Figure filenames, in the order `pipeline.run_study` writes them, with the
# caption each earns per the design spec's figure list (section 8).
_FIGURES = [
    ("actogram", "Actogram: per-animal 24-hour x day distance heatmap, "
                 "night phase shaded."),
    ("phase_profiles", "24-hour genotype mean +/- SEM profile, per primary measure, "
                       "night phase shaded."),
    ("hour_clusters", "The same 24-hour mean +/- SEM profile, with the cluster-based "
                      "permutation test's observed hour-cluster(s) shaded -- red if "
                      "FDR q < 0.05, grey otherwise. The SEM band by itself is NOT a "
                      "significance test -- see Hour-of-day cluster permutation test "
                      "below."),
    ("cage_paired", "Cage-paired dot plot: animals as points, grouped by cage and "
                    "coloured by genotype -- the same contrast the statistics test."),
    ("effect_forest", "Effect-size forest: stratified difference and its 95% cluster "
                      "bootstrap CI, per measure, ranked by p-value; red marks FDR "
                      "q < 0.05."),
    ("daily_trajectories", "Daily trajectories: one line per animal across the "
                           "recording window, coloured by genotype."),
    ("qc_panel", "QC panel: rack-mask fraction, recording breaks, out-of-range "
                "values, mean valid hours per day, fraction of values "
                "missing (NaN), excluded tags, and non-finite in_rack bins, "
                "one bar per cage; the title flags when bin width varies "
                "across cages."),
]


def _format_cell(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        text = "n/a" if pd.isna(value) else f"{value:.4g}"
    elif value is None:
        text = "n/a"
    else:
        text = str(value)
    return text.replace("|", "\\|")


# p-value-shaped columns get their own compact formatter (3 significant
# figures, switching to scientific notation for small magnitudes -- `g`
# already does this) rather than `_format_cell`'s generic 4g: the owner's
# complaint was specifically that a long decimal p-value overlaps its
# neighbours in the rendered PDF. Both the raw and both pooled p columns are
# listed here so the fix applies wherever any of them is displayed, not just
# in the tables split out below.
_P_VALUE_COLUMNS = {"p_value", "p_adjusted_bh", "pooled_p_value", "pooled_p_adjusted_bh"}


def _format_p(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{float(value):.3g}"


def _p_stars(value) -> str:
    """Significance stars for a p-value: `*` p < 0.05, `**` p < 0.005,
    `***` p < 0.0005, blank otherwise (including NaN/None) -- for reading at a
    glance; the numeric p-value is kept alongside (via `_format_p`) for quoting.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    value = float(value)
    if value < 0.0005:
        return "***"
    if value < 0.005:
        return "**"
    if value < 0.05:
        return "*"
    return ""


_STAR_LEGEND = ("*Significance stars, applied to both the raw and the FDR-adjusted "
               "p-value: `*` p < 0.05, `**` p < 0.005, `***` p < 0.0005; blank = not "
               "significant at any of these thresholds. The numeric p-value is kept "
               "alongside (3 significant figures) for quoting.*")


def _with_significance_stars(frame: pd.DataFrame, p_column: str, fdr_column: str,
                             p_stars_column: str, fdr_stars_column: str) -> pd.DataFrame:
    """Return a COPY of `frame` with two new star columns added.

    Always copies rather than mutating in place: `frame` here is a live
    result frame (e.g. `result["scan"]`) that other sections read from too,
    and must not pick up display-only columns as a side effect of rendering
    one table.
    """
    frame = frame.copy()
    frame[p_stars_column] = (frame[p_column].map(_p_stars) if p_column in frame.columns
                             else "")
    frame[fdr_stars_column] = (frame[fdr_column].map(_p_stars) if fdr_column in frame.columns
                               else "")
    return frame


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Render a DataFrame as a markdown table by hand (no `tabulate` dependency)."""
    if frame.empty:
        return "*(no rows)*"
    columns = columns or list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"

    def cell(row, column):
        return _format_p(row[column]) if column in _P_VALUE_COLUMNS else _format_cell(row[column])

    body = ["| " + " | ".join(cell(row, c) for c in columns) + " |"
            for _, row in frame.iterrows()]
    return "\n".join([header, sep, *body])


def _data_and_coverage(study: StudyConfig, result: dict) -> str:
    profile = result["profile"]
    daily = result["daily"]
    rack = result["qc"]["rack"]

    loaded_cages = sorted({spec.cage for spec in study.available_cages})
    cages_line = (", ".join(str(c) for c in loaded_cages) if loaded_cages
                 else "none (no export has arrived)")
    n_animals = len(profile)

    missing = study.missing_cages
    missing_line = (", ".join(str(c) for c in missing) if missing
                    else "none -- every configured cage's export has arrived")

    all_phase = daily[daily["phase"] == "all"]
    windows = (all_phase.groupby("cage")["date"].agg(["min", "max", "nunique"])
              .rename(columns={"min": "first_day", "max": "last_day",
                               "nunique": "days_retained"})
              .reset_index())

    rack_fraction = (rack.groupby("cage", as_index=False)["fraction_out"].mean()
                     .rename(columns={"fraction_out": "mean_fraction_out_of_rack"}))
    windows = windows.merge(rack_fraction, on="cage", how="left")

    lines = [
        "## Data and coverage",
        "",
        f"- Cages analysed (export loaded): {cages_line} ({n_animals} animals total).",
        f"- Cages missing (export not received): {missing_line}.",
        f"- Habituation days dropped: {study.habituation_days}.",
        f"- Analysis window: "
        f"{study.window_start or 'start of recording'} to "
        f"{study.window_end or 'end of recording'}.",
        "",
        "Recording window, days retained after habituation, and rack-mask fraction, "
        "per cage:",
        "",
        _markdown_table(windows, ["cage", "first_day", "last_day", "days_retained",
                                  "mean_fraction_out_of_rack"]),
        "",
    ]
    return "\n".join(lines)


def _quality_control(result: dict) -> str:
    """Bin width per cage and the missing-value audit, stated explicitly.

    Both come straight from `qc.qc_tables` (via `result["qc"]`): bin width
    from `qc.bin_regularity` (already computed, previously not surfaced
    anywhere a reader would see it), and the missing-value fraction from the
    new `qc.missing_values` audit. A mixed-resolution study (this study's real
    exports: cage 58616 uses 600 s bins, the other five use 3600 s) or a
    silently-coerced non-numeric cell (a literal `'NaN'` string, for example)
    must be STATED here, not left for a reader to discover by noticing a
    figure looks wrong.
    """
    regularity = result["qc"]["regularity"]
    missing = result["qc"]["missing"]

    lines = ["## Quality control", ""]

    if regularity.empty:
        lines += ["No bins were loaded, so bin width cannot be described.", ""]
    else:
        widths_by_cage = {cage: sorted(set(block["duration_s"]))
                          for cage, block in regularity.groupby("cage")}
        distinct = sorted({w for widths in widths_by_cage.values() for w in widths})
        if len(distinct) > 1:
            per_cage = "; ".join(
                f"cage {cage}: {'/'.join(f'{w:.0f}' for w in widths)} s"
                for cage, widths in sorted(widths_by_cage.items()))
            lines += [
                f"**Bin width varies across cages in this run:** {per_cage}. Every "
                "figure and metric in this report normalises to a duration-weighted, "
                "per-valid-hour basis before comparing cages -- a raw per-bin value "
                "is NOT comparable across different bin widths (pooling a 600 s-bin "
                "cage with a 3600 s-bin cage in raw per-bin units would manufacture a "
                "difference that is pure bin-width artefact).",
                "",
            ]
        else:
            lines += [f"Every cage in this run shares one bin width: "
                     f"{distinct[0]:.0f} s.", ""]

    total_bins = int(missing["bins"].sum()) if not missing.empty else 0
    total_missing = int(missing["missing"].sum()) if not missing.empty else 0
    if total_missing > 0:
        worst = (missing[missing["missing"] > 0]
                .sort_values("fraction_missing", ascending=False).head(10))
        lines += [
            f"**Missing values.** {total_missing} of {total_bins} loaded bins "
            f"({total_missing / total_bins:.2%}) carry a non-numeric value (e.g. a "
            "literal `'NaN'` string), coerced to NaN at parse time (`io.load_cage`). "
            "These bins are excluded from the duration weight in every downstream "
            "aggregate, not counted as zero; a cage/tag/measure with no non-missing "
            "bin at all yields NaN, never 0.0. Worst-affected cage/tag/measure "
            "combinations:",
            "",
            _markdown_table(worst, ["cage", "tag", "measure", "bins", "missing",
                                    "fraction_missing"]),
            "",
        ]
    else:
        lines += ["**Missing values.** No non-numeric values were found in any "
                 "loaded export.", ""]

    lines += [_non_finite_rack_paragraph(result["qc"].get("rack_non_finite")), ""]
    lines += [_excluded_tags_paragraph(result["qc"].get("excluded_tags")), ""]
    lines += [_distance_resolution_paragraph(result["qc"].get("distance_resolution")), ""]

    return "\n".join(lines)


def _distance_resolution_paragraph(distance_resolution: pd.DataFrame | None) -> str:
    """State any NATIVE (pre-resampling) bin-width mismatch and what it taints.

    `distance_cm` is a PATH LENGTH, and path length is sampling-rate dependent
    (the coastline problem) -- see `qc.distance_resolution_confound`, which
    this reads verbatim (`result["qc"]["distance_resolution"]`). A finer
    native resolution captures micro-movement a coarser native resolution
    never sees, and downstream resampling to a common width
    (`metrics.resample_bins`) cannot undo it -- the vendor already computed
    each bin's path length at its own native resolution. This is an expected
    RESULT for a mixed-resolution study, not an error, so it is stated here
    plainly rather than left for a reader to discover by noticing a figure
    looks wrong -- and again next to the pooled-comparison section below,
    since that is where it actually bites.
    """
    if distance_resolution is None or distance_resolution.empty:
        return ("**Distance-resolution confound.** No bin-width data was "
                "available to check for a native-resolution mismatch across "
                "cages.")
    if not bool(distance_resolution["bin_width_differs_in_study"].any()):
        return (
            "**Distance-resolution confound.** Every cage in this run shares "
            "one NATIVE (pre-resampling) bin width, so `distance_cm` and "
            "everything built from it (`io.DISTANCE_DERIVED_PROFILE_MEASURES`: "
            "the circadian family L5, M10, RA, mesor, amplitude, IS, IV, "
            "acrophase_hour, plus `distance_dark_light_ratio` and "
            "`distance_cv_daily`) carries no resolution confound in this run."
        )
    by_width: dict[float, list[int]] = {}
    for row in distance_resolution.itertuples():
        by_width.setdefault(float(row.native_bin_width_s), []).append(int(row.cage))
    groups_desc = "; ".join(
        f"{width:.0f} s: cage(s) {', '.join(str(c) for c in sorted(cages))}"
        for width, cages in sorted(by_width.items()))
    affected = distance_resolution["distance_derived_measures_affected"].iloc[0]
    return (
        "**Distance-resolution confound.** Cages in this run differ in NATIVE "
        f"(pre-resampling) bin width: {groups_desc}. `distance_cm` is a PATH "
        "LENGTH, which is sampling-rate dependent (the coastline problem): a "
        "finer native resolution captures micro-movement the same behaviour, "
        "recorded natively at a coarser resolution, never sees, and "
        "`metrics.resample_bins` cannot undo this after the fact -- the "
        "vendor already computed each bin's path length at its own native "
        f"resolution before this pipeline ever sees it. Affected measures: "
        f"{affected}. A WITHIN-CAGE (cage-stratified) comparison of these "
        "measures remains valid -- see `group_comparison.csv` -- because "
        "every cage's own bins share one native width, so a within-cage "
        "difference is never a resolution artefact. A POOLED or other "
        "cross-cage comparison of any of these measures is NOT safe: see the "
        "`pooled_distance_bin_width_confound` column in "
        "`group_comparison_pooled.csv` (True marks exactly the affected "
        "rows) and the Secondary analysis section below."
    )


def _non_finite_rack_paragraph(non_finite: pd.DataFrame | None) -> str:
    """Name any cage/tag whose `in_rack` reading was non-finite (NaN, +-inf).

    Controller ruling (see `qc.valid_bins`): such a bin is excluded from
    analysis -- it is not evidence the cage was in the rack -- but must be
    reported by name and kept separate from a genuine out-of-rack (0) bin,
    never silently dropped and never folded into the out-of-rack fraction
    above.
    """
    if non_finite is None or non_finite.empty:
        return ("**Non-finite rack readings.** Every `in_rack` value in every "
                "loaded export was finite; none were excluded on this basis.")
    total = int(non_finite["n_non_finite_bins"].sum())
    parts = [f"cage {int(row.cage)} tag {int(row.tag)} ({int(row.n_non_finite_bins)} bin(s))"
             for row in non_finite.itertuples()]
    return (
        f"**Non-finite rack readings.** {total} bin(s) had a non-finite `in_rack` "
        "value (NaN or +-inf) and were excluded from analysis -- not treated as "
        "in-rack, and NOT counted in the out-of-rack fraction above (a different "
        "failure mode, reported separately per the study owner's ruling): "
        + "; ".join(parts) + ". See `qc_rack_non_finite.csv` for the full table."
    )


def _excluded_tags_paragraph(excluded: pd.DataFrame | None) -> str:
    """Name any export tag not listed in the animal CSV and why it was dropped.

    Per the study owner's ruling (`animals.check_join`), the animal CSV is the
    sole authority on which animals exist: a tag present in an export but not
    listed there is not a tracked animal -- it may be an animal removed from
    the cage or one whose ear tag was lost -- and is kept out of group
    analysis entirely, regardless of how much data it carries. This must be
    stated, not left for a reader to notice the animal counts do not match the
    raw export.
    """
    if excluded is None or excluded.empty:
        return ("**Excluded tags.** Every tag present in a loaded export is listed "
                "in the animal CSV; no tag was excluded from analysis.")
    parts = [f"cage {int(row.cage)} tag {int(row.tag)} "
             f"({row.missing_fraction:.1%} of {int(row.bins)} bins missing)"
             for row in excluded.itertuples()]
    return (
        f"**Excluded tags.** {len(excluded)} tag(s) appear in a loaded export but are "
        "NOT listed in the animal CSV, per the study owner's ruling that the CSV is "
        "the sole authority on which animals exist -- each may be an animal removed "
        "from the cage or one whose ear tag was lost and buried to the ground. These "
        "are excluded from all analysis (profile, statistics, figures) regardless of "
        "how much data they carry, and never reach `build_profile`: "
        + "; ".join(parts) + ". See `qc_excluded_tags.csv` for the full table."
    )


def _count_kind_measures(measures: list[str]) -> list[str]:
    """Base measure names among `measures` whose daily rule is per-valid-hour.

    `metrics._aggregate_measure` turns a `count`-kind measure into a rate per
    valid hour (see `MEASURE_KIND` in `io.py`), not a per-day total. This reads
    that mapping directly rather than hardcoding which measures it applies to,
    so the sentence below cannot drift if a measure's kind ever changes.

    `measures` should be every measure ACTUALLY REPORTED (the profile table's
    own columns), not just `study.primary_measures + study.scan_measures`:
    with an empty `scan_measures` (the common case -- see
    `pipeline._default_scan_measures`), that config-only list omits
    `climbing_events_*`, `rearing_events_*`, and `sleep_bouts_*` whenever none
    of them happens to be a primary measure, even though they are still
    per-valid-hour columns sitting right there in the profile table and the
    full scan. A reader would then read e.g. `rearing_events_dark` as a
    nightly count with nothing in the Methods section to correct them.
    """
    bases = sorted({_base_measure(m) for m in measures})
    return [b for b in bases if MEASURE_KIND.get(b) == "count"]


def _count_kind_sentence(per_hour: list[str]) -> str:
    if not per_hour:
        return ("No measure in this report's primary or scan set uses the count "
                "aggregation rule (rate per valid hour).")
    base = (
        f"Per the export's aggregation rule (`io.MEASURE_KIND`), these measures are "
        f"summed over bins and expressed as a rate PER VALID HOUR, not per day -- "
        f"their `_all`/`_light`/`_dark` profile columns are the bin total divided by "
        f"the cage's valid hours in that window: {', '.join(per_hour)}.")
    if "aggression_events" in per_hour:
        base += (" In particular, `aggression_events_all` is events per hour, "
                 "NOT per day.")
    return base


def _measure_glossary(reported_measures: list[str]) -> str:
    """Published definition of every raw exported measure actually reported.

    `reported_measures` is the profile table's own columns (see `_methods`),
    stripped of phase suffix via `_base_measure` -- the same set the per-hour
    disclosure above (`_count_kind_sentence`) is built from -- so this glossary
    lists exactly the measures this run's report carries, never the full
    22-measure catalogue regardless of what was actually analysed. Derived
    circadian statistics (`IS`, `IV`, `RA`, `acrophase_hour`, ...) are not raw
    exported measures and have no entry in `io.MEASURE_DEFINITIONS`, so they
    are silently skipped here rather than raising.

    Definitions come from the instrument paper -- `delali2025` -- via
    `io.MEASURE_DEFINITIONS`, kept in sync with `io.MEASURES`/`io.MEASURE_KIND`
    by `test_io.py` so this table cannot silently drop a measure the pipeline
    still produces.
    """
    bases = sorted({_base_measure(m) for m in reported_measures})
    known = [b for b in bases if b in MEASURE_DEFINITIONS]
    if not known:
        return ""
    rows = pd.DataFrame({"measure": known,
                         "definition": [MEASURE_DEFINITIONS[b] for b in known]})
    lines = [
        "**Measure definitions (published source).** Every measure reported "
        f"below is defined per the instrument paper, {_DELALIC_CITATION} "
        "(citekey `delali2025`) -- not reinterpreted from the vendor's export "
        "column header, which for two measures (`accel_cm_s`, "
        "`angular_velocity_cm_s`) understates its own unit; see the note in "
        "those two rows.",
        "",
        _markdown_table(rows, ["measure", "definition"]),
        "",
    ]
    return "\n".join(lines)


def _bin_timing_sentence(result: dict) -> str:
    """Describe observed bin span and start-to-start cadence, read off the data.

    A bin's own span (`Datetime to` minus `Datetime from`, via `qc.bin_regularity`)
    and the gap between consecutive bin starts are two different quantities. The
    real cage-58616 export has a constant 600 s span for every bin but an
    irregular cadence (most gaps 10 minutes, some 11, a few 9), so this must be
    generated from `result["native_long"]` rather than asserted, and the two
    quantities must stay distinct in the sentence.

    Deliberately reads `result["native_long"]` (the rack-masked frame BEFORE
    any `resample_bin_s` re-binning), not `result["long"]` (which is
    post-resampling when resampling is configured): this sentence describes
    what actually arrived from Oldenlabs, per the pipeline's own rule that QC
    and bin-width reporting describe the raw data, not the analysis
    resolution. The resampling itself, when in effect, is stated separately
    (see `_resampling_sentence`).

    Cadence is judged PER CAGE, against that cage's own modal gap -- the real
    study mixes a 10-minute-cadence cage (58616) with five 60-minute-cadence
    cages, and pooling every cage's gaps into one modal value before judging
    deviations would score an hourly cage's own normal, gapless 60-minute
    cadence as a "coverage hole" relative to a different cage's 10-minute
    mode (the same spacing-vs-duration conflation `643ab77` removed from the
    Methods text, resurfacing here across cages instead of within one).
    `qc.recording_breaks` already uses this per-cage-relative pattern for the
    same reason.
    """
    long = result["native_long"]
    if long.empty:
        return "No bins were loaded in this run, so bin span and cadence cannot be described."

    regularity = qc.bin_regularity(long)
    durations_min = sorted(set(regularity["duration_s"] / 60.0))
    total_bins = int(regularity["bins"].sum())

    if len(durations_min) == 1:
        duration = durations_min[0]
        span_line = (
            f"Every one of the {total_bins} bins in this run spans exactly "
            f"{duration:.4g} minutes ({int(round(duration * 60))} s) -- there is no "
            f"bin-DURATION jitter in this export. Every rate or mean is still "
            f"weighted by each bin's actual duration rather than a plain mean over "
            f"bins, as a safeguard should a future export vary, not as a response "
            f"to jitter observed here."
        )
    else:
        span_line = (
            f"Bin duration is NOT uniform in this run: across {total_bins} bins it "
            f"ranges from {durations_min[0]:.4g} to {durations_min[-1]:.4g} minutes "
            f"(`qc.bin_regularity`), so every rate or mean is weighted by each bin's "
            f"actual duration rather than a plain mean over bins."
        )

    bins_sorted = (long.drop_duplicates(subset=["cage", "bin_start"])
                   .sort_values(["cage", "bin_start"]))

    cage_lines = []
    for cage, cage_bins in bins_sorted.groupby("cage", sort=True):
        gaps = cage_bins["bin_start"].diff().dt.total_seconds() / 60.0
        gaps = gaps.dropna()
        if gaps.empty:
            continue
        gap_counts = gaps.round(6).value_counts().sort_index()
        total_gaps = int(gap_counts.sum())
        if len(gap_counts) == 1:
            cage_lines.append(
                f"cage {cage}: uniform at {gap_counts.index[0]:.4g} minutes across "
                f"all {total_gaps} gaps")
        else:
            modal_gap = gap_counts.idxmax()
            off_parts = []
            for value, count in gap_counts.items():
                if value == modal_gap:
                    continue
                delta = value - modal_gap
                kind = "coverage hole" if delta > 0 else "overlap"
                off_parts.append(
                    f"{int(count)} at {value:.4g} min ({abs(delta):.4g}-min {kind})")
            cage_lines.append(
                f"cage {cage}: {int(gap_counts[modal_gap])} of {total_gaps} gaps are "
                f"{modal_gap:.4g} minutes (its own modal cadence); " + "; ".join(off_parts))
    if not cage_lines:
        return span_line

    cadence_line = (
        "Start-to-start CADENCE between consecutive bins is a distinct quantity from "
        "bin duration, and each cage is judged against its OWN modal gap -- this study "
        "mixes cages of different bin widths, so pooling every cage's gaps into one "
        "modal value before judging deviations would misscore one cage's normal cadence "
        "as a gap relative to another's: " + "; ".join(cage_lines) + ".")
    return span_line + " " + cadence_line


def _resampling_sentence(result: dict) -> str:
    """State whether `resample_bin_s` is in effect, its target width, and which
    cages it actually changed.

    Per the study owner's ruling ("downsample minutely to hourly ... keep the
    analysis to hourly at this moment"), a study configured with
    `resample_bin_s` re-bins every cage's masked frame to that common,
    clock-aligned width before any metric is computed (`metrics.resample_bins`),
    so the daily/circadian family and every figure see one resolution even
    when the raw exports do not (the real study: cage 58616 at 600 s, the
    other five at 3600 s). `resampled_cages` (from `pipeline.run_study`) names
    only the cages whose NATIVE width actually differed from the target --
    a cage already at the target width is technically resampled too but that
    is a no-op in value terms, so it is not named as "re-binned" here.
    """
    bin_s = result.get("resample_bin_s")
    if bin_s is None:
        return ("**Resampling.** `resample_bin_s` is not set for this run: every "
                "cage's native bin width is used as-is, and no re-binning was applied.")
    resampled = result.get("resampled_cages") or []
    if resampled:
        cages_line = ", ".join(str(c) for c in resampled)
        detail = (f"Cage(s) actually re-binned (native width differed from the target): "
                  f"{cages_line}.")
    else:
        detail = ("No loaded cage's native width differed from the target, so this "
                  "run's resampling was a no-op in value terms for every cage.")
    return (
        f"**Resampling.** Every cage's masked frame was downsampled to a common, "
        f"clock-aligned {bin_s} s bin width (`resample_bin_s` = {bin_s} in the study "
        f"configuration) before any metric (daily aggregation, the circadian family, "
        f"every figure) was computed. {detail} Aggregation follows the same "
        f"duration-weighted, presence-aware rule as every other aggregation step in "
        f"this pipeline (`metrics.resample_bins`): a target bin with no present "
        f"source bin is NaN, never 0.0."
    )


def _methods(study: StudyConfig, result: dict) -> str:
    scan = result["scan"]
    # Every measure column ACTUALLY REPORTED in the profile table -- not just
    # `study.primary_measures + study.scan_measures` -- so the disclosure
    # below covers climbing_events/rearing_events/sleep_bouts even when
    # scan_measures is empty and none of them happens to be a primary
    # measure. See _count_kind_measures.
    reported_measures = [c for c in result["profile"].columns if c not in IDENTITY]
    per_hour = _count_kind_measures(reported_measures)
    per_hour_line = _count_kind_sentence(per_hour)

    if "measure" in scan.columns and not scan.empty:
        example = scan.iloc[0]
        assignments_line = (
            f"For example, `{example['measure']}` enumerates {int(example['assignments'])} "
            f"label assignments; the smallest attainable two-sided p-value for that "
            f"measure is `min_attainable_p` = {example['min_attainable_p']:.6g}, reported "
            f"by the code rather than assumed."
        )
    else:
        assignments_line = (
            "No measure in this run has a contributing cage pair, so no assignment "
            "count or `min_attainable_p` was computed (see Results)."
        )

    lines = [
        "## Methods",
        "",
        f"**Aggregation.** {_bin_timing_sentence(result)} Maxima take the maximum. "
        "Counts (see below) become a rate per valid hour, so a day that lost bins to "
        "the rack mask is not scored as a day of low activity. Bins where "
        "`Cage in rack == 0` are excluded from every metric before aggregation. Note "
        "that `speed_max_cm_s` is already outlier-robust at the per-BIN level -- per "
        "the instrument paper (`delali2025`), each bin's own value is the mean of the "
        "top 1% of frame-wise speeds during active periods within that bin, not a "
        "single-frame maximum -- so \"maxima take the maximum\" describes only this "
        "pipeline's further aggregation of that already-robust per-bin value across "
        "bins, never a raw single-frame peak.",
        "",
        _resampling_sentence(result),
        "",
        per_hour_line,
        "",
        _measure_glossary(reported_measures),
        "",
        f"**Phase definition.** Light phase is `hour in [{study.lights_on}, "
        f"{study.lights_off})`; every other hour is dark phase (a "
        f"{study.lights_off - study.lights_on}:"
        f"{24 - (study.lights_off - study.lights_on)} schedule).",
        "",
        "**Cage-stratified statistic.** The experimental unit is the animal, but "
        "animals are nested in cages, and cage composition ranges over which "
        "genotypes share it. The primary statistic is the weighted mean of "
        "within-cage (mut - wt) differences, weighted by "
        "`n_mut * n_wt / (n_mut + n_wt)` per cage. A cage housing a single genotype "
        "has weight zero and drops out of the test by construction -- see "
        "Interpretation safeguards for which cage(s) that is in this run. Every "
        "per-measure row below reports both `n_mut`/`n_wt` (every animal with a "
        "non-NaN value, including any zero-weight cage) and "
        "`n_mut_contributing`/`n_wt_contributing` (animals in cages that actually "
        "carry weight in the test); read the CONTRIBUTING counts as the sample "
        "size the comparison actually used. Genotype "
        "labels are permuted only WITHIN a cage, so cage-level differences (housing, "
        "cohort, cage-mate composition) cannot masquerade as a genotype effect. The "
        "design enumerates `prod(C(n_cage, k_cage))` over contributing cages -- small "
        "enough to enumerate exactly for this study's cage sizes, so the test is "
        "EXACT, not sampled.",
        "",
        assignments_line,
        "",
        "The 2/assignments floor for the smallest attainable two-sided p-value holds "
        "only when every contributing cage is genotype-balanced (equal mut and wt "
        "counts), because only then is the fully sign-flipped labelling itself one of "
        "the enumerated assignments, guaranteed to tie the observed statistic under "
        "the absolute value. When any contributing cage is unbalanced, only the "
        "observed labelling is guaranteed to produce that tie, and the correct floor "
        "is 1/assignments, not 2/assignments. `compare()` returns this floor per "
        "measure as `min_attainable_p`; this report always prints that returned "
        "value and never assumes 2/assignments.",
        "",
        "**Effect size and interval.** Hedges' g is computed over CONTRIBUTING cages "
        "only (weight > 0), using the stratified difference itself as its numerator, "
        "so the effect size describes the same quantity as the test. Its pooled "
        "standard deviation additionally drops any contributing cage where either "
        "genotype cell has fewer than two animals (a within-genotype variance needs "
        "at least two animals to estimate), so the denominator can be pooled over "
        "fewer cages than the numerator's weighted sum. Alongside it, a percentile "
        "cluster bootstrap (resample cages with replacement, then animals within "
        "genotype within cage, 5000 draws) gives a 95% interval on the stratified "
        "difference. With few contributing cages this percentile interval is "
        "ANTI-CONSERVATIVE: independent simulations run during this project's build "
        "put coverage in the low nineties against a nominal 95% at five contributing "
        "cages. The interval below should be read as approximate, not exact, "
        "especially when `cages_contributing` is small.",
        "",
        "**Sensitivity analysis.** `MixedLM(value ~ genotype, groups=cage)` "
        "(statsmodels) is reported beside the permutation result for every measure. "
        "Where the two disagree, the disagreement is reported rather than choosing "
        "the more favourable answer after the fact. A failed or boundary fit is "
        "reported as `lmm_converged = False` with the warning text kept, not hidden "
        "behind a number drawn from a singular Hessian.",
        "",
        "**Multiplicity.** The scan across every candidate measure is corrected with "
        "Benjamini-Hochberg FDR (`p_adjusted_bh`) and is HYPOTHESIS-GENERATING, not "
        "confirmatory. The pre-specified PRIMARY measures, named in the study "
        "configuration before the run, are reported first and on their own terms.",
        "",
    ]
    return "\n".join(lines)


# Split in two so neither block overflows the rendered PDF's page width (the
# owner's report: "primary measures are very hard to read as a table the
# numbers are overlapping to each other"). Both blocks repeat `measure` so
# either reads on its own, including across a page break.
_EFFECT_DISPLAY_COLUMNS = [
    "measure", "n_mut", "n_wt", "cages_contributing",
    "mean_mut", "mean_wt", "difference", "ci_low", "ci_high",
]
_INFERENCE_DISPLAY_COLUMNS = [
    "measure", "hedges_g", "p_value", "p_stars", "p_adjusted_bh", "fdr_stars",
    "lmm_p_value", "lmm_converged",
]


def _results(study: StudyConfig, result: dict, output: Path) -> str:
    scan = result["scan"]
    lines = ["## Results", ""]

    if "note" in scan.columns:
        # `pipeline.run_study`'s own skip path: no loaded cage contains both
        # genotypes at all, so `groups.scan` was never called (see `run_study`).
        note = scan.attrs.get("note", "no loaded cage contains both genotypes; "
                                      "group comparison skipped")
        lines += [
            f"No genotype comparison was computed: {note}.",
            "",
            "This is an expected RESULT for a single-genotype cage, not an error -- "
            "see Interpretation safeguards below for which cage(s) this applies to. "
            "Only descriptive tables and figures follow.",
            "",
        ]
    elif "measure" in scan.columns and scan.empty:
        # `groups.scan` itself ran (cages ARE genotype-mixed) but every measure was
        # individually skipped -- constant, absent, or no contributing cage. The
        # real reason lives in `scan.attrs["skipped_measures"]`, not in a "no cage
        # has both genotypes" fallback, which would be false here.
        skipped = scan.attrs.get("skipped_measures", [])
        reason = ("; ".join(skipped) if skipped
                 else "every candidate measure was skipped for an unrecorded reason")
        lines += [
            f"No genotype comparison was computed: {reason}.",
            "",
            "At least one loaded cage does contain both genotypes; every candidate "
            "measure was individually dropped by the scan (see the reason above), "
            "not because no comparison was possible. Only descriptive tables and "
            "figures follow.",
            "",
        ]
    else:
        scan_display = _with_significance_stars(scan, "p_value", "p_adjusted_bh",
                                                 "p_stars", "fdr_stars")
        primary = scan_display[scan_display["measure"].isin(study.primary_measures)]
        primary = primary.set_index("measure").reindex(study.primary_measures).reset_index()
        primary = primary.dropna(subset=["p_value"])
        lines += [
            "### Primary measures (pre-specified)",
            "",
            "Each table is split into two blocks, both keyed on `measure`, so "
            "neither overflows the page: effect sizes first, then inference.",
            "",
        ]
        if primary.empty:
            lines += [
                "*(none of the pre-specified primary measures produced a row -- see "
                "the scan below and its skipped-measure notes.)*",
                "",
            ]
        else:
            lines += [
                "**Effect sizes:**",
                "",
                _markdown_table(primary, _EFFECT_DISPLAY_COLUMNS),
                "",
                "**Inference:**",
                "",
                _markdown_table(primary, _INFERENCE_DISPLAY_COLUMNS),
                "",
                _STAR_LEGEND,
                "",
            ]
        lines += [
            "### Full scan (hypothesis-generating, FDR-corrected)",
            "",
            "**Effect sizes:**",
            "",
            _markdown_table(scan_display, _EFFECT_DISPLAY_COLUMNS),
            "",
            "**Inference:**",
            "",
            _markdown_table(scan_display, _INFERENCE_DISPLAY_COLUMNS),
            "",
            _STAR_LEGEND,
            "",
        ]
        skipped = scan["skipped_measures"].iloc[0] if "skipped_measures" in scan.columns else ""
        if skipped:
            lines += [f"Measures skipped by the scan: {skipped}.", ""]

    lines += ["### Figures", ""]
    figures_dir = output / "figures"
    for name, caption in _FIGURES:
        if (figures_dir / f"{name}.png").exists():
            lines += [f"![{caption}](figures/{name}.png)", "", caption, ""]
    return "\n".join(lines)


_CLUSTER_EFFECT_COLUMNS = ["measure", "hours", "n_hours", "sign", "mass"]
_CLUSTER_INFERENCE_COLUMNS = ["measure", "p_value", "p_stars", "p_adjusted_bh", "fdr_stars",
                              "n_permutations", "exact"]


def _whole_day_row(scan: pd.DataFrame, primary_measure: str):
    """The whole-day stratified-scan row for one phase-suffixed primary measure,
    or None if the scan carries no `measure` column (e.g. the all-wildtype
    skip path) or that measure produced no row."""
    if "measure" not in scan.columns:
        return None
    match = scan[scan["measure"] == primary_measure]
    return None if match.empty else match.iloc[0]


def _cluster_permutation_section(study: StudyConfig, result: dict) -> str:
    """Hour-of-day cluster-based permutation test -- companion to the primary
    (whole-day) cage-stratified comparison in `_results` above.

    Every number here is read off `result["cluster_permutation"]`
    (`groups.cluster_permutation_scan`'s table, BH-corrected across measures) and
    `result["cluster_permutation_detail"]` (`groups.cluster_permutation` called
    directly per measure -- kept because the scan table only carries a
    permutation count for a measure that actually produced a cluster; a measure
    with none still needs an honest count, not a borrowed one -- see
    `pipeline.run_study`), never typed by hand, so this section cannot drift
    from the analysis that was actually run.
    """
    cluster = result["cluster_permutation"]
    detail = result["cluster_permutation_detail"]
    scan = result["scan"]

    base_to_primary: dict[str, list[str]] = {}
    for measure in study.primary_measures:
        base_to_primary.setdefault(_base_measure(measure), []).append(measure)
    tested = list(base_to_primary)

    lines = [
        "## Hour-of-day cluster permutation test (companion to the primary comparison)",
        "",
        "The whole-day comparison above collapses each measure to one number; a "
        "genotype difference confined to particular hours of the day can average "
        "out to nothing in that number. This test looks for exactly that pattern.",
        "",
        "**What it does.** At each of the 24 clock hours, the same cage-stratified "
        "weighted genotype difference the primary comparison uses "
        "(`groups.stratified_difference`'s statistic) is computed. Contiguous runs "
        "of hours whose standardised difference exceeds a threshold form a "
        "cluster; a cluster's mass is the summed statistic of its member hours. "
        "The null distribution comes from permuting genotype labels WITHIN CAGE "
        "(never across cages, matching the primary test's own design) and "
        "recomputing the maximum absolute cluster mass on every permutation; an "
        "observed cluster's p-value is the fraction of permuted maxima at least "
        "as large as its own mass.",
        "",
    ]

    threshold = groups.CLUSTER_FORMING_THRESHOLD
    if detail:
        threshold = next(iter(detail.values()))["threshold"]
    lines += [
        f"**The cluster-forming threshold is a DECLARED parameter, not tuned to "
        f"this study's data: {threshold:g} null standard deviations from that "
        "hour's own null mean.** It was fixed before this test was run on this "
        "study's data, not chosen after seeing the result.",
        "",
        "**Taking the maximum cluster mass over the permutation null is what "
        "corrects for testing all 24 hours -- and however many clusters happen "
        "to form -- at once**, the max-statistic correction standard to "
        "cluster-based permutation tests (Maris & Oostenveld 2007). Clock hours "
        "are CIRCULAR (hour 23 is adjacent to hour 0), so a cluster may wrap "
        "midnight; both the test and the figure below handle that explicitly.",
        "",
        "**n is animals, not animal-hours.** Each animal contributes exactly ONE "
        "value per clock hour -- averaged across every day it was recorded -- "
        "before any statistic sees the data, so the sample size at every hour is "
        "the number of animals, never the far larger count of animal-hours.",
        "",
        "**Non-overlapping SEM bands in the figure below are NOT themselves a "
        "significance test.** SEM error bars that merely touch correspond to "
        "roughly p ~= 0.3, not p ~= 0.05 (Cumming & Finch 2005); roughly +/-2 "
        "standard errors of separation between the two genotype means is needed "
        "for p ~= 0.05. The shaded cluster(s) below, where present, are the "
        "actual test result -- the mean +/- SEM band by itself is descriptive "
        "only.",
        "",
    ]

    if detail:
        signatures = {(d["exact"], d["n_permutations"]) for d in detail.values()}
        if len(signatures) == 1:
            exact, n_permutations = next(iter(signatures))
            method = "enumerated EXACTLY" if exact else "SAMPLED (`neu_oldenlabs.RNG_SEED`)"
            lines += [
                f"**Permutations.** {n_permutations:,} within-cage label "
                f"assignments, {method}, used for every measure tested below.",
                "",
            ]
        else:
            parts = [f"{measure}: {d['n_permutations']:,} "
                    f"({'exact' if d['exact'] else 'sampled'})"
                    for measure, d in detail.items()]
            lines += [
                "**Permutations.** The count varies by measure (a measure's own "
                "missing-hour dropout can change cage composition): "
                + "; ".join(parts) + ".",
                "",
            ]
    else:
        lines += ["**Permutations.** No measure was successfully tested below.", ""]

    if not tested:
        lines += ["No primary measures were configured, so this test did not run.", ""]
        return "\n".join(lines)

    has_cluster_rows = "measure" in cluster.columns and not cluster.empty
    if has_cluster_rows:
        cluster_display = _with_significance_stars(cluster, "p_value", "p_adjusted_bh",
                                                    "p_stars", "fdr_stars")
        lines += [
            "### Observed clusters, all tested measures",
            "",
            "**Effect sizes:**",
            "",
            _markdown_table(cluster_display, _CLUSTER_EFFECT_COLUMNS),
            "",
            "**Inference:**",
            "",
            _markdown_table(cluster_display, _CLUSTER_INFERENCE_COLUMNS),
            "",
            _STAR_LEGEND,
            "",
        ]
    else:
        cluster_display = cluster
        lines += [
            "*(No hour-cluster exceeded the threshold for any tested measure.)*",
            "",
        ]

    lines += ["### Per-measure comparison against the whole-day result", ""]
    for measure in tested:
        primaries = base_to_primary[measure]
        whole_day_rows = {primary: _whole_day_row(scan, primary) for primary in primaries}
        whole_day_parts = []
        for primary, row in whole_day_rows.items():
            if row is None:
                whole_day_parts.append(f"{primary}: not available in the whole-day scan")
            else:
                whole_day_parts.append(
                    f"{primary}: p={_format_p(row['p_value'])}, "
                    f"q={_format_p(row['p_adjusted_bh'])} whole-day")
        whole_day_text = "; ".join(whole_day_parts)
        whole_day_significant = any(
            row is not None and not pd.isna(row["p_adjusted_bh"]) and row["p_adjusted_bh"] < 0.05
            for row in whole_day_rows.values())

        if measure not in detail:
            lines += [
                f"**{measure}.** Not tested for an hour-cluster (absent from this "
                f"study's export, or no loaded cage contains both genotypes). "
                f"Whole-day: {whole_day_text}.",
                "",
            ]
            continue

        measure_clusters = (cluster_display[cluster_display["measure"] == measure]
                            if has_cluster_rows else cluster_display.iloc[0:0])
        if measure_clusters.empty:
            lines += [
                f"**{measure}.** No hour-cluster exceeded the threshold. "
                f"Whole-day: {whole_day_text}.",
                "",
            ]
            continue

        cluster_parts = []
        any_significant = False
        for row in measure_clusters.itertuples():
            significant = not pd.isna(row.p_adjusted_bh) and row.p_adjusted_bh < 0.05
            any_significant = any_significant or significant
            wraps = " (wraps midnight)" if "," in str(row.hours) else ""
            cluster_parts.append(
                f"hours {row.hours}{wraps} (sign {'+' if row.sign > 0 else '-'}, "
                f"mass {row.mass:.3g}, p={_format_p(row.p_value)}, "
                f"q={_format_p(row.p_adjusted_bh)})")
        clusters_text = "; ".join(cluster_parts)

        flag = ""
        if any_significant and not whole_day_significant:
            flag = (" This measure shows a significant hour-cluster with NO "
                    "significant whole-day effect -- exactly the pattern this "
                    "test exists to detect.")
        lines += [
            f"**{measure}.** Hour-cluster(s): {clusters_text}. "
            f"Whole-day: {whole_day_text}.{flag}",
            "",
        ]

    if has_cluster_rows:
        skipped = (cluster["skipped_measures"].iloc[0]
                  if "skipped_measures" in cluster.columns else "")
    else:
        skipped = "; ".join(cluster.attrs.get("skipped_measures", []) or [])
    if skipped:
        lines += [f"Measures/hours skipped by the cluster scan: {skipped}.", ""]

    return "\n".join(lines)


# Split, like the primary-measures and full-scan tables in `_results`, so
# neither block overflows the page; both repeat `measure`. Both EFFECT lists
# carry `pooled_distance_bin_width_confound` right after `measure` -- see
# `qc.distance_resolution_confound` / `groups.compare_pooled` -- so a reader
# scanning either rendered table sees the flag before the numbers it
# qualifies, not just in the raw `group_comparison_pooled.csv`.
_POOLED_COMPARISON_EFFECT_COLUMNS = [
    "measure", "pooled_distance_bin_width_confound", "difference", "ci_low", "ci_high",
    "pooled_difference", "pooled_ci_low", "pooled_ci_high",
]
_POOLED_COMPARISON_INFERENCE_COLUMNS = [
    "measure", "hedges_g", "p_value", "p_stars", "p_adjusted_bh", "fdr_stars",
    "pooled_hedges_g", "pooled_p_value", "pooled_p_stars", "pooled_p_adjusted_bh",
    "pooled_fdr_stars",
]

_POOLED_SCAN_EFFECT_COLUMNS = [
    "measure", "pooled_distance_bin_width_confound", "n_mut", "n_wt",
    "pooled_mean_mut", "pooled_mean_wt",
    "pooled_difference", "pooled_ci_low", "pooled_ci_high",
]
_POOLED_SCAN_INFERENCE_COLUMNS = [
    "measure", "pooled_hedges_g", "pooled_p_value", "pooled_p_stars",
    "pooled_p_adjusted_bh", "pooled_fdr_stars", "pooled_draws", "pooled_min_attainable_p",
]


def _secondary_pooled_analysis(study: StudyConfig, result: dict) -> str:
    """SECONDARY section: the cage-ignoring pooled comparison.

    Run at the study owner's explicit request ("run analysis not within but
    all wildtypes and all mutants irrespectful of where they are coming
    from"), never as a replacement for the primary cage-stratified result
    above. Every number is read off `result["scan"]` / `result["scan_pooled"]`
    -- the frames `pipeline.run_study` actually computed -- never typed by
    hand, so this section cannot drift from the analysis it describes.
    """
    scan = result["scan"]
    pooled = result["scan_pooled"]
    weights = _cage_weights(result["profile"])

    lines = [
        "## Secondary analysis: pooled comparison ignoring cage (owner-requested)",
        "",
        "**This section was run at the study owner's explicit request** -- "
        "\"run analysis not within but all wildtypes and all mutants irrespectful "
        "of where they are coming from\" -- and is a SECONDARY, WEAKER analysis. "
        "It is never a replacement for the primary cage-stratified comparison "
        "above: it pools every wildtype animal against every mutant animal with "
        "cage identity IGNORED ENTIRELY, discarding the cage-stratification the "
        "primary analysis uses precisely because cages differ for reasons that "
        "have nothing to do with genotype. As a result, this analysis CANNOT "
        "separate a genotype effect from a cage effect confounded with genotype.",
        "",
        _distance_resolution_paragraph(result["qc"].get("distance_resolution")),
        "",
    ]

    if weights is not None:
        single_genotype = weights[(weights["n_mut"] == 0) | (weights["n_wt"] == 0)]
        if not single_genotype.empty:
            parts = [
                f"cage {int(row.cage)} contributes {int(row.n_wt)} wildtype and "
                f"{int(row.n_mut)} mutant animal(s)"
                for row in single_genotype.itertuples()
            ]
            lines += [
                "**Cage-effect confound in this run.** " + "; ".join(parts) + ". "
                "Any cage-specific effect (housing, cohort, whatever) in a "
                "single-genotype cage loads entirely onto that genotype's pooled "
                "mean below, with no within-cage contrast to net it against. The "
                "stratified analysis above assigns that cage weight zero for "
                "exactly this reason; the pooled analysis below cannot, by "
                "construction.",
                "",
            ]
        else:
            lines += [
                "**Cage-effect confound in this run.** Every loaded cage in this "
                "run houses both genotypes, so no single cage's idiosyncrasies "
                "load entirely onto one genotype's pooled mean below -- but the "
                "pooled test still cannot distinguish a genotype effect from a "
                "cage effect that varies smoothly across cages, only the "
                "extreme single-genotype-cage case.",
                "",
            ]

    draws = None
    if "pooled_draws" in pooled.columns and not pooled.empty:
        draws = int(pooled["pooled_draws"].iloc[0])
    if draws is not None:
        lines += [
            f"**Monte Carlo, not exact.** The pooled p-value below is estimated "
            f"from {draws:,} label-shuffles across ALL animals (cage ignored), "
            f"seeded from `neu_oldenlabs.RNG_SEED` for reproducibility -- NOT an "
            f"exact enumeration like the stratified test's p-value above. It can "
            f"never read exactly 0 (`pooled_min_attainable_p` = "
            f"1/{draws + 1:,} = {1 / (draws + 1):.6g}).",
            "",
        ]
    else:
        lines += [
            "**Monte Carlo, not exact.** The pooled comparison, when computed, "
            "estimates its p-value from a seeded label-shuffle across all "
            "animals rather than an exact enumeration -- see below for whether "
            "any measure in this run produced one.",
            "",
        ]

    lines += [
        "**Where the two analyses disagree, trust the stratified one.** The "
        "stratified (cage-controlled) result above is the one that removes "
        "cage as a confound; the pooled result below does not. Read a "
        "pooled-only finding as a candidate cage effect wearing a genotype "
        "label, not as independent confirmation.",
        "",
    ]

    has_stratified = "measure" in scan.columns and not scan.empty
    has_pooled = "measure" in pooled.columns and not pooled.empty

    if not has_pooled:
        reason = "; ".join(pooled.attrs.get("skipped_measures", []) or
                           ["no measure produced a pooled comparison"])
        lines += [
            f"No pooled comparison was computed for any measure: {reason}.",
            "",
        ]
        return "\n".join(lines)

    if has_stratified:
        scan_display = _with_significance_stars(scan, "p_value", "p_adjusted_bh",
                                                 "p_stars", "fdr_stars")
        pooled_display_for_merge = _with_significance_stars(
            pooled, "pooled_p_value", "pooled_p_adjusted_bh",
            "pooled_p_stars", "pooled_fdr_stars")
        merged = scan_display.merge(pooled_display_for_merge, on="measure", how="outer")
        primary = merged[merged["measure"].isin(study.primary_measures)]
        primary = primary.set_index("measure").reindex(study.primary_measures).reset_index()
        lines += [
            "### Primary measures: stratified vs. pooled, side by side",
            "",
            "**Effect sizes:**",
            "",
            _markdown_table(primary, _POOLED_COMPARISON_EFFECT_COLUMNS),
            "",
            "**Inference:**",
            "",
            _markdown_table(primary, _POOLED_COMPARISON_INFERENCE_COLUMNS),
            "",
            _STAR_LEGEND,
            "",
        ]

        strat_sig = set(scan.loc[scan["p_adjusted_bh"] < 0.05, "measure"])
        pooled_sig = set(pooled.loc[pooled["pooled_p_adjusted_bh"] < 0.05, "measure"])
        only_stratified = sorted(strat_sig - pooled_sig)
        only_pooled = sorted(pooled_sig - strat_sig)
        lines += [
            "**Where the two disagree (FDR q < 0.05 in one but not the other).**",
            "",
            f"- Significant only in the stratified (cage-controlled) scan: "
            f"{', '.join(only_stratified) if only_stratified else 'none'}.",
            f"- Significant only in the pooled (cage-ignoring) scan: "
            f"{', '.join(only_pooled) if only_pooled else 'none'}"
            + (" -- per the caveat above, read this as a candidate cage confound, "
               "not independent confirmation." if only_pooled else "."),
            "",
        ]
    else:
        lines += [
            "*(No stratified scan was available to compare against -- see Results "
            "above for why.)*",
            "",
        ]

    n_pooled_flagged = int((pooled["pooled_p_adjusted_bh"] < 0.05).sum())
    flagged_measures = pooled.loc[pooled["pooled_p_adjusted_bh"] < 0.05, "measure"].tolist()
    flagged_line = (", ".join(flagged_measures) if flagged_measures else "none")
    pooled_display = _with_significance_stars(pooled, "pooled_p_value", "pooled_p_adjusted_bh",
                                              "pooled_p_stars", "pooled_fdr_stars")
    lines += [
        f"**Pooled scan, FDR-significant (q < 0.05):** {n_pooled_flagged} of "
        f"{len(pooled)} scanned measure(s): {flagged_line}.",
        "",
        "### Full pooled scan (hypothesis-generating, FDR-corrected)",
        "",
        "**Effect sizes:**",
        "",
        _markdown_table(pooled_display, _POOLED_SCAN_EFFECT_COLUMNS),
        "",
        "**Inference:**",
        "",
        _markdown_table(pooled_display, _POOLED_SCAN_INFERENCE_COLUMNS),
        "",
        _STAR_LEGEND,
        "",
    ]
    skipped = pooled["skipped_measures"].iloc[0] if "skipped_measures" in pooled.columns else ""
    if skipped:
        lines += [f"Measures skipped by the pooled scan: {skipped}.", ""]

    return "\n".join(lines)


def _cage_weights(profile: pd.DataFrame) -> pd.DataFrame | None:
    """The per-cage mut/wt composition, via `groups.cage_weights`, or None if empty.

    Uses a dummy `value` column of 1.0 (never NaN) rather than any specific
    measure's values, so the composition it reports reflects the animal table's
    true genotype composition -- not one measure's missing-data pattern.
    """
    if profile.empty:
        return None
    composition = profile[["cage", "genotype"]].copy()
    composition["value"] = 1.0
    return groups.cage_weights(composition)


def _zero_weight_sentence(weights: pd.DataFrame | None) -> str:
    if weights is None:
        return "No animals were loaded, so no cage composition can be described."
    zero = weights[weights["weight"] <= 0]
    if zero.empty:
        return ("Every loaded cage houses both genotypes and contributes a "
                "within-cage contrast to the comparison; no cage drops out "
                "by construction in this run.")
    parts = [f"cage {int(row.cage)} ({int(row.n_mut)} mut / {int(row.n_wt)} wt)"
             for row in zero.itertuples()]
    plural = len(parts) > 1
    noun = "cages" if plural else "cage"
    verb = "house" if plural else "houses"
    contribute = "contribute" if plural else "contributes"
    pronoun = "Each" if plural else "It"
    return (
        f"The following {noun} {verb} a single genotype and therefore {contribute} "
        f"NO within-cage contrast: {', '.join(parts)}. Weight is zero and "
        f"{pronoun.lower()} drops out of the genotype comparison by construction "
        f"rather than silently shifting a mean; its animals still appear in every "
        f"descriptive table and figure above.")


def _dyadic_composition_clause(weights: pd.DataFrame | None) -> str:
    """Describe this run's actual cage composition, read off `groups.cage_weights`.

    The dyadic caveat itself is a fixed methodological fact and stays fixed; only
    its claims ABOUT THIS RUN's cages -- whether any cage is genotype-mixed, and
    how composition varies -- are generated from the profile, so they cannot
    contradict the zero-weight-cage sentence above or assert a mutant exists when
    none was loaded.
    """
    if weights is None:
        return "No animals were loaded, so cage composition cannot be described."
    parts = [f"cage {int(row.cage)} ({int(row.n_mut)} mut / {int(row.n_wt)} wt)"
             for row in weights.itertuples()]
    composition_line = "; ".join(parts)
    if int(weights["n_mut"].sum()) == 0:
        return (f"In this run every loaded cage is single-genotype wildtype "
                f"({composition_line}), so this dependence does not act on any "
                f"animal in the current data -- it applies as soon as a "
                f"mutant-carrying cage is loaded.")
    distinct = {(int(row.n_mut), int(row.n_wt)) for row in weights.itertuples()}
    if len(distinct) == 1:
        return f"In this run every loaded cage shares the same composition ({composition_line})."
    return f"In this run, cage composition varies across cages: {composition_line}."


def _safeguards(study: StudyConfig, result: dict) -> str:
    profile = result["profile"]
    scan = result["scan"]
    weights = _cage_weights(profile)

    zero_weight_line = _zero_weight_sentence(weights)
    composition_clause = _dyadic_composition_clause(weights)

    fdr_line = ("The full scan is FDR-corrected (Benjamini-Hochberg, `p_adjusted_bh`) and "
               "is HYPOTHESIS-GENERATING: it is a screen across candidate measures, not a "
               "confirmatory test. The pre-specified primary measures are reported first "
               "and are the ones this study is powered and intended to answer.")
    if "measure" in scan.columns and not scan.empty:
        n_flagged = int((scan["p_adjusted_bh"] < 0.05).sum())
        fdr_line += f" In this run, {n_flagged} of {len(scan)} scanned measure(s) meet q < 0.05."

    lines = [
        "## Interpretation safeguards",
        "",
        f"- **Cages with no within-cage contrast.** {zero_weight_line}",
        "",
        "- **Social distance and aggression are DYADIC, not per-animal.** Social "
        "distance and aggression are properties of a dyad, not of a single animal: "
        "an animal's value depends on which genotypes its cage-mates carry, not on "
        f"its own genotype alone. {composition_clause} Within-cage permutation is "
        "the correct test precisely because of this dyadic dependence: it compares "
        "each animal only against its own cage-mates, never against an animal in a "
        "differently-composed cage.",
        "",
        f"- **Non-significant results.** {fdr_line} A non-significant result -- for a "
        "primary measure or a scanned one -- is reported as absence of evidence, "
        "never as evidence of absence.",
        "",
        "- **Missing exports.** Cages whose exports have not arrived are named in "
        "Data and coverage above rather than silently omitted from this report.",
        "",
    ]
    return "\n".join(lines)


def build_study_report(study: StudyConfig, output: Path, cache_dir: Path | None = None) -> Path:
    """Run the pipeline and assemble `output/report.md` from the frames it returns.

    Every number and every named cage in this report is read off `result` -- the
    frames `pipeline.run_study` produces -- rather than typed by hand, so the
    report cannot drift from the analysis it describes.
    """
    output = Path(output)
    result = run_study(study, output, cache_dir=cache_dir)

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sections = [
        f"# {study.name}: Oldenlabs home-cage report",
        "",
        f"Generated {date} (UTC).",
        "",
        _data_and_coverage(study, result),
        _quality_control(result),
        _methods(study, result),
        _results(study, result, output),
        _cluster_permutation_section(study, result),
        _secondary_pooled_analysis(study, result),
        _safeguards(study, result),
    ]
    text = "\n".join(sections) + "\n"
    report_path = output / "report.md"
    report_path.write_text(text)
    return report_path
