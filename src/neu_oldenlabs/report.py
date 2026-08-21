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
from .io import MEASURE_KIND
from .pipeline import _base_measure, run_study

# Figure filenames, in the order `pipeline.run_study` writes them, with the
# caption each earns per the design spec's figure list (section 8).
_FIGURES = [
    ("actogram", "Actogram: per-animal 24-hour x day distance heatmap, "
                 "night phase shaded."),
    ("phase_profiles", "24-hour genotype mean +/- SEM profile, per primary measure, "
                       "night phase shaded."),
    ("cage_paired", "Cage-paired dot plot: animals as points, grouped by cage and "
                    "coloured by genotype -- the same contrast the statistics test."),
    ("effect_forest", "Effect-size forest: stratified difference and its 95% cluster "
                      "bootstrap CI, per measure, ranked by p-value; red marks FDR "
                      "q < 0.05."),
    ("daily_trajectories", "Daily trajectories: one line per animal across the "
                           "recording window, coloured by genotype."),
    ("qc_panel", "QC panel: rack-mask fraction, recording breaks, out-of-range "
                "values, and mean valid hours per day, one bar per cage."),
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


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Render a DataFrame as a markdown table by hand (no `tabulate` dependency)."""
    if frame.empty:
        return "*(no rows)*"
    columns = columns or list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_format_cell(row[c]) for c in columns) + " |"
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


def _count_kind_measures(measures: list[str]) -> list[str]:
    """Base measure names among `measures` whose daily rule is per-valid-hour.

    `metrics._aggregate_measure` turns a `count`-kind measure into a rate per
    valid hour (see `MEASURE_KIND` in `io.py`), not a per-day total. This reads
    that mapping directly rather than hardcoding which measures it applies to,
    so the sentence below cannot drift if a measure's kind ever changes.
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


def _bin_timing_sentence(result: dict) -> str:
    """Describe observed bin span and start-to-start cadence, read off the data.

    A bin's own span (`Datetime to` minus `Datetime from`, via `qc.bin_regularity`)
    and the gap between consecutive bin starts are two different quantities. The
    real cage-58616 export has a constant 600 s span for every bin but an
    irregular cadence (most gaps 10 minutes, some 11, a few 9), so this must be
    generated from `result["long"]` rather than asserted, and the two quantities
    must stay distinct in the sentence.
    """
    long = result["long"]
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
    gaps = (bins_sorted.groupby("cage")["bin_start"]
            .apply(lambda s: s.diff().dt.total_seconds() / 60.0)
            .dropna())
    if gaps.empty:
        return span_line

    gap_counts = gaps.round(6).value_counts().sort_index()
    total_gaps = int(gap_counts.sum())
    if len(gap_counts) == 1:
        cadence_line = (
            f"Start-to-start CADENCE between consecutive bins is likewise uniform, "
            f"at {gap_counts.index[0]:.4g} minutes across all {total_gaps} gaps.")
    else:
        modal_gap = gap_counts.idxmax()
        off_parts = []
        for value, count in gap_counts.items():
            if value == modal_gap:
                continue
            delta = value - modal_gap
            kind = "coverage hole" if delta > 0 else "overlap"
            off_parts.append(f"{int(count)} at {value:.4g} min ({abs(delta):.4g}-min {kind})")
        cadence_line = (
            f"Start-to-start CADENCE between consecutive bins is a distinct quantity "
            f"from bin duration and is NOT perfectly uniform here: {int(gap_counts[modal_gap])} "
            f"of {total_gaps} gaps are {modal_gap:.4g} minutes; " + "; ".join(off_parts) + ".")
    return span_line + " " + cadence_line


def _methods(study: StudyConfig, result: dict) -> str:
    scan = result["scan"]
    all_measures = list(dict.fromkeys(list(study.primary_measures) + list(study.scan_measures)))
    per_hour = _count_kind_measures(all_measures)
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
        "`Cage in rack == 0` are excluded from every metric before aggregation.",
        "",
        per_hour_line,
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
        "Interpretation safeguards for which cage(s) that is in this run. Genotype "
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


_COMPARE_DISPLAY_COLUMNS = [
    "measure", "n_mut", "n_wt", "cages_contributing", "difference", "ci_low", "ci_high",
    "hedges_g", "p_value", "min_attainable_p", "assignments", "p_adjusted_bh",
    "lmm_coefficient", "lmm_se", "lmm_p_value", "lmm_converged", "lmm_note",
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
        primary = scan[scan["measure"].isin(study.primary_measures)]
        primary = primary.set_index("measure").reindex(study.primary_measures).reset_index()
        primary = primary.dropna(subset=["p_value"])
        lines += [
            "### Primary measures (pre-specified)",
            "",
            _markdown_table(primary, _COMPARE_DISPLAY_COLUMNS)
            if not primary.empty else
            "*(none of the pre-specified primary measures produced a row -- see the "
            "scan below and its skipped-measure notes.)*",
            "",
            "### Full scan (hypothesis-generating, FDR-corrected)",
            "",
            _markdown_table(scan, _COMPARE_DISPLAY_COLUMNS),
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
        _methods(study, result),
        _results(study, result, output),
        _safeguards(study, result),
    ]
    text = "\n".join(sections) + "\n"
    report_path = output / "report.md"
    report_path.write_text(text)
    return report_path
