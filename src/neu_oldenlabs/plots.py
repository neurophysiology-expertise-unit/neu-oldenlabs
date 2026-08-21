"""The six study figures, each with its values written alongside it as a CSV.

Every plotted quantity is also a table: `_save` writes `<name>.csv` next to
`<name>.png` for every figure, so nothing in a report is only a picture.

One genotype palette and one dark-phase shading style are shared by every
figure that uses them, so a reader never has to re-learn what a colour or a
shaded band means between panels. The pair (wt=blue, mut=orange) is the
default categorical palette's first two slots, validated colourblind-safe and
readable on a white surface (`dataviz` skill, `scripts/validate_palette.js`).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from . import RNG_SEED, metrics
from .io import MEASURE_KIND

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GENOTYPE_COLOR = {"wt": "#2a78d6", "mut": "#eb6834"}
GENOTYPE_ORDER = ("wt", "mut")
DARK_BAND = {"color": "#0b0b0b", "alpha": 0.08, "zorder": 0}
LIGHTS_ON = 7
LIGHTS_OFF = 19


def _save(fig, table: pd.DataFrame, output: Path, name: str) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / f"{name}.csv", index=False)
    with warnings.catch_warnings():
        # A figure-spanning colorbar (actogram) legitimately steals space from
        # multiple Axes, which tight_layout cannot account for; this is a
        # known, benign interaction, not a real layout problem.
        warnings.filterwarnings("ignore", message="This figure includes Axes that are "
                                "not compatible with tight_layout")
        fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=180)
    plt.close(fig)


def _shade_dark(ax, lights_on: int, lights_off: int, span: float = 24.0) -> None:
    """Shade the dark phase on an hour-of-day axis running 0..span.

    `span` is 24 for a single 24-hour panel and 48 for a double-plotted one
    (two consecutive days side by side); the night band is drawn once per
    24-hour block so it repeats correctly in either case.
    """
    for block_start in np.arange(0, span, 24):
        ax.axvspan(block_start, block_start + lights_on, **DARK_BAND)
        ax.axvspan(block_start + lights_off, block_start + 24, **DARK_BAND)


def _genotype_legend(ax, present: set[str]) -> None:
    handles = [plt.Line2D([0], [0], marker="o", linestyle="none",
                          color=GENOTYPE_COLOR[genotype], label=genotype)
              for genotype in GENOTYPE_ORDER if genotype in present]
    if handles:
        ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8)


def actogram(profile_long: pd.DataFrame, output: Path, lights_on: int = LIGHTS_ON,
            lights_off: int = LIGHTS_OFF) -> None:
    """One 24-hour, day-by-day activity heatmap per animal.

    `profile_long` is the rack-masked long frame. Activity is `distance_cm`
    reduced into hour-of-day x calendar-date cells as a DURATION-WEIGHTED rate
    per valid hour -- total distance in the cell divided by the cell's total
    bin duration in hours, the same normalisation `metrics._aggregate_measure`
    uses for a "count"-kind measure and `circadian.hourly_activity` uses for
    the circadian signal -- never a plain mean over however many bins happen
    to fall in the cell. A plain per-bin mean makes the value's unit "per
    bin", which is not comparable across cages with different bin widths: a
    600-second-bin cage and a 3600-second-bin cage carrying the SAME
    underlying per-hour activity would show a ~6x difference in raw per-bin
    means for no biological reason. The night band is drawn over every panel
    from the study's own light schedule.
    """
    distance = profile_long[profile_long["measure"] == "distance_cm"].copy()
    table = pd.DataFrame(columns=["cage", "tag", "date", "hour", "value"])
    if not distance.empty:
        distance["date"] = distance["bin_start"].dt.date.astype(str)
        distance["hour"] = distance["bin_start"].dt.hour
        # A cell's duration weight must count only bins whose value is actually
        # present (`metrics._aggregate_measure`'s rule) -- otherwise a cell that
        # is entirely NaN (e.g. a literal 'NaN' string in the export) sums to
        # total=0.0 (pandas sums an all-NaN group to 0.0) over a NONZERO
        # seconds (every bin's full duration, present or not), landing on a
        # hard, plausible 0.0 instead of the honest NaN.
        distance["duration_present"] = distance["duration_s"].where(
            distance["value"].notna(), 0.0)
        grouped = distance.groupby(["cage", "tag", "date", "hour"], as_index=False).agg(
            total=("value", "sum"), seconds=("duration_present", "sum"))
        grouped["value"] = grouped["total"] / (grouped["seconds"] / 3600.0).replace(0, np.nan)
        table = grouped[["cage", "tag", "date", "hour", "value"]]

    animals = sorted(set(zip(table["cage"], table["tag"])))
    if not animals:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "no distance_cm data", ha="center", va="center")
        ax.axis("off")
        _save(fig, table, output, "actogram")
        return

    ncols = min(3, len(animals))
    nrows = -(-len(animals) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.2 * nrows),
                             squeeze=False)
    vmax = table["value"].max()
    image = None
    for index, (cage, tag) in enumerate(animals):
        ax = axes[index // ncols][index % ncols]
        block = table[(table["cage"] == cage) & (table["tag"] == tag)]
        grid = block.pivot(index="date", columns="hour", values="value").sort_index()
        grid = grid.reindex(columns=range(24))
        image = ax.imshow(grid.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=vmax,
                          extent=(0, 24, len(grid), 0))
        _shade_dark(ax, lights_on, lights_off, span=24)
        ax.set_title(f"cage {cage}, tag {tag}", fontsize=9)
        ax.set_xlabel("hour of day", fontsize=8)
        ax.set_ylabel("day", fontsize=8)
        ax.set_xlim(0, 24)
    for index in range(len(animals), nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.6,
                     label="distance_cm (cm / valid hour)")
    _save(fig, table, output, "actogram")


def _hourly_value_per_animal(block: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Reduce raw bins to one duration-weighted value per animal, per hour-of-day.

    Delegates the reduction itself to `metrics.hourly_by_animal` (the same
    per-measure rule `metrics._aggregate_measure` uses: duration-weighted for
    rate/mean, the maximum for max, a rate per valid hour for count -- see
    that function's docstring), so the result does not carry a bin-width
    artefact the way a flat per-bin mean would, then joins genotype back in
    (`block` must already carry it, one value per (cage, tag)). Returns one
    row per (cage, tag, genotype, hour) -- the animal is the unit this
    reduces to, so a caller can then compute mean/SEM ACROSS ANIMALS rather
    than across pooled bins. `groups.hourly_profile_by_animal` reuses the
    same `metrics.hourly_by_animal` call for the cluster-permutation test, so
    the per-animal, per-hour value is computed in exactly one place.
    """
    per_animal = metrics.hourly_by_animal(block, measure)
    genotype = block[["cage", "tag", "genotype"]].drop_duplicates()
    merged = per_animal.merge(genotype, on=["cage", "tag"], how="left")
    return merged[["cage", "tag", "genotype", "hour", "value"]]


def _hourly_mean_sem_table(hourly: pd.DataFrame, measures: list[str]) -> pd.DataFrame:
    """Mean, SEM, and n ACROSS ANIMALS, per measure/genotype/hour-of-day.

    Shared by `phase_profiles` and `hour_clusters` so the two figures cannot
    silently drift apart in how they compute the same underlying numbers.
    Bins are first reduced to one duration-weighted value per ANIMAL at each
    hour-of-day (`_hourly_value_per_animal`), then mean and SEM are computed
    across those per-animal values -- see `phase_profiles` for why both
    steps matter (bin-width artefact and SEM-across-animals-not-bins).
    """
    frame = hourly.copy()
    if not frame.empty:
        frame["hour"] = frame["bin_start"].dt.hour

    rows = []
    for measure in measures:
        block = frame[frame["measure"] == measure]
        per_animal = _hourly_value_per_animal(block, measure)
        for (genotype, hour), sub in per_animal.groupby(["genotype", "hour"]):
            rows.append({"measure": measure, "genotype": genotype, "hour": hour,
                        "mean": sub["value"].mean(), "sem": sub["value"].sem(),
                        "n": sub["value"].count()})
    return pd.DataFrame(rows, columns=["measure", "genotype", "hour", "mean", "sem", "n"])


def phase_profiles(hourly: pd.DataFrame, output: Path, measures: list[str],
                   lights_on: int = LIGHTS_ON, lights_off: int = LIGHTS_OFF) -> None:
    """Mean +/- SEM hour-of-day profile per measure, one line per genotype.

    `hourly` is the rack-masked long frame joined with genotype (it must carry
    a `genotype` column already, since only the pipeline knows which animal
    table applies). One subplot per measure in `measures`.

    See `_hourly_mean_sem_table` for how the mean/SEM/n table is built: bins
    are first reduced to one duration-weighted value per ANIMAL at each
    hour-of-day, then mean and SEM are computed across those per-animal
    values. Two things this fixes at once: (1) a flat mean over raw bin
    values is "per bin", not "per hour", so pooling cages with different bin
    widths manufactures a difference that is pure bin-width artefact; (2)
    the experimental unit is the animal, so the SEM must be taken across
    animals, not across pooled bins from every animal and every day --
    pooling bins makes the band far too narrow (the real study's CSV
    recorded n=1200 for 5 animals) and overstates how precisely the mean is
    known.

    `hour_clusters` is the companion figure: the same mean +/- SEM profile,
    but with the cluster-permutation test's significant hour-clusters marked
    -- this figure alone does not test anything; see that function's
    docstring for the caveat that non-overlapping SEM bands are NOT a test.
    """
    measures = [m for m in measures if m in set(hourly.get("measure", []))]
    table = _hourly_mean_sem_table(hourly, measures)

    if not measures:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "no measures available", ha="center", va="center")
        ax.axis("off")
        _save(fig, table, output, "phase_profiles")
        return

    fig, axes = plt.subplots(len(measures), 1, figsize=(6, 2.4 * len(measures)), squeeze=False)
    present = set(table["genotype"]) if not table.empty else set()
    for row_index, measure in enumerate(measures):
        ax = axes[row_index][0]
        block = table[table["measure"] == measure]
        for genotype in GENOTYPE_ORDER:
            line = block[block["genotype"] == genotype].sort_values("hour")
            if line.empty:
                continue
            ax.plot(line["hour"], line["mean"], color=GENOTYPE_COLOR[genotype],
                   marker="o", markersize=3, linewidth=1.5, label=genotype)
            ax.fill_between(line["hour"], line["mean"] - line["sem"],
                            line["mean"] + line["sem"], color=GENOTYPE_COLOR[genotype],
                            alpha=0.2, linewidth=0)
        _shade_dark(ax, lights_on, lights_off, span=24)
        ax.set_xlim(0, 23)
        label = measure
        if MEASURE_KIND.get(measure) == "count":
            label += " (per valid hour)"
        ax.set_ylabel(label, fontsize=8)
        ax.set_xlabel("hour of day" if row_index == len(measures) - 1 else "")
        _genotype_legend(ax, present)
    _save(fig, table, output, "phase_profiles")


def _cluster_spans(hours: list) -> list:
    """Contiguous integer-hour runs in `hours` as (start, end) axis spans,
    each covering that hour's full bin width (hour h covers [h, h+1)).

    `hours` is `groups.cluster_permutation`'s already-circularly-resolved
    cluster hour list (e.g. [22, 23, 0, 1] for a cluster wrapping midnight).
    A run that wraps past midnight is split into two spans -- e.g.
    [22, 23, 0, 1] becomes [(22, 24), (0, 2)] -- so it can be drawn with two
    ordinary `axvspan` calls on an axis that does not itself wrap.
    """
    if not hours:
        return []
    ordered = sorted(hours)
    if len(ordered) == 24:
        return [(0.0, 24.0)]

    gaps = [(ordered[i + 1] - ordered[i], i) for i in range(len(ordered) - 1)]
    wrap_gap = (ordered[0] + 24 - ordered[-1], len(ordered) - 1)
    _, split_index = max(gaps + [wrap_gap], key=lambda item: item[0]) if gaps else wrap_gap
    rotated = ordered[split_index + 1:] + ordered[:split_index + 1]

    runs = []
    start = prev = rotated[0]
    for hour in rotated[1:]:
        if hour == (prev + 1) % 24:
            prev = hour
            continue
        runs.append((start, prev))
        start = prev = hour
    runs.append((start, prev))

    spans = []
    for a, b in runs:
        if a <= b:
            spans.append((float(a), float(b) + 1.0))
        else:  # wraps past midnight
            spans.append((float(a), 24.0))
            spans.append((0.0, float(b) + 1.0))
    return spans


def hour_clusters(hourly: pd.DataFrame, cluster_scan: pd.DataFrame, output: Path,
                  measures: list[str], lights_on: int = LIGHTS_ON,
                  lights_off: int = LIGHTS_OFF) -> None:
    """The `phase_profiles` mean +/- SEM hour-of-day profile, with the
    cluster-permutation test's significant hour-cluster(s) marked.

    `cluster_scan` is `groups.cluster_permutation_scan`'s output table (one
    row per OBSERVED cluster that exceeded the cluster-forming threshold,
    already BH-corrected ACROSS MEASURES in `p_adjusted_bh`) -- this figure
    only VISUALISES clusters the stats module already found; it recomputes
    nothing. A cluster is shaded on its measure's panel -- red if
    `p_adjusted_bh < 0.05`, grey otherwise, matching `effect_forest`'s
    red/grey convention -- and split across the midnight wrap into two spans
    when it wraps (`_cluster_spans`). A measure with no row in `cluster_scan`
    at all (every hour stayed under the cluster-forming threshold, or the
    measure was skipped -- see `cluster_scan.attrs.get("skipped_measures")`)
    is plotted with no shading at all.

    CAVEAT, stated here and in the report alongside this figure:
    non-overlapping SEM bands are NOT themselves a test -- SEM bars that just
    touch correspond to roughly p ~= 0.3, not p ~= 0.05 (Cumming & Finch
    2005; the cluster-permutation test is what corrects for both the
    autocorrelation across 24 hours and the multiple-comparison burden of
    testing all of them). The shaded cluster(s) here are the actual test
    result; the mean +/- SEM bands are descriptive only.
    """
    measures = [m for m in measures if m in set(hourly.get("measure", []))]
    table = _hourly_mean_sem_table(hourly, measures)

    if not measures:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "no measures available", ha="center", va="center")
        ax.axis("off")
        _save(fig, table, output, "hour_clusters")
        return

    has_clusters = "measure" in cluster_scan.columns and not cluster_scan.empty

    fig, axes = plt.subplots(len(measures), 1, figsize=(6, 2.4 * len(measures)), squeeze=False)
    present = set(table["genotype"]) if not table.empty else set()
    for row_index, measure in enumerate(measures):
        ax = axes[row_index][0]
        block = table[table["measure"] == measure]
        for genotype in GENOTYPE_ORDER:
            line = block[block["genotype"] == genotype].sort_values("hour")
            if line.empty:
                continue
            ax.plot(line["hour"], line["mean"], color=GENOTYPE_COLOR[genotype],
                   marker="o", markersize=3, linewidth=1.5, label=genotype)
            ax.fill_between(line["hour"], line["mean"] - line["sem"],
                            line["mean"] + line["sem"], color=GENOTYPE_COLOR[genotype],
                            alpha=0.2, linewidth=0)
        _shade_dark(ax, lights_on, lights_off, span=24)
        if has_clusters:
            for cluster in cluster_scan[cluster_scan["measure"] == measure].itertuples():
                significant = (not pd.isna(cluster.p_adjusted_bh)
                              and cluster.p_adjusted_bh < 0.05)
                color = "#d03b3b" if significant else "#8c8c8c"
                for start, end in _cluster_spans(cluster.hour_list):
                    ax.axvspan(start, end, color=color, alpha=0.18, zorder=1)
                label_hour = sorted(cluster.hour_list)[len(cluster.hour_list) // 2]
                stars = "*" if significant else ""
                ax.annotate(f"p={cluster.p_value:.3g}{stars}",
                           xy=(label_hour, 1.0), xycoords=("data", "axes fraction"),
                           xytext=(0, 2), textcoords="offset points",
                           ha="center", fontsize=6, color=color)
        ax.set_xlim(0, 23)
        label = measure
        if MEASURE_KIND.get(measure) == "count":
            label += " (per valid hour)"
        ax.set_ylabel(label, fontsize=8)
        ax.set_xlabel("hour of day" if row_index == len(measures) - 1 else "")
        _genotype_legend(ax, present)
    _save(fig, table, output, "hour_clusters")


def cage_paired(profile: pd.DataFrame, output: Path, measures: list[str]) -> None:
    """One panel per measure: animals as points, grouped by cage, coloured by genotype.

    This is the panel the group comparison statistics answer to: it shows the
    same within-cage contrast `groups.compare` computes, so an all-wildtype
    cage is visible here as a column with only one colour in it.
    """
    measures = [m for m in measures if m in profile.columns]
    rows = []
    for measure in measures:
        for _, animal in profile.iterrows():
            rows.append({"measure": measure, "cage": animal["cage"],
                        "animal_id": animal["animal_id"], "genotype": animal["genotype"],
                        "value": animal[measure]})
    table = pd.DataFrame(rows, columns=["measure", "cage", "animal_id", "genotype", "value"])

    if not measures:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "no measures available", ha="center", va="center")
        ax.axis("off")
        _save(fig, table, output, "cage_paired")
        return

    cages = sorted(profile["cage"].unique())
    positions = {cage: index for index, cage in enumerate(cages)}
    rng = np.random.default_rng(RNG_SEED)

    fig, axes = plt.subplots(1, len(measures), figsize=(3.2 * len(measures), 3.2), squeeze=False)
    present = set(table["genotype"]) if not table.empty else set()
    for col_index, measure in enumerate(measures):
        ax = axes[0][col_index]
        block = table[table["measure"] == measure].dropna(subset=["value"])
        for _, point in block.iterrows():
            jitter = rng.uniform(-0.12, 0.12)
            ax.scatter(positions[point["cage"]] + jitter, point["value"],
                      color=GENOTYPE_COLOR.get(point["genotype"], "#808080"),
                      s=28, edgecolor="white", linewidth=0.5, zorder=2)
        ax.set_xticks(range(len(cages)))
        ax.set_xticklabels([str(cage) for cage in cages], fontsize=8)
        ax.set_xlim(-0.5, len(cages) - 0.5)
        ax.set_title(measure, fontsize=9)
        ax.set_xlabel("cage", fontsize=8)
        if col_index == 0:
            ax.set_ylabel("value", fontsize=8)
        _genotype_legend(ax, present)
    _save(fig, table, output, "cage_paired")


def effect_forest(scan: pd.DataFrame, output: Path) -> None:
    """Forest plot of Hedges' g and its own bootstrap CI, ranked by p-value.

    Plotting the raw `difference` mixes centimetres, percentages, and
    unitless indices on one shared axis -- with a real scan, distance sets
    the scale and every percentage/index collapses onto the zero line.
    Hedges' g is unitless and already computed (`groups.hedges_g`), so it is
    what goes on the shared axis instead. `g_ci_low`/`g_ci_high`
    (`groups.cluster_bootstrap_g_ci`, via `groups._cluster_bootstrap`) are
    g's OWN percentile bootstrap CI -- every draw recomputes g from its own
    resampled pooled SD, not the raw-difference CI rescaled by the point
    estimate's fixed `hedges_g / difference` ratio, which would silently
    treat the pooled SD as constant across draws when it in fact resamples
    with everything else, and would collapse to zero width whenever
    `difference` is exactly 0.

    Writes nothing when there is no per-measure comparison to plot: either the
    scan table carries only a `note` explaining a skip (no `measure` column),
    or every measure was individually skipped (a `measure` column with zero
    rows). Either way, the figures/tables stay paired exactly.
    """
    if "measure" not in scan.columns or scan.empty:
        return

    table = scan[["measure", "difference", "ci_low", "ci_high", "hedges_g",
                 "g_ci_low", "g_ci_high", "p_value", "p_adjusted_bh"]].copy()
    ordered = table.sort_values("p_value").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6, 0.5 * len(ordered) + 1.5))
    y = np.arange(len(ordered))[::-1]
    for row, y_position in zip(ordered.itertuples(), y):
        color = "#d03b3b" if row.p_adjusted_bh < 0.05 else "#52514e"
        if pd.isna(row.hedges_g):
            ax.scatter([0], [y_position], marker="x", color="#999999", s=30, zorder=2)
            continue
        if pd.isna(row.g_ci_low) or pd.isna(row.g_ci_high):
            # g itself is defined but no bootstrap draw had a poolable cage
            # (vanishingly rare, but possible) -- plot the point with no
            # interval rather than feeding NaN bounds into errorbar.
            ax.scatter([row.hedges_g], [y_position], marker="o", color=color,
                      s=25, zorder=2)
            continue
        ax.errorbar([row.hedges_g], [y_position],
                   xerr=[[row.hedges_g - row.g_ci_low], [row.g_ci_high - row.hedges_g]],
                   fmt="o", color=color, elinewidth=1.5, capsize=3, markersize=5, zorder=2)
    ax.axvline(0, color="#0b0b0b", linewidth=1, linestyle="--", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["measure"], fontsize=8)
    ax.set_xlabel("mut - wt (Hedges' g, 95% CI; x marks g undefined)")
    _save(fig, ordered, output, "effect_forest")


def daily_trajectories(daily: pd.DataFrame, output: Path, measure: str) -> None:
    """One line per animal, day-by-day, for one measure (all-phase), by genotype.

    `daily` must already carry `genotype` (joined in by the pipeline, which is
    the only place that knows the animal table).
    """
    block = daily[(daily["measure"] == measure) & (daily["phase"] == "all")].copy()
    table = block[["cage", "tag", "genotype", "date", "value"]].sort_values(
        ["cage", "tag", "date"])

    fig, ax = plt.subplots(figsize=(7, 3.5))
    if table.empty:
        ax.text(0.5, 0.5, f"no data for {measure}", ha="center", va="center")
        ax.axis("off")
        _save(fig, table, output, "daily_trajectories")
        return

    present = set(table["genotype"])
    for (cage, tag), animal in table.groupby(["cage", "tag"]):
        animal = animal.sort_values("date")
        color = GENOTYPE_COLOR.get(animal["genotype"].iloc[0], "#808080")
        ax.plot(animal["date"], animal["value"], color=color, linewidth=1.2, alpha=0.85,
               marker="o", markersize=3)
    ax.set_ylabel(measure)
    ax.set_xlabel("date")
    fig.autofmt_xdate(rotation=30)
    _genotype_legend(ax, present)
    _save(fig, table, output, "daily_trajectories")


def qc_panel(qc: dict[str, pd.DataFrame], output: Path) -> None:
    """One row per cage: fraction of bins out of rack, breaks, range violations,
    coverage, fraction of values missing (NaN), and excluded (non-CSV-listed)
    tags -- plus bin width, called out explicitly because a mixed-resolution
    study must be stated, not silently averaged: pooling a 600-second-bin cage
    with a 3600-second-bin cage in a per-bin figure would manufacture a
    difference that is pure bin width (see `actogram`, `phase_profiles`).

    Built from the `qc.qc_tables` outputs, folded into one per-cage summary
    so the figure and its CSV both describe "how much can I trust this
    cage's data" at a glance. `excluded_tags` (from `qc.excluded_tag_table`,
    via `animals.check_join`'s asymmetric join rule) is a count of tags per
    cage that were present in the export but not listed in the animal CSV,
    and were therefore excluded from every other table and figure here --
    this panel is the one place their existence is visible in a figure.
    `rack_non_finite` (from `qc.non_finite_rack`) counts bins per cage whose
    `in_rack` reading was non-finite (NaN, +-inf) -- excluded from analysis
    the same as an out-of-rack bin, but a different failure mode, so it is
    plotted as its own panel rather than folded into
    `mean_fraction_out_of_rack`.
    """
    rack, regularity, breaks, coverage, ranges, missing = (
        qc["rack"], qc["regularity"], qc["breaks"], qc["coverage"], qc["ranges"],
        qc["missing"])
    excluded = qc.get("excluded_tags", pd.DataFrame(columns=["cage", "tag"]))
    non_finite = qc.get("rack_non_finite",
                        pd.DataFrame(columns=["cage", "tag", "n_non_finite_bins"]))

    cages = sorted(rack["cage"].unique()) if not rack.empty else []
    fraction_out = (rack.groupby("cage")["fraction_out"].mean().reindex(cages).fillna(0.0))
    n_breaks = (breaks.groupby("cage").size().reindex(cages).fillna(0).astype(int)
               if not breaks.empty else pd.Series(0, index=cages))
    n_violations = (ranges.groupby("cage")["n_violations"].sum().reindex(cages).fillna(0)
                    if not ranges.empty else pd.Series(0, index=cages))
    valid_hours = (coverage.groupby("cage")["valid_hours"].mean().reindex(cages).fillna(0.0)
                  if not coverage.empty else pd.Series(0.0, index=cages))
    fraction_missing = (
        (missing.groupby("cage")["missing"].sum() / missing.groupby("cage")["bins"].sum())
        .reindex(cages).fillna(0.0) if not missing.empty else pd.Series(0.0, index=cages))
    n_excluded = (excluded.groupby("cage").size().reindex(cages).fillna(0).astype(int)
                 if not excluded.empty else pd.Series(0, index=cages, dtype=int))
    n_non_finite = (non_finite.groupby("cage")["n_non_finite_bins"].sum()
                    .reindex(cages).fillna(0).astype(int)
                    if not non_finite.empty else pd.Series(0, index=cages, dtype=int))

    bin_widths = {}
    if not regularity.empty:
        for cage, block in regularity.groupby("cage"):
            bin_widths[cage] = sorted(set(block["duration_s"]))
    bin_width_label = {cage: "/".join(f"{w:.0f}" for w in widths)
                       for cage, widths in bin_widths.items()}

    table = pd.DataFrame({
        "cage": cages, "mean_fraction_out_of_rack": fraction_out.to_numpy(),
        "n_recording_breaks": n_breaks.to_numpy(),
        "n_range_violations": n_violations.to_numpy(),
        "mean_valid_hours_per_day": valid_hours.to_numpy(),
        "mean_fraction_missing_values": fraction_missing.to_numpy(),
        "n_excluded_tags": n_excluded.to_numpy(),
        "n_non_finite_rack_bins": n_non_finite.to_numpy(),
        "bin_width_s": [bin_width_label.get(cage, "") for cage in cages],
    })

    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    panels = [("mean_fraction_out_of_rack", "fraction of bins out of rack"),
             ("n_recording_breaks", "recording breaks (relative to cage's own bin width)"),
             ("n_range_violations", "out-of-range values"),
             ("mean_valid_hours_per_day", "mean valid hours / day"),
             ("mean_fraction_missing_values", "fraction of values missing (NaN)"),
             ("n_excluded_tags", "tags excluded (not in animal CSV)"),
             ("n_non_finite_rack_bins", "non-finite in_rack bins (excluded, not out-of-rack)")]
    x = np.arange(len(cages))
    flat_axes = axes.flat
    for (column, title), ax in zip(panels, flat_axes):
        if cages:
            ax.bar(x, table[column], color="#2a78d6")
            ax.set_xticks(x)
            ax.set_xticklabels([str(cage) for cage in cages], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("cage", fontsize=8)
    for ax in flat_axes:  # remaining, unused subplot(s)
        ax.axis("off")

    distinct_widths = sorted({w for widths in bin_widths.values() for w in widths})
    if len(distinct_widths) > 1:
        summary = "; ".join(f"cage {cage}: {bin_width_label[cage]} s" for cage in cages
                            if cage in bin_width_label)
        fig.suptitle(f"BIN WIDTH VARIES ACROSS CAGES -- raw per-bin values are not "
                     f"comparable: {summary}", fontsize=8, color="#d03b3b")
    elif distinct_widths:
        fig.suptitle(f"Bin width: {distinct_widths[0]:.0f} s across all cages", fontsize=8)
    _save(fig, table, output, "qc_panel")
