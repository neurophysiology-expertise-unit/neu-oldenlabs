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

from . import RNG_SEED
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
        grouped = distance.groupby(["cage", "tag", "date", "hour"], as_index=False).agg(
            total=("value", "sum"), seconds=("duration_s", "sum"))
        grouped["value"] = grouped["total"] / (grouped["seconds"] / 3600.0)
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

    Pools bins across every calendar date at the same hour-of-day for one
    animal, using the same per-measure rule `metrics._aggregate_measure` uses
    (duration-weighted for rate/mean, the maximum for max, a rate per valid
    hour for count), so the result does not carry a bin-width artefact the
    way a flat per-bin mean would. Returns one row per (cage, tag, genotype,
    hour) -- the animal is the unit this reduces to, so a caller can then
    compute mean/SEM ACROSS ANIMALS rather than across pooled bins.
    """
    kind = MEASURE_KIND.get(measure)
    frame = block.copy()
    frame["weighted"] = frame["value"] * frame["duration_s"]
    grouped = frame.groupby(["cage", "tag", "genotype", "hour"], as_index=False).agg(
        weighted=("weighted", "sum"), seconds=("duration_s", "sum"),
        total=("value", "sum"), peak=("value", "max"))
    if kind in ("rate", "mean"):
        grouped["value"] = grouped["weighted"] / grouped["seconds"].replace(0, np.nan)
    elif kind == "max":
        grouped["value"] = grouped["peak"]
    elif kind == "count":
        grouped["value"] = grouped["total"] / (grouped["seconds"] / 3600.0).replace(0, np.nan)
    else:
        raise ValueError(f"no aggregation rule for measure kind {kind!r}")
    return grouped[["cage", "tag", "genotype", "hour", "value"]]


def phase_profiles(hourly: pd.DataFrame, output: Path, measures: list[str],
                   lights_on: int = LIGHTS_ON, lights_off: int = LIGHTS_OFF) -> None:
    """Mean +/- SEM hour-of-day profile per measure, one line per genotype.

    `hourly` is the rack-masked long frame joined with genotype (it must carry
    a `genotype` column already, since only the pipeline knows which animal
    table applies). One subplot per measure in `measures`.

    Bins are first reduced to one duration-weighted value per ANIMAL at each
    hour-of-day (`_hourly_value_per_animal`), then mean and SEM are computed
    across those per-animal values. Two things this fixes at once: (1) a flat
    mean over raw bin values is "per bin", not "per hour", so pooling cages
    with different bin widths manufactures a difference that is pure bin-width
    artefact; (2) the experimental unit is the animal, so the SEM must be
    taken across animals, not across pooled bins from every animal and every
    day -- pooling bins makes the band far too narrow (the real study's CSV
    recorded n=1200 for 5 animals) and overstates how precisely the mean is
    known.
    """
    measures = [m for m in measures if m in set(hourly.get("measure", []))]
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
    table = pd.DataFrame(rows, columns=["measure", "genotype", "hour", "mean", "sem", "n"])

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
    """Forest plot of Hedges' g and its CI (in g units), ranked by p-value.

    Plotting the raw `difference` mixes centimetres, percentages, and
    unitless indices on one shared axis -- with a real scan, distance sets
    the scale and every percentage/index collapses onto the zero line.
    Hedges' g is unitless and already computed (`groups.hedges_g`), so it is
    what goes on the shared axis instead. The bootstrap CI (`ci_low`/`ci_high`)
    is on the same scale as `difference`, not `hedges_g`, so it is rescaled by
    `hedges_g / difference` -- the same ratio that converts the point estimate
    itself from raw units to SD units -- to keep the interval's width, in SD
    units, consistent with the point.

    Writes nothing when there is no per-measure comparison to plot: either the
    scan table carries only a `note` explaining a skip (no `measure` column),
    or every measure was individually skipped (a `measure` column with zero
    rows). Either way, the figures/tables stay paired exactly.
    """
    if "measure" not in scan.columns or scan.empty:
        return

    table = scan[["measure", "difference", "ci_low", "ci_high", "hedges_g",
                 "p_value", "p_adjusted_bh"]].copy()
    scale = np.where(table["difference"] != 0,
                     table["hedges_g"] / table["difference"].replace(0, np.nan), 0.0)
    table["g_ci_low"] = table["ci_low"] * scale
    table["g_ci_high"] = table["ci_high"] * scale
    ordered = table.sort_values("p_value").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6, 0.5 * len(ordered) + 1.5))
    y = np.arange(len(ordered))[::-1]
    for row, y_position in zip(ordered.itertuples(), y):
        color = "#d03b3b" if row.p_adjusted_bh < 0.05 else "#52514e"
        if pd.isna(row.hedges_g):
            ax.scatter([0], [y_position], marker="x", color="#999999", s=30, zorder=2)
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
    coverage, and fraction of values missing (NaN) -- plus bin width, called
    out explicitly because a mixed-resolution study must be stated, not
    silently averaged: pooling a 600-second-bin cage with a 3600-second-bin
    cage in a per-bin figure would manufacture a difference that is pure bin
    width (see `actogram`, `phase_profiles`).

    Built from the `qc.qc_tables` outputs, folded into one per-cage summary
    so the figure and its CSV both describe "how much can I trust this
    cage's data" at a glance.
    """
    rack, regularity, breaks, coverage, ranges, missing = (
        qc["rack"], qc["regularity"], qc["breaks"], qc["coverage"], qc["ranges"],
        qc["missing"])

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
        "bin_width_s": [bin_width_label.get(cage, "") for cage in cages],
    })

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6))
    panels = [("mean_fraction_out_of_rack", "fraction of bins out of rack"),
             ("n_recording_breaks", "recording breaks (relative to cage's own bin width)"),
             ("n_range_violations", "out-of-range values"),
             ("mean_valid_hours_per_day", "mean valid hours / day"),
             ("mean_fraction_missing_values", "fraction of values missing (NaN)")]
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
