"""Quality control. Reports what is wrong; repairs nothing.

The one thing QC does change downstream is the rack mask: bins recorded while the
cage was out of the rack are not observations of the animal's behaviour, so they
are excluded from every metric rather than averaged into one. A bin whose
`in_rack` reading is non-finite (NaN, +-inf) is excluded the same way -- it is
not evidence the cage was in the rack -- but is counted SEPARATELY from a
genuine out-of-rack (0) reading (see `valid_bins`, `non_finite_rack`): folding
it into the out-of-rack count would hide a different failure mode (a missing
or corrupt reading) behind a normal one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import DISTANCE_DERIVED_PROFILE_MEASURES, MEASURE_KIND, PERCENT_MEASURES


def valid_bins(long: pd.DataFrame) -> pd.DataFrame:
    """Build the rack-validity mask from the `in_rack` measure.

    Three outcomes per bin, kept as separate signals:
    - finite and > 0 -> valid: an observation of the animal's behaviour.
    - finite and <= 0 (the real data only ever has 0) -> excluded, counted as
      "out of rack" (see `rack_summary`).
    - non-finite (NaN, +-inf) -> excluded, counted SEPARATELY as "non-finite"
      (see `non_finite_rack`). This used to raise: a NaN rack status was
      previously coerced to "out of rack" and silently dropped bins, so a
      guard was added to stop that happening unnoticed. Controller ruling:
      the guard's purpose was to stop such bins being silently excluded, not
      to make them fatal -- on the real six-cage study this fires on exactly
      one cell at the very first recorded bin of each of cages 58597, 58627
      and 62923, which would otherwise block the entire analysis. So the bin
      is excluded (never assumed valid, exactly as the guard intended) and
      reported by name instead of raising.
    """
    rack = long[long["measure"] == "in_rack"]
    if rack.empty:
        raise ValueError("no 'in_rack' measure in this export; cannot build the rack mask")
    mask = rack[["cage", "tag", "bin_start", "value"]].copy()
    values = mask.pop("value")
    finite = np.isfinite(values)
    mask["non_finite"] = ~finite
    mask["valid"] = finite & (values > 0)
    return mask


def apply_rack_mask(long: pd.DataFrame) -> pd.DataFrame:
    mask = valid_bins(long)
    merged = long.merge(mask, on=["cage", "tag", "bin_start"], how="left", validate="many_to_one")
    if merged["valid"].isna().any():
        raise ValueError("bins with no rack status; the export is internally inconsistent")
    kept = merged[merged["valid"] & (merged["measure"] != "in_rack")]
    return kept.drop(columns=["valid", "non_finite"]).reset_index(drop=True)


def rack_summary(long: pd.DataFrame) -> pd.DataFrame:
    """Per (cage, tag): total bins and how many were genuinely out of rack.

    `bins_out_of_rack` counts only finite, non-positive `in_rack` readings --
    it deliberately excludes non-finite readings (see `valid_bins`,
    `non_finite_rack`), which are a different failure mode and are reported
    in their own table so the two are never conflated.
    """
    mask = valid_bins(long)
    out_of_rack = (~mask["valid"]) & (~mask["non_finite"])
    grouped = mask.assign(out_of_rack=out_of_rack).groupby(
        ["cage", "tag"], as_index=False).agg(
        bins=("valid", "size"), bins_out_of_rack=("out_of_rack", "sum"))
    grouped["bins_out_of_rack"] = grouped["bins_out_of_rack"].astype(int)
    grouped["fraction_out"] = grouped["bins_out_of_rack"] / grouped["bins"]
    return grouped


def non_finite_rack(long: pd.DataFrame) -> pd.DataFrame:
    """Per (cage, tag): how many bins have a non-finite `in_rack` value.

    These bins are excluded from analysis -- a non-finite reading is not
    evidence the cage was in the rack -- but must never be folded into
    `rack_summary`'s out-of-rack count (controller ruling, see `valid_bins`).
    Only cage/tag pairs with at least one non-finite bin are listed, so an
    empty result means every loaded `in_rack` reading was finite.
    """
    mask = valid_bins(long)
    bad = mask[mask["non_finite"]]
    columns = ["cage", "tag", "n_non_finite_bins"]
    if bad.empty:
        return pd.DataFrame(columns=columns)
    grouped = (bad.groupby(["cage", "tag"], as_index=False).size()
               .rename(columns={"size": "n_non_finite_bins"}))
    return grouped[columns]


def bin_regularity(long: pd.DataFrame) -> pd.DataFrame:
    bins = long.drop_duplicates(subset=["cage", "bin_start"])
    return (bins.groupby(["cage", "duration_s"], as_index=False)
            .size().rename(columns={"size": "bins"}))


def native_bin_width(long: pd.DataFrame) -> pd.DataFrame:
    """Per cage: the modal (most-populous) NATIVE bin duration, in seconds.

    `bin_regularity` already reports one row per (cage, duration_s) observed
    in the raw export -- almost always a single row per cage, since a cage's
    bins are normally all one width. This collapses that to one row per cage
    by taking the duration the most bins actually share, defensively, in case
    a handful of bins in some export ever carries a different span.

    Must be called on the RAW frame, before any `metrics.resample_bins`
    re-binning: "native" means what the vendor actually recorded, not this
    study's downstream analysis resolution -- see `distance_resolution_confound`.
    """
    regularity = bin_regularity(long)
    if regularity.empty:
        return pd.DataFrame(columns=["cage", "native_bin_width_s"])
    widest = regularity.loc[regularity.groupby("cage")["bins"].idxmax(),
                            ["cage", "duration_s"]]
    return widest.rename(columns={"duration_s": "native_bin_width_s"}).reset_index(drop=True)


def distance_resolution_confound(long: pd.DataFrame) -> pd.DataFrame:
    """Flag cages whose NATIVE (pre-resampling) bin width differs from the rest.

    `distance_cm` is a PATH LENGTH, and path length is sampling-rate dependent
    -- the coastline problem: summing many short-bin path-length estimates
    captures micro-movement that the same behaviour, recorded at a coarser
    native resolution, never sees, because the vendor already computed each
    bin's path length at ITS OWN native resolution before this pipeline ever
    touches the data. `metrics.resample_bins` (the downstream re-binning to a
    common width) cannot undo this: it can only average already-inflated
    finer-resolution path lengths down to a coarser cadence, not recover the
    coarser-resolution path length that would have been measured natively.

    Measured on the real study (WILDTYPE animals only, so genotype cannot
    explain the gap): cage 58616, natively at 600 s bins, reads L5 (lowest
    5-hour mean of hourly distance) 1.54x the other five cages' (natively
    3600 s) L5, with zero overlap between the two groups; M10 (highest 10-hour
    mean) only 1.03x. The inflation concentrates in the nadir because
    quiet-hour movement is almost all micro-movement, while peak-hour movement
    is dominated by large translations the coarser resolution already
    captures well.

    This is not a data-quality defect to repair -- a mixed-vendor-resolution
    study is an expected RESULT, not an error -- so it is reported, never
    raised, and the run completes either way.

    Returns one row per cage: its native bin width, whether ANY cage in this
    study has a different native width (a study-level fact, repeated on every
    row so a reader scanning the CSV sees it without cross-referencing
    anything else), and the fixed list of profile measures the mismatch
    taints (`io.DISTANCE_DERIVED_PROFILE_MEASURES` -- `distance_cm` itself and
    everything the circadian family builds from it). A WITHIN-CAGE
    (cage-stratified) comparison of these measures stays valid regardless --
    every cage's own bins share one native width, so a within-cage difference
    is never a resolution artefact -- only a POOLED or other cross-cage
    comparison is unsafe. Empty input returns an empty frame with the same
    columns.
    """
    columns = ["cage", "native_bin_width_s", "bin_width_differs_in_study",
              "distance_derived_measures_affected"]
    widths = native_bin_width(long)
    if widths.empty:
        return pd.DataFrame(columns=columns)
    differs = bool(widths["native_bin_width_s"].nunique() > 1)
    widths["bin_width_differs_in_study"] = differs
    widths["distance_derived_measures_affected"] = (
        ", ".join(DISTANCE_DERIVED_PROFILE_MEASURES) if differs else "")
    return widths[columns]


def recording_breaks(long: pd.DataFrame, threshold_min: float | None = None,
                     factor: float = 1.5) -> pd.DataFrame:
    """Flag start-to-start gaps that exceed a threshold, per cage.

    ``threshold_min=None`` (the default) makes the threshold RELATIVE to each
    cage's own bin width: a gap counts as a break only once it exceeds
    ``factor`` times that cage's median bin duration. A fixed absolute
    threshold (the previous behaviour) does not generalise across a study
    with more than one bin width -- a hardcoded 15-minute threshold flags
    EVERY gap in an hourly-binned cage (its normal 60-minute cadence is
    already 4x over 15 minutes), drowning real breaks in false positives.
    Pass an explicit ``threshold_min`` to keep the old fixed-threshold
    behaviour for a single known bin width.
    """
    rows = []
    for cage, frame in long.drop_duplicates(subset=["cage", "bin_start"]).groupby("cage"):
        ordered = frame.sort_values("bin_start")
        gap = (ordered["bin_start"].shift(-1) - ordered["bin_start"]).dt.total_seconds() / 60
        if threshold_min is None:
            bin_minutes = ordered["duration_s"] / 60.0
            cage_threshold = float(bin_minutes.median()) * factor if not bin_minutes.empty else 15.0
        else:
            cage_threshold = threshold_min
        flagged = ordered.assign(gap_min=gap)
        flagged = flagged[flagged["gap_min"] > cage_threshold]
        for _, row in flagged.iterrows():
            rows.append({"cage": cage, "after_bin": row["bin_start"],
                         "gap_min": float(row["gap_min"])})
    return pd.DataFrame(rows, columns=["cage", "after_bin", "gap_min"])


def coverage(long: pd.DataFrame) -> pd.DataFrame:
    bins = long.drop_duplicates(subset=["cage", "tag", "bin_start"]).copy()
    bins["date"] = bins["bin_start"].dt.date
    table = bins.groupby(["cage", "tag", "date"], as_index=False).agg(
        valid_bins=("bin_start", "size"), valid_seconds=("duration_s", "sum"))
    table["valid_hours"] = table.pop("valid_seconds") / 3600.0
    return table


def range_violations(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    percent = long[long["measure"].isin(PERCENT_MEASURES)]
    bad = percent[(percent["value"] < 0) | (percent["value"] > 100)]
    for (cage, tag, measure), frame in bad.groupby(["cage", "tag", "measure"]):
        rows.append({"cage": cage, "tag": tag, "measure": measure,
                     "rule": "percent outside [0, 100]", "n_violations": len(frame)})
    negative = long[(~long["measure"].isin(PERCENT_MEASURES)) & (long["value"] < 0)]
    for (cage, tag, measure), frame in negative.groupby(["cage", "tag", "measure"]):
        rows.append({"cage": cage, "tag": tag, "measure": measure,
                     "rule": "negative value", "n_violations": len(frame)})
    return pd.DataFrame(rows, columns=["cage", "tag", "measure", "rule", "n_violations"])


def missing_values(long: pd.DataFrame) -> pd.DataFrame:
    """Per (cage, tag, measure): how many bins have a missing value, and what fraction.

    ``io.load_cage`` coerces a malformed cell (a literal ``'NaN'`` string, for
    example) to NaN with ``errors="coerce"``, silently. Nothing else in QC can
    see this: ``range_violations`` cannot flag a NaN (a NaN comparison is
    False), and ``coverage`` counts the bin as valid because the rack mask
    only cares about ``in_rack``, not about the measure's own value. This is
    the audit that makes a silently-coerced value visible, so a cage/tag/
    measure that is missing in every bin -- which would otherwise average to
    an invisible NaN (or, before the aggregation fix, an indistinguishable-
    from-real 0.0) -- is named explicitly instead.
    """
    grouped = long.groupby(["cage", "tag", "measure"], as_index=False).agg(
        bins=("value", "size"), missing=("value", lambda v: int(v.isna().sum())))
    grouped["fraction_missing"] = grouped["missing"] / grouped["bins"]
    return grouped


def excluded_tag_table(long: pd.DataFrame, excluded: list[tuple[int, int]]) -> pd.DataFrame:
    """Per (cage, tag) excluded by `animals.check_join`: how much data it carries.

    A tag present in an export but not listed in the animal CSV is not a
    tracked animal (the study owner's ruling -- see `animals.check_join`) and
    never reaches `build_profile` or any other analysis, however much data it
    carries. It must not vanish silently, so this reports, for each excluded
    tag, how many bins it has and what fraction of those bins actually carry a
    value -- typically ~99.9% missing for the real study's three phantom tags,
    which is the evidence that these are stray slots (an animal removed from
    the cage, or one whose ear tag was lost) rather than a parsing artefact.

    ``long`` must be the UNFILTERED long frame (before excluded tags are
    dropped) -- this is the one place their data is still read.

    A bin counts as "present" for a tag if AT LEAST ONE of its non-mask
    measures (everything except `in_rack`) has a non-missing value in that
    bin. `in_rack` is excluded from the presence check because it describes
    the cage's rack state, not the animal -- an excluded tag's slot can carry
    a normal `in_rack` reading with no behavioural data behind it at all, and
    counting that would understate how empty these tags actually are.
    """
    columns = ["cage", "tag", "bins", "present_bins", "missing_fraction"]
    if not excluded:
        return pd.DataFrame(columns=columns)
    keys = pd.DataFrame(sorted(set(excluded)), columns=["cage", "tag"])
    subset = long.merge(keys, on=["cage", "tag"], how="inner")
    subset = subset[subset["measure"].map(MEASURE_KIND) != "mask"]
    rows = []
    for (cage, tag), block in subset.groupby(["cage", "tag"]):
        bins = int(block["bin_start"].nunique())
        present = int(block.loc[block["value"].notna(), "bin_start"].nunique())
        missing_fraction = (1 - present / bins) if bins else float("nan")
        rows.append({"cage": cage, "tag": tag, "bins": bins,
                     "present_bins": present, "missing_fraction": missing_fraction})
    return pd.DataFrame(rows, columns=columns)


def qc_tables(long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    masked = apply_rack_mask(long)
    return {"rack": rack_summary(long), "regularity": bin_regularity(long),
            "breaks": recording_breaks(long), "coverage": coverage(masked),
            "ranges": range_violations(long), "missing": missing_values(long),
            "rack_non_finite": non_finite_rack(long),
            "distance_resolution": distance_resolution_confound(long)}
