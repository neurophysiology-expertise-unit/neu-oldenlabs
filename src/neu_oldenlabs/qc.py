"""Quality control. Reports what is wrong; repairs nothing.

The one thing QC does change downstream is the rack mask: bins recorded while the
cage was out of the rack are not observations of the animal's behaviour, so they
are excluded from every metric rather than averaged into one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import PERCENT_MEASURES


def valid_bins(long: pd.DataFrame) -> pd.DataFrame:
    rack = long[long["measure"] == "in_rack"]
    if rack.empty:
        raise ValueError("no 'in_rack' measure in this export; cannot build the rack mask")
    mask = rack[["cage", "tag", "bin_start", "value"]].copy()
    values = mask.pop("value")
    non_finite = ~np.isfinite(values)
    if non_finite.any():
        bad = mask[non_finite]
        cage_count = bad["cage"].nunique()
        tag_count = bad["tag"].nunique()
        bin_count = len(bad)
        raise ValueError(
            f"non-finite in_rack values: {bin_count} bins across {tag_count} tag(s) "
            f"in {cage_count} cage(s)"
        )
    mask["valid"] = values > 0
    return mask


def apply_rack_mask(long: pd.DataFrame) -> pd.DataFrame:
    mask = valid_bins(long)
    merged = long.merge(mask, on=["cage", "tag", "bin_start"], how="left", validate="many_to_one")
    if merged["valid"].isna().any():
        raise ValueError("bins with no rack status; the export is internally inconsistent")
    kept = merged[merged["valid"] & (merged["measure"] != "in_rack")]
    return kept.drop(columns="valid").reset_index(drop=True)


def rack_summary(long: pd.DataFrame) -> pd.DataFrame:
    mask = valid_bins(long)
    grouped = mask.groupby(["cage", "tag"], as_index=False).agg(
        bins=("valid", "size"), bins_out_of_rack=("valid", lambda v: int((~v).sum())))
    grouped["fraction_out"] = grouped["bins_out_of_rack"] / grouped["bins"]
    return grouped


def bin_regularity(long: pd.DataFrame) -> pd.DataFrame:
    bins = long.drop_duplicates(subset=["cage", "bin_start"])
    return (bins.groupby(["cage", "duration_s"], as_index=False)
            .size().rename(columns={"size": "bins"}))


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


def qc_tables(long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    masked = apply_rack_mask(long)
    return {"rack": rack_summary(long), "regularity": bin_regularity(long),
            "breaks": recording_breaks(long), "coverage": coverage(masked),
            "ranges": range_violations(long), "missing": missing_values(long)}
