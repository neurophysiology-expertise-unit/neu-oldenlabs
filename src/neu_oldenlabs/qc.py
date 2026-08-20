"""Quality control. Reports what is wrong; repairs nothing.

The one thing QC does change downstream is the rack mask: bins recorded while the
cage was out of the rack are not observations of the animal's behaviour, so they
are excluded from every metric rather than averaged into one.
"""
from __future__ import annotations

import pandas as pd

from .io import PERCENT_MEASURES


def valid_bins(long: pd.DataFrame) -> pd.DataFrame:
    rack = long[long["measure"] == "in_rack"]
    if rack.empty:
        raise ValueError("no 'in_rack' measure in this export; cannot build the rack mask")
    mask = rack[["cage", "tag", "bin_start", "value"]].copy()
    mask["valid"] = mask.pop("value") > 0
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


def recording_breaks(long: pd.DataFrame, threshold_min: float = 15.0) -> pd.DataFrame:
    rows = []
    for cage, frame in long.drop_duplicates(subset=["cage", "bin_start"]).groupby("cage"):
        ordered = frame.sort_values("bin_start")
        gap = (ordered["bin_start"].shift(-1) - ordered["bin_start"]).dt.total_seconds() / 60
        flagged = ordered.assign(gap_min=gap)
        flagged = flagged[flagged["gap_min"] > threshold_min]
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


def qc_tables(long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    masked = apply_rack_mask(long)
    return {"rack": rack_summary(long), "regularity": bin_regularity(long),
            "breaks": recording_breaks(long), "coverage": coverage(masked),
            "ranges": range_violations(long)}
