"""Circadian description of the activity signal.

Activity is hourly distance travelled. Two families are reported: a cosinor fit,
which assumes a sinusoid and returns amplitude and phase, and the non-parametric
IS / IV / RA family, which assumes nothing about waveform. They answer different
questions and disagreeing is informative, so both are kept.

These metrics are descriptive. They are only interpretable as circadian claims
when the light schedule and a complete-day recording window are confirmed, which
for this study they are: 12:12, lights on 07:00.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def hourly_activity(long: pd.DataFrame) -> pd.DataFrame:
    """Distance per hour, on a regular hourly grid, per animal."""
    distance = long[long["measure"] == "distance_cm"].copy()
    if distance.empty:
        raise ValueError("no distance_cm measure; cannot build an activity signal")
    distance["hour_start"] = distance["bin_start"].dt.floor("h")
    grouped = distance.groupby(["cage", "tag", "hour_start"], as_index=False).agg(
        distance=("value", "sum"), seconds=("duration_s", "sum"))
    grouped["distance_per_hour"] = grouped["distance"] / (grouped["seconds"] / 3600.0)
    return grouped[["cage", "tag", "hour_start", "distance_per_hour"]]


def _as_hourly(series: pd.Series) -> pd.Series:
    ordered = series.sort_index()
    return ordered.asfreq("h") if ordered.index.freq is None else ordered


def cosinor(series: pd.Series) -> dict:
    """Least-squares fit of ``mesor + amplitude * cos(2*pi*(t - acrophase)/24)``."""
    values = series.to_numpy(dtype=float)
    hours = series.index.hour.to_numpy(dtype=float) + series.index.minute.to_numpy() / 60.0
    radians = 2 * np.pi * hours / 24.0
    design = np.column_stack([np.ones_like(radians), np.cos(radians), np.sin(radians)])
    finite = np.isfinite(values)
    coefficients, *_ = np.linalg.lstsq(design[finite], values[finite], rcond=None)
    mesor, cosine, sine = coefficients
    amplitude = float(np.hypot(cosine, sine))
    acrophase = float(np.arctan2(sine, cosine) * 24 / (2 * np.pi)) % 24
    return {"mesor": float(mesor), "amplitude": amplitude, "acrophase_hour": acrophase}


def interdaily_stability(series: pd.Series) -> float:
    """How reproducibly the 24-hour pattern repeats across days. 0 to 1."""
    values = _as_hourly(series)
    hourly_mean = values.groupby(values.index.hour).mean()
    grand = values.mean()
    n = values.count()
    numerator = n * ((hourly_mean - grand) ** 2).sum()
    denominator = len(hourly_mean) * ((values - grand) ** 2).sum()
    return float(numerator / denominator) if denominator else float("nan")


def intradaily_variability(series: pd.Series) -> float:
    """Fragmentation: mean squared hour-to-hour change over the variance."""
    values = _as_hourly(series).to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return float("nan")
    numerator = len(finite) * np.sum(np.diff(finite) ** 2)
    denominator = (len(finite) - 1) * np.sum((finite - finite.mean()) ** 2)
    return float(numerator / denominator) if denominator else float("nan")


def relative_amplitude(series: pd.Series) -> dict:
    """Highest ten and lowest five consecutive hours, and their contrast."""
    values = _as_hourly(series)
    ten = values.rolling(10, min_periods=10).mean()
    five = values.rolling(5, min_periods=5).mean()
    m10 = float(ten.max())
    l5 = float(five.min())
    total = m10 + l5
    return {"M10": m10, "L5": l5, "RA": float((m10 - l5) / total) if total else float("nan")}


def circadian_profile(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    hourly = hourly_activity(long)
    for (cage, tag), frame in hourly.groupby(["cage", "tag"], sort=False):
        series = frame.set_index("hour_start")["distance_per_hour"]
        row = {"cage": cage, "tag": tag}
        row.update(cosinor(series))
        row["IS"] = interdaily_stability(series)
        row["IV"] = intradaily_variability(series)
        row.update(relative_amplitude(series))
        rows.append(row)
    return pd.DataFrame(rows)
