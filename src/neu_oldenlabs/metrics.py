"""Aggregate binned measures to one value per animal, day, and light phase.

Three aggregation rules, chosen by measure type, because using one rule for all
of them is the easiest way to manufacture an effect:

* rates and means are weighted by each bin's actual duration -- a bin's own span
  (`Datetime to` minus `Datetime from`) need not be a fixed ten minutes, so a flat
  mean would silently overweight a short bin if one ever appeared; in the real
  cage-58616 export every bin happens to span exactly 600 s, so this weighting is
  a no-op there and a safeguard for a future export, not a response to duration
  jitter observed here (the START-TO-START gap between bins is a different
  quantity and is the one that varies in that export; see `qc.bin_regularity`);
* maxima take the maximum;
* counts become a rate per valid hour, so a day that lost bins to the rack mask is
  not scored as a day of low activity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import MEASURE_KIND

LIGHTS_ON = 7
LIGHTS_OFF = 19


def add_time_fields(long: pd.DataFrame, lights_on: int = LIGHTS_ON,
                    lights_off: int = LIGHTS_OFF) -> pd.DataFrame:
    frame = long.copy()
    frame["date"] = frame["bin_start"].dt.date
    frame["hour"] = frame["bin_start"].dt.hour
    light = (frame["hour"] >= lights_on) & (frame["hour"] < lights_off)
    frame["phase"] = np.where(light, "light", "dark")
    return frame


KEYS = ["cage", "tag", "date"]


def _aggregate_measure(block: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Apply one measure's rule to every (cage, tag, date) group at once.

    A bin's ``value`` can be missing (``io.load_cage`` coerces a malformed
    cell, e.g. a literal ``'NaN'`` string, to NaN). The duration weight for a
    group must count ONLY bins whose value is present -- ``seconds`` here is
    the total duration of bins that actually have a value, not every bin in
    the group -- otherwise a rate/mean's denominator includes minutes that
    contributed nothing to its numerator, deflating the mean in proportion to
    the missing fraction. A group with no present value at all then has
    ``seconds == 0``, so every branch below divides by a value that
    ``.replace(0, np.nan)`` turns into NaN, and the result is NaN, never the
    ``0.0`` that ``pandas.Series.sum()`` would otherwise silently produce for
    an all-NaN group.
    """
    kind = MEASURE_KIND[measure]
    frame = block.copy()
    frame["weighted"] = frame["value"] * frame["duration_s"]
    frame["duration_present"] = frame["duration_s"].where(frame["value"].notna(), 0.0)
    grouped = frame.groupby(KEYS, as_index=False).agg(
        weighted=("weighted", "sum"), seconds=("duration_present", "sum"),
        total=("value", "sum"), peak=("value", "max"))
    grouped["valid_hours"] = grouped["seconds"] / 3600.0
    if kind in ("rate", "mean"):
        grouped["value"] = grouped["weighted"] / grouped["seconds"].replace(0, np.nan)
    elif kind == "max":
        grouped["value"] = grouped["peak"]
    elif kind == "count":
        grouped["value"] = grouped["total"] / grouped["valid_hours"].replace(0, np.nan)
    else:
        raise ValueError(f"no aggregation rule for measure kind {kind!r}")
    grouped["measure"] = measure
    return grouped[KEYS + ["measure", "value", "valid_hours"]]


def daily(long: pd.DataFrame, lights_on: int = LIGHTS_ON,
          lights_off: int = LIGHTS_OFF) -> pd.DataFrame:
    frame = add_time_fields(long, lights_on, lights_off)
    parts = []
    for phase in ("all", "light", "dark"):
        subset = frame if phase == "all" else frame[frame["phase"] == phase]
        if subset.empty:
            continue
        for measure, block in subset.groupby("measure", sort=False):
            aggregated = _aggregate_measure(block, measure)
            aggregated.insert(3, "phase", phase)
            parts.append(aggregated)
    table = pd.concat(parts, ignore_index=True)
    return table[["cage", "tag", "date", "phase", "measure", "value", "valid_hours"]]


def drop_habituation(table: pd.DataFrame, days: int) -> pd.DataFrame:
    """Drop the first ``days`` calendar days of each cage's recording.

    The first day is a partial day and the animals have just been introduced to
    the cage, so it measures handling, not phenotype. ``days=0`` means "drop
    nothing" -- it does not still drop the leading partial day.
    """
    if days <= 0:
        return table
    kept = []
    for cage, frame in table.groupby("cage", sort=False):
        dates = sorted(set(frame["date"]))
        excluded = set(dates[:days + 1])
        kept.append(frame[~frame["date"].isin(excluded)])
    return pd.concat(kept, ignore_index=True)


def restrict_window(table: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    frame = table
    if start is not None:
        frame = frame[frame["date"] >= pd.Timestamp(start).date()]
    if end is not None:
        frame = frame[frame["date"] <= pd.Timestamp(end).date()]
    return frame.reset_index(drop=True)
