"""One row per animal: the behavioural phenotype the statistics compare.

Each daily measure is averaged across days, separately for the whole day and for
each light phase, then joined to the circadian fits and to genotype. Averaging
across days rather than pooling all bins gives every day equal weight, so a single
long day cannot dominate an animal's phenotype.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRIMARY_MEASURES = ("distance_cm_all", "inactive_pct_dark", "IS",
                    "social_distance_cm_all", "aggression_events_all")

IDENTITY = ["animal_id", "cage", "tag", "animal_number", "ear_mark",
            "genotype", "genotype_raw", "n_days"]


def _check_total_join(daily_keys: set, circadian_keys: set, animal_keys: set) -> None:
    """Fail unless the daily data, circadian table, and animal table name the same animals.

    A duplicate or missing animal must raise, not be silently dropped or NaN-filled.
    `validate="one_to_one"` on the merges only guards against duplicate keys; it does
    nothing to check that every key on one side has a partner on the other. This is
    the other half of that guard, in the shape of `animals.check_join`.
    """
    all_keys = daily_keys | circadian_keys | animal_keys
    offenders = {}
    for key in all_keys:
        missing_from = []
        if key not in daily_keys:
            missing_from.append("daily data")
        if key not in circadian_keys:
            missing_from.append("circadian table")
        if key not in animal_keys:
            missing_from.append("animal table")
        if missing_from:
            offenders[key] = missing_from
    if offenders:
        details = [f"{cage}-{tag} missing from {', '.join(sides)}"
                   for (cage, tag), sides in sorted(offenders.items())]
        raise ValueError(f"animal join is not total: {'; '.join(details)}")


def build_profile(daily_table: pd.DataFrame, circadian_table: pd.DataFrame,
                  animal_table: pd.DataFrame, min_days: int = 5) -> pd.DataFrame:
    """Assemble one profile row per animal from the three per-animal inputs.

    Contract: `daily_table`, `circadian_table`, and `animal_table` must describe
    exactly the same set of `(cage, tag)` animals. This function does NOT filter or
    reconcile them — it is not this function's job to decide which animals belong in
    the study. If the study's animal table lists animals for cages that were never
    loaded (e.g. a cage export is missing), that mismatch must be resolved by the
    caller (the pipeline) before calling `build_profile`, by restricting the animal
    table to the cages actually present. Given mismatched inputs, `build_profile`
    raises rather than silently dropping or NaN-filling the offending animals.
    """
    daily_table = daily_table.copy()
    daily_table["column"] = daily_table["measure"] + "_" + daily_table["phase"]

    daily_keys = set(zip(daily_table["cage"], daily_table["tag"]))
    circadian_keys = set(zip(circadian_table["cage"], circadian_table["tag"]))
    animal_keys = set(zip(animal_table["cage"], animal_table["tag"]))
    _check_total_join(daily_keys, circadian_keys, animal_keys)

    means = (daily_table.groupby(["cage", "tag", "column"], as_index=False)["value"].mean()
             .pivot(index=["cage", "tag"], columns="column", values="value"))
    means.columns.name = None

    # A day only counts toward `n_days` if at least one "all"-phase measure has
    # a real (non-NaN) value that day. Without the `value.notna()` filter, a
    # calendar day where every measure was missing for the whole day (e.g. a
    # sensor dead for an animal's entire recording, see metrics._aggregate_measure)
    # still produced a row -- with value now correctly NaN rather than a fabricated
    # 0.0 -- and would still count as a "day of data" here, letting the min_days
    # guard below be satisfied by days that contain no information at all.
    has_data = daily_table["phase"].eq("all") & daily_table["value"].notna()
    day_counts = (daily_table[has_data]
                  .groupby(["cage", "tag"])["date"].nunique().rename("n_days"))

    distance = daily_table[(daily_table["measure"] == "distance_cm")
                           & (daily_table["phase"] == "all")]
    variability = distance.groupby(["cage", "tag"])["value"].agg(
        lambda values: float(np.std(values, ddof=1) / np.mean(values))
        if len(values) > 1 and np.mean(values) else float("nan")).rename("distance_cv_daily")

    table = means.join(day_counts).join(variability).reset_index()
    # An animal with no "all"-phase rows has no entry in day_counts at all, which
    # joins as NaN rather than 0. NaN < min_days is False, so the min-days blanking
    # below would silently skip exactly the animal with the least data unless a
    # missing count is treated as zero days.
    table["n_days"] = table["n_days"].fillna(0)

    if "distance_cm_dark" in table and "distance_cm_light" in table:
        table["distance_dark_light_ratio"] = table["distance_cm_dark"] / table["distance_cm_light"]
    if "inactive_pct_all" in table and "sleep_bouts_all" in table:
        # inactive_pct is a percentage of the hour; sleep_bouts is bouts per hour.
        table["sleep_bout_minutes_all"] = (table["inactive_pct_all"] / 100.0 * 60.0
                                           / table["sleep_bouts_all"].replace(0, np.nan))

    table = table.merge(circadian_table, on=["cage", "tag"], how="left", validate="one_to_one")
    table = table.merge(animal_table, on=["cage", "tag"], how="inner", validate="one_to_one")
    table["animal_id"] = table["cage"].astype(str) + "-" + table["tag"].astype(str)

    measure_columns = [c for c in table.columns if c not in IDENTITY]
    thin = table["n_days"] < min_days
    table.loc[thin, measure_columns] = np.nan

    ordered = IDENTITY + sorted(measure_columns)
    return table[ordered].sort_values(["cage", "tag"]).reset_index(drop=True)
