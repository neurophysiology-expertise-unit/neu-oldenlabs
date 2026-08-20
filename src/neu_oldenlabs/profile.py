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


def build_profile(daily_table: pd.DataFrame, circadian_table: pd.DataFrame,
                  animal_table: pd.DataFrame, min_days: int = 5) -> pd.DataFrame:
    daily_table = daily_table.copy()
    daily_table["column"] = daily_table["measure"] + "_" + daily_table["phase"]

    means = (daily_table.groupby(["cage", "tag", "column"], as_index=False)["value"].mean()
             .pivot(index=["cage", "tag"], columns="column", values="value"))
    means.columns.name = None

    day_counts = (daily_table[daily_table["phase"] == "all"]
                  .groupby(["cage", "tag"])["date"].nunique().rename("n_days"))

    distance = daily_table[(daily_table["measure"] == "distance_cm")
                           & (daily_table["phase"] == "all")]
    variability = distance.groupby(["cage", "tag"])["value"].agg(
        lambda values: float(np.std(values, ddof=1) / np.mean(values))
        if len(values) > 1 and np.mean(values) else float("nan")).rename("distance_cv_daily")

    table = means.join(day_counts).join(variability).reset_index()

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
