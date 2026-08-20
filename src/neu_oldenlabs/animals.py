"""Animal identity and genotype, read from the study's CSV.

The CSV writes the cage number only on the first row of each block, so the cage
column is forward-filled. Genotype is normalised through one visible map and the
raw string is kept beside it, because a normalisation nobody can audit is a
normalisation nobody should trust.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

GENOTYPE_MAP = {"W/w": "wt", "W/W": "wt", "Mut/W": "mut"}

COLUMNS = ["cage", "animal_number", "ear_mark", "genotype_raw", "tag"]


def load_animals(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if len(raw.columns) < 5:
        raise ValueError(f"{csv_path}: expected at least 5 columns, got {len(raw.columns)}")
    table = raw.iloc[:, :5].copy()
    table.columns = COLUMNS
    table = table[table["animal_number"].str.strip() != ""]
    for column in COLUMNS:
        table[column] = table[column].str.strip()
    table["cage"] = table["cage"].replace("", pd.NA).ffill()
    if table["cage"].isna().any():
        raise ValueError(f"{csv_path}: rows before the first cage number")
    table["cage"] = table["cage"].astype("int64")
    table["animal_number"] = table["animal_number"].astype("int64")
    table["tag"] = table["tag"].astype("int64")

    unknown = sorted(set(table["genotype_raw"]) - set(GENOTYPE_MAP))
    if unknown:
        raise ValueError(f"{csv_path}: unknown genotype {unknown}; allowed {sorted(GENOTYPE_MAP)}")
    table["genotype"] = table["genotype_raw"].map(GENOTYPE_MAP)

    duplicated = table.duplicated(subset=["cage", "tag"], keep=False)
    if duplicated.any():
        offenders = table.loc[duplicated, ["cage", "tag"]].to_dict("records")
        raise ValueError(f"{csv_path}: duplicate tag within a cage: {offenders}")
    return table[["cage", "animal_number", "ear_mark", "genotype_raw", "genotype", "tag"]]


def check_join(table: pd.DataFrame, cage: int, tags: set[int]) -> None:
    """Fail unless the export's tags and the animal table's tags match exactly.

    A partial join would silently analyse fewer animals than the study contains,
    which is the one failure mode that never announces itself in a figure.
    """
    listed = set(table.loc[table["cage"] == cage, "tag"].tolist())
    if not listed:
        raise ValueError(f"cage {cage} has no animals in the animal table")
    extra = sorted(tags - listed)
    if extra:
        raise ValueError(f"cage {cage}: tags only in export, not in animal table: {extra}")
    absent = sorted(listed - tags)
    if absent:
        raise ValueError(f"cage {cage}: tags only in animal table, not in export: {absent}")
