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


def check_join(table: pd.DataFrame, cage: int, tags: set[int]) -> list[int]:
    """Reconcile one cage's export tags against the animal table. The join is
    deliberately ASYMMETRIC, per the study owner's ruling that the animal CSV
    is the sole authority on which animals exist: "if the slot is not defined
    in csv then it should not exist."

    - A tag present in the export but NOT listed in the CSV is not a tracked
      animal -- it may be an animal removed from the cage, or one whose ear tag
      was lost and its slot re-used -- and is returned here as an EXCLUDED tag
      rather than raised on, however much data it carries. The caller must keep
      it out of every downstream analysis (profile, statistics, figures) and
      report it rather than let it vanish silently (see
      `qc.excluded_tag_table`).
    - A tag the CSV LISTS for this cage but which never appears in the export
      is still a genuine discrepancy -- the study is missing that animal's
      entire record -- and this direction still RAISES. Silently analysing
      fewer animals than the study contains is the one failure mode that never
      announces itself in a figure.

    Returns the sorted list of excluded (export-only) tags.
    """
    listed = set(table.loc[table["cage"] == cage, "tag"].tolist())
    if not listed:
        raise ValueError(f"cage {cage} has no animals in the animal table")
    absent = sorted(listed - tags)
    if absent:
        raise ValueError(f"cage {cage}: tags only in animal table, not in export: {absent}")
    return sorted(tags - listed)
