"""Read Oldenlabs per-frame tracklet text files (`<uid>_coordinates_<from>_<to>.txt`).

The binned workbook (`io.py`) summarises each animal per 10- or 60-minute bin.
The tracklet export underneath it has one whitespace-separated row per video
frame (10 Hz): centroid, body and head boxes, heading, the vendor's per-frame
speed/distance, and per-frame binary behaviour labels for every animal.

Three facts about the files shape this reader:

- One export window can come back as SEVERAL text files. Tracking stops for a
  while (cage changes around 11:00 are the common case) and the file is split
  at the gap. Reading only the first file silently drops the rest of the day,
  so every `*_coordinates_*.txt` in a zip is read and concatenated.
- `frame` restarts at 1 in each file, so it is not a time index. Time comes from
  `datetime_from`, and a gap between files stays a gap: missing frames are not
  stillness, and nothing here fills them.
- Column names are checked against the known vendor schema. An unknown name
  raises, as in `io.py`: a renamed column must crash, not vanish. Values are
  carried as exported (positions in pixels, the vendor's own units otherwise);
  nothing is converted or renamed.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HZ = 10
TIME_FORMAT = "%Y-%m-%d-%H:%M:%S.%f"
_M = r"mouse(\d+)"
SCHEMA = [re.compile(p) for p in (
    r"frame", r"datetime_from", r"datetime_to",
    rf"centroid_{_M}_[xy]",
    rf"centroid_rectified_{_M}_[xy]",
    rf"bbox_{_M}_(topx|topy|width|height)",
    rf"bbox_head_(topx|topy|width|height)_{_M}",
    rf"head_centroid_[xy]_{_M}",
    rf"heading_{_M}",
    rf"(speed|acceleration_magnitude|angular_velocity|dist)_{_M}",
    rf"binary_(time_near_enr_tube|rearing_activity|climbing_activity|water_activity"
    rf"|food_activity|time_in_front_side_of_cage|time_in_center_of_cage"
    rf"|time_in_corner_of_cage|time_sleep|time_aggression)_{_M}",
)]


def check_schema(header: list[str]) -> None:
    unknown = [c for c in header if not any(p.fullmatch(c) for p in SCHEMA)]
    if unknown:
        raise ValueError(f"unknown tracklet column(s): {unknown}")
    if "datetime_from" not in header:
        raise ValueError("tracklet file has no datetime_from column")


def _sources(path: Path) -> list[tuple[str, callable]]:
    if path.suffix == ".zip":
        z = zipfile.ZipFile(path)
        names = sorted(n for n in z.namelist()
                       if "_coordinates_" in Path(n).name and n.endswith(".txt"))
        if not names:
            raise ValueError(f"{path}: no *_coordinates_*.txt inside")
        return [(n, lambda n=n: io.TextIOWrapper(z.open(n))) for n in names]
    return [(path.name, lambda: path.open())]


def read_coordinates(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Every frame of one export (zip or txt), all parts, in time order.

    Returns the requested vendor columns (all of them if `columns` is None) plus
    `t` (frame start time) and `part` (0-based index of the source file).
    `frame`, `datetime_from` and `datetime_to` are replaced by `t`.
    """
    path = Path(path)
    out = []
    for part, (name, opener) in enumerate(_sources(path)):
        with opener() as fh:
            header = fh.readline().split()
        check_schema(header)
        wanted = [c for c in header if c not in ("frame", "datetime_from", "datetime_to")]
        if columns is not None:
            missing = sorted(set(columns) - set(wanted))
            if missing:
                raise KeyError(f"{name}: no column(s) {missing}")
            wanted = [c for c in wanted if c in set(columns)]
        with opener() as fh:
            df = pd.read_csv(fh, sep=r"\s+", usecols=["datetime_from", *wanted],
                             dtype={c: "float32" for c in wanted})
        t = pd.to_datetime(df.pop("datetime_from"), format=TIME_FORMAT)
        out.append(pd.concat([pd.DataFrame({"t": t, "part": part}), df], axis=1))
    df = pd.concat(out, ignore_index=True).sort_values("t", kind="stable", ignore_index=True)
    if df.t.duplicated().any():
        raise ValueError(f"{path}: parts overlap in time ({df.t.duplicated().sum()} repeated frames)")
    return df


def gaps(t: pd.Series, min_s: float = 1.0) -> pd.DataFrame:
    """Stretches with no frames longer than `min_s`: last frame before, first after."""
    t = pd.Series(t).reset_index(drop=True)
    step = t.diff().dt.total_seconds()
    i = np.flatnonzero(step.to_numpy() > min_s)
    return pd.DataFrame({"last_before": t.iloc[i - 1].to_numpy(),
                         "first_after": t.iloc[i].to_numpy(),
                         "duration_s": step.iloc[i].to_numpy()})


def mice(df: pd.DataFrame) -> list[int]:
    """Animal slots present in a tracklet table (from its centroid columns)."""
    return sorted({int(m.group(1)) for c in df
                   if (m := re.fullmatch(rf"centroid_{_M}_x", c))})
