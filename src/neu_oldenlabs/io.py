"""Read one Oldenlabs cage export into a tidy long table.

The export is wide: a two-row header where row 1 repeats the animal tag once per
measure block and row 2 names the measure, then one row per time bin. Each bin's
own duration (`Datetime to` minus `Datetime from`) is carried on every row rather
than assumed to be a fixed ten minutes -- in the real cage-58616 export every bin
happens to span exactly 600 s, but a future export need not -- so every
downstream aggregation weights by the bin's actual duration as a safeguard, not
as a response to jitter observed here. That span is a different quantity from
the gap between consecutive bins' start times, which the real export does vary
by a minute here and there (see `qc.bin_regularity` and `qc.recording_breaks`).

Measure names are mapped through an explicit dictionary. An unrecognised name
raises: a vendor software update that renames a column must surface as a crash,
not as a quietly shorter table.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from . import __version__

MEASURES = {
    "Aggression time (%):": "aggression_time_pct",
    "Number of agression events (1/0):": "aggression_events",
    "Average movement acceleration (cm/s):": "accel_cm_s",
    "Average movement angular velocity (cm/s):": "angular_velocity_cm_s",
    "Average overall speed (cm/s):": "speed_overall_cm_s",
    "Average speed when awake (cm/s):": "speed_awake_cm_s",
    "Max movement speed (cm/s):": "speed_max_cm_s",
    "Total distance traveled (cm):": "distance_cm",
    "Cage center occupancy (%):": "occupancy_center_pct",
    "Cage front half occupancy (%):": "occupancy_front_half_pct",
    "Time spent in cage corners (%):": "occupancy_corners_pct",
    "Cage in rack (1/0):": "in_rack",
    "Climbing events:": "climbing_events",
    "Time spent climbing (%):": "time_climbing_pct",
    "Rearing events:": "rearing_events",
    "Time spent rearing (%):": "time_rearing_pct",
    "Enrichment tube interaction time (%):": "enrichment_tube_pct",
    "Sleep bouts:": "sleep_bouts",
    "Sleep time - spent inactive (%):": "inactive_pct",
    "Social Distance (cm):": "social_distance_cm",
    "Time spent eating (%):": "time_eating_pct",
    "Time spent drinking water (%):": "time_drinking_pct",
}

MEASURE_KIND = {
    "aggression_time_pct": "rate",
    "aggression_events": "count",
    "accel_cm_s": "mean",
    "angular_velocity_cm_s": "mean",
    "speed_overall_cm_s": "mean",
    "speed_awake_cm_s": "mean",
    "speed_max_cm_s": "max",
    "distance_cm": "count",
    "occupancy_center_pct": "rate",
    "occupancy_front_half_pct": "rate",
    "occupancy_corners_pct": "rate",
    "in_rack": "mask",
    "climbing_events": "count",
    "time_climbing_pct": "rate",
    "rearing_events": "count",
    "time_rearing_pct": "rate",
    "enrichment_tube_pct": "rate",
    "sleep_bouts": "count",
    "inactive_pct": "rate",
    "social_distance_cm": "mean",
    "time_eating_pct": "rate",
    "time_drinking_pct": "rate",
}

PERCENT_MEASURES = tuple(slug for slug in MEASURE_KIND if slug.endswith("_pct"))

COLUMNS = ["cage", "tag", "bin_start", "bin_end", "duration_s", "measure", "value"]


def cage_id_from_filename(path: Path) -> int:
    match = re.match(r"^(\d+)_", Path(path).name)
    if not match:
        raise ValueError(f"{path}: filename does not start with a cage number")
    return int(match.group(1))


def load_cage(path: Path, cage: int) -> pd.DataFrame:
    """Load one cage export. ``cage`` is authoritative; the filename must agree."""
    path = Path(path)
    stamped = cage_id_from_filename(path)
    if stamped != cage:
        raise ValueError(f"{path}: filename says cage {stamped}, configuration says {cage}")

    book = load_workbook(path, read_only=True, data_only=True)
    sheet = book[book.sheetnames[0]]
    rows = sheet.iter_rows(values_only=True)
    tag_row = next(rows)
    name_row = next(rows)

    unknown = sorted({str(name) for name in name_row[2:]
                      if name is not None and str(name) not in MEASURES})
    if unknown:
        raise ValueError(f"{path}: unknown measure {unknown}; known names are {sorted(MEASURES)}")

    keys = []
    for index in range(2, len(name_row)):
        name = name_row[index]
        if name is None:
            continue
        keys.append((index, int(tag_row[index]), MEASURES[str(name)]))

    starts, ends, blocks = [], [], {key: [] for key in keys}
    for row in rows:
        if row[0] is None:
            continue
        if row[1] is None:
            raise ValueError(
                f"{path}: bin starting {row[0]} has a blank 'Datetime to' (bin_end); "
                f"a missing bin_end silently drops out of the duration weighting while "
                f"the bin's value stays in the total, inflating count-per-hour rates"
            )
        starts.append(row[0])
        ends.append(row[1])
        for key in keys:
            blocks[key].append(row[key[0]])
    book.close()

    bin_start = pd.to_datetime(pd.Series(starts))
    bin_end = pd.to_datetime(pd.Series(ends))
    duration_s = (bin_end - bin_start).dt.total_seconds()

    frames = []
    for (_, tag, measure), values in blocks.items():
        frames.append(pd.DataFrame({
            "cage": cage, "tag": tag, "bin_start": bin_start, "bin_end": bin_end,
            "duration_s": duration_s, "measure": measure,
            "value": pd.to_numeric(pd.Series(values), errors="coerce"),
        }))
    frame = pd.concat(frames, ignore_index=True)
    frame["cage"] = frame["cage"].astype("int64")
    frame["tag"] = frame["tag"].astype("int64")
    return frame[COLUMNS]


def load_cage_cached(path: Path, cage: int, cache_dir: Path | None = None) -> pd.DataFrame:
    """Parquet-cache the parse, keyed by the export's content hash AND package version.

    The reference export is 29 MB of XML and takes long enough that an
    interactive re-run would otherwise pay for it every time. The package
    version is part of the key: a fix to the parsing logic (e.g. how missing
    values are handled) changes what `load_cage` returns for the SAME input
    bytes, and a stale cache keyed on content alone would silently keep
    serving the old, buggy parse while `provenance.json` stamps the new
    version -- a `version` mismatch that provenance could never catch because
    it never sees the cache.
    """
    if cache_dir is None:
        return load_cage(path, cage)
    from .provenance import sha256

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"cage{cage}_{__version__}_{sha256(path)[:16]}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    frame = load_cage(path, cage)
    frame.to_parquet(cached, index=False)
    return frame
