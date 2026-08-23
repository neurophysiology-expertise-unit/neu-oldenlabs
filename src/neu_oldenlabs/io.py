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

Measure SEMANTICS -- what each exported column actually means -- come from the
instrument paper: Delalic, Florea et al. (2025), "Smart Lids for deep
multi-animal phenotyping in standard home cages", Front. Behav. Neurosci.,
doi:10.3389/fnbeh.2025.1696654 (citekey `delali2025`). See `MEASURE_DEFINITIONS`
below for the published definition of each measure. Before this paper was
available, this repository had no independent source for what a measure meant
and simply carried the vendor's column header forward; per AGENTS.md, vendor
semantics are preserved, never reinterpreted -- now that a published source
exists, the documentation is corrected to match it.

Two of the vendor's own export column headers understate their units: both
`accel_cm_s` ("Average movement acceleration (cm/s):") and
`angular_velocity_cm_s` ("Average movement angular velocity (cm/s):") are
labelled "(cm/s)" in the export, but the paper gives acceleration in cm/s^2 and
angular velocity in deg/s. The slugs are NOT renamed here -- they mirror the
vendor's own export columns by design, and renaming them would break that
correspondence -- only the units stated in documentation (this module,
`MEASURE_DEFINITIONS`, the design spec, and the report) are corrected.
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

# The published definition of each exported measure, from the instrument paper:
# Delalic, Florea et al. (2025), "Smart Lids for deep multi-animal phenotyping in
# standard home cages", Front. Behav. Neurosci., doi:10.3389/fnbeh.2025.1696654
# (citekey `delali2025`). Keyed by slug, one entry per `MEASURES` value, plus
# `in_rack`, which is not one of the paper's 21 behavioural measures -- it is the
# rack-presence mask this pipeline uses to exclude bins, and its definition below
# says so rather than inventing a paper source for it. `test_io.py` asserts this
# dict, `MEASURES`, and `MEASURE_KIND` cannot drift apart (same key set).
MEASURE_DEFINITIONS: dict[str, str] = {
    "accel_cm_s": (
        "Acceleration (cm/s^2): average linear acceleration while the animal is "
        "moving; excludes quiescent/sleep periods. The vendor's own export column "
        "header says \"(cm/s)\" -- that is a labelling error in the export, not "
        "the paper's unit; the true unit is cm/s^2."
    ),
    "angular_velocity_cm_s": (
        "Angular motion (deg/s): average movement angular velocity while moving. "
        "The vendor's own export column header says \"(cm/s)\" -- that is a "
        "labelling error in the export, not the paper's unit; the true unit is "
        "deg/s."
    ),
    "speed_overall_cm_s": (
        "Average movement speed (cm/s): mean linear speed over the time bin, "
        "including inactivity."
    ),
    "speed_awake_cm_s": (
        "Average awake movement speed (cm/s): mean linear speed over the time "
        "bin, excluding inactivity."
    ),
    "speed_max_cm_s": (
        "Max movement speed (cm/s): the mean of the top 1% of frame-wise "
        "movement speeds during active periods, not a raw single-frame maximum -- "
        "this reduces noise from single-frame spikes and makes the measure "
        "outlier-robust by construction."
    ),
    "distance_cm": (
        "Distance traveled (cm): total travel path length over the observation "
        "period."
    ),
    "aggression_events": (
        "Aggression events (count): individual bouts of chasing, mounting, "
        "fighting, or a combination, involving two or more animals; bouts last "
        "20 s to 5 min and the whole bout counts as one event."
    ),
    "aggression_time_pct": (
        "Aggression time (%): percentage of time engaged in aggression."
    ),
    "social_distance_cm": (
        "Social distance (cm): average distance of the mouse's centroid to the "
        "nearest mouse's centroid. Does not consider directionality -- a low "
        "value does not imply interaction; its purpose is to identify social "
        "distancing or withdrawal."
    ),
    "occupancy_center_pct": (
        "Cage center time (%): time in the centre zone, defined as >=2 cm from "
        "all cage walls."
    ),
    "occupancy_front_half_pct": (
        "Cage front time (%): time in the front half of the cage."
    ),
    "occupancy_corners_pct": (
        "Cage corner time (%): time in the approximately 9 cm^2 area of each of "
        "the 4 cage corners."
    ),
    "enrichment_tube_pct": (
        "Enrichment tube interaction time (%): time on, in, or immediately "
        "around the enrichment tube."
    ),
    "time_climbing_pct": (
        "Climb time (%): time climbing on the food hopper metal grid, defined as "
        "both feet off the floor and reaching the upper portion of the hopper."
    ),
    "climbing_events": (
        "Climbing events (count): discrete climbing bouts on the food hopper "
        "metal grid, as defined for climb time."
    ),
    "rearing_events": (
        "Rearing events (count): rears against one of the three side walls; "
        "excludes rears against the food hopper, the red (enrichment) tube, and "
        "rears in the cage centre. The authors note that centre rears and hopper "
        "rears were not separately annotated or detected."
    ),
    "time_rearing_pct": (
        "Time spent rearing (%): percentage of time rearing against one of the "
        "three side walls, under the same exclusions as rearing events (excludes "
        "the food hopper, the red tube, and the cage centre)."
    ),
    "time_drinking_pct": (
        "Drinking time (%): time interacting with the bottle nozzle or "
        "autowater spout. The authors state this reflects behavioural "
        "engagement, not actual intake."
    ),
    "time_eating_pct": (
        "Eating time (%): time interacting with food at the hopper, including "
        "\"grinding\" (chewing without ingestion). The authors state this "
        "reflects behavioural engagement, not actual intake."
    ),
    "sleep_bouts": (
        "Sleep bouts (count): discrete inactivity bouts lasting >=2 minutes; the "
        "2-minute threshold follows Pack et al. (2007)."
    ),
    "inactive_pct": (
        "Sleep time (%): percentage of recording time classified as sleep, i.e. "
        "continuous inactivity lasting >=2 minutes -- a sleep proxy with that "
        "threshold, not a measure of generic immobility."
    ),
    "in_rack": (
        "Cage in rack (1/0): NOT one of the paper's 21 behavioural measures -- "
        "this is the rack-presence mask (Cage in rack == 0 marks a bin the cage "
        "was pulled from the rack, e.g. for husbandry) that this pipeline uses to "
        "exclude bins from every metric before aggregation."
    ),
}

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
