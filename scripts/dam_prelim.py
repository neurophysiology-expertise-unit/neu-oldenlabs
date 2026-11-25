#!/usr/bin/env python3
"""Preliminary ELS vs CTL dam comparison from daily tracklet exports (one dam per cage).

Hourly per dam: distance (vendor per-frame `dist`, the measure that reproduces the
binned workbook), sleep, corner (nest) and food/water time, and the fraction of frames
the animal is tracked. Pups share the cage and are untagged, so every measure is the
tracker's slot 1, not a verified dam identity: treat as preliminary.

    PYTHONPATH=src python scripts/dam_prelim.py <tracklets_daily dir> <groups.csv> <out dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neu_oldenlabs.tracklets import mice, read_coordinates

COLS = {"distance_cm": "dist", "sleep_pct": "binary_time_sleep", "corner_pct": "binary_time_in_corner_of_cage",
        "food_pct": "binary_food_activity", "water_pct": "binary_water_activity"}
GROUP_COLOUR = {"CTL": "#0072B2", "ELS": "#E69F00"}          # Okabe-Ito


def hourly(path: Path) -> pd.DataFrame:
    df = read_coordinates(path)
    m = mice(df)[0]                                            # one dam per cage: slot 1
    h = df.t.dt.floor("h")
    out = pd.DataFrame({"tracked_pct": df[f"centroid_mouse{m}_x"].notna().groupby(h).mean() * 100,
                        "frames": df.groupby(h).size()})
    for name, col in COLS.items():
        s = df[f"{col}_mouse{m}"]
        out[name] = s.groupby(h).sum() if name == "distance_cm" else s.groupby(h).mean() * 100
    return out.rename_axis("hour").reset_index().assign(device_uid=path.name[:8])


def main(src: Path, groups: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    grp = pd.read_csv(groups).set_index("device_uid").group
    hrs = pd.concat([hourly(p) for p in sorted(src.glob("*.zip"))], ignore_index=True)
    hrs["group"] = hrs.device_uid.map(grp)
    n0 = len(hrs)
    # full hours with the dam tracked most of the time: recording began 2026-11-12 ~16:00, and
    # an export covers the whole day with untracked rows before that
    hrs = hrs[(hrs.frames >= 0.9 * 36000) & (hrs.tracked_pct >= 50)]
    print(f"kept {len(hrs)}/{n0} dam-hours (full hour, tracked >= 50%)")
    hrs.to_csv(out / "hourly_by_dam.csv", index=False)

    # daily means per dam, with a presence flag: a cage with no real animal shows ~no distance
    hrs["date"] = hrs.hour.dt.date
    day = hrs.groupby(["device_uid", "group", "date"]).agg(
        distance_cm=("distance_cm", "sum"), sleep_pct=("sleep_pct", "mean"), corner_pct=("corner_pct", "mean"),
        food_pct=("food_pct", "mean"), water_pct=("water_pct", "mean"), tracked_pct=("tracked_pct", "mean"),
        hours=("hour", "size")).reset_index()
    day.to_csv(out / "daily_by_dam.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for ax, (meas, label) in zip(axes.flat, [("distance_cm", "distance (m/day)"), ("sleep_pct", "sleep (% of time)"),
                                             ("corner_pct", "in corner / nest (% of time)"), ("food_pct", "eating (% of time)")]):
        for uid, d in day.groupby("device_uid"):
            y = d[meas] / (100 if meas == "distance_cm" else 1)
            ax.plot(pd.to_datetime(d.date), y, color=GROUP_COLOUR[d.group.iloc[0]], lw=1, alpha=0.45)
        for g, d in day.groupby("group"):
            mu = d.groupby("date")[meas].mean() / (100 if meas == "distance_cm" else 1)
            ax.plot(pd.to_datetime(mu.index), mu.values, color=GROUP_COLOUR[g], lw=2.5, label=f"{g} (n={d.device_uid.nunique()})")
        ax.set_ylabel(label); ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.autofmt_xdate(); fig.suptitle("Dams, ELS vs CTL: daily values (thin = one dam, thick = group mean). PRELIMINARY", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "daily_ELS_vs_CTL.png", dpi=200); plt.close(fig)

    hrs["hod"] = hrs.hour.dt.hour
    prof = hrs.groupby(["group", "device_uid", "hod"]).distance_cm.mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.axvspan(19, 24, color="k", alpha=0.06); ax.axvspan(0, 7, color="k", alpha=0.06)
    for g, d in prof.groupby("group"):
        p = d.pivot(index="hod", columns="device_uid", values="distance_cm")
        mu, se = p.mean(axis=1), p.std(axis=1, ddof=1) / np.sqrt(p.shape[1])
        ax.plot(mu.index, mu, color=GROUP_COLOUR[g], lw=2, label=f"{g} (n={p.shape[1]})")
        ax.fill_between(mu.index, mu - se, mu + se, color=GROUP_COLOUR[g], alpha=0.2)
    ax.set_xlabel("hour of day (grey = lights off)"); ax.set_ylabel("distance (cm/h)"); ax.set_xticks([0, 7, 12, 19, 23])
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("24-h activity profile, mean ± SEM across dams. PRELIMINARY", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "profile24h_ELS_vs_CTL.png", dpi=200); plt.close(fig)
    prof.to_csv(out / "profile24h_by_dam.csv", index=False)
    print(day.groupby(["group", "device_uid"])[["distance_cm", "sleep_pct", "corner_pct", "tracked_pct", "hours"]].mean().round(1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
