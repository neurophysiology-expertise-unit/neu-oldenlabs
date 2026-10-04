from __future__ import annotations

import zipfile
from datetime import datetime, timedelta

import pytest

from neu_oldenlabs.tracklets import gaps, mice, read_coordinates

HEADER = ["frame", "centroid_mouse1_x", "centroid_mouse1_y", "centroid_mouse2_x",
          "centroid_mouse2_y", "datetime_from", "datetime_to", "speed_mouse1",
          "speed_mouse2", "binary_time_sleep_mouse1", "binary_time_sleep_mouse2"]


def part(start: datetime, n: int, header=HEADER) -> str:
    rows = [" ".join(header)]
    for i in range(n):
        t = start + timedelta(seconds=i / 10)
        stamp = lambda x: x.strftime("%Y-%m-%d-%H:%M:%S.") + f"{x.microsecond // 1000:03d}"
        vals = {"frame": str(i + 1), "datetime_from": stamp(t),
                "datetime_to": stamp(t + timedelta(seconds=0.1))}
        rows.append(" ".join(vals.get(c, f"{i}.5") for c in header))
    return "\n".join(rows) + "\n"


@pytest.fixture
def split_zip(tmp_path):
    p = tmp_path / "abc12345_20260710T010000.zip"
    with zipfile.ZipFile(p, "w") as z:
        # later part written first: order must come from time, not zip order
        z.writestr("abc12345_coordinates_20260710-113550_20260710-190000.txt",
                   part(datetime(2026, 7, 10, 11, 35, 50), 30))
        z.writestr("abc12345_coordinates_20260710-010000_20260710-111550.txt",
                   part(datetime(2026, 7, 10, 1, 0, 0), 20))
        z.writestr("abc12345_video_20260710-010000_20260710-190000.mp4", b"")
    return p


def test_reads_every_part_in_time_order_and_keeps_the_gap(split_zip):
    df = read_coordinates(split_zip)
    assert len(df) == 50                              # both parts, not the first only
    assert df.t.is_monotonic_increasing
    assert df.part.tolist() == [0] * 20 + [1] * 30
    assert "frame" not in df and "datetime_to" not in df   # frame restarts per part
    g = gaps(df.t)
    assert len(g) == 1
    assert g.first_after[0] == datetime(2026, 7, 10, 11, 35, 50)
    assert mice(df) == [1, 2]


def test_column_subset(split_zip):
    df = read_coordinates(split_zip, columns=["speed_mouse2"])
    assert list(df) == ["t", "part", "speed_mouse2"]
    with pytest.raises(KeyError):
        read_coordinates(split_zip, columns=["speed_mouse9"])


def test_unknown_column_raises(tmp_path):
    p = tmp_path / "x_coordinates_a_b.txt"
    p.write_text(part(datetime(2026, 7, 10), 3, HEADER + ["novel_measure_mouse1"]))
    with pytest.raises(ValueError, match="novel_measure_mouse1"):
        read_coordinates(p)


def test_overlapping_parts_raise(tmp_path):
    p = tmp_path / "x.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("x_coordinates_1.txt", part(datetime(2026, 7, 10), 10))
        z.writestr("x_coordinates_2.txt", part(datetime(2026, 7, 10, 0, 0, 0, 500000), 10))
    with pytest.raises(ValueError, match="overlap"):
        read_coordinates(p)
