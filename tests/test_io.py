from __future__ import annotations

import pytest
from openpyxl import load_workbook

from neu_oldenlabs.io import MEASURE_KIND, MEASURES, cage_id_from_filename, load_cage


def test_measure_map_has_the_twenty_two_exported_measures():
    assert len(MEASURES) == 22
    assert len(set(MEASURES.values())) == 22
    assert set(MEASURE_KIND) == set(MEASURES.values())
    assert MEASURES["Total distance traveled (cm):"] == "distance_cm"
    assert MEASURE_KIND["distance_cm"] == "count"
    assert MEASURE_KIND["in_rack"] == "mask"
    # The exporter's spelling of "agression" is preserved, not corrected.
    assert "Number of agression events (1/0):" in MEASURES


def test_load_cage_returns_tidy_long_rows(make_workbook):
    path = make_workbook(tags=(1, 2, 3), bins=4)
    frame = load_cage(path, cage=58616)
    assert list(frame.columns) == ["cage", "tag", "bin_start", "bin_end",
                                   "duration_s", "measure", "value"]
    assert len(frame) == 4 * 3 * 22
    assert set(frame["tag"]) == {1, 2, 3}
    assert set(frame["measure"]) == set(MEASURES.values())
    assert (frame["cage"] == 58616).all()


def test_bin_duration_follows_the_actual_timestamps(make_workbook):
    path = make_workbook(tags=(1,), bins=3, minutes=(10, 9, 11))
    frame = load_cage(path, cage=58616)
    durations = (frame.drop_duplicates("bin_start")
                 .sort_values("bin_start")["duration_s"].tolist())
    assert durations == [600.0, 540.0, 660.0]


def test_unknown_measure_name_raises(make_workbook, tmp_path):
    path = make_workbook(tags=(1,), bins=2)
    book = load_workbook(path)
    book.active.cell(row=2, column=3, value="Some New Vendor Measure (%):")
    book.save(path)
    with pytest.raises(ValueError, match="unknown measure"):
        load_cage(path, cage=58616)


def test_cage_id_mismatch_with_filename_raises(make_workbook):
    path = make_workbook(name="058616_CA_x_cg1_1_1_1_cage.xlsx", tags=(1,), bins=2)
    with pytest.raises(ValueError, match="filename"):
        load_cage(path, cage=99999)


def test_cage_id_from_filename_strips_the_leading_zero():
    assert cage_id_from_filename("058616_CA_l47P_cg00001_1_1_3423_cage.xlsx") == 58616
