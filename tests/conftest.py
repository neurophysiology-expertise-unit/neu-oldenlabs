"""Synthetic Oldenlabs workbooks with the real header geometry.

Tests never read study data. This builds the same two-row header the exporter
writes: row 1 repeats the animal tag once per measure block, row 2 names the
measure, and columns A/B carry the bin bounds.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from openpyxl import Workbook

from neu_oldenlabs.io import MEASURES

SOURCE_NAMES = list(MEASURES)


@pytest.fixture
def make_workbook(tmp_path):
    def build(name="058616_CA_x_cg1_1_1_1_cage.xlsx", tags=(1, 2), bins=6,
              start=datetime(2026, 7, 11, 16, 56), minutes=(10,), values=None,
              measures=SOURCE_NAMES):
        book = Workbook()
        sheet = book.active
        sheet.cell(row=1, column=2, value="Animal ID")
        sheet.cell(row=2, column=1, value="Datetime from")
        sheet.cell(row=2, column=2, value="Datetime to")
        column = 3
        for measure in measures:
            for tag in tags:
                sheet.cell(row=1, column=column, value=tag)
                sheet.cell(row=2, column=column, value=measure)
                column += 1
        clock = start
        for index in range(bins):
            width = minutes[index % len(minutes)]
            end = clock + timedelta(minutes=width)
            row = 3 + index
            sheet.cell(row=row, column=1, value=clock.strftime("%Y-%m-%d %H:%M:%S"))
            sheet.cell(row=row, column=2, value=end.strftime("%Y-%m-%d %H:%M:%S"))
            column = 3
            for measure in measures:
                for tag in tags:
                    if values is None:
                        cell_value = 1.0
                    else:
                        cell_value = values(measure, tag, index)
                    sheet.cell(row=row, column=column, value=cell_value)
                    column += 1
            clock = end
        path = tmp_path / name
        book.save(path)
        return path
    return build


@pytest.fixture
def synthetic_study(tmp_path, make_workbook):
    """Two genotype-mixed synthetic cages plus one whose export has not arrived.

    Values vary with tag and bin so no measure is constant across animals; a
    constant measure is dropped by the scan and would make the pipeline tests
    pass for the wrong reason.
    """
    import json

    from neu_oldenlabs.config import load_study

    def varying(measure, tag, index):
        if measure == "Cage in rack (1/0):":
            return 1.0
        return 1.0 + tag + (index % 7)

    exports = {}
    for cage in (1001, 1002):
        exports[cage] = make_workbook(name=f"{cage}_x_cage.xlsx", tags=(1, 2, 3),
                                      bins=6 * 24 * 5, values=varying)
    animals = tmp_path / "animals.csv"
    rows = ["cage #,animal #,,genotype,tag #,Tag date,Monitoring start"]
    number = 1
    for cage in (1001, 1002):
        for tag, genotype in ((1, "W/w"), (2, "Mut/W"), (3, "Mut/W")):
            prefix = str(cage) if tag == 1 else ""
            rows.append(f"{prefix},{number},L,{genotype},{tag},,")
            number += 1
    animals.write_text("\n".join(rows) + "\n")

    config = tmp_path / "study.json"
    config.write_text(json.dumps({
        "name": "synthetic", "animals_csv": str(animals),
        "cages": [{"cage": 1001, "export": str(exports[1001])},
                  {"cage": 1002, "export": str(exports[1002])},
                  {"cage": 1003, "export": None}],
        "habituation_days": 0, "lights_on": 7, "lights_off": 19, "min_days": 1,
        "primary_measures": ["distance_cm_all"], "scan_measures": ["distance_cm_all"],
    }))
    return load_study(config)


@pytest.fixture
def all_wildtype_study(tmp_path, make_workbook):
    """One all-wildtype cage, mirroring the real study's current live shape.

    Cage 58616 is the only export that exists in the real study today, and it
    houses five wildtype animals and no mutants (see the composition table in
    tests/test_groups.py). This is not a hypothetical edge case: it is the
    exact -- and currently only -- path the real pipeline takes, so it earns
    its own fixture rather than being folded into `synthetic_study`, whose
    two cages are both genotype-mixed.
    """
    import json

    from neu_oldenlabs.config import load_study

    def varying(measure, tag, index):
        if measure == "Cage in rack (1/0):":
            return 1.0
        return 1.0 + tag + (index % 7)

    export = make_workbook(name="58616_x_cage.xlsx", tags=(1, 2, 3, 4, 5),
                           bins=6 * 24 * 5, values=varying)
    animals = tmp_path / "animals.csv"
    rows = ["cage #,animal #,,genotype,tag #,Tag date,Monitoring start"]
    for index, tag in enumerate((1, 2, 3, 4, 5), start=1):
        prefix = "58616" if tag == 1 else ""
        rows.append(f"{prefix},{index},L,W/w,{tag},,")
    animals.write_text("\n".join(rows) + "\n")

    config = tmp_path / "study.json"
    config.write_text(json.dumps({
        "name": "all_wildtype", "animals_csv": str(animals),
        "cages": [{"cage": 58616, "export": str(export)}],
        "habituation_days": 0, "lights_on": 7, "lights_off": 19, "min_days": 1,
        "primary_measures": ["distance_cm_all"], "scan_measures": ["distance_cm_all"],
    }))
    return load_study(config)
