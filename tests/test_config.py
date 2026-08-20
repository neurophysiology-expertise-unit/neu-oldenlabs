from __future__ import annotations

import json

import pytest

from neu_oldenlabs.config import load_study


def write_config(tmp_path, **overrides):
    export = tmp_path / "058616_x_cage.xlsx"
    export.write_text("")
    animals = tmp_path / "animals.csv"
    animals.write_text("")
    payload = {"name": "study", "animals_csv": "animals.csv",
               "cages": [{"cage": 58616, "export": "058616_x_cage.xlsx"},
                         {"cage": 58623, "export": None}],
               "habituation_days": 1, "lights_on": 7, "lights_off": 19,
               "min_days": 5, "primary_measures": ["distance_cm_all"], "scan_measures": []}
    payload.update(overrides)
    path = tmp_path / "study.json"
    path.write_text(json.dumps(payload))
    return path


def test_relative_paths_resolve_against_the_config_file(tmp_path):
    study = load_study(write_config(tmp_path))
    assert study.animals_csv == (tmp_path / "animals.csv").resolve()
    assert study.cages[0].export == (tmp_path / "058616_x_cage.xlsx").resolve()


def test_cages_without_an_export_are_reported_as_missing(tmp_path):
    study = load_study(write_config(tmp_path))
    assert [c.cage for c in study.available_cages] == [58616]
    assert study.missing_cages == [58623]


def test_a_declared_export_that_does_not_exist_raises(tmp_path):
    path = write_config(tmp_path)
    (tmp_path / "058616_x_cage.xlsx").unlink()
    with pytest.raises(FileNotFoundError, match="058616"):
        load_study(path)


def test_a_missing_required_key_raises(tmp_path):
    path = write_config(tmp_path)
    payload = json.loads(path.read_text())
    del payload["lights_on"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="lights_on"):
        load_study(path)
