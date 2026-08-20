from __future__ import annotations

import json

from neu_oldenlabs.provenance import sha256, write_provenance


def test_write_provenance_records_hashes_version_and_parameters(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("hello")
    output = tmp_path / "out"
    output.mkdir()

    write_provenance(output, [source], {"command": "qc", "habituation_days": 1})

    payload = json.loads((output / "provenance.json").read_text())
    assert payload["version"] == "0.1.0"
    assert payload["seed"] == 20260820
    assert payload["parameters"]["command"] == "qc"
    assert payload["inputs"][str(source.resolve())] == sha256(source)
    assert payload["created_utc"].endswith("+00:00")


def test_write_provenance_fails_on_missing_input(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    try:
        write_provenance(output, [tmp_path / "absent.txt"], {})
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for a missing input")
