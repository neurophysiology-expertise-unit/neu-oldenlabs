"""The study configuration: which cages, which files, which parameters.

Everything the analysis needs to know that is not in the data lives here, in one
explicit file. Nothing is inferred from a filename, and a cage whose export has
not arrived is declared with a null export rather than omitted, so a partial study
is visible in the configuration instead of being invisible in the output.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REQUIRED = ("name", "animals_csv", "cages", "habituation_days",
            "lights_on", "lights_off", "min_days", "primary_measures")


@dataclass(frozen=True)
class CageSpec:
    cage: int
    export: Path | None


@dataclass(frozen=True)
class StudyConfig:
    name: str
    path: Path
    animals_csv: Path
    cages: tuple[CageSpec, ...]
    window_start: str | None
    window_end: str | None
    habituation_days: int
    lights_on: int
    lights_off: int
    min_days: int
    primary_measures: tuple[str, ...]
    scan_measures: tuple[str, ...]

    @property
    def available_cages(self) -> list[CageSpec]:
        return [spec for spec in self.cages if spec.export is not None]

    @property
    def missing_cages(self) -> list[int]:
        return [spec.cage for spec in self.cages if spec.export is None]


def load_study(path: Path) -> StudyConfig:
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    missing = [key for key in REQUIRED if key not in payload]
    if missing:
        raise ValueError(f"{path}: missing required key(s) {missing}")
    root = path.parent

    def resolve(value):
        return (root / value).resolve() if value is not None else None

    animals_csv = resolve(payload["animals_csv"])
    if not animals_csv.exists():
        raise FileNotFoundError(f"{path}: animals_csv does not exist: {animals_csv}")

    cages = []
    for entry in payload["cages"]:
        export = resolve(entry.get("export"))
        if export is not None and not export.exists():
            raise FileNotFoundError(
                f"{path}: export for cage {entry['cage']} does not exist: {export}")
        cages.append(CageSpec(cage=int(entry["cage"]), export=export))

    return StudyConfig(
        name=payload["name"], path=path, animals_csv=animals_csv, cages=tuple(cages),
        window_start=payload.get("window_start"), window_end=payload.get("window_end"),
        habituation_days=int(payload["habituation_days"]),
        lights_on=int(payload["lights_on"]), lights_off=int(payload["lights_off"]),
        min_days=int(payload["min_days"]),
        primary_measures=tuple(payload["primary_measures"]),
        scan_measures=tuple(payload.get("scan_measures") or ()),
    )
