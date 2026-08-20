# neu-oldenlabs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `neu-oldenlabs`, a Python package that ingests Oldenlabs home-cage
monitoring Excel exports, quality-controls them, computes a per-animal behavioural
profile, and compares mutant against wildtype animals with a cage-stratified exact
permutation test, emitting figures, tables, and a rendered report.

**Architecture:** A linear pipeline of single-purpose modules with table-in/table-out
interfaces: `io` (xlsx → tidy long) → `qc` (masks and reports) → `metrics` (duration-weighted
daily aggregation) + `circadian` (cosinor, IS/IV/RA) → `profile` (one row per animal) →
`groups` (statistics) → `plots` + `report`. `animals` and `config` supply identity and
run parameters. Nothing downstream of `io` touches a file path; nothing upstream of
`groups` knows about genotype groups.

**Tech Stack:** Python 3.11, pandas ≥ 2, numpy, scipy, statsmodels, matplotlib, openpyxl,
pyarrow, pytest. Conda env `neu-oldenlabs`. Reports rendered with pandoc.

**Spec:** `docs/superpowers/specs/2026-08-20-neu-oldenlabs-design.md` — read it before
starting. Executors read both documents.

**Two refinements to the spec's module list**, made while decomposing: profile assembly
gets its own `profile.py` (a different responsibility from daily aggregation, and it is
where `neu-intellicage` puts it), and `study.json` parsing gets its own `config.py`
(so no other module parses JSON). Everything else follows the spec's structure exactly.

## Global Constraints

- Python ≥ 3.11; pandas ≥ 2.2; matplotlib ≥ 3.7; numpy ≥ 1.24; scipy ≥ 1.11; statsmodels ≥ 0.14.
- **Never commit data.** No Oldenlabs export, no `DaCruz_Epilepsy.csv`, no generated
  animal-level table, no ear-tag identity may enter this repository. `.gitignore` blocks
  `outputs/`, `data/`, `*.xlsx`, `*.csv`, `*.png`, `*.pdf`, `*.parquet`.
- **Tests use synthetic fixtures only** — never a slice of real animal data.
- **Duration-weighted aggregation everywhere.** Bins jitter between 9 and 11 minutes;
  weights are always the actual `bin_end - bin_start`.
- **An unrecognised measure name is a hard error**, never a silent drop.
- **Genotype map is exactly** `{"W/w": "wt", "W/W": "wt", "Mut/W": "mut"}`; any other
  string raises. `genotype_raw` is retained on every animal row.
- **Light schedule:** 12:12, lights on 07:00, lights off 19:00. Dark phase is the active phase.
- **RNG seed** is fixed at `20260820` and recorded in `provenance.json`.
- Every plotted quantity is also written as a CSV beside the figure.
- Every CLI command writes `provenance.json` (input paths + sha256, config hash, package
  version, parameters, seed).
- Line length 100. `from __future__ import annotations` at the top of every module.

---

### Task 1: Repository scaffolding, environment, and provenance

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `AGENTS.md`, `LICENSE`
- Create: `src/neu_oldenlabs/__init__.py`, `src/neu_oldenlabs/provenance.py`
- Create: `tests/test_provenance.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing.
- Produces: `neu_oldenlabs.__version__: str` (`"0.1.0"`);
  `provenance.write_provenance(output: Path, inputs: list[Path], parameters: dict) -> None`
  writing `output/provenance.json`;
  `provenance.sha256(path: Path) -> str`.

- [ ] **Step 1: Create the conda environment**

```bash
conda create -y -n neu-oldenlabs python=3.11
conda run -n neu-oldenlabs python -m pip install \
  "pandas>=2.2" "numpy>=1.24" "scipy>=1.11" "statsmodels>=0.14" \
  "matplotlib>=3.7" "openpyxl>=3.1" "pyarrow>=14" "pytest>=7"
conda run -n neu-oldenlabs python -c "import pandas, numpy, scipy, statsmodels, matplotlib, openpyxl, pyarrow; print('deps ok')"
```

Expected: `deps ok`. Every later command in this plan runs under
`conda run -n neu-oldenlabs`.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "neu-oldenlabs"
version = "0.1.0"
description = "Reproducible loading, quality control, and analysis of Oldenlabs home-cage monitoring data"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [{name = "Neurophysiology Expertise Unit"}]
dependencies = [
  "numpy>=1.24",
  "pandas>=2.2",
  "scipy>=1.11",
  "statsmodels>=0.14",
  "matplotlib>=3.7",
  "openpyxl>=3.1",
  "pyarrow>=14",
]

[project.optional-dependencies]
test = ["pytest>=7"]

[project.scripts]
neu-oldenlabs = "neu_oldenlabs.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Write `.gitignore`**

```
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
.coverage
build/
dist/
outputs/
data/
cache/
*.xlsx
*.csv
*.parquet
*.png
*.pdf
```

- [ ] **Step 4: Write `AGENTS.md`**

```markdown
# AGENTS.md — neu-oldenlabs

- This repository is reusable analysis code, not a study or manuscript repo.
- Never commit Oldenlabs exports, the animal/genotype CSV, ear-tag identities, or
  generated animal-level tables. Data are supplied by path at runtime.
- Preserve Oldenlabs source semantics. Do not silently reinterpret or rename an
  exported measure. An unknown measure name is an error, not something to skip.
- Bins are nominally 10 minutes but jitter between 9 and 11. Every aggregation is
  weighted by the actual bin duration. A plain mean over bins is a bug.
- Bins where `Cage in rack == 0` are excluded from every metric.
- Every plotted quantity must also be written as a machine-readable table.
- Analysis runs record input hashes and parameters in `provenance.json`.
- The genotype comparison is cage-stratified: labels permute only within a cage.
  Never report a genotype comparison computed from cages whose exports are missing;
  report which cages are absent.
- Crash loudly on real errors. A missing cage export or an all-wildtype cage is an
  expected RESULT, logged, not an error.
```

- [ ] **Step 5: Write `src/neu_oldenlabs/__init__.py`**

```python
"""Reproducible analysis of Oldenlabs home-cage monitoring exports."""
from __future__ import annotations

__version__ = "0.1.0"
RNG_SEED = 20260820
```

- [ ] **Step 6: Write the failing test `tests/test_provenance.py`**

```python
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
```

- [ ] **Step 7: Run the test and confirm it fails**

Run: `conda run -n neu-oldenlabs python -m pytest tests/test_provenance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neu_oldenlabs.provenance'`

- [ ] **Step 8: Write `src/neu_oldenlabs/provenance.py`**

```python
"""Stamp every analysis output with what produced it.

Without this, a figure and the export it came from can drift apart with nothing
to show which version of the data a delivered report contains.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import RNG_SEED, __version__


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(output: Path, inputs: list[Path], parameters: dict) -> None:
    """Record inputs and parameters for one analysis step.

    A missing input is an error rather than an omitted hash: a provenance file
    that silently lists fewer inputs than the run used is worse than none.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for item in inputs:
        item = Path(item)
        if not item.exists():
            raise FileNotFoundError(f"provenance input does not exist: {item}")
        hashes[str(item.resolve())] = sha256(item)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "version": __version__,
        "seed": RNG_SEED,
        "inputs": hashes,
        "parameters": parameters,
    }
    (output / "provenance.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
```

- [ ] **Step 9: Install the package and run the tests**

Run:
```bash
conda run -n neu-oldenlabs python -m pip install -e '.[test]'
conda run -n neu-oldenlabs python -m pytest tests/ -v
```
Expected: 2 passed.

- [ ] **Step 10: Update `README.md`**

Replace its contents with a description of the package, an install block
(`conda create` + `pip install -e '.[test]'`), and a "No data in this repository"
note. Leave the command list to be filled in by Task 11.

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "feat: scaffold neu-oldenlabs package with provenance"
```

---

### Task 2: Animal table and genotype normalisation

**Files:**
- Create: `src/neu_oldenlabs/animals.py`
- Create: `tests/test_animals.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `animals.GENOTYPE_MAP: dict[str, str]` = `{"W/w": "wt", "W/W": "wt", "Mut/W": "mut"}`
  - `animals.load_animals(csv_path: Path) -> pd.DataFrame` with columns
    `cage:int64, animal_number:int64, ear_mark:str, genotype_raw:str, genotype:str, tag:int64`
  - `animals.check_join(table: pd.DataFrame, cage: int, tags: set[int]) -> None` — raises
    `ValueError` when the tags present in an export and the tags listed for that cage
    are not identical sets.

- [ ] **Step 1: Write the failing test `tests/test_animals.py`**

```python
from __future__ import annotations

import pytest

from neu_oldenlabs.animals import check_join, load_animals

CSV = """cage #,animal #,,genotype,tag #,Tag date,Monitoring start
58616,659,L,W/w,2,260706,
,660,R,W/w,5,,
58623,668,L,W/w,3,,
,669,R,Mut/W,1,,
,672,RR,W/W,4,,
"""


def write_csv(tmp_path, text=CSV):
    path = tmp_path / "animals.csv"
    path.write_text(text)
    return path


def test_cage_is_forward_filled_and_types_are_integers(tmp_path):
    table = load_animals(write_csv(tmp_path))
    assert list(table["cage"]) == [58616, 58616, 58623, 58623, 58623]
    assert table["tag"].tolist() == [2, 5, 3, 1, 4]
    assert table["animal_number"].dtype.kind == "i"


def test_upper_and_lower_case_wildtype_both_normalise_to_wt(tmp_path):
    table = load_animals(write_csv(tmp_path))
    by_number = table.set_index("animal_number")
    assert by_number.loc[672, "genotype"] == "wt"
    assert by_number.loc[672, "genotype_raw"] == "W/W"
    assert by_number.loc[659, "genotype"] == "wt"
    assert by_number.loc[669, "genotype"] == "mut"


def test_unknown_genotype_string_raises(tmp_path):
    bad = CSV.replace("Mut/W", "mut/w ?")
    with pytest.raises(ValueError, match="unknown genotype"):
        load_animals(write_csv(tmp_path, bad))


def test_duplicate_tag_within_a_cage_raises(tmp_path):
    bad = CSV.replace(",660,R,W/w,5,,", ",660,R,W/w,2,,")
    with pytest.raises(ValueError, match="duplicate tag"):
        load_animals(write_csv(tmp_path, bad))


def test_check_join_requires_identical_tag_sets(tmp_path):
    table = load_animals(write_csv(tmp_path))
    check_join(table, 58616, {2, 5})
    with pytest.raises(ValueError, match="only in export"):
        check_join(table, 58616, {2, 5, 9})
    with pytest.raises(ValueError, match="only in animal table"):
        check_join(table, 58616, {2})
```

- [ ] **Step 2: Run and confirm failure**

Run: `conda run -n neu-oldenlabs python -m pytest tests/test_animals.py -v`
Expected: FAIL — no module `neu_oldenlabs.animals`.

- [ ] **Step 3: Write `src/neu_oldenlabs/animals.py`**

```python
"""Animal identity and genotype, read from the study's CSV.

The CSV writes the cage number only on the first row of each block, so the cage
column is forward-filled. Genotype is normalised through one visible map and the
raw string is kept beside it, because a normalisation nobody can audit is a
normalisation nobody should trust.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

GENOTYPE_MAP = {"W/w": "wt", "W/W": "wt", "Mut/W": "mut"}

COLUMNS = ["cage", "animal_number", "ear_mark", "genotype_raw", "tag"]


def load_animals(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if len(raw.columns) < 5:
        raise ValueError(f"{csv_path}: expected at least 5 columns, got {len(raw.columns)}")
    table = raw.iloc[:, :5].copy()
    table.columns = COLUMNS
    table = table[table["animal_number"].str.strip() != ""]
    for column in COLUMNS:
        table[column] = table[column].str.strip()
    table["cage"] = table["cage"].replace("", pd.NA).ffill()
    if table["cage"].isna().any():
        raise ValueError(f"{csv_path}: rows before the first cage number")
    table["cage"] = table["cage"].astype("int64")
    table["animal_number"] = table["animal_number"].astype("int64")
    table["tag"] = table["tag"].astype("int64")

    unknown = sorted(set(table["genotype_raw"]) - set(GENOTYPE_MAP))
    if unknown:
        raise ValueError(f"{csv_path}: unknown genotype {unknown}; allowed {sorted(GENOTYPE_MAP)}")
    table["genotype"] = table["genotype_raw"].map(GENOTYPE_MAP)

    duplicated = table.duplicated(subset=["cage", "tag"], keep=False)
    if duplicated.any():
        offenders = table.loc[duplicated, ["cage", "tag"]].to_dict("records")
        raise ValueError(f"{csv_path}: duplicate tag within a cage: {offenders}")
    return table[["cage", "animal_number", "ear_mark", "genotype_raw", "genotype", "tag"]]


def check_join(table: pd.DataFrame, cage: int, tags: set[int]) -> None:
    """Fail unless the export's tags and the animal table's tags match exactly.

    A partial join would silently analyse fewer animals than the study contains,
    which is the one failure mode that never announces itself in a figure.
    """
    listed = set(table.loc[table["cage"] == cage, "tag"].tolist())
    if not listed:
        raise ValueError(f"cage {cage} has no animals in the animal table")
    extra = sorted(tags - listed)
    if extra:
        raise ValueError(f"cage {cage}: tags only in export, not in animal table: {extra}")
    absent = sorted(listed - tags)
    if absent:
        raise ValueError(f"cage {cage}: tags only in animal table, not in export: {absent}")
```

- [ ] **Step 4: Run the tests**

Run: `conda run -n neu-oldenlabs python -m pytest tests/test_animals.py -v`
Expected: 5 passed.

- [ ] **Step 5: Verify against the real CSV (read-only, output not committed)**

```bash
conda run -n neu-oldenlabs python -c "
from neu_oldenlabs.animals import load_animals
t = load_animals('/mnt/sysfs01/users/cagatay/external/cruz/oldenlabs/260820_study_2/DaCruz_Epilepsy.csv')
print(len(t), t.groupby('genotype').size().to_dict())
print(t.groupby(['cage','genotype']).size().unstack(fill_value=0))
"
```
Expected: `25 {'mut': 10, 'wt': 15}` and cage 58616 showing `mut=0, wt=5`.

- [ ] **Step 6: Commit**

```bash
git add src/neu_oldenlabs/animals.py tests/test_animals.py
git commit -m "feat: load animal table with audited genotype normalisation"
```

---

### Task 3: Excel ingest to tidy long format

**Files:**
- Create: `src/neu_oldenlabs/io.py`
- Create: `tests/conftest.py`
- Create: `tests/test_io.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `io.MEASURES: dict[str, str]` — source measure name → slug (22 entries)
  - `io.MEASURE_KIND: dict[str, str]` — slug → one of `"rate" | "mean" | "max" | "count" | "mask"`
  - `io.load_cage(path: Path, cage: int) -> pd.DataFrame` with columns
    `cage:int64, tag:int64, bin_start:datetime64[ns], bin_end:datetime64[ns],
    duration_s:float64, measure:str, value:float64`
  - `io.cage_id_from_filename(path: Path) -> int`
  - `io.load_cage_cached(path: Path, cage: int, cache_dir: Path | None) -> pd.DataFrame`
  - `tests/conftest.py::make_workbook(path, tags, bins, values=None) -> Path` fixture helper
    reproducing the real header geometry.

- [ ] **Step 1: Write the fixture helper `tests/conftest.py`**

This file gains two fixtures. `make_workbook` is used from this task onward;
`synthetic_study` is written now, in the same file, but is not used until Task 10 —
its imports are inside the function body precisely so it costs nothing until then.

```python
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
```

- [ ] **Step 2: Write the failing test `tests/test_io.py`**

```python
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
```

- [ ] **Step 3: Run and confirm failure**

Run: `conda run -n neu-oldenlabs python -m pytest tests/test_io.py -v`
Expected: FAIL — no module `neu_oldenlabs.io`.

- [ ] **Step 4: Write `src/neu_oldenlabs/io.py`**

```python
"""Read one Oldenlabs cage export into a tidy long table.

The export is wide: a two-row header where row 1 repeats the animal tag once per
measure block and row 2 names the measure, then one row per time bin. Bins are
nominally ten minutes and observed to vary between nine and eleven, so the bin
duration is carried on every row and every downstream aggregation weights by it.

Measure names are mapped through an explicit dictionary. An unrecognised name
raises: a vendor software update that renames a column must surface as a crash,
not as a quietly shorter table.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

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
    """Parquet-cache the parse, keyed by the export's content hash.

    The reference export is 29 MB of XML and takes long enough that an
    interactive re-run would otherwise pay for it every time.
    """
    if cache_dir is None:
        return load_cage(path, cage)
    from .provenance import sha256

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"cage{cage}_{sha256(path)[:16]}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    frame = load_cage(path, cage)
    frame.to_parquet(cached, index=False)
    return frame
```

- [ ] **Step 5: Run the tests**

Run: `conda run -n neu-oldenlabs python -m pytest tests/test_io.py -v`
Expected: 6 passed.

- [ ] **Step 6: Verify against the real export**

```bash
conda run -n neu-oldenlabs python -c "
from neu_oldenlabs.io import load_cage
f = load_cage('/mnt/sysfs01/users/cagatay/external/cruz/oldenlabs/260820_study_2/058616_CA_l47P_cg00001_071120261653_082020261112_3423_cage.xlsx', 58616)
print(len(f), f['measure'].nunique(), sorted(f['tag'].unique()))
print(f['bin_start'].min(), f['bin_end'].max())
print(f['duration_s'].value_counts().to_dict())
print('nan values:', int(f['value'].isna().sum()))
"
```
Expected: 629,420 rows, 22 measures, tags `[1,2,3,4,5]`, window
`2026-07-11 16:56` → `2026-08-20 11:08`, durations `{600.0: …, 660.0: 36, 540.0: 4}`,
0 NaN.

- [ ] **Step 7: Commit**

```bash
git add src/neu_oldenlabs/io.py tests/conftest.py tests/test_io.py
git commit -m "feat: ingest Oldenlabs cage exports to tidy long format"
```

---

### Task 4: Quality control

**Files:**
- Create: `src/neu_oldenlabs/qc.py`
- Create: `tests/test_qc.py`

**Interfaces:**
- Consumes: `io.MEASURE_KIND`, `io.PERCENT_MEASURES`, the `load_cage` frame.
- Produces:
  - `qc.valid_bins(long: pd.DataFrame) -> pd.DataFrame` — `cage, tag, bin_start, valid:bool`
  - `qc.apply_rack_mask(long: pd.DataFrame) -> pd.DataFrame` — the long frame with
    invalid bins dropped and the `in_rack` measure removed
  - `qc.rack_summary(long) -> pd.DataFrame` — `cage, tag, bins, bins_out_of_rack, fraction_out`
  - `qc.bin_regularity(long) -> pd.DataFrame` — `cage, duration_s, bins` plus
    `qc.recording_breaks(long, threshold_min=15) -> pd.DataFrame`
  - `qc.coverage(long) -> pd.DataFrame` — `cage, tag, date, valid_bins, valid_hours`
  - `qc.range_violations(long) -> pd.DataFrame` — `cage, tag, measure, rule, n_violations`
  - `qc.qc_tables(long) -> dict[str, pd.DataFrame]` keyed
    `rack, regularity, breaks, coverage, ranges`

- [ ] **Step 1: Write the failing test `tests/test_qc.py`**

```python
from __future__ import annotations

from datetime import datetime

from neu_oldenlabs.io import load_cage
from neu_oldenlabs.qc import (apply_rack_mask, bin_regularity, coverage, range_violations,
                              recording_breaks, rack_summary)


def out_of_rack_in_bin_two(measure, tag, index):
    if measure == "Cage in rack (1/0):":
        return 0.0 if index == 1 else 1.0
    return 5.0


def test_rack_mask_drops_out_of_rack_bins_and_the_mask_measure(make_workbook):
    path = make_workbook(tags=(1, 2), bins=4, values=out_of_rack_in_bin_two)
    long = load_cage(path, 58616)
    masked = apply_rack_mask(long)
    assert "in_rack" not in set(masked["measure"])
    assert masked["bin_start"].nunique() == 3
    assert len(masked) == 3 * 2 * 21


def test_rack_summary_reports_the_excluded_fraction(make_workbook):
    path = make_workbook(tags=(1, 2), bins=4, values=out_of_rack_in_bin_two)
    summary = rack_summary(load_cage(path, 58616))
    assert summary["bins_out_of_rack"].tolist() == [1, 1]
    assert summary["fraction_out"].round(2).tolist() == [0.25, 0.25]


def test_bin_regularity_counts_each_observed_duration(make_workbook):
    path = make_workbook(tags=(1,), bins=4, minutes=(10, 10, 9, 11))
    table = bin_regularity(load_cage(path, 58616)).set_index("duration_s")
    assert table.loc[600.0, "bins"] == 2
    assert table.loc[540.0, "bins"] == 1
    assert table.loc[660.0, "bins"] == 1


def test_recording_break_is_flagged_when_a_gap_exceeds_the_threshold(make_workbook):
    path = make_workbook(tags=(1,), bins=3, minutes=(10, 25, 10))
    breaks = recording_breaks(load_cage(path, 58616), threshold_min=15)
    assert len(breaks) == 1
    assert breaks.iloc[0]["gap_min"] == 25.0


def test_coverage_counts_valid_bins_and_hours_per_animal_day(make_workbook):
    path = make_workbook(tags=(1, 2), bins=6, start=datetime(2026, 7, 11, 23, 40))
    table = coverage(apply_rack_mask(load_cage(path, 58616)))
    assert set(table["date"].astype(str)) == {"2026-07-11", "2026-07-12"}
    assert table["valid_bins"].sum() == 12
    assert round(table["valid_hours"].sum(), 3) == 2.0


def test_percentages_outside_zero_to_hundred_are_reported_not_corrected(make_workbook):
    def over_range(measure, tag, index):
        return 150.0 if measure == "Time spent eating (%):" else 1.0

    path = make_workbook(tags=(1,), bins=2, values=over_range)
    long = load_cage(path, 58616)
    violations = range_violations(long)
    row = violations[violations["measure"] == "time_eating_pct"].iloc[0]
    assert row["n_violations"] == 2
    assert long.loc[long["measure"] == "time_eating_pct", "value"].max() == 150.0
```

- [ ] **Step 2: Run and confirm failure**

Run: `conda run -n neu-oldenlabs python -m pytest tests/test_qc.py -v`
Expected: FAIL — no module `neu_oldenlabs.qc`.

- [ ] **Step 3: Write `src/neu_oldenlabs/qc.py`**

```python
"""Quality control. Reports what is wrong; repairs nothing.

The one thing QC does change downstream is the rack mask: bins recorded while the
cage was out of the rack are not observations of the animal's behaviour, so they
are excluded from every metric rather than averaged into one.
"""
from __future__ import annotations

import pandas as pd

from .io import PERCENT_MEASURES


def valid_bins(long: pd.DataFrame) -> pd.DataFrame:
    rack = long[long["measure"] == "in_rack"]
    if rack.empty:
        raise ValueError("no 'in_rack' measure in this export; cannot build the rack mask")
    mask = rack[["cage", "tag", "bin_start", "value"]].copy()
    mask["valid"] = mask.pop("value") > 0
    return mask


def apply_rack_mask(long: pd.DataFrame) -> pd.DataFrame:
    mask = valid_bins(long)
    merged = long.merge(mask, on=["cage", "tag", "bin_start"], how="left", validate="many_to_one")
    if merged["valid"].isna().any():
        raise ValueError("bins with no rack status; the export is internally inconsistent")
    kept = merged[merged["valid"] & (merged["measure"] != "in_rack")]
    return kept.drop(columns="valid").reset_index(drop=True)


def rack_summary(long: pd.DataFrame) -> pd.DataFrame:
    mask = valid_bins(long)
    grouped = mask.groupby(["cage", "tag"], as_index=False).agg(
        bins=("valid", "size"), bins_out_of_rack=("valid", lambda v: int((~v).sum())))
    grouped["fraction_out"] = grouped["bins_out_of_rack"] / grouped["bins"]
    return grouped


def bin_regularity(long: pd.DataFrame) -> pd.DataFrame:
    bins = long.drop_duplicates(subset=["cage", "bin_start"])
    return (bins.groupby(["cage", "duration_s"], as_index=False)
            .size().rename(columns={"size": "bins"}))


def recording_breaks(long: pd.DataFrame, threshold_min: float = 15.0) -> pd.DataFrame:
    rows = []
    for cage, frame in long.drop_duplicates(subset=["cage", "bin_start"]).groupby("cage"):
        ordered = frame.sort_values("bin_start")
        gap = (ordered["bin_start"].shift(-1) - ordered["bin_start"]).dt.total_seconds() / 60
        flagged = ordered.assign(gap_min=gap)
        flagged = flagged[flagged["gap_min"] > threshold_min]
        for _, row in flagged.iterrows():
            rows.append({"cage": cage, "after_bin": row["bin_start"],
                         "gap_min": float(row["gap_min"])})
    return pd.DataFrame(rows, columns=["cage", "after_bin", "gap_min"])


def coverage(long: pd.DataFrame) -> pd.DataFrame:
    bins = long.drop_duplicates(subset=["cage", "tag", "bin_start"]).copy()
    bins["date"] = bins["bin_start"].dt.date
    table = bins.groupby(["cage", "tag", "date"], as_index=False).agg(
        valid_bins=("bin_start", "size"), valid_seconds=("duration_s", "sum"))
    table["valid_hours"] = table.pop("valid_seconds") / 3600.0
    return table


def range_violations(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    percent = long[long["measure"].isin(PERCENT_MEASURES)]
    bad = percent[(percent["value"] < 0) | (percent["value"] > 100)]
    for (cage, tag, measure), frame in bad.groupby(["cage", "tag", "measure"]):
        rows.append({"cage": cage, "tag": tag, "measure": measure,
                     "rule": "percent outside [0, 100]", "n_violations": len(frame)})
    negative = long[(~long["measure"].isin(PERCENT_MEASURES)) & (long["value"] < 0)]
    for (cage, tag, measure), frame in negative.groupby(["cage", "tag", "measure"]):
        rows.append({"cage": cage, "tag": tag, "measure": measure,
                     "rule": "negative value", "n_violations": len(frame)})
    return pd.DataFrame(rows, columns=["cage", "tag", "measure", "rule", "n_violations"])


def qc_tables(long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    masked = apply_rack_mask(long)
    return {"rack": rack_summary(long), "regularity": bin_regularity(long),
            "breaks": recording_breaks(long), "coverage": coverage(masked),
            "ranges": range_violations(long)}
```

- [ ] **Step 4: Run the tests**

Run: `conda run -n neu-oldenlabs python -m pytest tests/test_qc.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/neu_oldenlabs/qc.py tests/test_qc.py
git commit -m "feat: quality control with rack mask, coverage, and range reporting"
```

---

### Task 5: Duration-weighted daily aggregation

**Files:**
- Create: `src/neu_oldenlabs/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `io.MEASURE_KIND`, the rack-masked long frame.
- Produces:
  - `metrics.add_time_fields(long, lights_on=7, lights_off=19) -> pd.DataFrame` adding
    `date:object, hour:int64, phase:str` where `phase` is `"light"` or `"dark"`
  - `metrics.daily(long, lights_on=7, lights_off=19) -> pd.DataFrame` with columns
    `cage, tag, date, phase, measure, value, valid_hours` where `phase` takes the three
    values `"all"`, `"light"`, `"dark"`
  - `metrics.drop_habituation(daily, days: int) -> pd.DataFrame`
  - `metrics.restrict_window(daily, start, end) -> pd.DataFrame`

Aggregation rule by `MEASURE_KIND`: `rate` and `mean` → duration-weighted mean;
`max` → maximum; `count` → **sum divided by valid hours** (a per-hour rate, so a day
with masked bins is not scored as low activity).

- [ ] **Step 1: Write the failing test `tests/test_metrics.py`**

```python
from __future__ import annotations

from datetime import datetime

import pandas as pd

from neu_oldenlabs.metrics import add_time_fields, daily, drop_habituation


def frame(rows):
    return pd.DataFrame(rows, columns=["cage", "tag", "bin_start", "bin_end",
                                       "duration_s", "measure", "value"])


def test_phase_split_uses_a_seven_to_nineteen_light_schedule():
    long = frame([
        (1, 1, datetime(2026, 7, 11, 6, 50), datetime(2026, 7, 11, 7, 0), 600.0, "distance_cm", 1),
        (1, 1, datetime(2026, 7, 11, 7, 0), datetime(2026, 7, 11, 7, 10), 600.0, "distance_cm", 1),
        (1, 1, datetime(2026, 7, 11, 19, 0), datetime(2026, 7, 11, 19, 10), 600.0, "distance_cm", 1),
    ])
    phases = add_time_fields(long)["phase"].tolist()
    assert phases == ["dark", "light", "dark"]


def test_rate_measures_are_weighted_by_bin_duration_not_averaged_flat():
    # 60 for nine minutes and 0 for eleven: the flat mean is 30.0, the correct
    # duration-weighted mean is 27.0.
    long = frame([
        (1, 1, datetime(2026, 7, 11, 20, 0), datetime(2026, 7, 11, 20, 9), 540.0,
         "inactive_pct", 60.0),
        (1, 1, datetime(2026, 7, 11, 20, 9), datetime(2026, 7, 11, 20, 20), 660.0,
         "inactive_pct", 0.0),
    ])
    table = daily(long)
    value = table[(table["phase"] == "all") & (table["measure"] == "inactive_pct")]["value"]
    assert round(float(value.iloc[0]), 6) == 27.0


def test_counts_become_a_rate_per_valid_hour():
    long = frame([
        (1, 1, datetime(2026, 7, 11, 20, 0), datetime(2026, 7, 11, 20, 30), 1800.0,
         "rearing_events", 6.0),
    ])
    table = daily(long)
    row = table[(table["phase"] == "all") & (table["measure"] == "rearing_events")].iloc[0]
    assert round(float(row["value"]), 6) == 12.0  # 6 events in half an hour
    assert round(float(row["valid_hours"]), 6) == 0.5


def test_count_rate_is_unchanged_when_half_the_bins_are_masked_away():
    both = frame([
        (1, 1, datetime(2026, 7, 11, 20, 0), datetime(2026, 7, 11, 20, 10), 600.0,
         "rearing_events", 2.0),
        (1, 1, datetime(2026, 7, 11, 20, 10), datetime(2026, 7, 11, 20, 20), 600.0,
         "rearing_events", 2.0),
    ])
    one = both.iloc[:1]
    value_both = daily(both).query("phase == 'all'")["value"].iloc[0]
    value_one = daily(one).query("phase == 'all'")["value"].iloc[0]
    assert round(float(value_both), 6) == round(float(value_one), 6) == 12.0


def test_max_measures_take_the_maximum():
    long = frame([
        (1, 1, datetime(2026, 7, 11, 20, 0), datetime(2026, 7, 11, 20, 10), 600.0,
         "speed_max_cm_s", 8.0),
        (1, 1, datetime(2026, 7, 11, 20, 10), datetime(2026, 7, 11, 20, 20), 600.0,
         "speed_max_cm_s", 19.0),
    ])
    table = daily(long).query("phase == 'all' and measure == 'speed_max_cm_s'")
    assert float(table["value"].iloc[0]) == 19.0


def test_habituation_drops_the_first_calendar_days_per_cage():
    table = pd.DataFrame({
        "cage": [1, 1, 1], "tag": [1, 1, 1],
        "date": [pd.Timestamp("2026-07-11").date(), pd.Timestamp("2026-07-12").date(),
                 pd.Timestamp("2026-07-13").date()],
        "phase": ["all"] * 3, "measure": ["distance_cm"] * 3,
        "value": [1.0, 2.0, 3.0], "valid_hours": [1.0, 24.0, 24.0],
    })
    kept = drop_habituation(table, days=1)
    assert [str(d) for d in kept["date"]] == ["2026-07-13"]
```

- [ ] **Step 2: Run and confirm failure**

Run: `conda run -n neu-oldenlabs python -m pytest tests/test_metrics.py -v`
Expected: FAIL — no module `neu_oldenlabs.metrics`.

- [ ] **Step 3: Write `src/neu_oldenlabs/metrics.py`**

```python
"""Aggregate binned measures to one value per animal, day, and light phase.

Three aggregation rules, chosen by measure type, because using one rule for all
of them is the easiest way to manufacture an effect:

* rates and means are weighted by the actual bin duration -- the bins jitter
  between nine and eleven minutes, so a flat mean silently overweights short bins;
* maxima take the maximum;
* counts become a rate per valid hour, so a day that lost bins to the rack mask is
  not scored as a day of low activity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import MEASURE_KIND

LIGHTS_ON = 7
LIGHTS_OFF = 19


def add_time_fields(long: pd.DataFrame, lights_on: int = LIGHTS_ON,
                    lights_off: int = LIGHTS_OFF) -> pd.DataFrame:
    frame = long.copy()
    frame["date"] = frame["bin_start"].dt.date
    frame["hour"] = frame["bin_start"].dt.hour
    light = (frame["hour"] >= lights_on) & (frame["hour"] < lights_off)
    frame["phase"] = np.where(light, "light", "dark")
    return frame


KEYS = ["cage", "tag", "date"]


def _aggregate_measure(block: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Apply one measure's rule to every (cage, tag, date) group at once."""
    kind = MEASURE_KIND[measure]
    frame = block.copy()
    frame["weighted"] = frame["value"] * frame["duration_s"]
    grouped = frame.groupby(KEYS, as_index=False).agg(
        weighted=("weighted", "sum"), seconds=("duration_s", "sum"),
        total=("value", "sum"), peak=("value", "max"))
    grouped["valid_hours"] = grouped["seconds"] / 3600.0
    if kind in ("rate", "mean"):
        grouped["value"] = grouped["weighted"] / grouped["seconds"].replace(0, np.nan)
    elif kind == "max":
        grouped["value"] = grouped["peak"]
    elif kind == "count":
        grouped["value"] = grouped["total"] / grouped["valid_hours"].replace(0, np.nan)
    else:
        raise ValueError(f"no aggregation rule for measure kind {kind!r}")
    grouped["measure"] = measure
    return grouped[KEYS + ["measure", "value", "valid_hours"]]


def daily(long: pd.DataFrame, lights_on: int = LIGHTS_ON,
          lights_off: int = LIGHTS_OFF) -> pd.DataFrame:
    frame = add_time_fields(long, lights_on, lights_off)
    parts = []
    for phase in ("all", "light", "dark"):
        subset = frame if phase == "all" else frame[frame["phase"] == phase]
        if subset.empty:
            continue
        for measure, block in subset.groupby("measure", sort=False):
            aggregated = _aggregate_measure(block, measure)
            aggregated.insert(3, "phase", phase)
            parts.append(aggregated)
    table = pd.concat(parts, ignore_index=True)
    return table[["cage", "tag", "date", "phase", "measure", "value", "valid_hours"]]


def drop_habituation(table: pd.DataFrame, days: int) -> pd.DataFrame:
    """Drop the first ``days`` calendar days of each cage's recording.

    The first day is a partial day and the animals have just been introduced to
    the cage, so it measures handling, not phenotype.
    """
    if days <= 0:
        return table
    kept = []
    for cage, frame in table.groupby("cage", sort=False):
        dates = sorted(set(frame["date"]))
        excluded = set(dates[:days + 1])
        kept.append(frame[~frame["date"].isin(excluded)])
    return pd.concat(kept, ignore_index=True)


def restrict_window(table: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    frame = table
    if start is not None:
        frame = frame[frame["date"] >= pd.Timestamp(start).date()]
    if end is not None:
        frame = frame[frame["date"] <= pd.Timestamp(end).date()]
    return frame.reset_index(drop=True)
```

Note on `drop_habituation`: `days=1` removes the first partial day **and** the first
full day — `dates[:days + 1]` — matching the spec's "drops the first partial day and
the first full day".

- [ ] **Step 4: Run the tests**

Run: `conda run -n neu-oldenlabs python -m pytest tests/test_metrics.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/neu_oldenlabs/metrics.py tests/test_metrics.py
git commit -m "feat: duration-weighted daily aggregation by measure kind"
```

---

### Task 6: Circadian metrics

**Files:**
- Create: `src/neu_oldenlabs/circadian.py`
- Create: `tests/test_circadian.py`

**Interfaces:**
- Consumes: the rack-masked long frame.
- Produces:
  - `circadian.hourly_activity(long) -> pd.DataFrame` — `cage, tag, hour_start, distance_per_hour`
  - `circadian.cosinor(series: pd.Series) -> dict` with keys `mesor, amplitude, acrophase_hour`
    (`series` indexed by hourly `DatetimeIndex`)
  - `circadian.interdaily_stability(series) -> float`
  - `circadian.intradaily_variability(series) -> float`
  - `circadian.relative_amplitude(series) -> dict` with keys `M10, L5, RA`
  - `circadian.circadian_profile(long) -> pd.DataFrame` — one row per `cage, tag` with
    columns `mesor, amplitude, acrophase_hour, IS, IV, M10, L5, RA`

Definitions (Van Someren): **IS** = between-day variance of the 24-h mean profile divided
by overall variance; **IV** = mean squared first difference divided by variance;
**M10 / L5** = highest 10-hour and lowest 5-hour running means; **RA** = (M10 − L5)/(M10 + L5).

- [ ] **Step 1: Write the failing test `tests/test_circadian.py`**

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from neu_oldenlabs.circadian import (cosinor, interdaily_stability, intradaily_variability,
                                     relative_amplitude)


def sine_series(days=10, amplitude=5.0, mesor=10.0, acrophase_hour=3.0):
    index = pd.date_range("2026-07-12", periods=24 * days, freq="h")
    radians = 2 * np.pi * (index.hour - acrophase_hour) / 24
    return pd.Series(mesor + amplitude * np.cos(radians), index=index)


def test_cosinor_recovers_a_known_amplitude_mesor_and_acrophase():
    fit = cosinor(sine_series())
    assert round(fit["mesor"], 6) == 10.0
    assert round(fit["amplitude"], 6) == 5.0
    assert round(fit["acrophase_hour"], 6) == 3.0


def test_interdaily_stability_is_one_for_a_perfectly_repeating_day():
    assert round(interdaily_stability(sine_series()), 6) == 1.0


def test_interdaily_stability_falls_when_days_differ():
    series = sine_series()
    rng = np.random.default_rng(0)
    noisy = series + rng.normal(0, 5, len(series))
    assert interdaily_stability(noisy) < interdaily_stability(series)


def test_intradaily_variability_is_higher_for_a_fragmented_rhythm():
    smooth = sine_series()
    fragmented = pd.Series(np.tile([0.0, 20.0], len(smooth) // 2), index=smooth.index)
    assert intradaily_variability(fragmented) > intradaily_variability(smooth)


def test_relative_amplitude_matches_a_hand_computed_block_profile():
    # Ten hours at 20, then fourteen at 0, repeated: M10 = 20, L5 = 0, RA = 1.
    day = [20.0] * 10 + [0.0] * 14
    index = pd.date_range("2026-07-12", periods=24 * 5, freq="h")
    series = pd.Series(day * 5, index=index)
    result = relative_amplitude(series)
    assert round(result["M10"], 6) == 20.0
    assert round(result["L5"], 6) == 0.0
    assert round(result["RA"], 6) == 1.0
```

- [ ] **Step 2: Run and confirm failure**

Run: `conda run -n neu-oldenlabs python -m pytest tests/test_circadian.py -v`
Expected: FAIL — no module `neu_oldenlabs.circadian`.

- [ ] **Step 3: Write `src/neu_oldenlabs/circadian.py`**

```python
"""Circadian description of the activity signal.

Activity is hourly distance travelled. Two families are reported: a cosinor fit,
which assumes a sinusoid and returns amplitude and phase, and the non-parametric
IS / IV / RA family, which assumes nothing about waveform. They answer different
questions and disagreeing is informative, so both are kept.

These metrics are descriptive. They are only interpretable as circadian claims
when the light schedule and a complete-day recording window are confirmed, which
for this study they are: 12:12, lights on 07:00.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def hourly_activity(long: pd.DataFrame) -> pd.DataFrame:
    """Distance per hour, on a regular hourly grid, per animal."""
    distance = long[long["measure"] == "distance_cm"].copy()
    if distance.empty:
        raise ValueError("no distance_cm measure; cannot build an activity signal")
    distance["hour_start"] = distance["bin_start"].dt.floor("h")
    grouped = distance.groupby(["cage", "tag", "hour_start"], as_index=False).agg(
        distance=("value", "sum"), seconds=("duration_s", "sum"))
    grouped["distance_per_hour"] = grouped["distance"] / (grouped["seconds"] / 3600.0)
    return grouped[["cage", "tag", "hour_start", "distance_per_hour"]]


def _as_hourly(series: pd.Series) -> pd.Series:
    ordered = series.sort_index()
    return ordered.asfreq("h") if ordered.index.freq is None else ordered


def cosinor(series: pd.Series) -> dict:
    """Least-squares fit of ``mesor + amplitude * cos(2*pi*(t - acrophase)/24)``."""
    values = series.to_numpy(dtype=float)
    hours = series.index.hour.to_numpy(dtype=float) + series.index.minute.to_numpy() / 60.0
    radians = 2 * np.pi * hours / 24.0
    design = np.column_stack([np.ones_like(radians), np.cos(radians), np.sin(radians)])
    finite = np.isfinite(values)
    coefficients, *_ = np.linalg.lstsq(design[finite], values[finite], rcond=None)
    mesor, cosine, sine = coefficients
    amplitude = float(np.hypot(cosine, sine))
    acrophase = float(np.arctan2(sine, cosine) * 24 / (2 * np.pi)) % 24
    return {"mesor": float(mesor), "amplitude": amplitude, "acrophase_hour": acrophase}


def interdaily_stability(series: pd.Series) -> float:
    """How reproducibly the 24-hour pattern repeats across days. 0 to 1."""
    values = _as_hourly(series)
    hourly_mean = values.groupby(values.index.hour).mean()
    grand = values.mean()
    n = values.count()
    numerator = n * ((hourly_mean - grand) ** 2).sum()
    denominator = len(hourly_mean) * ((values - grand) ** 2).sum()
    return float(numerator / denominator) if denominator else float("nan")


def intradaily_variability(series: pd.Series) -> float:
    """Fragmentation: mean squared hour-to-hour change over the variance."""
    values = _as_hourly(series).to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return float("nan")
    numerator = len(finite) * np.sum(np.diff(finite) ** 2)
    denominator = (len(finite) - 1) * np.sum((finite - finite.mean()) ** 2)
    return float(numerator / denominator) if denominator else float("nan")


def relative_amplitude(series: pd.Series) -> dict:
    """Highest ten and lowest five consecutive hours, and their contrast."""
    values = _as_hourly(series)
    ten = values.rolling(10, min_periods=10).mean()
    five = values.rolling(5, min_periods=5).mean()
    m10 = float(ten.max())
    l5 = float(five.min())
    total = m10 + l5
    return {"M10": m10, "L5": l5, "RA": float((m10 - l5) / total) if total else float("nan")}


def circadian_profile(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    hourly = hourly_activity(long)
    for (cage, tag), frame in hourly.groupby(["cage", "tag"], sort=False):
        series = frame.set_index("hour_start")["distance_per_hour"]
        row = {"cage": cage, "tag": tag}
        row.update(cosinor(series))
        row["IS"] = interdaily_stability(series)
        row["IV"] = intradaily_variability(series)
        row.update(relative_amplitude(series))
        rows.append(row)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run the tests**

Run: `conda run -n neu-oldenlabs python -m pytest tests/test_circadian.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/neu_oldenlabs/circadian.py tests/test_circadian.py
git commit -m "feat: cosinor and non-parametric IS/IV/RA circadian metrics"
```

---

### Task 7: Per-animal phenotype profile

**Files:**
- Create: `src/neu_oldenlabs/profile.py`
- Create: `tests/test_profile.py`

**Interfaces:**
- Consumes: `metrics.daily` output, `circadian.circadian_profile` output,
  `animals.load_animals` output.
- Produces:
  - `profile.build_profile(daily_table, circadian_table, animal_table) -> pd.DataFrame`
    — one row per animal, indexed by `animal_id` (the string `f"{cage}-{tag}"`), with
    identity columns `animal_id, cage, tag, animal_number, ear_mark, genotype,
    genotype_raw, n_days` and one column per measure named `<measure>_<phase>`
    (phases `all`, `light`, `dark`), plus derived columns
    `distance_dark_light_ratio`, `sleep_bout_minutes_all`, `distance_cv_daily`,
    and the circadian columns `mesor, amplitude, acrophase_hour, IS, IV, M10, L5, RA`.
  - `profile.PRIMARY_MEASURES: tuple[str, ...]` =
    `("distance_cm_all", "inactive_pct_dark", "IS", "social_distance_cm_all",
    "aggression_events_all")`

An animal contributes a measure only if it has at least `min_days` (default 5) days
with data; otherwise the cell is `NaN` and `n_days` records what it had.

- [ ] **Step 1: Write the failing test `tests/test_profile.py`**

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from neu_oldenlabs.profile import PRIMARY_MEASURES, build_profile


def daily_table(values_by_phase, days=10, cage=1, tag=1):
    rows = []
    for offset in range(days):
        date = (pd.Timestamp("2026-07-13") + pd.Timedelta(days=offset)).date()
        for phase, measures in values_by_phase.items():
            for measure, value in measures.items():
                hours = 24.0 if phase == "all" else 12.0
                rows.append({"cage": cage, "tag": tag, "date": date, "phase": phase,
                             "measure": measure, "value": value, "valid_hours": hours})
    return pd.DataFrame(rows)


def animal_table(cage=1, tag=1, genotype="mut"):
    return pd.DataFrame([{"cage": cage, "animal_number": 999, "ear_mark": "L",
                          "genotype_raw": "Mut/W", "genotype": genotype, "tag": tag}])


def circadian_table(cage=1, tag=1):
    return pd.DataFrame([{"cage": cage, "tag": tag, "mesor": 1.0, "amplitude": 2.0,
                          "acrophase_hour": 3.0, "IS": 0.5, "IV": 0.8,
                          "M10": 9.0, "L5": 1.0, "RA": 0.8}])


def test_profile_has_one_row_per_animal_with_identity_and_measures():
    daily = daily_table({"all": {"distance_cm": 100.0, "inactive_pct": 40.0},
                         "dark": {"inactive_pct": 20.0},
                         "light": {"inactive_pct": 60.0}})
    table = build_profile(daily, circadian_table(), animal_table())
    assert len(table) == 1
    row = table.iloc[0]
    assert row["animal_id"] == "1-1"
    assert row["genotype"] == "mut"
    assert row["distance_cm_all"] == 100.0
    assert row["inactive_pct_dark"] == 20.0
    assert row["IS"] == 0.5
    assert row["n_days"] == 10


def test_dark_light_ratio_is_derived_from_the_phase_columns():
    daily = daily_table({"all": {"distance_cm": 100.0},
                         "dark": {"distance_cm": 150.0},
                         "light": {"distance_cm": 50.0}})
    row = build_profile(daily, circadian_table(), animal_table()).iloc[0]
    assert round(row["distance_dark_light_ratio"], 6) == 3.0


def test_sleep_bout_duration_is_inactive_time_divided_by_bouts():
    # 50% inactive over an hour with 2 bouts per hour -> 15 minutes per bout.
    daily = daily_table({"all": {"inactive_pct": 50.0, "sleep_bouts": 2.0}})
    row = build_profile(daily, circadian_table(), animal_table()).iloc[0]
    assert round(row["sleep_bout_minutes_all"], 6) == 15.0


def test_day_to_day_variability_of_distance_is_reported():
    rows = []
    for offset, value in enumerate([100.0, 200.0, 100.0, 200.0, 100.0, 200.0]):
        date = (pd.Timestamp("2026-07-13") + pd.Timedelta(days=offset)).date()
        rows.append({"cage": 1, "tag": 1, "date": date, "phase": "all",
                     "measure": "distance_cm", "value": value, "valid_hours": 24.0})
    table = build_profile(pd.DataFrame(rows), circadian_table(), animal_table())
    expected = np.std([100, 200, 100, 200, 100, 200], ddof=1) / 150.0
    assert round(float(table.iloc[0]["distance_cv_daily"]), 6) == round(expected, 6)


def test_an_animal_with_too_few_days_gets_nan_not_a_number():
    daily = daily_table({"all": {"distance_cm": 100.0}}, days=3)
    row = build_profile(daily, circadian_table(), animal_table(), min_days=5).iloc[0]
    assert np.isnan(row["distance_cm_all"])
    assert row["n_days"] == 3


def test_primary_measures_are_all_present_as_columns():
    daily = daily_table({"all": {"distance_cm": 1.0, "social_distance_cm": 2.0,
                                 "aggression_events": 3.0, "inactive_pct": 4.0},
                         "dark": {"inactive_pct": 5.0}})
    table = build_profile(daily, circadian_table(), animal_table())
    for measure in PRIMARY_MEASURES:
        assert measure in table.columns
```

- [ ] **Step 2: Run and confirm failure**

Run: `conda run -n neu-oldenlabs python -m pytest tests/test_profile.py -v`
Expected: FAIL — no module `neu_oldenlabs.profile`.

- [ ] **Step 3: Write `src/neu_oldenlabs/profile.py`**

```python
"""One row per animal: the behavioural phenotype the statistics compare.

Each daily measure is averaged across days, separately for the whole day and for
each light phase, then joined to the circadian fits and to genotype. Averaging
across days rather than pooling all bins gives every day equal weight, so a single
long day cannot dominate an animal's phenotype.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRIMARY_MEASURES = ("distance_cm_all", "inactive_pct_dark", "IS",
                    "social_distance_cm_all", "aggression_events_all")

IDENTITY = ["animal_id", "cage", "tag", "animal_number", "ear_mark",
            "genotype", "genotype_raw", "n_days"]


def build_profile(daily_table: pd.DataFrame, circadian_table: pd.DataFrame,
                  animal_table: pd.DataFrame, min_days: int = 5) -> pd.DataFrame:
    daily_table = daily_table.copy()
    daily_table["column"] = daily_table["measure"] + "_" + daily_table["phase"]

    means = (daily_table.groupby(["cage", "tag", "column"], as_index=False)["value"].mean()
             .pivot(index=["cage", "tag"], columns="column", values="value"))
    means.columns.name = None

    day_counts = (daily_table[daily_table["phase"] == "all"]
                  .groupby(["cage", "tag"])["date"].nunique().rename("n_days"))

    distance = daily_table[(daily_table["measure"] == "distance_cm")
                           & (daily_table["phase"] == "all")]
    variability = distance.groupby(["cage", "tag"])["value"].agg(
        lambda values: float(np.std(values, ddof=1) / np.mean(values))
        if len(values) > 1 and np.mean(values) else float("nan")).rename("distance_cv_daily")

    table = means.join(day_counts).join(variability).reset_index()

    if "distance_cm_dark" in table and "distance_cm_light" in table:
        table["distance_dark_light_ratio"] = table["distance_cm_dark"] / table["distance_cm_light"]
    if "inactive_pct_all" in table and "sleep_bouts_all" in table:
        # inactive_pct is a percentage of the hour; sleep_bouts is bouts per hour.
        table["sleep_bout_minutes_all"] = (table["inactive_pct_all"] / 100.0 * 60.0
                                           / table["sleep_bouts_all"].replace(0, np.nan))

    table = table.merge(circadian_table, on=["cage", "tag"], how="left", validate="one_to_one")
    table = table.merge(animal_table, on=["cage", "tag"], how="inner", validate="one_to_one")
    table["animal_id"] = table["cage"].astype(str) + "-" + table["tag"].astype(str)

    measure_columns = [c for c in table.columns if c not in IDENTITY]
    thin = table["n_days"] < min_days
    table.loc[thin, measure_columns] = np.nan

    ordered = IDENTITY + sorted(measure_columns)
    return table[ordered].sort_values(["cage", "tag"]).reset_index(drop=True)
```

- [ ] **Step 4: Run the tests**

Run: `conda run -n neu-oldenlabs python -m pytest tests/test_profile.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/neu_oldenlabs/profile.py tests/test_profile.py
git commit -m "feat: assemble per-animal phenotype profile"
```

---

### Task 8: Cage-stratified statistics

**Files:**
- Create: `src/neu_oldenlabs/groups.py`
- Create: `tests/test_groups.py`

**Interfaces:**
- Consumes: `profile.build_profile` output.
- Produces:
  - `groups.cage_weights(values) -> pd.DataFrame` — per cage `n_mut, n_wt, weight`
    with `weight = n_mut * n_wt / (n_mut + n_wt)`
  - `groups.stratified_difference(values) -> float` — `Σ w_c (mean_mut_c − mean_wt_c) / Σ w_c`
  - `groups.count_assignments(values) -> int` — `Π C(n_c, n_mut_c)`
  - `groups.exact_stratified_permutation(values) -> tuple[float, int, float]` —
    `(p_value, n_assignments, min_attainable_p)`
  - `groups.hedges_g(values) -> float` — on cage-centred values
  - `groups.cluster_bootstrap_ci(values, draws=20000, level=0.95) -> tuple[float, float]`
  - `groups.mixed_model(values) -> dict` with keys `coefficient, se, p_value, converged`
  - `groups.benjamini_hochberg(p_values: np.ndarray) -> np.ndarray`
  - `groups.compare(profile, measure) -> dict` (one contrast row)
  - `groups.scan(profile, measures) -> pd.DataFrame` sorted by `p_value` with a
    `p_adjusted_bh` column

`values` throughout is a DataFrame with columns `animal_id, cage, genotype, value`,
genotype in `{"mut", "wt"}`, NaN values already dropped.

- [ ] **Step 1: Write the failing test `tests/test_groups.py`**

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neu_oldenlabs.groups import (benjamini_hochberg, cage_weights, cluster_bootstrap_ci,
                                  compare, count_assignments, exact_stratified_permutation,
                                  stratified_difference)

# The real study composition: 58616 is 5 wt / 0 mut, 58623 and 66336 are 2 wt / 3 mut,
# 58627 is 3 wt / 1 mut, 62923 is 1 wt / 1 mut, 58597 is 2 wt / 2 mut.
COMPOSITION = {58616: (5, 0), 58623: (2, 3), 66336: (2, 3),
               58627: (3, 1), 62923: (1, 1), 58597: (2, 2)}


def study_values(effect=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for cage, (n_wt, n_mut) in COMPOSITION.items():
        cage_offset = rng.normal(0, 10)
        for index in range(n_wt):
            rows.append({"animal_id": f"{cage}-w{index}", "cage": cage, "genotype": "wt",
                         "value": cage_offset + rng.normal(0, 1)})
        for index in range(n_mut):
            rows.append({"animal_id": f"{cage}-m{index}", "cage": cage, "genotype": "mut",
                         "value": cage_offset + effect + rng.normal(0, 1)})
    return pd.DataFrame(rows)


def test_the_design_admits_exactly_4800_within_cage_assignments():
    assert count_assignments(study_values()) == 4800


def test_an_all_wildtype_cage_has_zero_weight_and_cannot_shift_the_statistic():
    values = study_values()
    weights = cage_weights(values).set_index("cage")
    assert weights.loc[58616, "weight"] == 0.0

    shifted = values.copy()
    shifted.loc[shifted["cage"] == 58616, "value"] += 1000.0
    assert round(stratified_difference(values), 9) == round(stratified_difference(shifted), 9)


def test_cage_offsets_do_not_create_an_effect():
    # Huge between-cage offsets, no genotype effect: the test must not fire.
    p, n, floor = exact_stratified_permutation(study_values(effect=0.0, seed=3))
    assert n == 4800
    assert round(floor, 6) == round(2 / 4800, 6)
    assert p > 0.2


def test_a_planted_effect_is_detected():
    p, _, _ = exact_stratified_permutation(study_values(effect=4.0, seed=1))
    assert p < 0.01


def test_permutation_p_is_never_zero_because_the_observed_split_counts():
    p, n, _ = exact_stratified_permutation(study_values(effect=50.0, seed=2))
    assert p >= 1 / n


def test_bootstrap_interval_is_reproducible_and_brackets_a_planted_effect():
    values = study_values(effect=4.0, seed=1)
    first = cluster_bootstrap_ci(values, draws=2000)
    second = cluster_bootstrap_ci(values, draws=2000)
    assert first == second
    assert first[0] < 4.0 < first[1]


def test_benjamini_hochberg_matches_a_hand_computed_example():
    raw = np.array([0.01, 0.02, 0.03, 0.04])
    adjusted = benjamini_hochberg(raw)
    assert np.allclose(adjusted, [0.04, 0.04, 0.04, 0.04])


def test_compare_returns_a_row_with_counts_effect_and_interval():
    values = study_values(effect=3.0, seed=5)
    profile = values.rename(columns={"value": "distance_cm_all"})
    row = compare(profile, "distance_cm_all")
    assert row["measure"] == "distance_cm_all"
    assert row["n_mut"] == 10 and row["n_wt"] == 15
    assert row["cages_contributing"] == 5
    assert row["ci_low"] < row["difference"] < row["ci_high"]
    assert row["min_attainable_p"] == 2 / 4800


def test_a_measure_with_a_single_contributing_cage_still_computes():
    single = pd.DataFrame([
        {"animal_id": "a", "cage": 1, "genotype": "mut", "value": 2.0},
        {"animal_id": "b", "cage": 1, "genotype": "mut", "value": 3.0},
        {"animal_id": "c", "cage": 1, "genotype": "wt", "value": 1.0},
        {"animal_id": "d", "cage": 1, "genotype": "wt", "value": 0.0},
    ])
    p, n, _ = exact_stratified_permutation(single)
    assert n == 6
    assert p == pytest.approx(2 / 6)
```

- [ ] **Step 2: Run and confirm failure**

Run: `conda run -n neu-oldenlabs python -m pytest tests/test_groups.py -v`
Expected: FAIL — no module `neu_oldenlabs.groups`.

- [ ] **Step 3: Write `src/neu_oldenlabs/groups.py`**

```python
"""Compare mutant against wildtype with the cage as a stratum.

Animals are nested in cages, and cages differ for reasons that have nothing to do
with genotype: housing, cohort, and -- for the social measures -- which genotypes
the cage-mates carry. So the statistic is the weighted mean of *within-cage*
differences, and the permutation shuffles genotype labels only inside a cage. A
cage housing one genotype only carries no within-cage contrast, gets weight zero,
and drops out of the test by construction rather than silently shifting a mean.

The design admits 10 * 10 * 4 * 2 * 6 = 4800 label assignments, which is small
enough to enumerate, so the test is exact rather than sampled.
"""
from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pandas as pd

from . import RNG_SEED


def _split(values: pd.DataFrame) -> pd.DataFrame:
    frame = values.dropna(subset=["value"]).copy()
    unexpected = sorted(set(frame["genotype"]) - {"mut", "wt"})
    if unexpected:
        raise ValueError(f"genotype must be 'mut' or 'wt', got {unexpected}")
    return frame


def cage_weights(values: pd.DataFrame) -> pd.DataFrame:
    frame = _split(values)
    counts = (frame.groupby(["cage", "genotype"]).size().unstack(fill_value=0)
              .reindex(columns=["mut", "wt"], fill_value=0).reset_index())
    counts.columns = ["cage", "n_mut", "n_wt"]
    total = counts["n_mut"] + counts["n_wt"]
    counts["weight"] = np.where(total > 0, counts["n_mut"] * counts["n_wt"] / total, 0.0)
    return counts


def _difference_from_labels(cage_values: dict, labels: dict, weights: dict) -> float:
    numerator = denominator = 0.0
    for cage, array in cage_values.items():
        weight = weights[cage]
        if weight == 0:
            continue
        is_mut = labels[cage]
        numerator += weight * (array[is_mut].mean() - array[~is_mut].mean())
        denominator += weight
    return numerator / denominator if denominator else float("nan")


def _prepare(values: pd.DataFrame):
    frame = _split(values)
    weights = cage_weights(frame).set_index("cage")["weight"].to_dict()
    cage_values, observed_labels = {}, {}
    for cage, block in frame.groupby("cage", sort=True):
        cage_values[cage] = block["value"].to_numpy(dtype=float)
        observed_labels[cage] = (block["genotype"] == "mut").to_numpy()
    return cage_values, observed_labels, weights


def stratified_difference(values: pd.DataFrame) -> float:
    cage_values, labels, weights = _prepare(values)
    return _difference_from_labels(cage_values, labels, weights)


def count_assignments(values: pd.DataFrame) -> int:
    cage_values, labels, _ = _prepare(values)
    total = 1
    for cage, array in cage_values.items():
        n, k = len(array), int(labels[cage].sum())
        total *= len(list(combinations(range(n), k)))
    return total


def exact_stratified_permutation(values: pd.DataFrame) -> tuple[float, int, float]:
    """Enumerate every within-cage relabelling. Returns (p, assignments, min p)."""
    cage_values, observed, weights = _prepare(values)
    observed_difference = abs(_difference_from_labels(cage_values, observed, weights))

    # Every cage is enumerated, including any that houses a single genotype. Such
    # a cage admits exactly C(n, n) == 1 relabelling and contributes nothing to the
    # statistic, so enumerating it costs nothing and keeps this count identical to
    # `count_assignments` without any correction after the fact.
    cages = sorted(cage_values)
    per_cage = []
    for cage in cages:
        n, k = len(cage_values[cage]), int(observed[cage].sum())
        options = []
        for pick in combinations(range(n), k):
            mask = np.zeros(n, dtype=bool)
            mask[list(pick)] = True
            options.append(mask)
        per_cage.append(options)

    extreme = total = 0
    for combination in product(*per_cage):
        labels = dict(zip(cages, combination))
        difference = abs(_difference_from_labels(cage_values, labels, weights))
        total += 1
        # Ties count as extreme: an exactly-equal split is not evidence against
        # the null and must not be scored in its favour.
        extreme += difference >= observed_difference - 1e-12
    return extreme / total, total, 2.0 / total


def hedges_g(values: pd.DataFrame) -> float:
    """Standardised effect on cage-centred values, so cage offsets do not inflate it."""
    frame = _split(values).copy()
    frame["centred"] = frame["value"] - frame.groupby("cage")["value"].transform("mean")
    mut = frame.loc[frame["genotype"] == "mut", "centred"].to_numpy()
    wt = frame.loc[frame["genotype"] == "wt", "centred"].to_numpy()
    if len(mut) < 2 or len(wt) < 2:
        return float("nan")
    pooled = np.sqrt((((len(mut) - 1) * mut.var(ddof=1)) + ((len(wt) - 1) * wt.var(ddof=1)))
                     / (len(mut) + len(wt) - 2))
    if pooled == 0:
        return float("nan")
    d = (mut.mean() - wt.mean()) / pooled
    correction = 1 - 3 / (4 * (len(mut) + len(wt)) - 9)
    return float(d * correction)


def cluster_bootstrap_ci(values: pd.DataFrame, draws: int = 5000,
                         level: float = 0.95) -> tuple[float, float]:
    """Resample cages, then animals within genotype within cage.

    Resampling cages is what makes this interval honest: the cage, not the animal,
    is the unit that repeats if the study were run again.

    5000 draws keeps a thirty-measure scan to seconds rather than minutes; the
    interval's third decimal is not what any conclusion here rests on.
    """
    frame = _split(values)
    weights = cage_weights(frame).set_index("cage")["weight"].to_dict()
    cages = [cage for cage in sorted(frame["cage"].unique()) if weights[cage] > 0]
    blocks = {cage: {genotype: block["value"].to_numpy(dtype=float)
                     for genotype, block in frame[frame["cage"] == cage].groupby("genotype")}
              for cage in cages}
    rng = np.random.default_rng(RNG_SEED)
    estimates = np.empty(draws)
    for draw in range(draws):
        picked = rng.choice(cages, len(cages), replace=True)
        numerator = denominator = 0.0
        for cage in picked:
            weight = weights[cage]
            mut = rng.choice(blocks[cage]["mut"], len(blocks[cage]["mut"]), replace=True)
            wt = rng.choice(blocks[cage]["wt"], len(blocks[cage]["wt"]), replace=True)
            numerator += weight * (mut.mean() - wt.mean())
            denominator += weight
        estimates[draw] = numerator / denominator
    tail = (1 - level) / 2
    return float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1 - tail))


def mixed_model(values: pd.DataFrame) -> dict:
    """Sensitivity check: value ~ genotype with cage as a random intercept."""
    import statsmodels.formula.api as smf

    frame = _split(values).copy()
    frame["is_mut"] = (frame["genotype"] == "mut").astype(float)
    try:
        fit = smf.mixedlm("value ~ is_mut", frame, groups=frame["cage"]).fit()
    except Exception as error:  # noqa: BLE001 - convergence failure is a result, not a crash
        return {"coefficient": float("nan"), "se": float("nan"),
                "p_value": float("nan"), "converged": False, "note": str(error)}
    return {"coefficient": float(fit.params["is_mut"]), "se": float(fit.bse["is_mut"]),
            "p_value": float(fit.pvalues["is_mut"]), "converged": bool(fit.converged)}


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.clip(ranked, 0, 1)
    return adjusted


def compare(profile: pd.DataFrame, measure: str) -> dict:
    values = profile[["animal_id", "cage", "genotype", measure]].rename(
        columns={measure: "value"}).dropna(subset=["value"])
    weights = cage_weights(values)
    contributing = int((weights["weight"] > 0).sum())
    if contributing == 0:
        raise ValueError(f"{measure}: no cage contains both genotypes; nothing to compare")
    p, assignments, floor = exact_stratified_permutation(values)
    low, high = cluster_bootstrap_ci(values)
    model = mixed_model(values)
    return {
        "measure": measure,
        "n_mut": int((values["genotype"] == "mut").sum()),
        "n_wt": int((values["genotype"] == "wt").sum()),
        "cages_contributing": contributing,
        "mean_mut": float(values.loc[values["genotype"] == "mut", "value"].mean()),
        "mean_wt": float(values.loc[values["genotype"] == "wt", "value"].mean()),
        "difference": stratified_difference(values),
        "ci_low": low, "ci_high": high,
        "hedges_g": hedges_g(values),
        "p_value": p, "assignments": assignments, "min_attainable_p": floor,
        "lmm_coefficient": model["coefficient"], "lmm_p_value": model["p_value"],
        "lmm_converged": model["converged"],
    }


def scan(profile: pd.DataFrame, measures: list[str]) -> pd.DataFrame:
    rows, skipped = [], []
    for measure in measures:
        if measure not in profile.columns:
            skipped.append(f"{measure} (absent)")
            continue
        if profile[measure].nunique(dropna=True) <= 1:
            skipped.append(f"{measure} (constant)")
            continue
        try:
            rows.append(compare(profile, measure))
        except ValueError as error:
            skipped.append(f"{measure} ({error})")
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_adjusted_bh"] = benjamini_hochberg(table["p_value"].to_numpy())
    table["skipped_measures"] = "; ".join(skipped)
    return table.sort_values("p_value").reset_index(drop=True)
```

- [ ] **Step 4: Run the tests**

Run: `conda run -n neu-oldenlabs python -m pytest tests/test_groups.py -v`
Expected: 9 passed. The 4800-assignment enumeration runs in a few seconds; if
`test_bootstrap_interval` is slow, it uses `draws=2000` deliberately.

- [ ] **Step 5: Commit**

```bash
git add src/neu_oldenlabs/groups.py tests/test_groups.py
git commit -m "feat: cage-stratified exact permutation statistics"
```

---

### Task 9: Study configuration

**Files:**
- Create: `src/neu_oldenlabs/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `config.CageSpec` — frozen dataclass with `cage: int, export: Path | None`
  - `config.StudyConfig` — frozen dataclass with `name, path, animals_csv, cages,
    window_start, window_end, habituation_days, lights_on, lights_off,
    primary_measures, scan_measures, min_days`
  - `config.load_study(path: Path) -> StudyConfig`
  - `StudyConfig.available_cages` / `StudyConfig.missing_cages` properties

`study.json` schema (all paths may be relative to the config file):

```json
{
  "name": "dacruz_study2",
  "animals_csv": "/path/DaCruz_Epilepsy.csv",
  "cages": [{"cage": 58616, "export": "/path/058616_..._cage.xlsx"},
            {"cage": 58623, "export": null}],
  "window_start": null,
  "window_end": null,
  "habituation_days": 1,
  "lights_on": 7,
  "lights_off": 19,
  "min_days": 5,
  "primary_measures": ["distance_cm_all", "inactive_pct_dark", "IS",
                       "social_distance_cm_all", "aggression_events_all"],
  "scan_measures": []
}
```

An empty `scan_measures` means "every numeric profile column".

- [ ] **Step 1: Write the failing test `tests/test_config.py`**

```python
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
```

- [ ] **Step 2: Run and confirm failure**

Run: `conda run -n neu-oldenlabs python -m pytest tests/test_config.py -v`
Expected: FAIL — no module `neu_oldenlabs.config`.

- [ ] **Step 3: Write `src/neu_oldenlabs/config.py`**

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `conda run -n neu-oldenlabs python -m pytest tests/test_config.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/neu_oldenlabs/config.py tests/test_config.py
git commit -m "feat: explicit study configuration with declared missing cages"
```

---

### Task 10: Pipeline assembly and figures

**Files:**
- Create: `src/neu_oldenlabs/pipeline.py`
- Create: `src/neu_oldenlabs/plots.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 2–9.
- Produces:
  - `pipeline.run_study(study: StudyConfig, output: Path, cache_dir: Path | None) -> dict`
    returning `{"long", "daily", "profile", "qc", "scan"}` of DataFrames and writing
    `daily_measures.csv`, `animal_profile.csv`, `qc_*.csv`, `group_comparison.csv`
  - `plots.actogram(profile_long, output)`, `plots.phase_profiles(hourly, output, measures)`,
    `plots.cage_paired(profile, output, measures)`, `plots.effect_forest(scan, output)`,
    `plots.daily_trajectories(daily, output, measure)`, `plots.qc_panel(qc, output)`
  - Each plot function writes `<name>.png` **and** `<name>.csv` into `output`.

**Before writing `plots.py`, load the `dataviz` skill** and follow it for palette,
axis, and legend decisions. Genotype colour must be consistent across every figure,
and dark-phase shading must use the same band style everywhere.

- [ ] **Step 1: Write the failing test `tests/test_pipeline.py`**

```python
from __future__ import annotations

from neu_oldenlabs.pipeline import run_study


def test_run_study_produces_profile_and_comparison_tables(synthetic_study, tmp_path):
    study = synthetic_study
    output = tmp_path / "out"
    result = run_study(study, output, cache_dir=None)

    assert len(result["profile"]) == 6
    assert set(result["profile"]["genotype"]) == {"mut", "wt"}
    assert (output / "animal_profile.csv").exists()
    assert (output / "daily_measures.csv").exists()
    assert (output / "group_comparison.csv").exists()
    assert (output / "provenance.json").exists()


def test_missing_cages_are_recorded_not_silently_dropped(synthetic_study, tmp_path):
    result = run_study(synthetic_study, tmp_path / "out", cache_dir=None)
    assert result["missing_cages"] == [1003]


def test_every_figure_writes_its_values_as_csv(synthetic_study, tmp_path):
    study = synthetic_study
    output = tmp_path / "out"
    run_study(study, output, cache_dir=None)
    figures = sorted(p.stem for p in (output / "figures").glob("*.png"))
    tables = sorted(p.stem for p in (output / "figures").glob("*.csv"))
    assert figures and figures == tables
```

- [ ] **Step 2: Run and confirm failure**

Run: `conda run -n neu-oldenlabs python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — no module `neu_oldenlabs.pipeline`.

- [ ] **Step 3: Write `src/neu_oldenlabs/plots.py`**

Six functions, each following the same shape: build the table it plots, write it
as `<name>.csv`, draw the figure, save `<name>.png` at `dpi=180`, close the figure.
Use one shared helper:

```python
def _save(fig, table, output, name):
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / f"{name}.csv", index=False)
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=180)
    plt.close(fig)
```

and one shared genotype palette plus a `_shade_dark(ax, lights_on, lights_off)`
helper used by every time-resolved panel. The six figures are named
`actogram`, `phase_profiles`, `cage_paired`, `effect_forest`,
`daily_trajectories`, `qc_panel`.

- [ ] **Step 4: Write `src/neu_oldenlabs/pipeline.py`**

`run_study` in order: load animals → for each available cage, `load_cage_cached`,
`check_join` on the tags found, collect → concatenate → `qc.qc_tables` on the raw
long frame → `qc.apply_rack_mask` → `metrics.daily` → `drop_habituation` →
`restrict_window` → `circadian.circadian_profile` on the masked long frame →
`profile.build_profile` → `groups.scan` over `scan_measures` (or every numeric
profile column when empty) → write all CSVs → call the six plot functions →
`write_provenance` with every export plus the animals CSV plus the config file.
Return the dict of frames, including `missing_cages`.

`groups.scan` must be skipped, with a recorded reason, when no cage contains both
genotypes — the all-wildtype-study case, which is exactly what cage 58616 alone gives.
In that case `group_comparison.csv` is still written, holding a single `note` column
saying why nothing was compared, and `plots.effect_forest` writes **neither** a PNG nor
a CSV, so the figure/table pairing stays exact.

- [ ] **Step 5: Run the tests**

Run: `conda run -n neu-oldenlabs python -m pytest tests/test_pipeline.py -v`
Expected: 3 passed.

- [ ] **Step 6: Run the whole suite**

Run: `conda run -n neu-oldenlabs python -m pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/neu_oldenlabs/pipeline.py src/neu_oldenlabs/plots.py tests/test_pipeline.py
git commit -m "feat: assemble the study pipeline with figures and tables"
```

---

### Task 11: Report and CLI

**Files:**
- Create: `src/neu_oldenlabs/report.py`
- Create: `src/neu_oldenlabs/cli.py`
- Create: `scripts/render_report.sh`
- Create: `tests/test_report.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `pipeline.run_study` output.
- Produces:
  - `report.build_study_report(study: StudyConfig, output: Path, cache_dir=None) -> Path`
    writing `output/report.md` and returning its path
  - `cli.main()` with subcommands `ingest`, `qc`, `profile`, `compare`, `study-report`

`report.md` sections, in order: title and date; **Data and coverage** (cages
analysed, cages missing, recording windows, rack-mask fractions, days retained
after habituation); **Methods** (aggregation rules, phase definition, the
cage-stratified statistic, the exact assignment count and `min_attainable_p`,
bootstrap and LMM); **Results** (primary measures table first, then the FDR-corrected
scan, then figures with captions); **Interpretation safeguards** (the dyadic-measure
caveat for social distance and aggression; non-significant results reported as
absence of evidence; the scan labelled hypothesis-generating; any cage contributing
no within-cage contrast named explicitly).

- [ ] **Step 1: Write the failing test `tests/test_report.py`**

```python
from __future__ import annotations

from neu_oldenlabs.report import build_study_report


def test_report_names_missing_cages_and_the_permutation_count(synthetic_study, tmp_path):
    output = tmp_path / "out"
    path = build_study_report(synthetic_study, output)
    text = path.read_text()
    assert "1003" in text                      # the missing cage is named
    assert "within-cage" in text.lower()       # the test is described
    assert "absence of evidence" in text.lower()
    assert "min_attainable_p" in text or "smallest attainable" in text.lower()


def test_report_states_the_dyadic_caveat_for_social_measures(synthetic_study, tmp_path):
    text = build_study_report(synthetic_study, tmp_path / "out").read_text().lower()
    assert "social distance" in text and "cage-mate" in text
```

- [ ] **Step 2: Run and confirm failure**

Run: `conda run -n neu-oldenlabs python -m pytest tests/test_report.py -v`
Expected: FAIL — no module `neu_oldenlabs.report`.

- [ ] **Step 3: Write `src/neu_oldenlabs/report.py`**

Assemble `report.md` from the frames `run_study` returns, rendering tables with
`DataFrame.to_markdown()`. The safeguards section is written from the computed
values — for example the sentence naming cages that contribute no within-cage
contrast is generated from `groups.cage_weights`, not typed by hand — so the
report cannot drift from the analysis.

- [ ] **Step 4: Write `src/neu_oldenlabs/cli.py`**

```python
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_study
from .io import load_cage_cached
from .pipeline import run_study
from .report import build_study_report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neu-oldenlabs")
    sub = p.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="parse one cage export to a Parquet table")
    ingest.add_argument("export")
    ingest.add_argument("--cage", type=int, required=True)
    ingest.add_argument("--output", required=True)

    for name in ("qc", "profile", "compare", "study-report"):
        cmd = sub.add_parser(name)
        cmd.add_argument("config")
        cmd.add_argument("--output", required=True)
        cmd.add_argument("--cache", default=None)
    return p


def main() -> None:
    args = parser().parse_args()
    output = Path(args.output)
    if args.command == "ingest":
        frame = load_cage_cached(Path(args.export), args.cage, output)
        output.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output / f"cage{args.cage}.parquet", index=False)
        print(f"wrote {output / f'cage{args.cage}.parquet'} ({len(frame)} rows)")
        return

    study = load_study(args.config)
    cache = Path(args.cache) if args.cache else None
    if args.command == "study-report":
        print(build_study_report(study, output, cache_dir=cache))
        return
    result = run_study(study, output, cache_dir=cache)
    if args.command == "qc":
        print(result["qc"]["rack"].to_string(index=False))
    elif args.command == "profile":
        print(result["profile"].to_string(index=False))
    else:
        print(result["scan"].to_string(index=False) if len(result["scan"])
              else "no cage contains both genotypes; no comparison computed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write `scripts/render_report.sh`**

Copy `neu-intellicage/scripts/render_report.sh` and change the pandoc
`--metadata title` to `"Oldenlabs home-cage report"`. Make it executable:
`chmod +x scripts/render_report.sh`.

- [ ] **Step 6: Run the tests**

Run: `conda run -n neu-oldenlabs python -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 7: Update `README.md`** with the full command list, the interpretation
safeguards (mirroring the spec's §7 in prose), and the privacy note.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat: study report and command-line interface"
```

---

### Task 12: Create the neubrain project and run cage 58616

**Files:**
- Create (in `neubrain`, on branch `oldenlabs/dacruz-study2`):
  `projects/oldenlabs/plan.md`, `projects/oldenlabs/protocol.md`,
  `projects/oldenlabs/.gitignore`,
  `projects/oldenlabs/analysis/experiments/dacruz_study2/study.json`

**Interfaces:**
- Consumes: the `neu-oldenlabs` CLI.
- Produces: the study configuration and the first real run.

- [ ] **Step 1: Create the branch**

```bash
cd /mnt/sysfs01/users/cagatay/code/neubrain
git checkout main && git pull
git checkout -b oldenlabs/dacruz-study2
```

- [ ] **Step 2: Write `projects/oldenlabs/.gitignore`**

```
analysis/experiments/*/outputs/
*.png
*.pdf
*.csv
*.parquet
!analysis/experiments/*/study.json
```

- [ ] **Step 3: Write `study.json`**

```json
{
  "name": "dacruz_study2",
  "animals_csv": "/mnt/sysfs01/users/cagatay/external/cruz/oldenlabs/260820_study_2/DaCruz_Epilepsy.csv",
  "cages": [
    {"cage": 58616, "export": "/mnt/sysfs01/users/cagatay/external/cruz/oldenlabs/260820_study_2/058616_CA_l47P_cg00001_071120261653_082020261112_3423_cage.xlsx"},
    {"cage": 58623, "export": null},
    {"cage": 66336, "export": null},
    {"cage": 58627, "export": null},
    {"cage": 62923, "export": null},
    {"cage": 58597, "export": null}
  ],
  "window_start": null,
  "window_end": null,
  "habituation_days": 1,
  "lights_on": 7,
  "lights_off": 19,
  "min_days": 5,
  "primary_measures": ["distance_cm_all", "inactive_pct_dark", "IS",
                       "social_distance_cm_all", "aggression_events_all"],
  "scan_measures": []
}
```

As each remaining export arrives, replace that cage's `null` with its path. Nothing
else changes.

- [ ] **Step 4: Write `projects/oldenlabs/plan.md` and `protocol.md`**

`plan.md`: the question, the design (6 cages, 25 animals, mut n=10 / wt n=15,
~40 days of continuous monitoring), the pre-specified primary measures, and the
current state (which exports have arrived). `protocol.md`: the Oldenlabs system,
the 10-minute binned export, the 22 measures, ear-tag identity, the light schedule,
and the analysis decisions with their rationale — pointing at the spec rather than
restating it.

- [ ] **Step 5: Run the pipeline on cage 58616**

```bash
cd /mnt/sysfs01/users/cagatay/code/neubrain/projects/oldenlabs/analysis/experiments/dacruz_study2
conda run -n neu-oldenlabs neu-oldenlabs study-report study.json --output outputs --cache cache
```

Expected: `outputs/report.md`, `outputs/figures/*.png` with matching `*.csv`,
`outputs/animal_profile.csv` with 5 rows, and a report that names the five missing
cages and states that no genotype comparison was computed because no cage
contains both genotypes.

- [ ] **Step 6: Render the report**

```bash
/mnt/sysfs01/users/cagatay/code/neu-oldenlabs/scripts/render_report.sh outputs
```

Expected: `outputs/report.html` and `outputs/report.pdf`.

- [ ] **Step 7: Sanity-check the numbers against the raw file**

Confirm from `animal_profile.csv` that each of the five animals has `n_days` around
38 (40 days minus habituation), `genotype == "wt"` for all five, and a
`distance_cm_all` in a plausible range for a mouse (order 10⁴–10⁵ cm/day). Any
value that looks impossible is investigated before the report is shown to anyone.

- [ ] **Step 8: Commit the project (config and docs only, never outputs)**

```bash
cd /mnt/sysfs01/users/cagatay/code/neubrain
git add projects/oldenlabs
git status --short   # confirm no CSV, PNG, PDF, or parquet is staged
git commit -m "Add oldenlabs project for da Cruz Study 2"
```

---

## Self-Review

**Spec coverage:** §2.1 ingest → Task 3; §2.2 animals → Task 2; §2.3 cage identity
cross-check → Task 3 (`cage_id_from_filename`); §2.4 missing cages → Tasks 9, 10, 11;
§3 architecture → Tasks 1–11; §4 ingest/cache → Task 3; §5 QC (rack, regularity,
coverage, window, ranges) → Task 4, with window and habituation in Task 5; §6.1
aggregation rules → Task 5; §6.2 profile → Tasks 6, 7; §7 statistics (stratified
difference, exact permutation, effect size, bootstrap, FDR, LMM, dyadic caveat) →
Task 8 and the report's safeguards section in Task 11; §8 figures → Task 10; §9
deliverables and CLI → Task 11; §10 testing → the test file in every task; §11
AGENTS rules → Task 1.

**Known gaps, deliberate:** the plan does not run the genotype comparison on real
data, because five of six exports do not exist yet. Task 12 ends with the pipeline
proven on cage 58616 and the configuration ready to take the remaining files.

---

## Execution Handoff

Plan complete. Two execution options:

1. **Subagent-Driven (recommended)** — a fresh subagent per task, reviewed between tasks.
2. **Inline Execution** — tasks executed in this session with checkpoints.
