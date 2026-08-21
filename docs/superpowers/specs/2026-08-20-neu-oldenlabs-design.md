# neu-oldenlabs — design

**Date:** 2026-08-20
**Study:** da Cruz lab (Center for Neuroscience, KU Leuven) — `DaCruz_Epilepsy`, Oldenlabs
home-cage monitoring, Study 2
**Question:** do animals carrying the mutant allele (`Mut/W`) differ behaviourally
from wildtype (`W/w`) cage-mates across a month of continuous home-cage monitoring?

---

## 1. Purpose and scope

`neu-oldenlabs` is reusable analysis code for Oldenlabs home-cage monitoring
exports: ingest, quality control, per-animal behavioural profiling, and
between-genotype statistics with a rendered report. It is a code repository, not
a study repository. Raw exports, the animal/genotype table, and any generated
animal-level tables stay outside Git and are supplied by path at runtime.

The study itself lives in the `neubrain` vault as `projects/oldenlabs/`, on branch
`oldenlabs/dacruz-study2`.

Non-goals: video re-analysis, re-derivation of Oldenlabs' own behaviour
classifiers, and any attempt to reinterpret what the vendor's syllabus extraction
means. The exported measures are taken as given and preserved with their source
semantics, exactly as `neu-intellicage` preserves `CornerCondition` and
`PlaceError`.

## 2. Data

### 2.1 Cage export (`.xlsx`)

One file per cage. Verified against
`058616_CA_l47P_cg00001_071120261653_082020261112_3423_cage.xlsx`:

- Single sheet, 112 columns × 5724 rows (`A1:DH5724`).
- Row 1: `Animal ID` — the ear-tag number 1–5, repeated once per measure block.
- Row 2: measure name (columns A and B are `Datetime from` / `Datetime to`).
- Rows 3+: 5722 data bins, `2026-07-11 16:56` → `2026-08-20 10:58` (39.75 days).
- Layout is **22 measures × 5 animals = 110 value columns**, blocked by measure.
- No missing cells in the reference file (0 of 629,420).

Bins are *nominally* 10 minutes but jitter: 5681 gaps of 10 min, 36 of 11 min,
4 of 9 min. **All aggregation must be duration-weighted using the actual
`Datetime to − Datetime from`**, never a plain mean over bins.

The 22 measures:

| Slug | Source name | Type |
|---|---|---|
| `aggression_time_pct` | Aggression time (%) | rate |
| `aggression_events` | Number of agression events (1/0) | count |
| `accel_cm_s` | Average movement acceleration (cm/s) | mean |
| `angular_velocity_cm_s` | Average movement angular velocity (cm/s) | mean |
| `speed_overall_cm_s` | Average overall speed (cm/s) | mean |
| `speed_awake_cm_s` | Average speed when awake (cm/s) | mean |
| `speed_max_cm_s` | Max movement speed (cm/s) | max |
| `distance_cm` | Total distance traveled (cm) | count/sum |
| `occupancy_center_pct` | Cage center occupancy (%) | rate |
| `occupancy_front_half_pct` | Cage front half occupancy (%) | rate |
| `occupancy_corners_pct` | Time spent in cage corners (%) | rate |
| `in_rack` | Cage in rack (1/0) | mask |
| `climbing_events` | Climbing events | count |
| `time_climbing_pct` | Time spent climbing (%) | rate |
| `rearing_events` | Rearing events | count |
| `time_rearing_pct` | Time spent rearing (%) | rate |
| `enrichment_tube_pct` | Enrichment tube interaction time (%) | rate |
| `sleep_bouts` | Sleep bouts | count |
| `inactive_pct` | Sleep time - spent inactive (%) | rate |
| `social_distance_cm` | Social Distance (cm) | mean |
| `time_eating_pct` | Time spent eating (%) | rate |
| `time_drinking_pct` | Time spent drinking water (%) | rate |

The mapping lives in one explicit dict. **An unrecognised measure name is a hard
error**, never a silent drop — a vendor software update that renames or adds a
column must surface as a crash, not as a quietly shorter table.

### 2.2 Animal table (`DaCruz_Epilepsy.csv`)

25 animals in 6 cages: 58616 (5), 58623 (5), 66336 (5), 58627 (4), 62923 (2),
58597 (4). Columns: cage #, animal #, ear mark (L/R/LR/LL/RR), genotype, tag #,
tag date, monitoring start. Cage number appears only on the first row of each
block and is forward-filled.

**Join key is `(cage, tag)`** — tag # in the CSV is the `Animal ID` in the export.
The join must be exact and total: every populated tag column in an export needs
exactly one CSV row and vice versa, or the run fails.

**Genotype normalisation.** The CSV contains three spellings: `W/w` (14),
`Mut/W` (10), `W/W` (1 — animal 672, cage 58623). Per the study owner, `W/w` and
`W/W` are the same wildtype genotype and the case difference is a typo. The
normalisation is a single visible dict:

```
"W/w" -> "wt", "W/W" -> "wt", "Mut/W" -> "mut"
```

`genotype_raw` is retained on every animal row so the normalisation is auditable,
and any string outside those three is a hard error.

Resulting groups: **mut n = 10, wt n = 15.**

### 2.3 Cage identity

Cage ID comes from the study configuration. The ID parsed from the filename
prefix (`058616_…`) is used **only as a cross-check that must agree**. Software
never infers study meaning from filenames.

### 2.4 What is on disk today

Only cage 58616's export exists as of 2026-08-20, and that cage is entirely
wildtype — it yields no genotype contrast. The remaining five exports are being
produced by the study owner. The pipeline is built for all six; until the files
land, `ingest`/`qc`/`profile` run per cage and `compare` reports which cages are
missing rather than silently comparing a subset.

## 3. Architecture

New standalone repository `neu-oldenlabs`, mirroring the `neu-intellicage`
layout. Oldenlabs data (binned wide time-series) and IntelliCage data (discrete
visit/nose-poke events) are different data models; only the statistics,
provenance, and report-rendering layer (~200 lines) is common, and that layer is
copied rather than shared. Extracting a `neu-behav-core` package is deliberately
deferred: refactoring a working repository on a sample size of one buys less than
it costs, and the option remains open when a third study shows what is genuinely
shared.

```
neu-oldenlabs/
  src/neu_oldenlabs/
    io.py          xlsx -> tidy long frame, Parquet cache
    animals.py     CSV -> animal table, genotype normalisation, (cage, tag) join
    qc.py          rack mask, bin regularity, coverage, window, range sanity
    metrics.py     duration-weighted daily aggregation by measure type
    circadian.py   cosinor, IS / IV / RA (M10, L5)
    groups.py      cage-stratified exact permutation, effect sizes, FDR, LMM
    plots.py       six figure families; every figure also writes its CSV
    report.py      report.md assembly
    provenance.py  input hashes, config hash, version, params, seed
    cli.py         ingest | qc | profile | compare | study-report
  tests/
  scripts/render_report.sh
  AGENTS.md  README.md  pyproject.toml  .gitignore  LICENSE
```

Each module has one purpose and a table-in/table-out interface, so it can be
tested without the others. `io.py` never computes; `metrics.py` never reads
files; `groups.py` sees only a per-animal profile table.

**Environment:** a new conda env `neu-oldenlabs` (Python 3.11, pandas ≥ 2,
numpy, scipy, statsmodels, matplotlib, openpyxl, pyarrow, pytest). Existing envs
carry either pandas 0.23 or no pandas.

## 4. Ingest

`io.load_cage(path, cage_id) -> DataFrame` with columns
`cage, tag, bin_start, bin_end, duration_s, measure, value`.

Header rows 1–2 form the column key; columns A/B give the bin bounds. ~630k rows
per cage, ~3.8M for six — comfortable in pandas. Results cache to Parquet keyed
by the input file's hash, so re-runs skip the 29 MB XML parse.

## 5. Quality control

Report-only; QC never modifies inputs and never repairs values.

- **Rack mask** — bins with `in_rack == 0` are excluded from every downstream
  metric. The excluded fraction is reported per cage and per animal. Cage 58616
  loses nothing (all 5722 bins in rack); other cages may differ.
- **Bin regularity** — distribution of bin durations and of inter-bin gaps. A gap
  greater than 15 min is flagged as a recording break.
- **Coverage** — valid bins per animal per day across the month, so an animal
  that drops out mid-study is visible before it shifts a group mean.
- **Window alignment** — cages start on different dates. The configuration
  declares a common analysis window; per-cage coverage of that window is
  reported. `habituation_days` (default 1) drops the first partial day and the
  first full day.
- **Range sanity** — `%` measures within [0, 100], speeds ≥ 0, counts integral.
  Violations are reported per cage, never auto-corrected.

## 6. Metrics

### 6.1 Daily aggregation

The rule depends on measure type. Using the wrong one is the easiest way to
manufacture an effect:

- **rate / occupancy (%)** — duration-weighted mean over valid bins.
- **speeds, acceleration, angular velocity, social distance** — duration-weighted
  mean. `speed_max_cm_s` takes the max.
- **event counts** (rearing, climbing, aggression, sleep bouts) — summed per day
  and expressed **per valid hour**, so a day with masked bins is not scored as
  low activity.
- **distance** — summed per day, also expressed per valid hour.

### 6.2 Phenotype profile

One row per animal, ~30 columns. Each measure is computed over 24 h and
separately for the light phase (07:00–19:00) and dark phase (19:00–07:00), with a
dark/light ratio where it is meaningful. Light schedule: **12:12, lights on
07:00**, per the study owner.

| Domain | Measures |
|---|---|
| Locomotion | distance/h, mean speed, speed when awake, max speed, acceleration, angular velocity |
| Sleep | inactive %, dark-phase inactive %, sleep bouts/h, mean bout duration |
| Circadian | cosinor amplitude and acrophase on hourly distance; non-parametric IS, IV, RA (M10, L5) |
| Vertical / exploratory | rearing events/h, time rearing %, climbing events/h, time climbing %, enrichment-tube % |
| Space use | centre, corner, front-half occupancy % |
| Consumption | eating %, drinking % |
| Social | mean social distance, dark-phase social distance |
| Aggression | aggression events/day, aggression time % |
| Stability | day-to-day CV of distance/day |

The circadian family (IS / IV / RA) is the same one `neu-intellicage` computes in
tier 1, applied here to hourly distance instead of corner visits.

## 7. Statistics

The experimental unit is the animal. Animals are nested in cages, and five of the
six cages are genotype-mixed.

- **Primary statistic — cage-stratified difference.** The weighted mean of
  within-cage (mut − wt) differences over the five mixed cages. Cage 58616 is
  all-wildtype, contributes no within-cage contrast, and therefore **drops out of
  the test by construction** rather than silently inflating the wildtype mean.
  Its animals still appear in every descriptive table and figure.
- **Exact stratified permutation.** Genotype labels shuffle only *within* cage, so
  cage-level differences — housing, cohort, cage-mate composition — cannot
  masquerade as genotype effects. The design admits
  `10 × 10 × 4 × 2 × 6 = 4800` label assignments (per cage: 58623 C(5,3)=10,
  66336 C(5,3)=10, 58627 C(4,1)=4, 62923 C(2,1)=2, 58597 C(4,2)=6, 58616 C(5,0)=1).
  4800 is fully enumerable, so the test is **exact rather than sampled**.
  Smallest attainable two-sided p = **1/4800 ≈ 0.00021**, not 2/4800: the 2/n
  floor assumes the fully sign-flipped labelling is itself an enumerated
  assignment, which holds only when every contributing cage is genotype-balanced.
  These cages are 3/2, 3/2, 1/3, 1/1 and 2/2 — two of them (1/1 and 2/2) are
  balanced, but three are not, and the condition requires all of them — so the
  mirror is not enumerated and the floor is 1/n. The code reports 2/n only in the balanced case. This value is
  printed next to every p-value, as `neu-intellicage` does, but here it is a note
  of adequacy rather than of a resolution floor.
- **Effect size and interval always**, whatever the p-value: Hedges' g plus a
  cluster bootstrap CI on the stratified difference (resample cages with
  replacement, then animals within cage). Hedges' g is computed over the
  **contributing cages only**, with the stratified difference as its numerator and
  the pooled within-cage SD as its denominator, so the effect size and the test
  describe the same quantity — pooling every animal instead lets a cage with no
  within-cage contrast move the effect size while leaving the statistic untouched.
  With only five contributing cages the percentile cluster bootstrap is
  **anti-conservative** — two independent simulations during review put coverage
  in the low nineties against a nominal 95% (91% and 93%) — and the report says so
  rather than quietly presenting the interval as exact. A non-significant
  result is reported as absence of evidence, never as evidence of absence.
- **Multiplicity.** Benjamini–Hochberg FDR across the ~30-measure scan. The scan
  is labelled hypothesis-generating. A small **pre-specified primary set** is
  named in the configuration before the run: `distance_per_hour`,
  `inactive_pct_dark`, `IS`, `social_distance_cm`, `aggression_events_per_day`.
- **Sensitivity analysis.** `MixedLM(value ~ genotype, groups=cage)` from
  statsmodels, reported beside the permutation result. Where the two disagree,
  the disagreement is reported; neither is chosen post hoc for being the nicer
  answer.
- **Dyadic-measure caveat.** Social distance and aggression are properties of a
  dyad, not of an animal: a mutant's social distance depends on its cage-mates'
  genotypes, and cage composition ranges from 0/5 to 3/5 mutant. Within-cage
  permutation is the correct test for precisely this reason, and the report states
  the caveat rather than leaving the reader to notice it.
- RNG seed is fixed and recorded in `provenance.json`.

## 8. Figures

Every plotted quantity is also written as a machine-readable CSV beside the
figure. Chart design follows the `dataviz` skill, and dark/light phase shading is
consistent across all time-resolved panels.

1. **Actogram heatmap** — animal × (day × hour) distance, blocked by genotype.
2. **24-hour profiles** — genotype mean ± SEM with dark-phase shading, per
   primary measure.
3. **Cage-paired dot plot** — one panel per measure, animals as points grouped by
   cage and coloured by genotype. This shows exactly the contrast the test uses,
   and makes the all-wildtype cage visibly non-contributing.
4. **Effect-size forest** — Hedges' g ± bootstrap CI per measure, FDR-flagged.
5. **Daily trajectories** — per-animal series across the month, coloured by
   genotype.
6. **QC panel** — coverage and rack-mask fraction per cage and animal.

## 9. Deliverables

CLI:

```
neu-oldenlabs ingest  <cage.xlsx> --cage <id> --output <dir>
neu-oldenlabs qc      <study.json> --output <dir>
neu-oldenlabs profile <study.json> --output <dir>
neu-oldenlabs compare <study.json> --output <dir>
neu-oldenlabs study-report <study.json> --output <dir>
scripts/render_report.sh <dir>
```

`study-report` emits `report.md`, figures, and tables; `render_report.sh` builds
`report.html` and `report.pdf` with pandoc so a delivered PDF is always
rebuildable from committed inputs. Every command writes `provenance.json`
recording input paths and hashes, the configuration hash, package version,
parameters, and the RNG seed.

`study.json` (in `neubrain/projects/oldenlabs/analysis/experiments/dacruz_study2/`)
declares: cage list with export paths, animal CSV path, analysis window,
habituation days, light schedule, genotype group definitions, the primary measure
set, and the full scan measure list.

## 10. Testing

Test-driven, with synthetic fixtures only — never a slice of real animal data.

- `test_io` — a tiny synthetic workbook with the real header geometry, including
  a 9-minute and an 11-minute bin; asserts tidy output, correct durations, and a
  hard error on an unknown measure name.
- `test_animals` — genotype normalisation including the `W/W` case; forward-fill
  of cage; failure on an unmatched tag in either direction.
- `test_qc` — rack mask excludes the right bins; a 20-minute gap is flagged;
  out-of-range `%` reported and not corrected.
- `test_metrics` — duration weighting differs from the naive mean on jittered
  bins; counts-per-valid-hour is invariant to masked bins.
- `test_circadian` — cosinor recovers a known amplitude and acrophase; IS/IV on
  a constructed signal match hand-computed values.
- `test_groups` — permutation enumerates exactly 4800 assignments on the real
  cage composition; an all-wildtype cage contributes nothing; a planted effect is
  detected and a null is not; BH matches a hand-computed example.

## 11. Rules for this repository (AGENTS.md)

- Reusable analysis code, not a study or manuscript repository.
- Never commit Oldenlabs exports, the animal/genotype CSV, ear-tag identities, or
  generated animal-level tables. Data are supplied by path at runtime.
- Preserve Oldenlabs source semantics. Do not silently reinterpret or rename
  exported measures; an unknown measure name is an error.
- Every plotted quantity must also be written as a machine-readable table.
- Analysis runs record input hashes and parameters in `provenance.json`.
- Never report a genotype comparison computed from cages whose exports are
  missing; report which cages are absent.
- Crash loudly on real errors. A missing cage export or an all-wildtype cage is
  an expected *result*, logged, not an error.
