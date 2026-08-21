# neu-oldenlabs

Reproducible loading, quality control, and analysis of Oldenlabs home-cage
monitoring exports (behavioural neuroscience). Given Oldenlabs export files and an
animal/genotype table supplied by path at runtime, this package validates and loads
them, computes duration-weighted metrics and circadian summaries, builds
cage-stratified genotype comparisons, and writes figures, tables, and a
`provenance.json` stamp for every analysis run.

Design: `docs/superpowers/specs/2026-08-20-neu-oldenlabs-design.md`
Plan: `docs/superpowers/plans/2026-08-20-neu-oldenlabs.md`

## Install

```bash
conda create -y -n neu-oldenlabs python=3.11
conda run -n neu-oldenlabs python -m pip install -e '.[test]'
```

Run tests with:

```bash
conda run -n neu-oldenlabs python -m pytest tests/ -v
```

## No data in this repository

Oldenlabs exports, the animal/genotype table, ear-tag identities, and any generated
animal-level table must never be committed here. All data are supplied by path at
runtime; see `AGENTS.md` for the durable rules this repository follows.

## Commands

```bash
# Parse one cage export to a Parquet table
neu-oldenlabs ingest <cage.xlsx> --cage <id> --output <dir>

# Quality control: rack-mask fraction, recording breaks, out-of-range values, coverage
neu-oldenlabs qc <study.json> --output <dir>

# One row per animal: the behavioural phenotype the statistics compare
neu-oldenlabs profile <study.json> --output <dir>

# Cage-stratified genotype comparison, scanned across every configured measure
neu-oldenlabs compare <study.json> --output <dir>

# The full deliverable: report.md, figures, and tables
neu-oldenlabs study-report <study.json> --output <dir>

# Render report.md to report.html and report.pdf (requires pandoc + xelatex)
scripts/render_report.sh <dir>
```

All of `qc`/`profile`/`compare`/`study-report` accept `--cache <dir>` to Parquet-cache
the parsed export (keyed by content hash), so a re-run does not re-parse a multi-tens-of-MB
`.xlsx` file. Every command except `ingest` writes `provenance.json` into its output
directory, recording input paths and hashes, the package version, parameters, and the
RNG seed; `ingest` only writes the parsed Parquet table (`cage<id>.parquet`).

`study.json` declares the cage list with export paths (or `null` for an export that has
not arrived yet), the animal/genotype CSV path, the analysis window, habituation days,
light schedule, the pre-specified primary measure set, and the full scan measure list.
See `docs/superpowers/specs/2026-08-20-neu-oldenlabs-design.md` §9 for the schema.

`report.py` renders its tables by hand rather than via `DataFrame.to_markdown()`,
which needs the optional `tabulate` package that is not a declared dependency of this
project; nothing in `study-report` requires installing it.

## Statistics and how to read the report

The experimental unit is the animal, but animals are nested in cages, and the
genotype comparison is **cage-stratified**: it is the weighted mean of *within-cage*
(mut − wt) differences, weighted by `n_mut * n_wt / (n_mut + n_wt)` per cage. A cage
housing a single genotype has weight zero and drops out of the test **by
construction** — its animals still appear in every descriptive table and figure, but
contribute no contrast to the comparison. In the real Oldenlabs study, cage 58616 is
exactly this case: five wildtype animals and no mutants.

- **Exact, not sampled.** Genotype labels are permuted only *within* a cage, so
  cage-level differences (housing, cohort, cage-mate composition) cannot masquerade
  as a genotype effect. For the real study's six cages (`58623`, `66336`, `58627`,
  `62923`, `58597`, `58616`), the design admits `10 × 10 × 4 × 2 × 6 × 1 = 4800`
  within-cage label assignments — small enough to enumerate exactly rather than
  sample.
- **The p-value floor is 1/4800, not 2/4800.** The smallest attainable two-sided
  p-value for this design is **1/4800 ≈ 0.00021**. The familiar 2/n floor only holds
  when the fully sign-flipped labelling (every contributing cage's mut/wt labels
  swapped) is itself one of the enumerated assignments, which requires every
  contributing cage to be genotype-balanced. The real study's contributing cages are
  3/2, 3/2, 1/3, 1/1, and 2/2: two of these (1/1 and 2/2) are individually balanced,
  but the other three (3/2, 3/2, 1/3) are not, and the 2/n floor requires *every*
  contributing cage to be balanced, not just some — so the floor for this design is
  1/n, not 2/n. The code computes and reports the correct floor per measure as
  `min_attainable_p`; the report always prints that returned value, never a
  literal 2/n.
- **Hedges' g is computed over contributing cages only**, using the stratified
  difference itself as its numerator, so the effect size describes the same
  quantity as the test. Its pooled standard deviation additionally drops any
  contributing cage where either genotype cell has fewer than two animals, so the
  denominator can span fewer cages than the numerator — do not read g as pooling
  every animal in the study.
- **The cluster bootstrap CI is anti-conservative with few cages.** With only five
  contributing cages, simulations run during this project's build put the
  percentile cluster-bootstrap CI's coverage in the low nineties against a nominal
  95%. The report states this rather than presenting the interval as exact.
- **Social distance and aggression are dyadic**, not per-animal: a mutant's value
  depends on which genotypes its cage-mates carry, and cage composition ranges from
  0/5 to 3/5 mutant across the study. Within-cage permutation is the correct test
  precisely because of this, and the report states the caveat rather than leaving
  the reader to notice it.
- **`aggression_events` is a count**, aggregated as events per valid hour (a day
  that lost bins to the rack mask is not scored as a day of low activity), so the
  profile column `aggression_events_all` is events **per hour**, not per day.
- **The scan is hypothesis-generating.** The ~30-measure scan is FDR-corrected
  (Benjamini–Hochberg); the pre-specified primary measures, named in `study.json`
  before the run, are reported first and on their own terms.
- **A non-significant result is absence of evidence, never evidence of absence.**
- **Missing exports are named, not silently dropped.** A cage whose export has not
  arrived is declared with a null export in `study.json`, is never analysed as a
  genotype comparison, and is named explicitly in the report's Data and coverage
  section.

## Privacy

Oldenlabs exports, the animal/genotype CSV, ear-tag identities, and any generated
animal-level table (including `report.md`, figures, and the CSV/Parquet tables this
package writes) are study data, not code, and must never be committed to this
repository. All data are supplied by path at runtime; `.gitignore` and `AGENTS.md`
enforce this at the repository level, but the discipline is on whoever runs the
CLI: point `--output` and `--cache` at a directory outside version control.
