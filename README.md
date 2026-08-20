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

(To be filled in by Task 11 once the CLI is implemented.)
