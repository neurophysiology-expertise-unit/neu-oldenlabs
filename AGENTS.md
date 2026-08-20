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
