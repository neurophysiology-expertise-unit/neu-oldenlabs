# AGENTS.md — neu-oldenlabs

- This repository is reusable analysis code, not a study or manuscript repo.
- Never commit Oldenlabs exports, the animal/genotype CSV, ear-tag identities, or
  generated animal-level tables. Data are supplied by path at runtime.
- Preserve Oldenlabs source semantics. Do not silently reinterpret or rename an
  exported measure. An unknown measure name is an error, not something to skip.
  The authority for what each measure means is the instrument paper -- Delalic,
  Florea et al. (2025), "Smart Lids for deep multi-animal phenotyping in
  standard home cages", Front. Behav. Neurosci., doi:10.3389/fnbeh.2025.1696654,
  citekey `delali2025` -- not the vendor's export column header, which is
  sometimes wrong (e.g. it labels both acceleration and angular velocity
  "cm/s"; the paper gives cm/s^2 and deg/s respectively). See
  `io.MEASURE_DEFINITIONS` for the published definition of each measure.
- Bin width is NOT fixed across exports and must never be assumed: the real
  study's six cage exports use two different bin widths -- cage 58616 uses
  600 s (10-minute) bins, the other five use 3600 s (60-minute, hourly)
  bins -- and even within one export, a bin's own duration (`Datetime to`
  minus `Datetime from`) can differ from its nominal width. Every
  aggregation is weighted by each bin's actual duration, never a fixed
  constant. A plain mean over bins is a bug, and so is any per-bin figure or
  table that pools cages of different bin width without normalising to a
  per-hour (or otherwise duration-weighted) basis first -- pooling raw
  per-bin values across bin widths manufactures a difference that is pure
  bin-width artefact, not signal.
- Bins where `Cage in rack == 0` are excluded from every metric.
- Every plotted quantity must also be written as a machine-readable table.
- Analysis runs record input hashes and parameters in `provenance.json`.
- The genotype comparison is cage-stratified: labels permute only within a cage.
  Never report a genotype comparison computed from cages whose exports are missing;
  report which cages are absent.
- Crash loudly on real errors. A missing cage export or an all-wildtype cage is an
  expected RESULT, logged, not an error.
