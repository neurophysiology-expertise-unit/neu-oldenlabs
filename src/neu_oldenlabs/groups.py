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

import warnings
from itertools import combinations, product

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from . import RNG_SEED
from . import metrics
from .io import DISTANCE_DERIVED_PROFILE_MEASURES


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


def _label_options(n: int, k: int) -> list[np.ndarray]:
    """Every boolean mask of length `n` with exactly `k` True values.

    Factored out of `exact_stratified_permutation` so `cluster_permutation`
    (below) can enumerate the identical per-cage label options without
    duplicating the combinatorics.
    """
    options = []
    for pick in combinations(range(n), k):
        mask = np.zeros(n, dtype=bool)
        mask[list(pick)] = True
        options.append(mask)
    return options


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
        per_cage.append(_label_options(n, k))

    extreme = total = 0
    for combination in product(*per_cage):
        labels = dict(zip(cages, combination))
        difference = abs(_difference_from_labels(cage_values, labels, weights))
        total += 1
        # Ties count as extreme: an exactly-equal split is not evidence against
        # the null and must not be scored in its favour.
        extreme += difference >= observed_difference - 1e-12

    # The floor is 2/total only if the fully sign-flipped assignment -- swap
    # every contributing cage's mut/wt labels for their complement -- is
    # itself one of the enumerated assignments. Flipping a cage's labels only
    # preserves that cage's mut count (and so stays a valid assignment) when
    # the cage is genotype-balanced (n_mut == n_wt); when it flips, the whole
    # per-cage weighted sum flips sign, tying with the observed difference
    # under abs() and guaranteeing 2 extreme assignments. If any contributing
    # cage is unbalanced, that global flip is not a valid assignment, no
    # other assignment is guaranteed to reproduce -observed_difference
    # exactly, and only the observed assignment itself is guaranteed to tie.
    balanced = all(
        int(observed[cage].sum()) == len(cage_values[cage]) - int(observed[cage].sum())
        for cage in cages if weights[cage] > 0
    )
    floor = 2.0 / total if balanced else 1.0 / total
    return extreme / total, total, floor


def hedges_g(values: pd.DataFrame) -> float:
    """Standardised version of the stratified difference, on the same cages.

    A cage with weight 0 contributes no within-cage contrast to
    `stratified_difference` and must contribute none here either: pooling its
    animals into the effect size would describe a different quantity than the
    p-value does. So this restricts to contributing cages (weight > 0), uses
    `stratified_difference` itself as the numerator, and pools the
    within-genotype variance across only those same cages for the
    denominator, with the small-sample correction applied to the animal count
    that actually entered that pooled variance.
    """
    cage_values, labels, weights = _prepare(values)
    contributing = [cage for cage in cage_values if weights[cage] > 0]
    if not contributing:
        return float("nan")
    difference = _difference_from_labels(cage_values, labels, weights)

    pooled_ss = 0.0
    pooled_df = 0
    n_total = 0
    for cage in contributing:
        array = cage_values[cage]
        is_mut = labels[cage]
        mut, wt = array[is_mut], array[~is_mut]
        if len(mut) < 2 or len(wt) < 2:
            continue  # a within-genotype variance needs at least 2 animals to estimate
        pooled_ss += (len(mut) - 1) * mut.var(ddof=1) + (len(wt) - 1) * wt.var(ddof=1)
        pooled_df += (len(mut) - 1) + (len(wt) - 1)
        n_total += len(mut) + len(wt)
    if pooled_df <= 0:
        return float("nan")

    pooled_sd = np.sqrt(pooled_ss / pooled_df)
    if pooled_sd == 0:
        return float("nan")
    d = difference / pooled_sd
    correction = 1 - 3 / (4 * n_total - 9)
    return float(d * correction)


def _cluster_bootstrap(values: pd.DataFrame, draws: int = 5000,
                       level: float = 0.95) -> dict[str, tuple[float, float]]:
    """Resample cages, then animals within genotype within cage.

    Resampling cages is what makes this interval honest: the cage, not the animal,
    is the unit that repeats if the study were run again.

    Computes the bootstrap distribution of BOTH `stratified_difference` and
    `hedges_g` from the SAME draws (one resampling pass, not two): every draw
    recomputes `hedges_g`'s own pooled SD from that draw's resampled mut/wt
    arrays (contributing cages only, pooling within-genotype variance across
    cages with >=2 animals per genotype -- which cages qualify is fixed by the
    original animal counts, since a bootstrap draw resamples each cage back to
    its own original size; only which cages appear, and how many times, varies
    draw to draw) and the correction on that draw's own pooled animal count --
    rather than rescaling the difference-CI by the point estimate's fixed
    g/difference ratio, which would silently treat the pooled SD as constant
    across draws when it is not. A draw where no cage qualifies for pooling
    (so g is undefined that draw) is dropped from the g quantiles rather than
    contaminating them with a placeholder value.

    5000 draws is not cheap: `compare` (bootstrap + exact permutation + mixed
    model) measured at ~4.7 s per measure, so a thirty-measure `scan` costs on
    the order of two minutes, not seconds. That cost buys more decimal places
    on the interval than any conclusion here rests on, but it is not lowered
    silently -- see the module-level note in `scan`.
    """
    frame = _split(values)
    weights = cage_weights(frame).set_index("cage")["weight"].to_dict()
    cages = [cage for cage in sorted(frame["cage"].unique()) if weights[cage] > 0]
    blocks = {cage: {genotype: block["value"].to_numpy(dtype=float)
                     for genotype, block in frame[frame["cage"] == cage].groupby("genotype")}
              for cage in cages}
    # Whether a cage qualifies for the pooled-variance sum is fixed by its own
    # original animal counts (a bootstrap draw resamples it back to that same
    # size), so this is computed once rather than inside the draw loop.
    poolable = {cage for cage in cages
               if len(blocks[cage]["mut"]) >= 2 and len(blocks[cage]["wt"]) >= 2}

    rng = np.random.default_rng(RNG_SEED)
    differences = np.empty(draws)
    g_estimates = np.full(draws, np.nan)
    for draw in range(draws):
        picked = rng.choice(cages, len(cages), replace=True)
        numerator = denominator = 0.0
        pooled_ss = 0.0
        pooled_df = 0
        n_total = 0
        for cage in picked:
            weight = weights[cage]
            mut = rng.choice(blocks[cage]["mut"], len(blocks[cage]["mut"]), replace=True)
            wt = rng.choice(blocks[cage]["wt"], len(blocks[cage]["wt"]), replace=True)
            numerator += weight * (mut.mean() - wt.mean())
            denominator += weight
            if cage in poolable:
                pooled_ss += (len(mut) - 1) * mut.var(ddof=1) + (len(wt) - 1) * wt.var(ddof=1)
                pooled_df += (len(mut) - 1) + (len(wt) - 1)
                n_total += len(mut) + len(wt)
        difference = numerator / denominator
        differences[draw] = difference
        if pooled_df > 0:
            pooled_sd = np.sqrt(pooled_ss / pooled_df)
            if pooled_sd > 0:
                correction = 1 - 3 / (4 * n_total - 9)
                g_estimates[draw] = (difference / pooled_sd) * correction

    tail = (1 - level) / 2
    result = {"difference": (float(np.quantile(differences, tail)),
                             float(np.quantile(differences, 1 - tail)))}
    finite_g = g_estimates[np.isfinite(g_estimates)]
    if finite_g.size == 0:
        result["hedges_g"] = (float("nan"), float("nan"))
    else:
        result["hedges_g"] = (float(np.quantile(finite_g, tail)),
                              float(np.quantile(finite_g, 1 - tail)))
    return result


def cluster_bootstrap_ci(values: pd.DataFrame, draws: int = 5000,
                         level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI for the stratified difference. See `_cluster_bootstrap`."""
    return _cluster_bootstrap(values, draws, level)["difference"]


def cluster_bootstrap_g_ci(values: pd.DataFrame, draws: int = 5000,
                           level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI for Hedges' g itself. See `_cluster_bootstrap`."""
    return _cluster_bootstrap(values, draws, level)["hedges_g"]


def mixed_model(values: pd.DataFrame) -> dict:
    """Sensitivity check: value ~ genotype with cage as a random intercept.

    statsmodels signals a failed or boundary fit by *warning*
    (ConvergenceWarning: "failed to converge", "on the boundary of the
    parameter space", "Hessian ... is not positive definite", ...), not by
    raising -- a bare try/except therefore misses it and reports
    `converged: True` with an se and p-value drawn from a singular Hessian.
    Every warning raised during the fit call is caught instead of left to
    print to stderr (which would spray across a thirty-measure scan), but
    only a `ConvergenceWarning` is treated as a convergence failure. Catching
    ANY warning category (the previous behaviour used
    `warnings.simplefilter("always")` and then checked whether the caught
    list was non-empty at all) means an unrelated warning -- a `FutureWarning`
    from a library upgrade, for instance -- would mark every row of a scan as
    non-converged and void the sensitivity analysis for reasons that have
    nothing to do with the fit. `converged` is forced to False, and the
    convergence warning text is kept in `note`, only when a real
    ConvergenceWarning was raised; other warnings are still swallowed (so
    they do not spray to stderr across a scan) but leave `converged`
    unaffected.
    """
    import statsmodels.formula.api as smf

    frame = _split(values).copy()
    frame["is_mut"] = (frame["genotype"] == "mut").astype(float)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = smf.mixedlm("value ~ is_mut", frame, groups=frame["cage"]).fit()
    except Exception as error:  # noqa: BLE001 - convergence failure is a result, not a crash
        return {"coefficient": float("nan"), "se": float("nan"),
                "p_value": float("nan"), "converged": False, "note": str(error)}
    convergence_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    converged = bool(fit.converged) and not convergence_warnings
    note = "; ".join(str(warning.message) for warning in convergence_warnings)
    return {"coefficient": float(fit.params["is_mut"]), "se": float(fit.bse["is_mut"]),
            "p_value": float(fit.pvalues["is_mut"]), "converged": converged, "note": note}


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
    # One resampling pass produces both CIs (see `_cluster_bootstrap`): calling
    # `cluster_bootstrap_ci` and `cluster_bootstrap_g_ci` separately here would
    # silently double the ~5000-draw bootstrap cost per measure.
    bootstrap = _cluster_bootstrap(values)
    low, high = bootstrap["difference"]
    g_low, g_high = bootstrap["hedges_g"]
    model = mixed_model(values)

    # `n_mut`/`n_wt` count every animal with a non-NaN value, INCLUDING
    # animals in a zero-weight (single-genotype) cage that contributes no
    # within-cage contrast to the test at all -- so with the real study's
    # composition, `n_mut`/`n_wt` can read 10/15 for a comparison that
    # actually only used 10 mut and 10 wt animals (the 5 extra wt are cage
    # 58616, all-wildtype, weight 0). `n_mut_contributing`/`n_wt_contributing`
    # restrict to animals in cages with weight > 0, i.e. the counts the test
    # actually used; the totals are kept alongside, distinctly labelled.
    contributing_cages = set(weights.loc[weights["weight"] > 0, "cage"])
    contributing_values = values[values["cage"].isin(contributing_cages)]

    row = {
        "measure": measure,
        "n_mut": int((values["genotype"] == "mut").sum()),
        "n_wt": int((values["genotype"] == "wt").sum()),
        "n_mut_contributing": int((contributing_values["genotype"] == "mut").sum()),
        "n_wt_contributing": int((contributing_values["genotype"] == "wt").sum()),
        "cages_contributing": contributing,
        "mean_mut": float(values.loc[values["genotype"] == "mut", "value"].mean()),
        "mean_wt": float(values.loc[values["genotype"] == "wt", "value"].mean()),
        "difference": stratified_difference(values),
        "ci_low": low, "ci_high": high,
        "hedges_g": hedges_g(values),
        "g_ci_low": g_low, "g_ci_high": g_high,
        "p_value": p, "assignments": assignments, "min_attainable_p": floor,
        "lmm_coefficient": model["coefficient"], "lmm_se": model["se"],
        "lmm_p_value": model["p_value"], "lmm_converged": model["converged"],
        "lmm_note": model["note"],
    }
    assert list(row.keys()) == _COMPARE_COLUMNS, (
        "_COMPARE_COLUMNS (the hand-maintained duplicate used to give an "
        "all-skipped scan the right shape) has drifted out of sync with the "
        f"keys compare() actually returns: {list(row.keys())} != {_COMPARE_COLUMNS}")
    return row


# The columns `compare` produces, in order -- kept alongside `scan` so an
# all-skipped scan can still return a frame with the right shape instead of
# one with no columns at all (which would KeyError on `p_adjusted_bh`).
# `compare` asserts its own return keys match this list on every call, so the
# two cannot silently drift apart.
_COMPARE_COLUMNS = [
    "measure", "n_mut", "n_wt", "n_mut_contributing", "n_wt_contributing",
    "cages_contributing", "mean_mut", "mean_wt", "difference",
    "ci_low", "ci_high", "hedges_g", "g_ci_low", "g_ci_high",
    "p_value", "assignments", "min_attainable_p",
    "lmm_coefficient", "lmm_se", "lmm_p_value", "lmm_converged", "lmm_note",
]


def scan(profile: pd.DataFrame, measures: list[str]) -> pd.DataFrame:
    """Run `compare` over every measure and rank by p-value.

    `compare` is not cheap (~4.7 s measured per call: exact enumeration +
    5000-draw bootstrap + a mixed model), so a thirty-measure scan costs on
    the order of two minutes. That is deliberate -- see `cluster_bootstrap_ci`
    -- and not something to silently claw back by lowering a default here.
    """
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
    if not rows:
        # No contributing measure: still return the expected columns (so
        # `frame["p_adjusted_bh"]` does not KeyError on an empty scan) and
        # keep the skip audit trail, attached as frame metadata since there
        # is no row left to broadcast it into.
        table = pd.DataFrame(columns=_COMPARE_COLUMNS + ["p_adjusted_bh", "skipped_measures"])
        table.attrs["skipped_measures"] = skipped
        return table
    table = pd.DataFrame(rows)
    table["p_adjusted_bh"] = benjamini_hochberg(table["p_value"].to_numpy())
    table["skipped_measures"] = "; ".join(skipped)
    return table.sort_values("p_value").reset_index(drop=True)


# ============================================================================
# Cage-stratified, cluster-based permutation test over the 24-hour profile
# ============================================================================
#
# The study owner looked at the `phase_profiles` figure -- mutant and wildtype
# hourly curves separating, with non-overlapping SEM bands -- and asked how to
# test that properly. Testing 24 clock hours with 24 separate tests is wrong
# twice over: adjacent hours are strongly autocorrelated, and 24 tests times
# every scanned measure is a large multiplicity burden. Non-overlapping SEM
# bands are NOT themselves a test: bars that just touch correspond to roughly
# p ~= 0.3, not p ~= 0.05.
#
# The tool here is the cluster-based permutation test (Maris & Oostenveld
# 2007, standard in EEG/MEG for exactly this "many correlated time points"
# problem), adapted to this study's cage-stratified design: the per-hour
# statistic is the SAME weighted within-cage difference `stratified_difference`
# uses (reusing `_difference_from_labels` for the observed value), and the
# null is within-cage label permutation, exactly as `exact_stratified_permutation`
# enumerates it.

HOURS_PER_DAY = 24
CLUSTER_FORMING_THRESHOLD = 2.0
CLUSTER_ENUMERATION_CAP = 20_000
CLUSTER_PERMUTATION_DRAWS = 5000


def hourly_profile_by_animal(long: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Animal x clock-hour table: one row per animal, one column per hour (0..23).

    `long` is the rack-masked (and habituation/window-restricted) bin-level
    frame joined with genotype -- the same shape `pipeline.run_study` passes
    to `plots.phase_profiles`. Delegates the within-animal, across-day
    reduction to `metrics.hourly_by_animal` (duration-weighted per
    `io.MEASURE_KIND`, the same call `plots._hourly_value_per_animal` makes
    for the `phase_profiles` figure), so every animal contributes exactly ONE
    value per clock hour -- averaged across every day it was recorded --
    before any statistic sees the data. This is the step that prevents
    pseudoreplication: n in `cluster_permutation` is animals (~25 in the real
    study), never animal-hours (~27,000).

    Returns columns "animal_id", "cage", "tag", "genotype", then one column
    per hour, named by the integer hour (0..23). An hour with no bin at all
    for an animal is NaN, not a fabricated 0 -- `cluster_permutation` drops
    any animal missing even one hour (see its docstring).
    """
    per_animal = metrics.hourly_by_animal(long, measure)
    identity = long[["cage", "tag", "genotype"]].drop_duplicates()
    wide = per_animal.pivot(index=["cage", "tag"], columns="hour", values="value")
    wide = wide.reindex(columns=range(HOURS_PER_DAY))
    wide = wide.reset_index().merge(identity, on=["cage", "tag"], how="left")
    wide["animal_id"] = wide["cage"].astype(str) + "-" + wide["tag"].astype(str)
    ordered = ["animal_id", "cage", "tag", "genotype"] + list(range(HOURS_PER_DAY))
    return wide[ordered]


def _weighted_hourly_difference(cage_arrays: dict, labels: dict, weights: dict) -> np.ndarray:
    """`_difference_from_labels` generalised to many columns (hours) at once.

    `cage_arrays[cage]` is (n_animals_in_cage, n_columns). Returns the
    weighted within-cage difference for EVERY column in one array operation
    -- the identical formula `_difference_from_labels` uses for a single
    column (a test below checks the two agree), just evaluated for all 24
    hours together, which is what makes enumerating thousands of within-cage
    permutations across 24 hours affordable.
    """
    n_columns = next(iter(cage_arrays.values())).shape[1]
    numerator = np.zeros(n_columns)
    denominator = 0.0
    for cage, array in cage_arrays.items():
        weight = weights[cage]
        if weight == 0:
            continue
        is_mut = labels[cage]
        numerator = numerator + weight * (array[is_mut].mean(axis=0) - array[~is_mut].mean(axis=0))
        denominator += weight
    if denominator == 0:
        return np.full(n_columns, np.nan)
    return numerator / denominator


def _find_clusters(statistic: np.ndarray, threshold: float) -> list:
    """Contiguous, same-signed runs of hours where |statistic| > threshold,
    treating the 24 hours as CIRCULAR -- hour 23 is adjacent to hour 0.

    A naive "double the 24-point signal to 48 points, then scan for runs"
    trick is NOT safe here: it can split one true wrapping cluster into two
    reported clusters (the wrapped portion, found spanning the doubling
    seam, AND a spurious duplicate of its non-wrapped tail found again at
    the very start of the array, since a linear scan has no predecessor for
    index 0 and so always starts a fresh run there even when hour 0 is
    really the middle of a run that started at hour 22). This was caught by
    the midnight-wrap test during development and is why the module test
    `test_a_wrapping_cluster_is_reported_once_not_twice` exists.

    Correct approach: rotate the 24-point array to start at a genuine
    boundary -- an index that does NOT continue the run ending at the
    previous (circularly) index -- so a single ordinary LINEAR scan for
    same-sign runs cannot be split by, or duplicated across, the array's
    artificial start/end. Handles the degenerate case where every hour
    exceeds the threshold with the same sign (no boundary exists at all --
    the whole circle is one cluster) as a special case first. NaN entries
    (an hour whose null standard deviation was zero, see
    `cluster_permutation`) never exceed the threshold and are simply
    excluded, like any other non-exceeding hour.
    """
    n = len(statistic)
    active = np.abs(statistic) > threshold
    sign = np.sign(statistic)
    if not active.any():
        return []
    if active.all() and len(set(sign)) == 1:
        return [{"hours": list(range(n)), "mass": float(statistic.sum()), "sign": int(sign[0])}]

    rotate_at = next(
        i for i in range(n)
        if not (active[i] and active[i - 1] and sign[i] == sign[i - 1])
    )
    order = [(rotate_at + offset) % n for offset in range(n)]
    rotated_active = active[order]
    rotated_sign = sign[order]
    rotated_stat = statistic[order]

    runs = []
    start = None
    for i in range(n):
        if rotated_active[i] and start is None:
            start = i
        elif rotated_active[i] and rotated_sign[i] != rotated_sign[start]:
            runs.append((start, i - 1))
            start = i
        elif not rotated_active[i] and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, n - 1))

    clusters = []
    for s, e in runs:
        hours = sorted(order[i] for i in range(s, e + 1))
        mass = float(rotated_stat[s:e + 1].sum())
        clusters.append({"hours": hours, "mass": mass, "sign": int(rotated_sign[s])})
    return clusters


def cluster_permutation(hourly: pd.DataFrame, threshold: float = CLUSTER_FORMING_THRESHOLD,
                        draws: int | None = None) -> dict:
    """Cage-stratified cluster-based permutation test over the 24-hour profile.

    `hourly` is `hourly_profile_by_animal`'s output (or any frame with the
    same shape): one row per animal, columns "animal_id", "cage", "genotype",
    and one column per clock hour (0..23), each animal's value already
    averaged across days. An animal missing a value at ANY hour is dropped
    entirely (count kept in the returned dict's `n_animals_dropped`), so
    every hour compares the SAME set of animals under the SAME cage labels --
    without that, a permutation's relabelling would not mean the same thing
    at every hour.

    **Per-hour statistic.** The observed difference at each hour is
    `_difference_from_labels` -- the SAME weighted within-cage statistic
    `stratified_difference` uses, weight = n_mut*n_wt/(n_mut+n_wt) per cage
    -- called once per hour, so the primary test's statistic is reused, not
    reimplemented. The null distribution (below) uses
    `_weighted_hourly_difference`, the identical formula generalised to all
    24 hours in one array operation, which is what makes enumerating
    thousands of permutations affordable; a test checks the two produce the
    same number for a spot check.

    **Cluster-forming statistic and threshold -- both DECLARED, not tuned to
    the data.** Each hour's observed difference is standardised into a
    z-like statistic using the MEAN and STANDARD DEVIATION of that hour's own
    difference across the null (within-cage label permutation) distribution:
    `z_h = (observed_h - null_mean_h) / null_sd_h`. Both are computed ONCE
    from the same permutation draws used for the cluster-mass null below,
    then applied identically to the observed data and to every permutation's
    own recomputed differences when re-clustering it -- so an hour with more
    between-animal noise cannot spuriously dominate a cluster just because
    its raw difference happens to be numerically larger. The cluster-forming
    threshold defaults to `CLUSTER_FORMING_THRESHOLD` = 2.0 (an hour must
    stand about 2 null standard deviations from its own null mean to seed a
    cluster); pass `threshold` to use a different one. An hour whose null
    standard deviation is exactly 0 (every permutation ties there, e.g. a
    cage-zero-weight-only hour) gets z = NaN and can never exceed threshold.

    **Circularity.** Clock hours are CIRCULAR -- hour 23 is adjacent to hour
    0 -- so a cluster may wrap midnight; `_find_clusters` handles this
    explicitly (see its docstring and the module's wrap tests).

    **Null.** Within-cage label permutation, exactly as
    `exact_stratified_permutation` enumerates it (the same per-cage
    combinations of which animals are labelled "mut", via `_label_options`):
    for every permutation, all 24 hourly differences are recomputed, the
    SAME threshold re-clusters them, and the MAXIMUM absolute cluster mass
    (0.0 if no cluster forms) is recorded. This design's real-study
    composition admits exactly 4800 within-cage label assignments -- small
    enough to enumerate EXACTLY -- so this enumerates exactly whenever the
    design admits at most `CLUSTER_ENUMERATION_CAP` assignments and `draws`
    is not given; above that cap, or whenever `draws` is given explicitly,
    `draws` permutations (default `CLUSTER_PERMUTATION_DRAWS`) are SAMPLED
    using `neu_oldenlabs.RNG_SEED`, so two calls on the same input are
    identical. The returned dict's `exact` flag and `n_permutations` state
    which was done and how many.

    **p-value.** For each OBSERVED cluster, p = (1 + the number of
    permutations whose maximum absolute cluster mass is >= that cluster's
    own |mass|) / (1 + n_permutations) -- ties counted as extreme, so p can
    never be exactly 0 and never falls below 1/(n_permutations+1). Taking
    the MAXIMUM over clusters (and implicitly over all 24 hours) on every
    permutation is what corrects for testing 24 hours -- and however many
    clusters happen to form -- at once, exactly as the max-statistic step
    does in Maris & Oostenveld (2007).

    Returns a dict: `clusters` (list of {"hours", "sign", "mass", "p"},
    sorted by ascending p), `n_permutations`, `exact` (bool), `threshold`,
    `n_animals`, `n_animals_dropped_missing_hour`, `observed_difference` (raw
    24-length per-hour difference), `observed_z`, `null_mean`, `null_sd`
    (the 24-length standardisation actually used).
    """
    frame = hourly.copy()
    unexpected = sorted(set(frame["genotype"]) - {"mut", "wt"})
    if unexpected:
        raise ValueError(f"genotype must be 'mut' or 'wt', got {unexpected}")

    hour_columns = list(range(HOURS_PER_DAY))
    missing_columns = [h for h in hour_columns if h not in frame.columns]
    if missing_columns:
        raise ValueError(f"hourly profile is missing hour column(s) {missing_columns}")

    complete = frame.dropna(subset=hour_columns).reset_index(drop=True)
    n_dropped = len(frame) - len(complete)
    if complete.empty:
        raise ValueError("no animal has a value at every hour; nothing to test")

    weight_frame = complete[["cage", "genotype"]].copy()
    weight_frame["value"] = 1.0
    weights = cage_weights(weight_frame).set_index("cage")["weight"].to_dict()
    if not any(weight > 0 for weight in weights.values()):
        raise ValueError("no cage contains both genotypes; nothing to compare")

    cage_arrays: dict = {}
    observed_labels: dict = {}
    for cage, block in complete.groupby("cage", sort=True):
        cage_arrays[cage] = block[hour_columns].to_numpy(dtype=float)
        observed_labels[cage] = (block["genotype"] == "mut").to_numpy()

    observed_diff = np.array([
        _difference_from_labels(
            {cage: array[:, h] for cage, array in cage_arrays.items()},
            observed_labels, weights)
        for h in hour_columns
    ])

    cages = sorted(cage_arrays)
    per_cage_options = []
    total = 1
    for cage in cages:
        n = cage_arrays[cage].shape[0]
        k = int(observed_labels[cage].sum())
        options = _label_options(n, k)
        per_cage_options.append(options)
        total *= len(options)

    if draws is None:
        exact = total <= CLUSTER_ENUMERATION_CAP
        n_permutations = total if exact else CLUSTER_PERMUTATION_DRAWS
    else:
        exact = False
        n_permutations = draws

    if exact:
        def _label_iter():
            for combination in product(*per_cage_options):
                yield dict(zip(cages, combination))
    else:
        rng = np.random.default_rng(RNG_SEED)

        def _label_iter():
            for _ in range(n_permutations):
                labels = {}
                for cage in cages:
                    n = cage_arrays[cage].shape[0]
                    k = int(observed_labels[cage].sum())
                    picked = rng.choice(n, size=k, replace=False)
                    mask = np.zeros(n, dtype=bool)
                    mask[picked] = True
                    labels[cage] = mask
                yield labels

    null_diffs = np.empty((n_permutations, HOURS_PER_DAY))
    for index, labels in enumerate(_label_iter()):
        null_diffs[index] = _weighted_hourly_difference(cage_arrays, labels, weights)

    null_mean = null_diffs.mean(axis=0)
    null_sd = null_diffs.std(axis=0, ddof=1) if n_permutations > 1 else np.zeros(HOURS_PER_DAY)
    null_sd_safe = np.where(null_sd > 0, null_sd, np.nan)

    observed_z = (observed_diff - null_mean) / null_sd_safe
    null_z = (null_diffs - null_mean) / null_sd_safe

    observed_clusters = _find_clusters(observed_z, threshold)

    null_max_abs_mass = np.empty(n_permutations)
    for index in range(n_permutations):
        row_clusters = _find_clusters(null_z[index], threshold)
        null_max_abs_mass[index] = max((abs(c["mass"]) for c in row_clusters), default=0.0)

    clusters = []
    for cluster in observed_clusters:
        extreme = int(np.sum(null_max_abs_mass >= abs(cluster["mass"]) - 1e-12))
        p = (extreme + 1) / (n_permutations + 1)
        clusters.append({"hours": cluster["hours"], "sign": cluster["sign"],
                         "mass": cluster["mass"], "p": p})
    clusters.sort(key=lambda c: c["p"])

    return {
        "clusters": clusters,
        "n_permutations": n_permutations,
        "exact": exact,
        "threshold": threshold,
        "n_animals": len(complete),
        "n_animals_dropped_missing_hour": n_dropped,
        "observed_difference": observed_diff,
        "observed_z": observed_z,
        "null_mean": null_mean,
        "null_sd": null_sd,
    }


def _format_hour_range(hours: list) -> str:
    """Render a cluster's hour list as contiguous ranges, wrap-aware.

    e.g. [2, 3, 4, 5, 6] -> "2-6"; [0, 1, 22, 23] -> "22-23,0-1" (the wrap is
    rendered as two clock-order pieces, not as the numerically sorted, and
    misleading, "0-1,22-23" or "22-1").
    """
    if not hours:
        return ""
    ordered = sorted(hours)
    if len(ordered) == HOURS_PER_DAY:
        return "0-23 (all hours)"

    # Rotate to start right after the single largest gap (the true "break"
    # in the circular run), so the walk below never has to cross the wrap
    # itself -- the wrap, if any, is then represented as a value going from
    # 23 down to 0 inside one contiguous walk.
    gaps = [(ordered[i + 1] - ordered[i], i) for i in range(len(ordered) - 1)]
    wrap_gap = (ordered[0] + HOURS_PER_DAY - ordered[-1], len(ordered) - 1)
    _, split_index = max(gaps + [wrap_gap], key=lambda item: item[0]) if gaps else wrap_gap
    rotated = ordered[split_index + 1:] + ordered[:split_index + 1]

    ranges = []
    start = prev = rotated[0]
    for hour in rotated[1:]:
        if hour == (prev + 1) % HOURS_PER_DAY:
            prev = hour
            continue
        ranges.append((start, prev))
        start = prev = hour
    ranges.append((start, prev))

    parts = []
    for a, b in ranges:
        if a == b:
            parts.append(f"{a}")
        elif a <= b:
            parts.append(f"{a}-{b}")
        else:  # wraps past midnight
            parts.append(f"{a}-23,0-{b}")
    return ",".join(parts)


def cluster_permutation_scan(long: pd.DataFrame, measures: list, *,
                             threshold: float = CLUSTER_FORMING_THRESHOLD,
                             draws: int | None = None) -> pd.DataFrame:
    """Run `cluster_permutation` over every measure, one row per observed
    cluster, BH-corrected ACROSS MEASURES.

    The within-measure correction across the 24 hours is already handled
    inside `cluster_permutation` itself (the max-cluster-mass step); the BH
    correction here is layered ACROSS the measures scanned, exactly like
    `scan`'s `p_adjusted_bh` -- a different multiplicity axis, so both
    corrections are needed and neither substitutes for the other.

    `long` is the rack-masked (and habituation/window-restricted) bin-level
    frame joined with genotype, as `hourly_profile_by_animal` expects. A
    measure absent from `long`, with no cage housing both genotypes, or that
    raised for any other data reason is recorded in the returned frame's
    `skipped_measures` attribute rather than raising, mirroring `scan`. A
    measure with zero observed clusters (nothing exceeded the threshold)
    contributes no row either, also recorded as skipped.
    """
    rows = []
    skipped = []
    columns = ["measure", "hours", "hour_list", "n_hours", "sign", "mass", "p_value",
              "n_permutations", "exact", "threshold"]
    for measure in measures:
        if measure not in set(long.get("measure", [])):
            skipped.append(f"{measure} (absent)")
            continue
        hourly = hourly_profile_by_animal(long, measure)
        try:
            result = cluster_permutation(hourly, threshold=threshold, draws=draws)
        except ValueError as error:
            skipped.append(f"{measure} ({error})")
            continue
        if not result["clusters"]:
            skipped.append(f"{measure} (no cluster exceeded the threshold)")
            continue
        for cluster in result["clusters"]:
            rows.append({
                "measure": measure,
                "hours": _format_hour_range(cluster["hours"]),
                # `hour_list` keeps the raw list of integer hours (not just
                # the compressed display string above) so a programmatic
                # consumer -- `plots.hour_clusters`, which shades exactly
                # these hours on the profile figure -- does not have to
                # re-parse the display string to recover them.
                "hour_list": cluster["hours"],
                "n_hours": len(cluster["hours"]),
                "sign": cluster["sign"],
                "mass": cluster["mass"],
                "p_value": cluster["p"],
                "n_permutations": result["n_permutations"],
                "exact": result["exact"],
                "threshold": threshold,
            })
    if not rows:
        table = pd.DataFrame(columns=columns + ["p_adjusted_bh", "skipped_measures"])
        table.attrs["skipped_measures"] = skipped
        return table
    table = pd.DataFrame(rows, columns=columns)
    table["p_adjusted_bh"] = benjamini_hochberg(table["p_value"].to_numpy())
    table["skipped_measures"] = "; ".join(skipped)
    return table.sort_values("p_value").reset_index(drop=True)


# ============================================================================
# SECONDARY, cage-ignoring ("pooled") comparison
# ============================================================================
#
# Added at the study owner's explicit request: "Can you also run analysis not
# within but all wildtypes and all mutants irrespectful of where they are
# coming from" -- i.e. pool all wildtype animals against all mutant animals,
# ignoring cage entirely.
#
# This is a DIFFERENT ESTIMAND from every function above, and a WEAKER one.
# Every function above permutes genotype labels only WITHIN a cage precisely
# because a cage is a confound -- cages differ in housing, cohort, and (for
# the dyadic measures) which genotypes the cage-mates carry, for reasons that
# have nothing to do with genotype. The functions below discard that
# safeguard on purpose: they pool every animal's value into one wildtype mean
# and one mutant mean, cage identity dropped, and so CANNOT separate a
# genotype effect from a cage effect. If a cage houses only one genotype (the
# real study's cage 58616: 5 wildtype, 0 mutant), every idiosyncrasy of that
# cage -- housing, cohort, whatever -- is baked into the wildtype mean here
# with nothing to net it against, and a pooled "significant" result may be
# entirely a cage effect wearing a genotype label. This is a SECONDARY,
# hypothesis-generating comparison, never a replacement for the cage-
# stratified analysis above; read it only alongside the stratified result,
# and prefer the stratified one wherever they disagree.
#
# Every name below is prefixed `pooled_` (or, for the scan table, every new
# column is prefixed `pooled_`) so a pooled row or column can never be
# confused with a stratified one when the two tables sit side by side.


def pooled_difference(values: pd.DataFrame) -> float:
    """mean(mut) - mean(wt) over every animal in `values`, cage ignored entirely.

    This is NOT `stratified_difference`: it does not weight or centre by cage,
    so a cage effect confounded with genotype (e.g. an all-wildtype cage that
    happens to run high or low) shifts this number exactly as if it were a
    genotype effect. See the module note above.
    """
    frame = _split(values)
    mut = frame.loc[frame["genotype"] == "mut", "value"].to_numpy(dtype=float)
    wt = frame.loc[frame["genotype"] == "wt", "value"].to_numpy(dtype=float)
    return float(mut.mean() - wt.mean())


def pooled_permutation(values: pd.DataFrame, draws: int = 100_000,
                       ) -> tuple[float, int, float]:
    """Monte Carlo (not exact) two-sided permutation test, cage ignored entirely.

    The real study's composition (15 wildtype, 10 mutant) admits C(25, 10) =
    3,268,760 label assignments -- too many to enumerate exactly for every
    measure in a multi-measure scan, so this SHUFFLES genotype labels across
    ALL animals (cage ignored) `draws` times instead of enumerating them.
    Seeded from `neu_oldenlabs.RNG_SEED`, so two calls on the same input
    produce identical output.

    Ties count as extreme (an exactly-equal shuffle is not evidence against
    the null) and p is computed as `(extreme + 1) / (draws + 1)` -- the
    observed labelling is itself always one of the "draws" that could occur,
    which is also why this can never read exactly 0. `min_attainable_p` is
    `1 / (draws + 1)`, the smallest value this Monte Carlo procedure can ever
    report, and is returned rather than assumed so a caller need not
    recompute it from `draws`.

    Like every function in this section, this ignores cage entirely: it
    cannot distinguish a genotype effect from a cage effect confounded with
    genotype. See the module note above.
    """
    frame = _split(values)
    array = frame["value"].to_numpy(dtype=float)
    is_mut = (frame["genotype"] == "mut").to_numpy()
    n_mut = int(is_mut.sum())
    n = len(array)
    if n_mut == 0 or n_mut == n:
        raise ValueError("pooled permutation needs at least one mut and one wt animal")

    observed = abs(float(array[is_mut].mean() - array[~is_mut].mean()))

    # Vectorised: one random value per (draw, animal) cell, argsort along the
    # animal axis gives `draws` independent random permutations of the animal
    # index at once -- a Python loop over 100k draws x 70 measures would be
    # far too slow (see the module docstring / spec).
    rng = np.random.default_rng(RNG_SEED)
    random_grid = rng.random((draws, n))
    order = np.argsort(random_grid, axis=1)
    shuffled = array[order]
    mut_draws = shuffled[:, :n_mut]
    wt_draws = shuffled[:, n_mut:]
    diffs = np.abs(mut_draws.mean(axis=1) - wt_draws.mean(axis=1))

    extreme = int(np.sum(diffs >= observed - 1e-12))
    p_value = (extreme + 1) / (draws + 1)
    min_attainable_p = 1.0 / (draws + 1)
    return p_value, draws, min_attainable_p


def pooled_hedges_g(values: pd.DataFrame) -> float:
    """Conventional pooled-SD Hedges' g over every animal, cage ignored entirely.

    Deliberately NOT cage-centred -- unlike `hedges_g`, which restricts to
    contributing (genotype-mixed) cages and pools within-genotype variance
    only across those, this pools every mut and every wt animal directly.
    That is the whole point of this secondary analysis: it describes the
    conventional, cage-blind effect size an all-wildtype and all-mutant pool
    would report, cage confounds included. Uses the same small-sample
    correction (`1 - 3/(4N-9)`) as `hedges_g`.
    """
    frame = _split(values)
    mut = frame.loc[frame["genotype"] == "mut", "value"].to_numpy(dtype=float)
    wt = frame.loc[frame["genotype"] == "wt", "value"].to_numpy(dtype=float)
    if len(mut) < 2 or len(wt) < 2:
        return float("nan")  # a within-genotype variance needs at least 2 animals
    pooled_ss = (len(mut) - 1) * mut.var(ddof=1) + (len(wt) - 1) * wt.var(ddof=1)
    pooled_df = (len(mut) - 1) + (len(wt) - 1)
    pooled_sd = np.sqrt(pooled_ss / pooled_df)
    if pooled_sd == 0:
        return float("nan")
    n_total = len(mut) + len(wt)
    d = (mut.mean() - wt.mean()) / pooled_sd
    correction = 1 - 3 / (4 * n_total - 9)
    return float(d * correction)


def pooled_bootstrap_ci(values: pd.DataFrame, draws: int = 5000,
                        level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI for `pooled_difference`, resampling ANIMALS within
    genotype -- NOT cages, unlike `cluster_bootstrap_ci`. Seeded from
    `neu_oldenlabs.RNG_SEED`, so two calls on the same input are identical.

    Resampling animals directly (rather than cages, then animals within cage)
    is consistent with this analysis pooling cage away in the first place --
    there is no cage structure left here to resample. See the module note
    above for what that costs.
    """
    frame = _split(values)
    mut = frame.loc[frame["genotype"] == "mut", "value"].to_numpy(dtype=float)
    wt = frame.loc[frame["genotype"] == "wt", "value"].to_numpy(dtype=float)
    if len(mut) == 0 or len(wt) == 0:
        raise ValueError("pooled bootstrap needs at least one mut and one wt animal")

    rng = np.random.default_rng(RNG_SEED)
    mut_draws = rng.choice(mut, size=(draws, len(mut)), replace=True)
    wt_draws = rng.choice(wt, size=(draws, len(wt)), replace=True)
    diffs = mut_draws.mean(axis=1) - wt_draws.mean(axis=1)

    tail = (1 - level) / 2
    return float(np.quantile(diffs, tail)), float(np.quantile(diffs, 1 - tail))


def compare_pooled(profile: pd.DataFrame, measure: str,
                   bin_width_confound: bool = False) -> dict:
    """The pooled, cage-ignoring counterpart to `compare`.

    SECONDARY analysis, run at the study owner's explicit request -- see the
    module note above for what pooling away cage costs: this cannot separate
    a genotype effect from a cage effect, and an all-wildtype (or all-mutant)
    cage's idiosyncrasies land entirely in that genotype's pooled mean.

    Every column is prefixed `pooled_` (except `measure`, `n_mut`, `n_wt`,
    which name the same quantity as `compare`'s and are kept unprefixed so
    the two tables can be joined on them) precisely so a pooled row can never
    be confused with a stratified one when the two sit side by side.

    `bin_width_confound` -- whether ANY loaded cage's native (pre-resampling)
    bin width differs from another's in this study (see
    `qc.distance_resolution_confound`) -- is a study-level fact the caller
    (`pipeline.run_study`) passes in; this function only decides whether THIS
    measure is one of the ones the confound taints
    (`io.DISTANCE_DERIVED_PROFILE_MEASURES`). `distance_cm` is a path length,
    which is sampling-rate dependent (the coastline problem), so a POOLED
    comparison of a distance-derived measure across cages of different native
    resolution is not safe even though the within-cage (stratified) `compare`
    above is unaffected -- `pooled_distance_bin_width_confound` marks exactly
    that row, never the stratified table (flagging the stratified table would
    be wrong: it is unaffected by construction).
    """
    values = profile[["animal_id", "cage", "genotype", measure]].rename(
        columns={measure: "value"}).dropna(subset=["value"])
    frame = _split(values)
    n_mut = int((frame["genotype"] == "mut").sum())
    n_wt = int((frame["genotype"] == "wt").sum())
    if n_mut == 0 or n_wt == 0:
        raise ValueError(
            f"{measure}: pooled comparison needs at least one mut and one wt animal")

    p_value, draws, floor = pooled_permutation(values)
    low, high = pooled_bootstrap_ci(values)

    row = {
        "measure": measure,
        "n_mut": n_mut,
        "n_wt": n_wt,
        "pooled_mean_mut": float(frame.loc[frame["genotype"] == "mut", "value"].mean()),
        "pooled_mean_wt": float(frame.loc[frame["genotype"] == "wt", "value"].mean()),
        "pooled_difference": pooled_difference(values),
        "pooled_ci_low": low, "pooled_ci_high": high,
        "pooled_hedges_g": pooled_hedges_g(values),
        "pooled_p_value": p_value, "pooled_draws": draws,
        "pooled_min_attainable_p": floor,
        "pooled_distance_bin_width_confound":
            bool(bin_width_confound and measure in DISTANCE_DERIVED_PROFILE_MEASURES),
    }
    assert list(row.keys()) == _COMPARE_POOLED_COLUMNS, (
        "_COMPARE_POOLED_COLUMNS (the hand-maintained duplicate used to give an "
        "all-skipped pooled scan the right shape) has drifted out of sync with the "
        f"keys compare_pooled() actually returns: {list(row.keys())} != "
        f"{_COMPARE_POOLED_COLUMNS}")
    return row


# The columns `compare_pooled` produces, in order -- kept alongside
# `scan_pooled` so an all-skipped pooled scan can still return a frame with
# the right shape. `compare_pooled` asserts its own return keys match this
# list on every call, so the two cannot silently drift apart. Every column
# past `n_wt` is prefixed `pooled_` so it can never be confused with a
# stratified column when the two tables sit side by side.
_COMPARE_POOLED_COLUMNS = [
    "measure", "n_mut", "n_wt",
    "pooled_mean_mut", "pooled_mean_wt", "pooled_difference",
    "pooled_ci_low", "pooled_ci_high", "pooled_hedges_g",
    "pooled_p_value", "pooled_draws", "pooled_min_attainable_p",
    "pooled_distance_bin_width_confound",
]


def scan_pooled(profile: pd.DataFrame, measures: list[str],
                bin_width_confound: bool = False) -> pd.DataFrame:
    """Run `compare_pooled` over every measure and rank by pooled p-value.

    Mirrors `scan`'s shape and skip behaviour exactly (absent/constant/no-
    contributing-animals measures are skipped with a reason, not silently
    dropped), but for the SECONDARY, cage-ignoring comparison -- see the
    module note above. BH correction is applied over the same scanned set,
    in its own `pooled_p_adjusted_bh` column so it cannot be confused with
    the stratified scan's `p_adjusted_bh`.

    `bin_width_confound` is passed straight through to `compare_pooled` for
    every measure -- see its docstring: a study-level fact (do any two loaded
    cages differ in native bin width?) that `compare_pooled` turns into a
    per-row, per-measure flag via `io.DISTANCE_DERIVED_PROFILE_MEASURES`.
    """
    rows, skipped = [], []
    for measure in measures:
        if measure not in profile.columns:
            skipped.append(f"{measure} (absent)")
            continue
        if profile[measure].nunique(dropna=True) <= 1:
            skipped.append(f"{measure} (constant)")
            continue
        try:
            rows.append(compare_pooled(profile, measure, bin_width_confound))
        except ValueError as error:
            skipped.append(f"{measure} ({error})")
    if not rows:
        table = pd.DataFrame(
            columns=_COMPARE_POOLED_COLUMNS + ["pooled_p_adjusted_bh", "skipped_measures"])
        table.attrs["skipped_measures"] = skipped
        return table
    table = pd.DataFrame(rows)
    table["pooled_p_adjusted_bh"] = benjamini_hochberg(table["pooled_p_value"].to_numpy())
    table["skipped_measures"] = "; ".join(skipped)
    return table.sort_values("pooled_p_value").reset_index(drop=True)
