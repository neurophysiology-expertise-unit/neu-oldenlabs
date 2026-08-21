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
