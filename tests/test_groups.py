from __future__ import annotations

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from neu_oldenlabs.groups import (_COMPARE_COLUMNS, benjamini_hochberg, cage_weights,
                                  cluster_bootstrap_ci, compare, count_assignments,
                                  exact_stratified_permutation, hedges_g, mixed_model, scan,
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


def test_cage_offsets_cannot_change_the_result():
    # Adding a constant to every animal in a cage leaves every within-cage
    # difference unchanged, so the observed statistic and all 4800 permuted
    # statistics must be identical before and after -- deterministically, for
    # every seed, unlike a threshold on a single p-value draw.
    values = study_values(effect=2.0, seed=7)
    offsets = {58616: 1000.0, 58623: -500.0, 66336: 250.0,
               58627: -75.0, 62923: 4000.0, 58597: -1234.5}
    shifted = values.copy()
    shifted["value"] = values["value"] + values["cage"].map(offsets)

    assert stratified_difference(values) == pytest.approx(stratified_difference(shifted))

    p, n, floor = exact_stratified_permutation(values)
    p_shifted, n_shifted, floor_shifted = exact_stratified_permutation(shifted)
    assert p == pytest.approx(p_shifted)
    assert n == n_shifted
    assert floor == pytest.approx(floor_shifted)


def test_a_null_dataset_is_not_significant():
    # Under the null the p-value is uniformly distributed on [0, 1], so any
    # single seed is a spot check, not a guarantee -- a threshold like
    # `p > 0.2` fails for about one seed in five by construction. seed=5 was
    # chosen by scanning seeds 0-19 with effect=0.0 for a typical (non-
    # extreme) draw: it gives p = 0.4025, comfortably above 0.05.
    #
    # The floor here is 1/4800, not 2/4800: the floor is only 2/n when every
    # contributing cage is genotype-balanced (n_mut == n_wt), because only
    # then does the fully sign-flipped labelling stay a valid assignment and
    # guarantee a second tie. The real composition has three unbalanced
    # contributing cages (58623, 66336, 58627 are 2/3, 2/3, 3/1), so only the
    # observed assignment itself is guaranteed extreme.
    p, n, floor = exact_stratified_permutation(study_values(effect=0.0, seed=5))
    assert n == 4800
    assert round(floor, 6) == round(1 / 4800, 6)
    assert p > 0.05


def test_identical_values_in_every_cage_give_p_equal_to_one():
    # A deterministic null, unlike the seeded spot check above: if mut and wt
    # are literally the same number in every cage, every within-cage
    # difference is exactly 0 for the observed labelling AND for all 4800
    # permutations, so every permuted statistic ties the observed one and
    # p == 1.0 exactly, for any run, not just a lucky seed.
    identical = pd.DataFrame([
        {"animal_id": f"{cage}-{genotype}{index}", "cage": cage, "genotype": genotype,
         "value": 5.0}
        for cage, (n_wt, n_mut) in COMPOSITION.items()
        for genotype, count in [("wt", n_wt), ("mut", n_mut)]
        for index in range(count)
    ])
    p, n, floor = exact_stratified_permutation(identical)
    assert n == 4800
    assert p == 1.0
    assert floor == pytest.approx(1 / 4800)


def test_permutation_floor_is_two_over_n_when_every_contributing_cage_is_balanced():
    # The other branch of the floor logic: a single 2-mut/2-wt cage is
    # genotype-balanced, so the fully sign-flipped labelling is itself one of
    # the 6 enumerated assignments and the floor is 2/6, not 1/6.
    balanced = pd.DataFrame([
        {"animal_id": "a", "cage": 1, "genotype": "mut", "value": 2.0},
        {"animal_id": "b", "cage": 1, "genotype": "mut", "value": 3.0},
        {"animal_id": "c", "cage": 1, "genotype": "wt", "value": 1.0},
        {"animal_id": "d", "cage": 1, "genotype": "wt", "value": 0.0},
    ])
    p, n, floor = exact_stratified_permutation(balanced)
    assert n == 6
    assert floor == pytest.approx(2 / 6)


def test_a_planted_effect_is_detected():
    p, _, _ = exact_stratified_permutation(study_values(effect=4.0, seed=1))
    assert p < 0.01


def test_permutation_p_is_never_zero_because_the_observed_split_counts():
    p, n, _ = exact_stratified_permutation(study_values(effect=50.0, seed=2))
    assert p >= 1 / n


def test_bootstrap_interval_is_reproducible_and_brackets_the_observed_statistic():
    # A percentile bootstrap CI is centred on the OBSERVED statistic, not the
    # true planted effect, and covers the truth only ~95% of the time -- one
    # seed missing the planted value is expected behaviour, not a bug. What
    # the interval actually guarantees: it is reproducible, it brackets the
    # observed estimate it was built from, it has a sane low < high ordering,
    # and for a large true effect it excludes zero.
    values = study_values(effect=4.0, seed=1)
    first = cluster_bootstrap_ci(values, draws=2000)
    second = cluster_bootstrap_ci(values, draws=2000)
    assert first == second
    assert first[0] < first[1]
    observed = stratified_difference(values)
    assert first[0] < observed < first[1]
    assert first[0] > 0.0


def test_benjamini_hochberg_matches_a_hand_computed_example():
    # [0.01, 0.02, 0.03, 0.04] is a degenerate example: every step-up value
    # p_i * n / i comes out to exactly 0.04 for every i, so the cumulative-
    # minimum step -- the only non-trivial line in `benjamini_hochberg` -- is
    # never exercised; the test would pass identically with that line deleted.
    # Here the step-up values before the cumulative min are
    # [0.04, 0.04, 0.12, 0.10]: rank 3's 0.12 must be pulled down to rank 4's
    # smaller 0.10 by the cumulative-min-from-the-largest-p-value step, which
    # is the only place that step does anything.
    raw = np.array([0.01, 0.02, 0.09, 0.10])
    adjusted = benjamini_hochberg(raw)
    assert np.allclose(adjusted, [0.04, 0.04, 0.10, 0.10])


def test_compare_returns_a_row_with_counts_effect_and_interval():
    values = study_values(effect=3.0, seed=5)
    profile = values.rename(columns={"value": "distance_cm_all"})
    row = compare(profile, "distance_cm_all")
    assert row["measure"] == "distance_cm_all"
    assert row["n_mut"] == 10 and row["n_wt"] == 15
    # cage 58616 (5 wt, 0 mut) has weight 0 and contributes no within-cage
    # contrast; the *_contributing counts must reflect only the animals the
    # test actually used, not the totals across every loaded cage.
    assert row["n_mut_contributing"] == 10 and row["n_wt_contributing"] == 10
    assert row["cages_contributing"] == 5
    assert row["ci_low"] < row["difference"] < row["ci_high"]
    # 1/4800, not 2/4800 -- the real composition has unbalanced contributing
    # cages (58623, 66336, 58627), so the fully sign-flipped labelling is not
    # itself a valid assignment and only the observed one is guaranteed
    # extreme. See test_a_null_dataset_is_not_significant for the full
    # argument, and test_permutation_floor_is_two_over_n_when... for the
    # balanced branch.
    assert row["min_attainable_p"] == 1 / 4800


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


def test_hedges_g_ignores_a_zero_weight_cage_entirely():
    # cage 58616 (all wildtype) contributes no within-cage contrast and gets
    # weight 0 in the p-value machinery; hedges_g must be blind to it the
    # same way, or the effect size and the p-value describe different
    # quantities. Shifting AND inflating that cage's spread must not move
    # hedges_g at all.
    values = study_values(effect=3.0, seed=5)
    contaminated = values.copy()
    zero_weight_cage = contaminated["cage"] == 58616
    contaminated.loc[zero_weight_cage, "value"] = (
        contaminated.loc[zero_weight_cage, "value"] * 50.0 + 1000.0)
    assert hedges_g(values) == pytest.approx(hedges_g(contaminated))


def test_hedges_g_moves_in_step_with_the_stratified_difference():
    # The effect size is standardised stratified_difference, not a flat,
    # unweighted pool of every animal -- so a change that moves one but not
    # the other would mean they describe different quantities.
    values = study_values(effect=3.0, seed=5)
    bigger_effect = study_values(effect=8.0, seed=5)
    assert stratified_difference(bigger_effect) > stratified_difference(values)
    assert hedges_g(bigger_effect) > hedges_g(values)


def test_bootstrap_ci_width_is_on_the_within_cage_scale_not_the_offset_scale():
    # If resampling ignored cage structure (pooling mut and wt draws across
    # all cages instead of within cage, within genotype), a huge between-cage
    # offset would leak straight into the resampled means and blow the
    # interval up to the offset scale. Cage-stratified resampling can't see
    # it: the offset cancels inside every per-cage (mut - wt) term exactly as
    # it does for `stratified_difference`. Verified this discriminates: a
    # naive animal-level bootstrap (pool all mut, pool all wt, resample each
    # pool directly, ignoring cage) on this same shifted dataset gives a CI
    # roughly 20000 units wide; the cage-stratified one stays under 10.
    values = study_values(effect=2.0, seed=7)
    offsets = {58616: 1000.0, 58623: -5000.0, 66336: 2500.0,
               58627: -750.0, 62923: 40000.0, 58597: -12345.0}
    shifted = values.copy()
    shifted["value"] = values["value"] + values["cage"].map(offsets)

    low, high = cluster_bootstrap_ci(shifted, draws=2000)
    assert high - low < 10.0


def test_mixed_model_reports_convergence_failure_as_a_result():
    # statsmodels signals a failed or boundary fit by *warning*, not by
    # raising, so a bare try/except misses it. Forcing a ConvergenceWarning
    # during the fit (statsmodels does not reliably fail to converge on data
    # this small, so this is done explicitly rather than hunting for a
    # data-dependent trigger) must flip `converged` to False, put the warning
    # text in `note`, and must not let the warning escape to the caller.
    values = study_values(effect=3.0, seed=5)
    original_fit = MixedLM.fit

    def fit_and_warn(self, *args, **kwargs):
        warnings.warn("Maximum Likelihood optimization failed to converge.",
                       ConvergenceWarning)
        return original_fit(self, *args, **kwargs)

    with patch.object(MixedLM, "fit", fit_and_warn):
        with warnings.catch_warnings(record=True) as leaked:
            warnings.simplefilter("always")
            result = mixed_model(values)

    assert result["converged"] is False
    assert "converge" in result["note"].lower()
    assert leaked == []  # the warning must be caught, not sprayed to the caller


def test_mixed_model_note_is_empty_on_an_ordinary_converged_fit():
    values = study_values(effect=3.0, seed=5)
    result = mixed_model(values)
    assert result["converged"] is True
    assert result["note"] == ""


def test_mixed_model_is_not_marked_non_converged_by_an_unrelated_warning():
    # `warnings.simplefilter("always")` catches EVERY warning category during
    # the fit call, not just convergence ones. The old code treated the mere
    # presence of any caught warning as a convergence failure, so a single
    # FutureWarning from a library upgrade would mark every row of a scan as
    # non-converged and void the sensitivity analysis for reasons that have
    # nothing to do with whether the fit actually converged.
    values = study_values(effect=3.0, seed=5)
    original_fit = MixedLM.fit

    def fit_and_warn(self, *args, **kwargs):
        warnings.warn("some unrelated deprecation notice", FutureWarning)
        return original_fit(self, *args, **kwargs)

    with patch.object(MixedLM, "fit", fit_and_warn):
        with warnings.catch_warnings(record=True) as leaked:
            warnings.simplefilter("always")
            result = mixed_model(values)

    assert result["converged"] is True
    assert result["note"] == ""
    assert leaked == []  # still swallowed, so it doesn't spray across a scan


def test_compare_columns_constant_matches_compares_actual_return_keys():
    # `_COMPARE_COLUMNS` is a hand-maintained duplicate of compare()'s return
    # keys, kept so an all-skipped scan can still return a frame with the
    # right shape. `compare` asserts this on every call; this test pins it
    # down directly so a future edit that adds/renames/reorders a field in
    # one place without the other fails immediately and obviously.
    values = study_values(effect=3.0, seed=5)
    profile = values.rename(columns={"value": "distance_cm_all"})
    row = compare(profile, "distance_cm_all")
    assert list(row.keys()) == _COMPARE_COLUMNS


def test_scan_returns_expected_columns_when_every_measure_is_skipped():
    values = study_values(effect=3.0, seed=5)
    profile = values.rename(columns={"value": "distance_cm_all"})
    table = scan(profile, ["measure_not_in_profile"])
    assert list(table.columns).count("p_adjusted_bh") == 1
    assert list(table.columns).count("skipped_measures") == 1
    assert table.empty
    assert table.attrs["skipped_measures"] == ["measure_not_in_profile (absent)"]
