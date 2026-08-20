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
