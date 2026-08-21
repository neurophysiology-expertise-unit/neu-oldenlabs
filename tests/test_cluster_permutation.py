"""The cage-stratified, cluster-based permutation test over the 24-hour profile.

Two levels of fixture:

* `hourly_study_values` builds `cluster_permutation`'s expected input directly
  (an animal x hour wide table with cage/genotype identity, one value per
  animal per hour already) -- used for tests that exercise the statistical
  machinery itself (clustering, the null, p-values), where the study's real
  cage composition matters but the bin-level averaging step does not.
* `long_frame_from_wide` expands a wide table like that into a synthetic
  bin-level long frame (cage, tag, genotype, measure, bin_start, duration_s,
  value) across several calendar days with day-to-day jitter that averages
  back to the wide table's values -- used for the pseudoreplication guard,
  which is specifically about the across-day averaging step in
  `hourly_profile_by_animal` / `metrics.hourly_by_animal`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neu_oldenlabs.groups import (HOURS_PER_DAY, _difference_from_labels, _find_clusters,
                                  _format_hour_range, _weighted_hourly_difference, cage_weights,
                                  cluster_permutation, cluster_permutation_scan,
                                  hourly_profile_by_animal)

# The real study composition (see test_groups.py's COMPOSITION docstring):
# 58616 is 5 wt / 0 mut, 58623 and 66336 are 2 wt / 3 mut, 58627 is 3 wt / 1
# mut, 62923 is 1 wt / 1 mut, 58597 is 2 wt / 2 mut. Reused here so the
# cluster test runs against the same design the whole-day test does -- 4800
# within-cage assignments, one all-wildtype (zero-weight) cage included.
COMPOSITION = {58616: (5, 0), 58623: (2, 3), 66336: (2, 3),
               58627: (3, 1), 62923: (1, 1), 58597: (2, 2)}


def hourly_study_values(effect=0.0, effect_hours=range(24), seed=0, cage_offset_sd=10.0,
                        hour_shape_sd=2.0, noise_sd=1.0):
    """An animal x hour wide table: columns "animal_id", "cage", "genotype", 0..23.

    Every animal in a cage shares that cage's constant offset (tests cage-
    offset immunity, exactly like `test_groups.study_values`) and a genuine
    circadian-like hour shape (so a real, non-flat 24-hour profile is always
    present, genotype effect or not); `effect` is added to mutant animals'
    values at every hour in `effect_hours` only.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cage, (n_wt, n_mut) in COMPOSITION.items():
        cage_offset = rng.normal(0, cage_offset_sd)
        hour_shape = rng.normal(0, hour_shape_sd, HOURS_PER_DAY)
        for genotype, count in [("wt", n_wt), ("mut", n_mut)]:
            for index in range(count):
                values = cage_offset + hour_shape + rng.normal(0, noise_sd, HOURS_PER_DAY)
                if genotype == "mut":
                    values = values.copy()
                    for hour in effect_hours:
                        values[hour] += effect
                row = {"animal_id": f"{cage}-{genotype}{index}", "cage": cage,
                       "genotype": genotype}
                row.update({hour: values[hour] for hour in range(HOURS_PER_DAY)})
                rows.append(row)
    return pd.DataFrame(rows)


def long_frame_from_wide(wide: pd.DataFrame, days: int = 3, seed: int = 0,
                         jitter_sd: float = 0.5, measure: str = "distance_cm") -> pd.DataFrame:
    """Expand a wide animal x hour table into a synthetic bin-level long frame.

    Each animal gets `days` calendar days of hourly bins; every day's value
    is the wide table's value plus independent jitter, so `days` days average
    back to (approximately) the wide table's own value, and the averaging
    step in `metrics.hourly_by_animal` is genuinely exercised rather than
    handed a single, already-final number. Each animal is assigned a stable
    per-cage `tag` (1, 2, 3, ... within its cage, in the order it appears in
    `wide`) so it has a distinct (cage, tag) identity.
    """
    rng = np.random.default_rng(seed)
    counters: dict = {}
    animal_tag: dict = {}
    for animal_id, cage in zip(wide["animal_id"], wide["cage"]):
        counters[cage] = counters.get(cage, 0) + 1
        animal_tag[animal_id] = counters[cage]

    rows = []
    for _, animal in wide.iterrows():
        tag = animal_tag[animal["animal_id"]]
        for day in range(days):
            date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)
            jitter = rng.normal(0, jitter_sd, HOURS_PER_DAY)
            for hour in range(HOURS_PER_DAY):
                rows.append({
                    "cage": animal["cage"], "tag": tag, "genotype": animal["genotype"],
                    "measure": measure, "bin_start": date + pd.Timedelta(hours=hour),
                    "duration_s": 3600.0, "value": animal[hour] + jitter[hour],
                })
    return pd.DataFrame(rows)


# ============================================================================
# _find_clusters: the circular clustering primitive
# ============================================================================


def test_find_clusters_recovers_a_simple_contiguous_run():
    z = np.zeros(24)
    z[2:7] = 3.0
    clusters = _find_clusters(z, 2.0)
    assert len(clusters) == 1
    assert clusters[0]["hours"] == [2, 3, 4, 5, 6]
    assert clusters[0]["sign"] == 1
    assert clusters[0]["mass"] == pytest.approx(15.0)


def test_find_clusters_wraps_midnight_as_one_cluster_not_two():
    # THE circularity test: hours 22, 23, 0, 1 exceed the threshold. A naive
    # linear scan (or an unsafe "double the array" implementation -- see
    # `_find_clusters`'s docstring for the bug this caught during
    # development) would report this as two separate clusters ([22, 23] and
    # [0, 1]) or would double-count part of it; the correct answer is ONE
    # cluster covering all four hours.
    z = np.zeros(24)
    z[[22, 23, 0, 1]] = 3.0
    clusters = _find_clusters(z, 2.0)
    assert len(clusters) == 1
    assert clusters[0]["hours"] == [0, 1, 22, 23]
    assert clusters[0]["mass"] == pytest.approx(12.0)


def test_find_clusters_whole_circle_same_sign_is_one_cluster():
    z = np.full(24, 3.0)
    clusters = _find_clusters(z, 2.0)
    assert len(clusters) == 1
    assert clusters[0]["hours"] == list(range(24))
    assert clusters[0]["mass"] == pytest.approx(72.0)


def test_find_clusters_separates_two_non_adjacent_runs_of_opposite_sign():
    z = np.zeros(24)
    z[3:6] = 3.0
    z[15:18] = -3.0
    clusters = _find_clusters(z, 2.0)
    assert len(clusters) == 2
    hours = {tuple(c["hours"]) for c in clusters}
    assert (3, 4, 5) in hours
    assert (15, 16, 17) in hours


def test_find_clusters_nothing_exceeding_threshold_gives_no_clusters():
    z = np.full(24, 0.5)
    assert _find_clusters(z, 2.0) == []


def test_find_clusters_nan_never_exceeds_threshold():
    z = np.full(24, np.nan)
    assert _find_clusters(z, 2.0) == []


# ============================================================================
# _weighted_hourly_difference agrees with _difference_from_labels
# ============================================================================


def test_weighted_hourly_difference_matches_the_scalar_statistic():
    hourly = hourly_study_values(effect=4.0, effect_hours=range(2, 7), seed=1)
    hour_columns = list(range(HOURS_PER_DAY))
    cage_arrays, labels = {}, {}
    for cage, block in hourly.groupby("cage"):
        cage_arrays[cage] = block[hour_columns].to_numpy(dtype=float)
        labels[cage] = (block["genotype"] == "mut").to_numpy()
    weight_frame = hourly[["cage", "genotype"]].copy()
    weight_frame["value"] = 1.0
    weights = cage_weights(weight_frame).set_index("cage")["weight"].to_dict()

    vector = _weighted_hourly_difference(cage_arrays, labels, weights)
    for hour in (0, 3, 10, 23):
        scalar = _difference_from_labels(
            {cage: array[:, hour] for cage, array in cage_arrays.items()}, labels, weights)
        assert scalar == pytest.approx(vector[hour])


# ============================================================================
# cluster_permutation: end-to-end statistical behaviour
# ============================================================================


def test_a_planted_localised_effect_is_recovered_as_one_cluster_with_small_p():
    hourly = hourly_study_values(effect=4.0, effect_hours=range(2, 7), seed=1)
    result = cluster_permutation(hourly)
    assert result["exact"] is True
    assert result["n_permutations"] == 4800
    assert len(result["clusters"]) >= 1
    top = result["clusters"][0]
    assert set(top["hours"]) >= {2, 3, 4, 5, 6}
    assert top["sign"] == 1
    assert top["p"] < 0.01


def test_a_planted_effect_wrapping_midnight_is_recovered_as_one_cluster():
    hourly = hourly_study_values(effect=4.0, effect_hours=[22, 23, 0, 1], seed=1)
    result = cluster_permutation(hourly)
    matching = [c for c in result["clusters"] if set(c["hours"]) >= {22, 23, 0, 1}]
    assert len(matching) == 1
    cluster = matching[0]
    assert cluster["hours"] == [0, 1, 22, 23]
    assert cluster["p"] < 0.01
    # No OTHER cluster independently covers the same wrapped hours -- the
    # bug this guards against split the wrap into two reported clusters.
    assert sum(1 for c in result["clusters"] if 22 in c["hours"] or 23 in c["hours"]
              or 0 in c["hours"] or 1 in c["hours"]) == 1


def test_large_cage_offsets_with_no_genotype_effect_are_not_significant():
    # The same confound-immunity property the primary (whole-day) test has:
    # huge, genotype-independent cage offsets must not manufacture a
    # significant cluster. seed=1 was chosen by scanning seeds 0-9 with
    # effect=0.0 and a large cage_offset_sd for a comfortably non-significant
    # draw (min p was 1.0, i.e. zero clusters even exceeded the threshold).
    hourly = hourly_study_values(effect=0.0, seed=1, cage_offset_sd=5000.0)
    result = cluster_permutation(hourly)
    assert all(c["p"] >= 0.05 for c in result["clusters"])


def test_cage_offsets_alone_cannot_move_the_observed_statistic():
    # Deterministic companion to the null-seed spot check above: adding a
    # constant to every animal in a cage leaves every within-cage difference
    # (and so every hourly cluster statistic) unchanged, for any seed --
    # mirrors test_groups.test_cage_offsets_cannot_change_the_result.
    hourly = hourly_study_values(effect=3.0, effect_hours=range(2, 7), seed=7)
    offsets = {58616: 1000.0, 58623: -500.0, 66336: 250.0,
              58627: -75.0, 62923: 4000.0, 58597: -1234.5}
    shifted = hourly.copy()
    for hour in range(HOURS_PER_DAY):
        shifted[hour] = hourly[hour] + hourly["cage"].map(offsets)

    result = cluster_permutation(hourly)
    result_shifted = cluster_permutation(shifted)
    assert len(result["clusters"]) == len(result_shifted["clusters"])
    for a, b in zip(result["clusters"], result_shifted["clusters"]):
        assert a["hours"] == b["hours"]
        assert a["sign"] == b["sign"]
        # Large offsets (up to 4000) subtracted back out leave floating-point
        # noise at the ~1e-10 relative level, not exact equality.
        assert a["mass"] == pytest.approx(b["mass"], rel=1e-6)
        assert a["p"] == pytest.approx(b["p"], rel=1e-6)


def test_pseudoreplication_guard_duplicating_days_does_not_change_p():
    # Build bin-level data (not the pre-averaged wide table) so the
    # across-day averaging step in `hourly_profile_by_animal` /
    # `metrics.hourly_by_animal` is genuinely exercised, then duplicate every
    # animal's days onto new calendar dates with the SAME per-day jitter
    # pattern. Averaging happens before `cluster_permutation` ever sees the
    # data, so the resulting per-animal, per-hour values -- and therefore
    # every cluster's mass and p -- must be (near-)identical, not just
    # "smaller p from more apparent data", which is exactly the
    # pseudoreplication failure mode this guards against.
    wide = hourly_study_values(effect=4.0, effect_hours=range(2, 7), seed=1)
    long = long_frame_from_wide(wide, days=3, seed=2)
    long_duplicated = pd.concat(
        [long, long.assign(bin_start=long["bin_start"] + pd.Timedelta(days=3))],
        ignore_index=True)

    hourly_once = hourly_profile_by_animal(long, "distance_cm")
    hourly_twice = hourly_profile_by_animal(long_duplicated, "distance_cm")

    result_once = cluster_permutation(hourly_once)
    result_twice = cluster_permutation(hourly_twice)

    assert result_once["n_permutations"] == result_twice["n_permutations"]
    assert len(result_once["clusters"]) == len(result_twice["clusters"])
    for a, b in zip(result_once["clusters"], result_twice["clusters"]):
        assert a["hours"] == b["hours"]
        assert a["p"] == pytest.approx(b["p"], rel=1e-6)
        assert a["mass"] == pytest.approx(b["mass"], rel=1e-6)


def test_p_is_never_zero_and_never_below_the_floor():
    hourly = hourly_study_values(effect=50.0, effect_hours=range(2, 7), seed=1)
    result = cluster_permutation(hourly)
    floor = 1.0 / (result["n_permutations"] + 1)
    for cluster in result["clusters"]:
        assert cluster["p"] > 0.0
        assert cluster["p"] >= floor - 1e-12


def test_reproducibility_exact_branch():
    hourly = hourly_study_values(effect=4.0, effect_hours=range(2, 7), seed=1)
    first = cluster_permutation(hourly)
    second = cluster_permutation(hourly)
    assert first["clusters"] == second["clusters"]
    assert first["exact"] is True


def test_reproducibility_sampled_branch():
    # Force the Monte Carlo path (rather than the exact 4800-assignment
    # enumeration this composition would otherwise trigger) by passing
    # `draws` explicitly, and check two calls on the same data are
    # bit-for-bit identical -- seeded from `neu_oldenlabs.RNG_SEED`.
    hourly = hourly_study_values(effect=4.0, effect_hours=range(2, 7), seed=1)
    first = cluster_permutation(hourly, draws=300)
    second = cluster_permutation(hourly, draws=300)
    assert first["clusters"] == second["clusters"]
    assert first["exact"] is False
    assert first["n_permutations"] == 300


def test_an_animal_missing_a_single_hour_is_dropped_not_nan_propagated():
    hourly = hourly_study_values(effect=4.0, effect_hours=range(2, 7), seed=1)
    holed = hourly.copy()
    holed.loc[holed.index[0], 5] = np.nan
    result = cluster_permutation(holed)
    assert result["n_animals_dropped_missing_hour"] == 1
    assert result["n_animals"] == len(hourly) - 1


def test_unexpected_genotype_value_raises():
    hourly = hourly_study_values(effect=0.0, seed=1)
    bad = hourly.copy()
    bad.loc[bad.index[0], "genotype"] = "het"
    with pytest.raises(ValueError, match="genotype must be"):
        cluster_permutation(bad)


def test_no_cage_with_both_genotypes_raises():
    single_cage = pd.DataFrame([
        {"animal_id": "1-wt0", "cage": 1, "genotype": "wt",
         **{h: 1.0 for h in range(24)}},
        {"animal_id": "1-wt1", "cage": 1, "genotype": "wt",
         **{h: 1.0 for h in range(24)}},
    ])
    with pytest.raises(ValueError, match="no cage contains both genotypes"):
        cluster_permutation(single_cage)


# ============================================================================
# _format_hour_range
# ============================================================================


def test_format_hour_range_simple_contiguous_block():
    assert _format_hour_range([2, 3, 4, 5, 6]) == "2-6"


def test_format_hour_range_single_hour():
    assert _format_hour_range([9]) == "9"


def test_format_hour_range_wraps_midnight():
    assert _format_hour_range([0, 1, 22, 23]) == "22-23,0-1"


def test_format_hour_range_all_hours():
    assert _format_hour_range(list(range(24))) == "0-23 (all hours)"


# ============================================================================
# hourly_profile_by_animal
# ============================================================================


def test_hourly_profile_by_animal_has_one_row_per_animal_and_24_hour_columns():
    wide = hourly_study_values(effect=0.0, seed=1)
    long = long_frame_from_wide(wide, days=2, seed=3)
    table = hourly_profile_by_animal(long, "distance_cm")
    assert set(table.columns) == {"animal_id", "cage", "tag", "genotype", *range(24)}
    assert len(table) == len(wide)


def test_hourly_profile_by_animal_averages_across_days():
    # Two days with values 10 and 20 at every hour must average to 15 --
    # this is the pseudoreplication-prevention step itself, checked directly
    # (not just indirectly via the p-value guard above).
    rows = []
    for day, value in [(0, 10.0), (1, 20.0)]:
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)
        for hour in range(24):
            rows.append({"cage": 1, "tag": 1, "genotype": "wt", "measure": "distance_cm",
                        "bin_start": date + pd.Timedelta(hours=hour),
                        "duration_s": 3600.0, "value": value})
    long = pd.DataFrame(rows)
    table = hourly_profile_by_animal(long, "distance_cm")
    assert len(table) == 1
    for hour in range(24):
        assert table.iloc[0][hour] == pytest.approx(15.0)


# ============================================================================
# cluster_permutation_scan
# ============================================================================


def test_cluster_permutation_scan_bh_corrects_across_measures():
    wide = hourly_study_values(effect=4.0, effect_hours=range(2, 7), seed=1)
    long = long_frame_from_wide(wide, days=2, seed=4, measure="distance_cm")
    long2 = long_frame_from_wide(wide, days=2, seed=4, measure="social_distance_cm")
    combined = pd.concat([long, long2], ignore_index=True)

    table = cluster_permutation_scan(combined, ["distance_cm", "social_distance_cm"])
    assert "p_adjusted_bh" in table.columns
    assert set(table["measure"]) <= {"distance_cm", "social_distance_cm"}
    assert (table["p_adjusted_bh"] >= table["p_value"]).all()


def test_cluster_permutation_scan_skips_an_absent_measure():
    wide = hourly_study_values(effect=0.0, seed=1)
    long = long_frame_from_wide(wide, days=2, seed=5)
    table = cluster_permutation_scan(long, ["measure_not_present"])
    assert table.empty
    assert "measure_not_present (absent)" in table.attrs["skipped_measures"]
