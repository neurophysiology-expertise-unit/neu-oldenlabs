from __future__ import annotations

import numpy as np
import pandas as pd

from neu_oldenlabs.circadian import (cosinor, interdaily_stability, intradaily_variability,
                                     relative_amplitude)


def sine_series(days=10, amplitude=5.0, mesor=10.0, acrophase_hour=3.0):
    index = pd.date_range("2026-07-12", periods=24 * days, freq="h")
    radians = 2 * np.pi * (index.hour - acrophase_hour) / 24
    return pd.Series(mesor + amplitude * np.cos(radians), index=index)


def test_cosinor_recovers_a_known_amplitude_mesor_and_acrophase():
    fit = cosinor(sine_series())
    assert round(fit["mesor"], 6) == 10.0
    assert round(fit["amplitude"], 6) == 5.0
    assert round(fit["acrophase_hour"], 6) == 3.0


def test_interdaily_stability_is_one_for_a_perfectly_repeating_day():
    assert round(interdaily_stability(sine_series()), 6) == 1.0


def test_interdaily_stability_falls_when_days_differ():
    series = sine_series()
    rng = np.random.default_rng(0)
    noisy = series + rng.normal(0, 5, len(series))
    assert interdaily_stability(noisy) < interdaily_stability(series)


def test_intradaily_variability_is_higher_for_a_fragmented_rhythm():
    smooth = sine_series()
    fragmented = pd.Series(np.tile([0.0, 20.0], len(smooth) // 2), index=smooth.index)
    assert intradaily_variability(fragmented) > intradaily_variability(smooth)


def test_relative_amplitude_matches_a_hand_computed_block_profile():
    # Ten hours at 20, then fourteen at 0, repeated: M10 = 20, L5 = 0, RA = 1.
    day = [20.0] * 10 + [0.0] * 14
    index = pd.date_range("2026-07-12", periods=24 * 5, freq="h")
    series = pd.Series(day * 5, index=index)
    result = relative_amplitude(series)
    assert round(result["M10"], 6) == 20.0
    assert round(result["L5"], 6) == 0.0
    assert round(result["RA"], 6) == 1.0
