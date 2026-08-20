"""Tests for target-blind recovery of the 1.2 Hz pump fundamental."""

import numpy as np

from decomb import pump_recovery


def test_high_harmonics_are_consecutive_teeth_between_20_and_95_hz():
    assert pump_recovery.high_harmonic_numbers(1.2, 500.0) == tuple(range(17, 80))


def test_adjacent_features_do_not_read_the_exact_fundamental():
    sampling_frequency_hz = 1000.0
    times_s = np.arange(40_000) / sampling_frequency_hz
    background = np.stack(
        [
            np.sin(2.0 * np.pi * 20.4 * times_s)
            + 0.6 * np.sin(2.0 * np.pi * 21.6 * times_s + 0.2),
            np.cos(2.0 * np.pi * 20.4 * times_s)
            + 0.8 * np.cos(2.0 * np.pi * 21.6 * times_s - 0.3),
        ]
    )
    injection = 7.0 * np.sin(2.0 * np.pi * 1.2 * times_s + 0.4)
    bounds = ((0, 20_000), (20_000, 40_000))

    original = pump_recovery.extract_adjacent_features(
        background,
        sampling_frequency_hz,
        bounds,
        fundamental_hz=1.2,
    )
    injected = pump_recovery.extract_adjacent_features(
        background + injection,
        sampling_frequency_hz,
        bounds,
        fundamental_hz=1.2,
    )

    np.testing.assert_allclose(injected, original, rtol=0.0, atol=1e-12)
