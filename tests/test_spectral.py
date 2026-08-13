"""Deterministic spectral primitives used by line detection."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import periodogram

from decomb import spectral


def _tone(frequency_hz: float, sfreq: float, n_times: int) -> np.ndarray:
    times_s = np.arange(n_times) / sfreq
    return np.sin(2.0 * np.pi * frequency_hz * times_s)


def test_hann_periodogram_recovers_a_tone_and_leading_dimensions():
    sfreq = 500.0
    n_times = 10_800
    data = np.stack(
        (
            _tone(20.0, sfreq, n_times),
            _tone(40.0, sfreq, n_times),
        )
    )

    frequencies_hz, power = spectral.hann_periodogram(data, sfreq)

    assert power.shape == (2, frequencies_hz.size)
    assert frequencies_hz[int(np.argmax(power[0]))] == pytest.approx(
        20.0,
        abs=frequencies_hz[1],
    )


def test_hann_periodogram_rejects_non_finite_data():
    with pytest.raises(ValueError, match="finite"):
        spectral.hann_periodogram(np.array([1.0, np.nan, 2.0]), 500.0)


def test_hann_periodogram_removes_each_windows_dc_level():
    frequencies_hz, power = spectral.hann_periodogram(np.ones(128), 128.0)

    assert frequencies_hz.shape == power.shape
    np.testing.assert_array_equal(power, np.zeros_like(power))


def test_hann_periodogram_doubles_the_last_positive_bin_for_odd_sample_counts():
    sample_count = 9
    sampling_frequency_hz = 100.0
    data = np.random.default_rng(7).normal(size=sample_count)

    _, observed = spectral.hann_periodogram(data, sampling_frequency_hz)
    _, expected = periodogram(
        data,
        fs=sampling_frequency_hz,
        window=np.hanning(sample_count),
        detrend="constant",
        return_onesided=True,
        scaling="density",
    )

    assert np.allclose(observed, expected)
