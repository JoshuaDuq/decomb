"""Signal-preserving removal before residual FIR notching."""

from __future__ import annotations

import numpy as np
import pytest

from decomb import recovery


def test_recovery_frequencies_must_be_sorted_unique_and_below_nyquist():
    with pytest.raises(ValueError, match="sorted unique"):
        recovery.validate_frequencies((20.0, 10.0, 20.0), 100.0)

    with pytest.raises(ValueError, match="strictly below Nyquist"):
        recovery.validate_frequencies((10.0, 50.0), 100.0)


def test_recovery_result_requires_matching_finite_channel_data():
    clean = np.zeros((2, 100))

    with pytest.raises(ValueError, match="same two-dimensional shape"):
        recovery.SignalRecoveryResult(clean, np.zeros(100), (10.0,))

    artifact = clean.copy()
    artifact[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        recovery.SignalRecoveryResult(clean, artifact, (10.0,))


def test_recovery_result_preserves_valid_arrays_and_frequencies():
    clean = np.ones((2, 100))
    artifact = np.full((2, 100), 0.5)

    result = recovery.SignalRecoveryResult(clean, artifact, (10.0, 20.0))

    np.testing.assert_array_equal(result.cleaned_data, clean)
    np.testing.assert_array_equal(result.artifact_data, artifact)
    assert result.frequencies_hz == (10.0, 20.0)


def _sinusoid_amplitude(
    data: np.ndarray,
    sampling_frequency_hz: float,
    frequency_hz: float,
) -> np.ndarray:
    times_s = np.arange(data.shape[-1]) / sampling_frequency_hz
    basis = np.exp(-2j * np.pi * frequency_hz * times_s)
    return 2.0 * np.abs(data @ basis) / data.shape[-1]


def test_multitaper_subtraction_reconstructs_change_and_reduces_target():
    sampling_frequency_hz = 200.0
    times_s = np.arange(int(12.0 * sampling_frequency_hz)) / sampling_frequency_hz
    neural = np.sin(2.0 * np.pi * 7.0 * times_s)
    artifact = 2.0 * np.sin(2.0 * np.pi * 25.0 * times_s + 0.3)
    data = np.vstack((neural + artifact, 0.5 * neural - artifact))
    original = data.copy()

    result = recovery.subtract_multitaper_sinusoids(
        data,
        sampling_frequency_hz,
        (25.0,),
        window_s=4.0,
    )

    np.testing.assert_array_equal(data, original)
    np.testing.assert_allclose(
        result.cleaned_data + result.artifact_data,
        original,
        rtol=0.0,
        atol=1e-14,
    )
    before = _sinusoid_amplitude(original, sampling_frequency_hz, 25.0)
    after = _sinusoid_amplitude(
        result.cleaned_data,
        sampling_frequency_hz,
        25.0,
    )
    assert np.all(after < 0.1 * before)
    assert result.frequencies_hz == (25.0,)


def test_multitaper_subtraction_rejects_short_signal():
    with pytest.raises(ValueError, match="shorter than one recovery window"):
        recovery.subtract_multitaper_sinusoids(
            np.zeros((2, 100)),
            100.0,
            (10.0,),
            window_s=2.0,
        )






















# --- reconstruction cost -------------------------------------------------------------
#
# The published MATLAB ends with `rXbar = sig - sum(reconXbar(idx,:))`: it reconstructs only
# the components it removes. Summing the keepers instead is equivalent in result but costs a
# diagonal average for every component at every recursion level, which is what makes the
# method intractable at EEG sampling rates.














