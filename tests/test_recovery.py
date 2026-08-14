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


def test_trigger_locked_basis_reconstructs_variable_harmonic_artifact():
    sampling_frequency_hz = 100.0
    repetition_time_s = 1.0
    cycle_samples = int(sampling_frequency_hz * repetition_time_s)
    cycle_count = 20
    times_s = np.arange(cycle_count * cycle_samples) / sampling_frequency_hz
    trigger_samples = np.arange(cycle_count) * cycle_samples
    cycle_index = np.repeat(np.arange(cycle_count), cycle_samples)
    amplitude = 1.5 + 0.4 * np.sin(2.0 * np.pi * cycle_index / 7.0)
    quadrature = 0.3 * np.cos(2.0 * np.pi * cycle_index / 5.0)
    scanner_artifact = (
        amplitude * np.sin(2.0 * np.pi * 10.0 * times_s)
        + quadrature * np.cos(2.0 * np.pi * 20.0 * times_s)
    )
    neural = 0.5 * np.sin(2.0 * np.pi * 7.3 * times_s + 0.2)
    data = np.vstack((neural + scanner_artifact, neural - scanner_artifact))

    result = recovery.subtract_trigger_locked_optimal_basis(
        data,
        sampling_frequency_hz,
        (10.0, 20.0),
        trigger_samples,
        repetition_time_s=repetition_time_s,
        component_count=2,
    )

    np.testing.assert_allclose(
        result.cleaned_data + result.artifact_data,
        data,
        rtol=0.0,
        atol=1e-14,
    )
    before = _sinusoid_amplitude(data, sampling_frequency_hz, 10.0)
    after = _sinusoid_amplitude(
        result.cleaned_data,
        sampling_frequency_hz,
        10.0,
    )
    assert np.all(after < 0.05 * before)


def test_trigger_locked_basis_requires_exact_trigger_lattice():
    with pytest.raises(ValueError, match="configured repetition time"):
        recovery.subtract_trigger_locked_optimal_basis(
            np.zeros((2, 500)),
            100.0,
            (10.0,),
            (0, 100, 201, 300),
            repetition_time_s=1.0,
            component_count=1,
        )


def test_trigger_locked_basis_covers_samples_before_first_trigger():
    sampling_frequency_hz = 100.0
    times_s = np.arange(450) / sampling_frequency_hz
    data = np.sin(2.0 * np.pi * 10.0 * times_s)[np.newaxis, :]

    result = recovery.subtract_trigger_locked_optimal_basis(
        data,
        sampling_frequency_hz,
        (10.0,),
        (50, 150, 250, 350),
        repetition_time_s=1.0,
        component_count=1,
    )

    assert np.linalg.norm(result.artifact_data[:, :50]) > 0.0


def test_recursive_trajectory_pca_reconstructs_change_and_reduces_target():
    sampling_frequency_hz = 100.0
    times_s = np.arange(int(40.0 * sampling_frequency_hz)) / sampling_frequency_hz
    neural = 0.4 * np.sin(2.0 * np.pi * 7.3 * times_s + 0.1)
    artifact = 1.5 * np.sin(2.0 * np.pi * 20.0 * times_s + 0.4)
    data = np.vstack((neural + artifact, 0.5 * neural - artifact))
    settings = recovery.TrajectoryPCASettings(segment_s=2.0)

    result = recovery.subtract_recursive_trajectory_pca(
        data,
        sampling_frequency_hz,
        (20.0,),
        settings,
    )

    np.testing.assert_allclose(
        result.cleaned_data + result.artifact_data,
        data,
        rtol=0.0,
        atol=1e-14,
    )
    before = _sinusoid_amplitude(data, sampling_frequency_hz, 20.0)
    after = _sinusoid_amplitude(
        result.cleaned_data,
        sampling_frequency_hz,
        20.0,
    )
    assert np.all(after < 0.1 * before)


def test_recursive_trajectory_pca_requires_one_complete_segment():
    settings = recovery.TrajectoryPCASettings(segment_s=2.0)

    with pytest.raises(ValueError, match="shorter than one trajectory segment"):
        recovery.subtract_recursive_trajectory_pca(
            np.zeros((2, 100)),
            100.0,
            (20.0,),
            settings,
        )
