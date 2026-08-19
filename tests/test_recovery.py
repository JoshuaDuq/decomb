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


def test_spatial_line_subspace_removes_line_and_preserves_orthogonal_neural_signal():
    sampling_frequency_hz = 200.0
    times_s = np.arange(int(20.0 * sampling_frequency_hz)) / sampling_frequency_hz
    artifact_topography = np.array([1.0, 0.5, -0.25])
    artifact_topography /= np.linalg.norm(artifact_topography)
    neural_topography = np.array([0.5, -1.0, 0.0])
    neural_topography -= artifact_topography * (
        artifact_topography @ neural_topography
    )
    neural_topography /= np.linalg.norm(neural_topography)
    artifact = 4.0 * np.sin(2.0 * np.pi * 25.0 * times_s + 0.2)
    neural = 0.7 * np.sin(2.0 * np.pi * 25.0 * times_s - 0.4)
    broadband = np.sin(2.0 * np.pi * 7.3 * times_s + 0.1)
    background = (
        artifact_topography[:, np.newaxis] * artifact
        + neural_topography[:, np.newaxis] * broadband
    )
    injected = background + neural_topography[:, np.newaxis] * neural

    model = recovery.fit_spatial_line_subspace(
        background,
        sampling_frequency_hz,
        (25.0,),
        window_s=4.0,
        rank=1,
    )
    cleaned_background = recovery.subtract_spatial_line_subspace(
        background,
        sampling_frequency_hz,
        model,
        window_s=4.0,
    )
    cleaned_injected = recovery.subtract_spatial_line_subspace(
        injected,
        sampling_frequency_hz,
        model,
        window_s=4.0,
    )

    recovered_neural = cleaned_injected.cleaned_data - cleaned_background.cleaned_data
    expected_neural = neural_topography[:, np.newaxis] * neural
    relative_neural_error = np.linalg.norm(
        recovered_neural - expected_neural
    ) / np.linalg.norm(expected_neural)
    assert relative_neural_error < 1e-3
    before = _sinusoid_amplitude(background, sampling_frequency_hz, 25.0)
    after = _sinusoid_amplitude(
        cleaned_background.cleaned_data,
        sampling_frequency_hz,
        25.0,
    )
    assert np.linalg.norm(after) < 0.01 * np.linalg.norm(before)
    expected_broadband = neural_topography[:, np.newaxis] * broadband
    relative_broadband_error = np.linalg.norm(
        cleaned_background.cleaned_data - expected_broadband
    ) / np.linalg.norm(expected_broadband)
    assert relative_broadband_error < 1e-3


def test_spatial_line_subspace_exposes_aligned_signal_identifiability_limit():
    sampling_frequency_hz = 200.0
    times_s = np.arange(int(20.0 * sampling_frequency_hz)) / sampling_frequency_hz
    artifact_topography = np.array([1.0, 0.5, -0.25])
    artifact_topography /= np.linalg.norm(artifact_topography)
    background = artifact_topography[:, np.newaxis] * np.sin(
        2.0 * np.pi * 25.0 * times_s
    )
    aligned_neural = artifact_topography[:, np.newaxis] * np.cos(
        2.0 * np.pi * 25.0 * times_s
    )
    model = recovery.fit_spatial_line_subspace(
        background,
        sampling_frequency_hz,
        (25.0,),
        window_s=4.0,
        rank=1,
    )

    cleaned_background = recovery.subtract_spatial_line_subspace(
        background,
        sampling_frequency_hz,
        model,
        window_s=4.0,
    )
    cleaned_injected = recovery.subtract_spatial_line_subspace(
        background + aligned_neural,
        sampling_frequency_hz,
        model,
        window_s=4.0,
    )

    recovered_neural = cleaned_injected.cleaned_data - cleaned_background.cleaned_data
    assert np.linalg.norm(recovered_neural) < 1e-3 * np.linalg.norm(aligned_neural)


def test_spatial_line_subspace_model_rejects_nonorthonormal_basis():
    with pytest.raises(ValueError, match="orthonormal"):
        recovery.SpatialLineSubspaceModel(
            (25.0,),
            np.array([[1.0], [1.0]]),
        )


def test_spatial_basis_matches_left_singular_subspace_without_time_svd():
    artifact = np.random.default_rng(18).normal(size=(4, 10_000))
    expected_vectors, _, _ = np.linalg.svd(artifact, full_matrices=False)

    basis = recovery._leading_spatial_basis(artifact, rank=2)

    np.testing.assert_allclose(
        basis @ basis.T,
        expected_vectors[:, :2] @ expected_vectors[:, :2].T,
        rtol=1e-10,
        atol=1e-12,
    )


def test_spatial_model_uses_one_joint_fit_for_all_authorized_lines(monkeypatch):
    data = np.random.default_rng(19).normal(size=(3, 1_000))
    calls = []

    def joint_fit(
        values, sampling_frequency_hz, frequencies_hz, *, window_s, n_jobs=1
    ):
        calls.append(tuple(frequencies_hz))
        return recovery.SignalRecoveryResult(
            np.zeros_like(values),
            values,
            tuple(frequencies_hz),
        )

    monkeypatch.setattr(recovery, "subtract_multitaper_sinusoids", joint_fit)

    recovery.fit_spatial_line_subspace(
        data,
        100.0,
        (10.0, 20.0, 30.0),
        window_s=4.0,
        rank=1,
    )

    assert calls == [(10.0, 20.0, 30.0)]


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
            settings,
        )


# --- reconstruction cost -------------------------------------------------------------
#
# The published MATLAB ends with `rXbar = sig - sum(reconXbar(idx,:))`: it reconstructs only
# the components it removes. Summing the keepers instead is equivalent in result but costs a
# diagonal average for every component at every recursion level, which is what makes the
# method intractable at EEG sampling rates.


def _golden_case():
    """A configuration where the method demonstrably removes something (52% of input std)."""
    sfreq = 200.0
    rng = np.random.default_rng(11)
    n = 1600
    t = np.arange(n) / sfreq
    data = (rng.normal(0, 1.0, n) + 0.8 * np.sin(2 * np.pi * 50.0 * t + 0.3))[None, :]
    return data, sfreq, recovery.TrajectoryPCASettings(segment_s=0.30)


def test_component_reconstructions_sum_to_the_original_signal():
    """The identity the optimisation rests on: every component summed gives the input back.

    Without this, subtracting the removed components is not the same operation as summing
    the kept ones, and the two implementations would silently disagree.
    """
    rng = np.random.default_rng(3)
    signal = rng.normal(0, 1.0, 500)
    dim = 40
    matrix = np.lib.stride_tricks.sliding_window_view(signal, dim).T
    centred = matrix - matrix.mean(axis=1, keepdims=True)
    _, vectors = np.linalg.eigh((centred @ centred.T) / matrix.shape[1])
    scores = matrix.T @ vectors

    total = np.zeros(signal.size)
    for index in range(dim):
        total += recovery._diagonal_average(scores[:, index], vectors[:, index])

    assert np.allclose(total, signal, atol=1e-9)


def test_diagonal_average_matches_an_explicit_anti_diagonal_mean():
    """Correctness of the reconstruction itself, independent of how it is computed."""
    rng = np.random.default_rng(5)
    score = rng.normal(size=60)
    vector = rng.normal(size=8)

    produced = recovery._diagonal_average(score, vector)

    outer = np.outer(vector, score)          # (dim, n_samples), entry (k, j) at time j + k
    expected = np.array(
        [
            np.mean([outer[k, s - k] for k in range(outer.shape[0])
                     if 0 <= s - k < outer.shape[1]])
            for s in range(score.size + vector.size - 1)
        ]
    )
    assert np.allclose(produced, expected, atol=1e-9)


def test_recursion_only_reconstructs_the_components_it_needs(monkeypatch):
    """Cost, not result: a component that is kept unchanged needs no reconstruction."""
    data, sfreq, settings = _golden_case()
    dim = int(round(sfreq * settings.segment_s))

    calls = {"n": 0}
    original = recovery._diagonal_average

    def counting(score, vector):
        calls["n"] += 1
        return original(score, vector)

    monkeypatch.setattr(recovery, "_diagonal_average", counting)
    recovery.subtract_recursive_trajectory_pca(data, sfreq, settings)

    # Summing the keepers costs one reconstruction per component per level, and every
    # above-floor component at depth 1 spawns a full depth-2 decomposition.
    naive = dim * dim
    assert calls["n"] < naive / 10, f"{calls['n']} reconstructions, naive would be ~{naive}"


def test_optimised_recursion_reproduces_the_reference_output():
    """Behaviour must not move: values captured from the summing implementation."""
    data, sfreq, settings = _golden_case()

    result = recovery.subtract_recursive_trajectory_pca(data, sfreq, settings)

    expected = np.array(
        [0.1108930669, 0.5405292639, 0.5008843502, 1.4305163169,
         1.7601285486, 0.4161564595, -0.8271054192, -0.0413352608]
    )
    assert np.allclose(result.cleaned_data[0][::200], expected, atol=1e-6)
    assert result.artifact_data[0].sum() == pytest.approx(0.031429038974377654, abs=1e-6)
    assert (result.artifact_data[0] ** 2).sum() == pytest.approx(548.4399904974105, rel=1e-9)


def test_excess_kurtosis_matches_scipy():
    """The gate's statistic, computed directly.

    `scipy.stats.kurtosis` carries per-call wrapper overhead that dominated the profile at
    13,629 calls per channel; the value it returns for the biased Fisher definition is a
    ratio of central moments and needs none of it.
    """
    from scipy.stats import kurtosis as scipy_kurtosis

    rng = np.random.default_rng(17)
    for sample in (
        rng.normal(size=500),
        rng.normal(size=500) ** 3,
        np.sin(np.linspace(0, 40 * np.pi, 500)),
        np.repeat([1.0, -1.0], 250),
    ):
        assert recovery._excess_kurtosis(sample) == pytest.approx(
            float(scipy_kurtosis(sample, fisher=True, bias=True)), abs=1e-10
        )


def test_excess_kurtosis_of_a_constant_series_is_finite():
    """A flat component has zero spread; the gate must not see a nan."""
    assert np.isfinite(recovery._excess_kurtosis(np.full(50, 2.5)))
