"""Cohort-scale false-detection and paired recovery trials."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from decomb import injection, notch, validation


def _narrow_settings(frequency_range_hz=(1.0, 20.0)) -> notch.HarmonicNotchSettings:
    return notch.HarmonicNotchSettings(
        estimation_window_s=54.0,
        familywise_error_rate=0.05,
        frequency_range_hz=frequency_range_hz,
    )


def _noise_raw(*, n_channels: int = 2, seed: int = 0, duration_s: float = 120.0):
    sampling_frequency_hz = 250.0
    n_samples = int(duration_s * sampling_frequency_hz)
    data = np.random.default_rng(seed).normal(scale=1e-6, size=(n_channels, n_samples))
    names = [f"C{index}" for index in range(n_channels)]
    return mne.io.RawArray(
        data,
        mne.create_info(names, sampling_frequency_hz, "eeg"),
        verbose="ERROR",
    )


def test_average_reference_has_zero_instantaneous_channel_mean():
    raw = _noise_raw(n_channels=3, seed=1)

    referenced = validation.average_reference(raw)

    np.testing.assert_allclose(referenced.get_data().mean(axis=0), 0.0, atol=1e-21)


def test_average_reference_requires_two_non_bad_eeg_channels():
    raw = _noise_raw(n_channels=1, seed=2)

    with pytest.raises(ValueError, match="at least two non-bad EEG channels"):
        validation.average_reference(raw)


def test_mne_spectrum_fit_reports_native_channel_decisions():
    raw = _noise_raw(n_channels=2, seed=0)
    times_s = raw.times
    raw._data[0] += 50e-6 * np.sin(2.0 * np.pi * 10.0 * times_s)

    result = validation.mne_spectrum_fit(raw, window_s=54.0, p_value=0.05)

    assert result.detected_channels == ("C0",)
    assert not np.array_equal(result.cleaned.get_data(picks=["C0"]), raw.get_data(picks=["C0"]))


def test_mne_spectrum_fit_detection_includes_the_nyquist_bin():
    sampling_frequency_hz = 100.0
    sample_count = 1_000
    raw = mne.io.RawArray(
        np.vstack(
            (
                50e-6 * (-1.0) ** np.arange(sample_count),
                np.zeros(sample_count),
            )
        ),
        mne.create_info(["C0", "C1"], sampling_frequency_hz, "eeg"),
        verbose="ERROR",
    )

    detected = validation.mne_spectrum_fit_detected_channels(
        raw,
        window_s=10.0,
        p_value=0.05,
    )

    assert detected == ("C0",)


def test_false_detection_trials_cover_every_channel_and_method():
    raw = _noise_raw(n_channels=3, seed=3)

    trials = validation.false_detection_trials(
        raw,
        _narrow_settings(),
        np.random.default_rng(4),
        recording_name="sub-test_run-1",
        participant="sub-test",
    )

    keys = {(trial.channel_name, trial.method) for trial in trials}
    assert keys == {
        (channel, method)
        for channel in ("C0", "C1", "C2")
        for method in validation.ALL_METHODS
    }
    assert all(trial.recording == "sub-test_run-1" for trial in trials)
    assert all(trial.participant == "sub-test" for trial in trials)


def test_paired_energy_decomposition_is_exact_for_identity_cleaning():
    background = np.random.default_rng(5).normal(size=(2, 1_000))
    times = np.arange(1_000) / 100.0
    basis = np.stack(
        [np.sin(2.0 * np.pi * 10.0 * times), np.cos(2.0 * np.pi * 10.0 * times)]
    )
    artifact = np.stack([2.0 * basis[0], -2.0 * basis[0]])
    injected = background + artifact
    valid = np.ones(1_000, dtype=bool)

    metrics = validation.paired_energy_metrics(
        background,
        injected,
        background,
        injected,
        basis,
        valid,
    )

    assert metrics.remaining_fraction == pytest.approx(1.0)
    assert metrics.collateral_fraction == pytest.approx(0.0, abs=1e-28)
    assert metrics.difference_energy_v2 == pytest.approx(metrics.injected_energy_v2)


def test_paired_energy_components_sum_to_the_cleaned_difference_energy():
    rng = np.random.default_rng(6)
    background = rng.normal(size=(3, 600))
    times = np.arange(600) / 100.0
    basis = np.stack(
        [np.sin(2.0 * np.pi * 7.0 * times), np.cos(2.0 * np.pi * 7.0 * times)]
    )
    artifact = np.stack([basis[0], -0.5 * basis[0], -0.5 * basis[0]])
    injected = background + artifact
    cleaned_background = background + rng.normal(scale=0.01, size=background.shape)
    cleaned_injected = injected + rng.normal(scale=0.01, size=injected.shape)
    valid = np.ones(600, dtype=bool)

    metrics = validation.paired_energy_metrics(
        background,
        injected,
        cleaned_background,
        cleaned_injected,
        basis,
        valid,
    )

    decomposed = metrics.injected_energy_v2 * (
        metrics.remaining_fraction + metrics.collateral_fraction
    )
    assert decomposed == pytest.approx(metrics.difference_energy_v2)


def test_recovery_trial_uses_paired_time_domain_energy():
    background = _noise_raw(n_channels=2, seed=7)
    spec = injection.SinusoidInjection(
        kind="stationary",
        frequency_hz=10.0,
        amplitude_v=50e-6,
    )

    trials = validation.recovery_trial(
        background,
        _narrow_settings(),
        spec,
        np.random.default_rng(8),
        recording_name="sub-test_run-1",
        participant="sub-test",
        channel_name="C0",
    )

    assert {trial.method for trial in trials} == set(validation.ALL_METHODS)
    for trial in trials:
        assert trial.injected_energy_v2 > 0.0
        assert trial.difference_energy_v2 >= 0.0
        assert trial.remaining_fraction >= 0.0
        assert trial.collateral_fraction >= 0.0
        assert np.isfinite(trial.artifact_to_background_db)


def test_recovery_trial_rejects_a_background_missing_the_channel():
    background = _noise_raw(n_channels=2, seed=9)
    spec = injection.SinusoidInjection(
        kind="stationary",
        frequency_hz=10.0,
        amplitude_v=50e-6,
    )

    with pytest.raises(ValueError, match="C9"):
        validation.recovery_trial(
            background,
            _narrow_settings(),
            spec,
            np.random.default_rng(10),
            recording_name="sub-test_run-1",
            participant="sub-test",
            channel_name="C9",
        )
