"""Cohort-scale false-detection and recovery trials."""

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


def test_false_detection_trials_covers_every_channel_and_correction():
    raw = _noise_raw(n_channels=3, seed=1)
    settings = _narrow_settings()

    trials = validation.false_detection_trials(
        raw,
        settings,
        np.random.default_rng(2),
        recording_name="sub-test_run-1",
        participant="sub-test",
    )

    keys = {(trial.channel_name, trial.correction) for trial in trials}
    assert keys == {
        (channel, correction)
        for channel in ("C0", "C1", "C2")
        for correction in ("holm", "bonferroni", "none")
    }
    assert all(trial.recording == "sub-test_run-1" for trial in trials)
    assert all(trial.participant == "sub-test" for trial in trials)


def test_recovery_trial_removes_most_of_a_strong_stationary_injection():
    background = _noise_raw(n_channels=1, seed=3)
    settings = _narrow_settings()
    spec = injection.SinusoidInjection(kind="stationary", frequency_hz=10.0, amplitude_v=5e-5)

    trials = validation.recovery_trial(
        background,
        settings,
        spec,
        np.random.default_rng(4),
        recording_name="sub-test_run-1",
        channel_name="C0",
    )

    by_correction = {trial.correction: trial for trial in trials}
    assert set(by_correction) == {"holm", "bonferroni", "none"}
    for trial in trials:
        assert trial.injected_energy_v2 > 0.0
        assert np.isfinite(trial.collateral_energy_v2)
    # A strong, unambiguous stationary line should be almost entirely removed under
    # every correction procedure: the differences between them show up in weaker cases.
    for trial in trials:
        assert trial.remaining_fraction < 0.25


def test_recovery_trial_reports_nan_remaining_fraction_when_nothing_was_injected():
    background = _noise_raw(n_channels=1, seed=5)
    settings = _narrow_settings()
    # An injection so weak it adds no measurable energy in its own band-power bin can
    # legitimately leave injected_energy_v2 at or below zero; the fraction is undefined,
    # not zero, and must say so rather than silently reporting perfect recovery.
    spec = injection.SinusoidInjection(kind="stationary", frequency_hz=10.0, amplitude_v=1e-12)

    trials = validation.recovery_trial(
        background,
        settings,
        spec,
        np.random.default_rng(6),
        recording_name="sub-test_run-1",
        channel_name="C0",
    )

    for trial in trials:
        if trial.injected_energy_v2 <= 0.0:
            assert np.isnan(trial.remaining_fraction)


def test_recovery_trial_rejects_a_background_missing_the_channel():
    background = _noise_raw(n_channels=1, seed=7)
    settings = _narrow_settings()
    spec = injection.SinusoidInjection(kind="stationary", frequency_hz=10.0, amplitude_v=5e-5)

    with pytest.raises(ValueError, match="C9"):
        validation.recovery_trial(
            background,
            settings,
            spec,
            np.random.default_rng(0),
            recording_name="sub-test_run-1",
            channel_name="C9",
        )
