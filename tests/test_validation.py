"""Cohort-scale false-detection and paired recovery trials."""

from __future__ import annotations

from types import SimpleNamespace

import mne
import numpy as np
import pytest

from decomb import injection, lines, notch, validation


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
    raw = mne.io.RawArray(
        data,
        mne.create_info(names, sampling_frequency_hz, "eeg"),
        verbose="ERROR",
    )
    raw.set_annotations(
        mne.Annotations(
            onset=np.arange(0.0, raw.times[-1], 0.9),
            duration=0.0,
            description="Volume/V  1",
        )
    )
    return raw


def test_false_detection_trials_cover_every_channel():
    raw = _noise_raw(n_channels=3, seed=3)

    trials = validation.false_detection_trials(
        raw,
        _narrow_settings(),
        np.random.default_rng(4),
        recording_name="sub-test_run-1",
        participant="sub-test",
    )

    assert {trial.channel_name for trial in trials} == {"C0", "C1", "C2"}
    assert all(trial.recording == "sub-test_run-1" for trial in trials)
    assert all(trial.participant == "sub-test" for trial in trials)


def test_false_detection_trials_count_scanner_authorization_for_every_channel(
    monkeypatch,
):
    raw = _noise_raw(n_channels=3, seed=18)
    null_model = lines.LineModel((), 1, 3, 10)
    monkeypatch.setattr(validation.surrogates, "surrogate_raw", lambda raw, rng: raw)
    monkeypatch.setattr(
        notch,
        "fit_harmonic_round",
        lambda raw, settings: SimpleNamespace(
            model=null_model,
            scanner_harmonics=object(),
        ),
    )
    trials = validation.false_detection_trials(
        raw,
        _narrow_settings(),
        np.random.default_rng(19),
        recording_name="sub-test_run-1",
        participant="sub-test",
    )

    assert all(trial.line_detected for trial in trials)


def test_paired_energy_decomposition_is_exact_for_identity_cleaning():
    background = np.random.default_rng(5).normal(size=(2, 1_000))
    times = np.arange(1_000) / 100.0
    basis = np.stack(
        [np.sin(2.0 * np.pi * 10.0 * times), np.cos(2.0 * np.pi * 10.0 * times)]
    )
    component = np.stack([2.0 * basis[0], -2.0 * basis[0]])
    injected = background + component
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
    component = np.stack([basis[0], -0.5 * basis[0], -0.5 * basis[0]])
    injected = background + component
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
    target = injection.FactorialInjectionTarget(
        kind="stationary",
        frequency_hz=10.0,
        component_to_background_db=-10.0,
        phase_rad=0.0,
    )

    result = validation.recovery_trial(
        background,
        _narrow_settings(),
        target,
        np.random.default_rng(8),
        recording_name="sub-test_run-1",
        participant="sub-test",
        channel_name="C0",
    )

    assert result.trial.recording == "sub-test_run-1"
    assert result.sequential_authorization is not None
    assert result.sequential_authorization.recording == "sub-test_run-1"
    assert result.sequential_authorization.phase_rad == 0.0
    trial = result.trial
    assert trial.injected_energy_v2 > 0.0
    assert trial.difference_energy_v2 >= 0.0
    assert trial.remaining_fraction >= 0.0
    assert trial.collateral_fraction >= 0.0
    assert trial.component_to_background_db == pytest.approx(-10.0)
    assert trial.phase_rad == 0.0


def test_recovery_trial_rejects_a_background_missing_the_channel():
    background = _noise_raw(n_channels=2, seed=9)
    target = injection.FactorialInjectionTarget(
        kind="stationary",
        frequency_hz=10.0,
        component_to_background_db=-10.0,
        phase_rad=0.0,
    )

    with pytest.raises(ValueError, match="C9"):
        validation.recovery_trial(
            background,
            _narrow_settings(),
            target,
            np.random.default_rng(10),
            recording_name="sub-test_run-1",
            participant="sub-test",
            channel_name="C9",
        )


def test_sequential_authorization_distinguishes_injected_and_unsupported_lines():
    model = lines.LineModel(
        channels=(
            lines.ChannelLineModel(
                channel_index=0,
                channel_name="C0",
                lines=(
                    lines.SupportedLine(10.0, 1e-9, 1e-6, (0,), None),
                    lines.SupportedLine(14.0, 1e-9, 1e-6, (1,), None),
                ),
                fundamental_hz=None,
                comb_corrected_p_value=None,
            ),
        ),
        window_count=2,
        channel_count=2,
        test_count_per_channel=100,
    )
    cleaning = SimpleNamespace(
        rounds=(SimpleNamespace(model=model, scanner_plan=None),)
    )
    spec = injection.SinusoidInjection("stationary", 10.0, 1e-6)

    supported, unsupported = validation.sequential_authorization_outcomes(
        cleaning,
        spec,
        frequency_bin_width_hz=1.0 / 54.0,
    )

    assert supported
    assert unsupported


def test_sequential_authorization_includes_trigger_anchored_comb_targets():
    settings = notch.HarmonicNotchSettings(
        estimation_window_s=10.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(1.0, 5.0),
        scanner_repetition_time_s=1.0,
        scanner_trigger_event_name="Scanner/Volume",
    )
    evidence = notch.ScannerHarmonicEvidence(1.0, 1e-10, (2, 4))
    scanner_plan = notch.plan_scanner_harmonic_notches(
        evidence,
        settings,
        maximum_hz=5.0,
    )
    model = lines.LineModel((), 1, 2, 5)
    cleaning = SimpleNamespace(
        rounds=(
            SimpleNamespace(
                model=model,
                scanner_harmonics=evidence,
                scanner_plan=scanner_plan,
            ),
        )
    )
    spec = injection.SinusoidInjection("stationary", 2.0, 1e-6)

    supported, unsupported = validation.sequential_authorization_outcomes(
        cleaning,
        spec,
        frequency_bin_width_hz=0.1,
    )

    assert supported
    assert unsupported
