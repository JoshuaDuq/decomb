"""Stationary, drifting, and intermittent sinusoid injections."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from decomb import injection


def test_stationary_injection_matches_a_pure_tone():
    spec = injection.SinusoidInjection(kind="stationary", frequency_hz=10.0, amplitude_v=2.0)
    sampling_frequency_hz = 100.0
    n_samples = 1_000

    realization = injection.realize_injection(
        spec, n_samples, sampling_frequency_hz, np.random.default_rng(0)
    )

    times_s = np.arange(n_samples) / sampling_frequency_hz
    expected = 2.0 * np.sin(2.0 * np.pi * 10.0 * times_s)
    np.testing.assert_allclose(realization.waveform_v, expected)
    assert realization.temporal_basis.shape == (2, n_samples)


def test_drifting_injection_sweeps_from_start_to_end_frequency():
    spec = injection.SinusoidInjection(
        kind="drifting", frequency_hz=10.0, amplitude_v=1.0, drift_hz=5.0
    )
    sampling_frequency_hz = 200.0
    n_samples = 20_000  # 100 s

    waveform = injection.realize_injection(
        spec, n_samples, sampling_frequency_hz, np.random.default_rng(1)
    ).waveform_v

    # Instantaneous frequency near the start should look like 10 Hz, near the end 15 Hz.
    early = waveform[: int(2.0 * sampling_frequency_hz)]
    late = waveform[-int(2.0 * sampling_frequency_hz) :]
    early_freq = np.fft.rfftfreq(early.size, d=1.0 / sampling_frequency_hz)
    late_freq = np.fft.rfftfreq(late.size, d=1.0 / sampling_frequency_hz)
    early_peak = early_freq[np.argmax(np.abs(np.fft.rfft(early)))]
    late_peak = late_freq[np.argmax(np.abs(np.fft.rfft(late)))]
    assert early_peak == pytest.approx(10.0, abs=1.0)
    assert late_peak == pytest.approx(15.0, abs=1.0)


def test_intermittent_injection_is_zero_outside_its_active_span():
    spec = injection.SinusoidInjection(
        kind="intermittent", frequency_hz=10.0, amplitude_v=1.0, occupancy=0.3
    )
    sampling_frequency_hz = 100.0
    n_samples = 1_000

    realization = injection.realize_injection(
        spec, n_samples, sampling_frequency_hz, np.random.default_rng(2)
    )
    waveform = realization.waveform_v

    active = np.abs(waveform) > 0.0
    # occupancy is a share of samples, not a guarantee every active sample is nonzero
    # (a sinusoid crosses zero), so check the active *span* rather than every sample.
    active_span = np.flatnonzero(active)
    assert active_span.size > 0
    span_samples = active_span[-1] - active_span[0] + 1
    assert span_samples == pytest.approx(0.3 * n_samples, rel=0.05)
    assert not active[: active_span[0]].any()
    assert not active[active_span[-1] + 1 :].any()
    assert not realization.temporal_basis[:, : active_span[0]].any()
    assert not realization.temporal_basis[:, active_span[-1] + 1 :].any()


def test_realized_waveform_lies_exactly_in_its_declared_subspace():
    spec = injection.SinusoidInjection(
        kind="drifting",
        frequency_hz=8.0,
        amplitude_v=3.0,
        drift_hz=2.0,
        phase_rad=0.7,
    )

    realization = injection.realize_injection(
        spec,
        2_000,
        200.0,
        np.random.default_rng(3),
    )
    coefficients = np.linalg.lstsq(
        realization.temporal_basis.T,
        realization.waveform_v,
        rcond=None,
    )[0]

    np.testing.assert_allclose(
        realization.temporal_basis.T @ coefficients,
        realization.waveform_v,
        atol=1e-12,
    )


def test_phase_modulated_injection_has_the_requested_instantaneous_frequency_bounds():
    spec = injection.SinusoidInjection(
        kind="phase_modulated",
        frequency_hz=10.0,
        amplitude_v=1.0,
        phase_modulation_hz=0.2,
        phase_deviation_rad=0.5,
    )

    realization = injection.realize_injection(
        spec,
        20_000,
        200.0,
        np.random.default_rng(31),
    )
    low_hz, high_hz = injection.injected_frequency_band_hz(
        spec,
        half_width_hz=0.0,
    )

    assert low_hz == pytest.approx(9.9)
    assert high_hz == pytest.approx(10.1)
    assert np.isfinite(realization.waveform_v).all()
    assert np.linalg.matrix_rank(realization.temporal_basis) == 2


def test_kind_and_parameter_combinations_are_validated():
    with pytest.raises(ValueError, match="kind"):
        injection.SinusoidInjection(kind="bursty", frequency_hz=10.0, amplitude_v=1.0)
    with pytest.raises(ValueError, match="drift_hz"):
        injection.SinusoidInjection(
            kind="stationary", frequency_hz=10.0, amplitude_v=1.0, drift_hz=1.0
        )
    with pytest.raises(ValueError, match="occupancy"):
        injection.SinusoidInjection(
            kind="drifting",
            frequency_hz=10.0,
            amplitude_v=1.0,
            drift_hz=1.0,
            occupancy=0.5,
        )
    with pytest.raises(ValueError, match="drift_hz"):
        injection.SinusoidInjection(
            kind="drifting",
            frequency_hz=10.0,
            amplitude_v=1.0,
            drift_hz=np.nan,
        )
    with pytest.raises(ValueError, match="component_to_background_db"):
        injection.FactorialInjectionTarget(
            kind="stationary",
            frequency_hz=10.0,
            component_to_background_db=np.nan,
        )
    with pytest.raises(ValueError, match="phase_modulation_hz"):
        injection.SinusoidInjection(
            kind="phase_modulated",
            frequency_hz=10.0,
            amplitude_v=1.0,
        )


@pytest.mark.parametrize(
    "spec",
    [
        injection.SinusoidInjection("stationary", 50.0, 1.0),
        injection.SinusoidInjection("drifting", 10.0, 1.0, drift_hz=-10.0),
        injection.SinusoidInjection("drifting", 40.0, 1.0, drift_hz=10.0),
    ],
)
def test_realization_rejects_trajectories_outside_zero_and_nyquist(spec):
    with pytest.raises(ValueError, match=r"strictly inside \(0, Nyquist\)"):
        injection.realize_injection(
            spec,
            1_000,
            100.0,
            np.random.default_rng(4),
        )


def test_injected_frequency_band_covers_the_drift_sweep():
    spec = injection.SinusoidInjection(
        kind="drifting", frequency_hz=10.0, amplitude_v=1.0, drift_hz=-3.0
    )

    low_hz, high_hz = injection.injected_frequency_band_hz(spec, half_width_hz=0.1)

    assert low_hz == pytest.approx(6.9)
    assert high_hz == pytest.approx(10.1)


def test_inject_into_raw_adds_the_waveform_to_one_channel_only():
    sampling_frequency_hz = 100.0
    data = np.zeros((2, 1_000))
    raw = mne.io.RawArray(
        data,
        mne.create_info(["C3", "C4"], sampling_frequency_hz, "eeg"),
        verbose="ERROR",
    )
    spec = injection.SinusoidInjection(kind="stationary", frequency_hz=10.0, amplitude_v=1.0)

    realization = injection.realize_injection(
        spec,
        raw.n_times,
        sampling_frequency_hz,
        np.random.default_rng(0),
    )
    injected = injection.inject_into_raw(raw, "C3", realization)

    assert not np.allclose(injected.get_data(picks=["C3"]), 0.0)
    np.testing.assert_allclose(injected.get_data(picks=["C4"]), 0.0)
    assert injected.ch_names == raw.ch_names
    assert injected.n_times == raw.n_times


def test_inject_into_raw_rejects_an_unknown_channel():
    raw = mne.io.RawArray(
        np.zeros((1, 100)),
        mne.create_info(["C3"], 100.0, "eeg"),
        verbose="ERROR",
    )
    spec = injection.SinusoidInjection(kind="stationary", frequency_hz=10.0, amplitude_v=1.0)

    with pytest.raises(ValueError, match="C4"):
        realization = injection.realize_injection(
            spec,
            raw.n_times,
            100.0,
            np.random.default_rng(0),
        )
        injection.inject_into_raw(raw, "C4", realization)


def test_spatially_balanced_injection_preserves_the_requested_target_amplitude():
    raw = mne.io.RawArray(
        np.zeros((3, 1_000)),
        mne.create_info(["C3", "C4", "Pz"], 100.0, "eeg"),
        verbose="ERROR",
    )
    realization = injection.realize_injection(
        injection.SinusoidInjection("stationary", 10.0, 2e-6),
        raw.n_times,
        100.0,
        np.random.default_rng(0),
    )

    injected = injection.inject_spatially_balanced(raw, "C3", realization)
    difference = injected.get_data() - raw.get_data()

    np.testing.assert_allclose(difference.mean(axis=0), 0.0, atol=1e-21)
    np.testing.assert_allclose(difference[0], realization.waveform_v)
