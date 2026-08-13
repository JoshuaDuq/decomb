"""Sinusoid-free Gaussian surrogates matched to a real channel's own spectrum."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from decomb import surrogates


def _line_channel(*, line_hz: float = 60.0, seed: int = 0) -> tuple[np.ndarray, float]:
    sampling_frequency_hz = 250.0
    times_s = np.arange(int(60.0 * sampling_frequency_hz)) / sampling_frequency_hz
    data = np.random.default_rng(seed).normal(scale=1e-6, size=times_s.size)
    data += 20e-6 * np.sin(2.0 * np.pi * line_hz * times_s)
    return data, sampling_frequency_hz


def test_background_spectrum_suppresses_a_narrowband_line():
    data, sampling_frequency_hz = _line_channel()

    raw_spectrum = np.abs(np.fft.rfft(data)) ** 2
    smoothed = surrogates.background_power_spectrum(data, sampling_frequency_hz)

    frequencies_hz = np.fft.rfftfreq(data.size, d=1.0 / sampling_frequency_hz)
    at_line = np.argmin(np.abs(frequencies_hz - 60.0))
    assert smoothed[at_line] < raw_spectrum[at_line] / 10.0


def test_synthesized_process_matches_its_target_spectrum_on_average():
    sampling_frequency_hz = 200.0
    sample_count = 4_000
    frequencies_hz = np.fft.rfftfreq(sample_count, d=1.0 / sampling_frequency_hz)
    target = 1.0 / (1.0 + frequencies_hz)  # an arbitrary smooth 1/f-like envelope

    rng = np.random.default_rng(3)
    realizations = np.stack(
        [
            np.abs(
                np.fft.rfft(
                    surrogates.synthesize_gaussian_process(target, sample_count, rng)
                )
            )
            ** 2
            for _ in range(200)
        ]
    )
    average_power = realizations.mean(axis=0)
    # Interior bins only: the DC and Nyquist bins are drawn from a one-degree-of-freedom
    # distribution and are individually noisier, but every bin's expectation matches.
    ratio = average_power[5:-5] / target[5:-5]
    assert np.median(ratio) == pytest.approx(1.0, rel=0.25)


def test_synthesized_process_has_no_persistent_line():
    sampling_frequency_hz = 250.0
    sample_count = int(60.0 * sampling_frequency_hz)
    data, _ = _line_channel(seed=5)
    envelope = surrogates.background_power_spectrum(data, sampling_frequency_hz)

    rng = np.random.default_rng(9)
    surrogate = surrogates.synthesize_gaussian_process(envelope, sample_count, rng)

    frequencies_hz = np.fft.rfftfreq(sample_count, d=1.0 / sampling_frequency_hz)
    surrogate_spectrum = np.abs(np.fft.rfft(surrogate)) ** 2
    at_line = np.argmin(np.abs(frequencies_hz - 60.0))
    neighbourhood = surrogate_spectrum[max(0, at_line - 50) : at_line + 50]
    # A real 60 Hz line stands far above its shoulders; a surrogate drawn from the
    # smoothed envelope should not reproduce a comparable outlier at that exact bin.
    assert surrogate_spectrum[at_line] < 20.0 * np.median(neighbourhood)


def test_surrogate_eeg_data_is_independent_per_channel():
    sampling_frequency_hz = 250.0
    data = np.random.default_rng(1).normal(scale=1e-6, size=(3, 15_000))
    rng = np.random.default_rng(2)

    surrogate = surrogates.surrogate_eeg_data(data, sampling_frequency_hz, rng)

    assert surrogate.shape == data.shape
    assert not np.allclose(surrogate[0], surrogate[1])


def test_surrogate_raw_preserves_length_channels_and_annotations():
    sampling_frequency_hz = 250.0
    n_times = 15_000
    data = np.random.default_rng(4).normal(scale=1e-6, size=(2, n_times))
    raw = mne.io.RawArray(
        data,
        mne.create_info(["C3", "C4"], sampling_frequency_hz, "eeg"),
        verbose="ERROR",
    )
    raw.set_annotations(
        mne.Annotations(onset=[10.0], duration=[2.0], description=["bad_acq_skip"])
    )

    surrogate = surrogates.surrogate_raw(raw, np.random.default_rng(6))

    assert surrogate.ch_names == raw.ch_names
    assert surrogate.n_times == raw.n_times
    assert float(surrogate.info["sfreq"]) == sampling_frequency_hz
    assert list(surrogate.annotations.description) == ["bad_acq_skip"]
    assert not np.allclose(surrogate.get_data(), raw.get_data())


def test_surrogate_raw_excludes_bad_eeg_channels_and_preserves_channel_metadata():
    raw = mne.io.RawArray(
        np.random.default_rng(7).normal(size=(3, 1_000)),
        mne.create_info(["C3", "C4", "EOG1"], 100.0, ["eeg", "eeg", "eog"]),
        verbose="ERROR",
    )
    raw.info["bads"] = ["C4"]

    surrogate = surrogates.surrogate_raw(raw, np.random.default_rng(8))

    assert surrogate.ch_names == ["C3"]
    assert surrogate.get_channel_types() == ["eeg"]


def test_surrogate_raw_requires_an_eeg_channel():
    sampling_frequency_hz = 100.0
    data = np.random.default_rng(0).normal(size=(1, 1_000))
    raw = mne.io.RawArray(
        data,
        mne.create_info(["EOG1"], sampling_frequency_hz, "eog"),
        verbose="ERROR",
    )

    with pytest.raises(ValueError, match="EEG channel"):
        surrogates.surrogate_raw(raw, np.random.default_rng(0))
