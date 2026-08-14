"""Multiplicity-controlled coherent and persistent line detection."""

from __future__ import annotations

import numpy as np
import pytest

from decomb import lines


def _windows(*, line_hz: float | None, seed: int = 0) -> tuple[np.ndarray, float]:
    sampling_frequency_hz = 200.0
    sample_count = int(4.0 * sampling_frequency_hz)
    times_s = np.arange(sample_count) / sampling_frequency_hz
    data = np.random.default_rng(seed).normal(
        scale=1.0,
        size=(3, 2, sample_count),
    )
    if line_hz is not None:
        data += 8.0 * np.sin(2.0 * np.pi * line_hz * times_s)
    return data, sampling_frequency_hz


def test_channel_holm_test_detects_an_isolated_line():
    data, sampling_frequency_hz = _windows(line_hz=37.25)

    result = lines.detect_lines(
        data,
        sampling_frequency_hz,
        frequency_range_hz=(1.0, 80.0),
        familywise_error_rate=0.05,
    )

    assert result.detections
    assert any(
        abs(detection.frequency_hz - 37.25) <= 1.0 / 4.0
        for detection in result.detections
    )
    assert all(
        detection.corrected_p_value < 0.05
        for detection in result.detections
    )


def test_channel_holm_test_does_not_invent_a_line_in_deterministic_noise():
    data, sampling_frequency_hz = _windows(line_hz=None, seed=11)

    result = lines.detect_lines(
        data,
        sampling_frequency_hz,
        frequency_range_hz=(1.0, 80.0),
        familywise_error_rate=0.05,
    )

    assert result.detections == ()


def test_persistent_peak_test_detects_phase_incoherent_narrowband_power():
    sampling_frequency_hz = 100.0
    sample_count = 1_000
    window_count = 97
    rng = np.random.default_rng(19)
    data = rng.normal(size=(window_count, 2, sample_count))
    target_bin = int(20.0 * sample_count / sampling_frequency_hz)
    for window in range(window_count):
        spectrum = np.fft.rfft(data[window], axis=-1)
        spectrum[:, target_bin - 1 : target_bin + 2] *= 30.0
        data[window] = np.fft.irfft(spectrum, n=sample_count, axis=-1)

    frequencies_hz, p_values = lines.persistent_peak_p_values(
        data,
        sampling_frequency_hz,
        frequency_range_hz=(1.0, 40.0),
    )

    frequency_index = int(np.argmin(np.abs(frequencies_hz - 20.0)))
    assert np.count_nonzero(p_values[:, :, frequency_index] < 1.0) >= 46
    assert np.min(p_values[:, :, frequency_index]) < 1e-8


def test_persistent_peak_evidence_is_reported_only_on_disjoint_test_windows():
    sampling_frequency_hz = 100.0
    sample_count = 1_000
    window_count = 98
    rng = np.random.default_rng(23)
    data = rng.normal(size=(window_count, 2, sample_count))
    target_bin = int(20.0 * sample_count / sampling_frequency_hz)
    test_indices = np.arange(2, window_count, 4)
    for window_index in test_indices:
        spectrum = np.fft.rfft(data[window_index], axis=-1)
        spectrum[:, target_bin - 1 : target_bin + 2] *= 30.0
        data[window_index] = np.fft.irfft(spectrum, n=sample_count, axis=-1)

    frequencies_hz, p_values = lines.persistent_peak_p_values(
        data,
        sampling_frequency_hz,
        frequency_range_hz=(1.0, 40.0),
    )

    frequency_index = int(np.argmin(np.abs(frequencies_hz - 20.0)))
    non_test_indices = np.setdiff1d(np.arange(window_count), test_indices)
    assert np.all(p_values[non_test_indices, :, frequency_index] == 1.0)
    assert np.all(p_values[test_indices, :, frequency_index] < 1e-8)


def test_persistent_peak_test_requires_disjoint_background_and_test_windows():
    with pytest.raises(ValueError, match="at least three windows"):
        lines.persistent_peak_p_values(
            np.zeros((2, 2, 1_000)),
            100.0,
            frequency_range_hz=(1.0, 40.0),
        )


def test_null_detection_builds_an_explicit_clean_recording_model():
    data, sampling_frequency_hz = _windows(line_hz=None, seed=11)
    result = lines.detect_lines(
        data,
        sampling_frequency_hz,
        frequency_range_hz=(1.0, 80.0),
        familywise_error_rate=0.05,
    )

    model = lines.build_line_model(
        result,
        channel_names=("C3", "C4"),
    )

    assert model.channels == ()
    assert model.line_count == 0
    assert model.channel_count == 2


def test_reported_p_values_use_the_complete_recording_family():
    data, sampling_frequency_hz = _windows(line_hz=37.25)
    frequencies_hz, raw_p_values = lines.line_test_p_values(
        data,
        sampling_frequency_hz,
        frequency_range_hz=(1.0, 80.0),
    )

    result = lines.detect_lines(
        data,
        sampling_frequency_hz,
        frequency_range_hz=(1.0, 80.0),
        familywise_error_rate=0.05,
    )

    assert raw_p_values.shape == (*data.shape[:2], frequencies_hz.size)
    assert result.total_test_count == raw_p_values.size
    assert result.test_count_per_channel == raw_p_values[:, 0, :].size
    strongest = min(result.detections, key=lambda detection: detection.corrected_p_value)
    expected = min(
        1.0,
        float(np.min(raw_p_values)) * result.total_test_count,
    )
    assert strongest.corrected_p_value == expected


def test_holm_correction_controls_the_recording_family(monkeypatch):
    frequencies_hz = np.array([10.0, 20.0])
    raw_p_values = np.array(
        [
            [[0.001, 0.50], [0.50, 0.60]],
            [[0.02, 0.80], [0.70, 0.90]],
        ]
    )

    monkeypatch.setattr(
        lines,
        "line_test_p_values",
        lambda *args, **kwargs: (frequencies_hz, raw_p_values),
    )

    result = lines.detect_lines(
        np.zeros((2, 2, 8)),
        8.0,
        frequency_range_hz=(1.0, 3.0),
        familywise_error_rate=0.05,
    )

    assert result.test_count_per_channel == 4
    assert result.total_test_count == 8
    assert len(result.detections) == 1
    detection = result.detections[0]
    assert detection.channel_index == 0
    assert detection.frequency_hz == 10.0
    assert detection.raw_p_value == 0.001
    assert detection.corrected_p_value == 0.008


def test_holm_correction_covers_the_complete_recording_family():
    result = lines.detect_lines_from_p_values(
        np.array([10.0]),
        np.array([[[0.03], [0.03]]]),
        familywise_error_rate=0.05,
    )

    assert result.detections == ()


def test_dc_is_not_treated_as_a_sinusoidal_line():
    data = np.ones((2, 3, 2_000))

    frequencies_hz, p_values = lines.thomson_f_p_values(
        data,
        200.0,
        frequency_range_hz=(0.0, 80.0),
    )

    assert frequencies_hz[0] > 0.0
    assert np.all(p_values == 1.0)


def test_nyquist_can_be_included_for_native_mne_spectrum_fit():
    sampling_frequency_hz = 200.0
    data = np.zeros((1, 1, 2_000))

    frequencies_hz, _ = lines.thomson_f_p_values(
        data,
        sampling_frequency_hz,
        frequency_range_hz=(0.0, sampling_frequency_hz / 2.0),
    )

    assert frequencies_hz[-1] == sampling_frequency_hz / 2.0


def test_underflowed_p_value_remains_valid_statistical_evidence():
    model = lines.build_line_model(
        lines.LineDetectionResult(
            (
                lines.LineDetection(
                    frequency_hz=40.0,
                    raw_p_value=0.0,
                    corrected_p_value=0.0,
                    window_index=0,
                    channel_index=0,
                ),
            ),
            tested_frequencies_hz=tuple(np.arange(1.0, 101.0)),
            window_count=10,
            channel_count=1,
        ),
        channel_names=("Cz",),
    )

    assert model.channels[0].lines[0].corrected_p_value == 0.0
