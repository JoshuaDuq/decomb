"""Descriptive signal-recovery evaluation without preservation gates."""

from __future__ import annotations

import numpy as np

from decomb import recovery_evaluation


def test_preservation_metrics_report_identity_without_thresholding():
    sampling_frequency_hz = 100.0
    times_s = np.arange(1_000) / sampling_frequency_hz
    data = np.vstack(
        (
            np.sin(2.0 * np.pi * 7.0 * times_s),
            np.cos(2.0 * np.pi * 12.0 * times_s),
        )
    )

    result = recovery_evaluation.measure_preservation(
        data,
        data.copy(),
        sampling_frequency_hz,
        (("theta", 4.0, 7.9), ("alpha", 8.0, 12.9)),
        window_s=2.0,
    )

    assert result.signal_correlation == 1.0
    assert result.normalized_change_rms == 0.0
    assert tuple(metric.name for metric in result.bands) == ("theta", "alpha")
    for metric in result.bands:
        assert abs(metric.power_change_db) < 1e-12
        assert abs(metric.phase_error_degrees) < 1e-12


def test_preservation_metrics_describe_band_power_loss_without_failing():
    sampling_frequency_hz = 100.0
    times_s = np.arange(1_000) / sampling_frequency_hz
    theta = np.sin(2.0 * np.pi * 6.0 * times_s)
    alpha = np.sin(2.0 * np.pi * 10.0 * times_s)
    original = (theta + alpha)[np.newaxis, :]
    cleaned = (0.5 * theta + alpha)[np.newaxis, :]

    result = recovery_evaluation.measure_preservation(
        original,
        cleaned,
        sampling_frequency_hz,
        (("theta", 4.0, 7.9), ("alpha", 8.0, 12.9)),
        window_s=2.0,
    )

    metrics = {metric.name: metric for metric in result.bands}
    assert np.isclose(metrics["theta"].power_change_db, -6.0206, atol=1e-3)
    assert abs(metrics["alpha"].power_change_db) < 1e-3
    assert 0.0 < result.normalized_change_rms < 1.0


def test_band_preservation_reports_complete_removal_without_requiring_correlation():
    sampling_frequency_hz = 100.0
    times_s = np.arange(1_000) / sampling_frequency_hz
    original = np.sin(2.0 * np.pi * 10.0 * times_s)[np.newaxis, :]

    metric = recovery_evaluation.measure_band_preservation(
        original,
        np.zeros_like(original),
        sampling_frequency_hz,
        ("alpha", 8.0, 12.9),
        window_s=2.0,
    )

    assert metric.power_ratio == 0.0
    assert metric.power_change_db < -3_000.0
    assert metric.phase_error_degrees is None
