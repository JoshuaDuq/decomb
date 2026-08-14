"""Descriptive artifact-removal and signal-preservation measurements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from decomb import recordings


@dataclass(frozen=True)
class BandPreservationMetric:
    """Power and phase change within one predefined analysis band."""

    name: str
    low_hz: float
    high_hz: float
    power_change_db: float
    phase_error_degrees: float | None


@dataclass(frozen=True)
class PreservationMetrics:
    """Descriptive metrics that do not decide whether a candidate passes."""

    signal_correlation: float
    normalized_change_rms: float
    bands: tuple[BandPreservationMetric, ...]


def _validated_pair(
    original: NDArray[np.floating],
    cleaned: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    original_values = np.asarray(original, dtype=np.float64)
    cleaned_values = np.asarray(cleaned, dtype=np.float64)
    if original_values.ndim != 2 or cleaned_values.shape != original_values.shape:
        raise ValueError("original and cleaned data must have the same 2D shape")
    if not np.isfinite(original_values).all() or not np.isfinite(cleaned_values).all():
        raise ValueError("original and cleaned data must be finite")
    return original_values, cleaned_values


def _validated_bands(
    bands: Sequence[tuple[str, float, float]],
    sampling_frequency_hz: float,
) -> tuple[tuple[str, float, float], ...]:
    nyquist_hz = sampling_frequency_hz / 2.0
    validated = []
    for name, low_hz, high_hz in bands:
        if not str(name).strip():
            raise ValueError("band names must not be empty")
        if not np.all(np.isfinite((low_hz, high_hz))) or not (
            0.0 <= low_hz < high_hz <= nyquist_hz
        ):
            raise ValueError("band edges must be finite, increasing, and below Nyquist")
        validated.append((str(name), float(low_hz), float(high_hz)))
    if not validated:
        raise ValueError("at least one analysis band is required")
    return tuple(validated)


def _windowed_spectra(
    data: NDArray[np.float64],
    bounds: tuple[tuple[int, int], ...],
) -> NDArray[np.complex128]:
    windows = np.stack([data[:, start:stop] for start, stop in bounds])
    windows -= windows.mean(axis=-1, keepdims=True)
    windows *= np.hanning(windows.shape[-1])
    return np.fft.rfft(windows, axis=-1)


def _band_metric(
    name: str,
    low_hz: float,
    high_hz: float,
    frequencies_hz: NDArray[np.float64],
    original_spectra: NDArray[np.complex128],
    cleaned_spectra: NDArray[np.complex128],
) -> BandPreservationMetric:
    inside = (frequencies_hz >= low_hz) & (frequencies_hz <= high_hz)
    if not inside.any():
        raise ValueError(f"no spectral bin lies in analysis band {name!r}")

    original = original_spectra[..., inside]
    cleaned = cleaned_spectra[..., inside]
    original_power = float(np.sum(np.abs(original) ** 2))
    cleaned_power = float(np.sum(np.abs(cleaned) ** 2))
    if original_power <= 0.0:
        raise ValueError(f"analysis band {name!r} has no original power")
    power_change_db = 10.0 * np.log10(
        max(cleaned_power, np.finfo(float).tiny) / original_power
    )

    weights = np.abs(original) * np.abs(cleaned)
    phase_error_degrees = None
    if float(weights.sum()) > 0.0:
        phase_errors = np.angle(cleaned * np.conj(original))
        phase_error_degrees = float(
            np.sqrt(np.average(phase_errors**2, weights=weights))
            * 180.0
            / np.pi
        )
    return BandPreservationMetric(
        name,
        low_hz,
        high_hz,
        float(power_change_db),
        phase_error_degrees,
    )


def measure_preservation(
    original: NDArray[np.floating],
    cleaned: NDArray[np.floating],
    sampling_frequency_hz: float,
    bands: Sequence[tuple[str, float, float]],
    *,
    window_s: float,
) -> PreservationMetrics:
    """Describe time-domain, band-power, and spectral-phase changes."""
    original_values, cleaned_values = _validated_pair(original, cleaned)
    window_samples = recordings.estimation_window_samples(
        sampling_frequency_hz,
        window_s,
    )
    bounds = recordings.overlapping_window_bounds(
        n_times=original_values.shape[-1],
        window_samples=window_samples,
        overlap=0.5,
    )
    validated_bands = _validated_bands(bands, sampling_frequency_hz)

    original_centered = original_values - original_values.mean()
    cleaned_centered = cleaned_values - cleaned_values.mean()
    original_energy = float(np.sum(original_centered**2))
    cleaned_energy = float(np.sum(cleaned_centered**2))
    if original_energy <= 0.0 or cleaned_energy <= 0.0:
        raise ValueError("correlation requires non-constant original and cleaned data")
    signal_correlation = float(
        np.sum(original_centered * cleaned_centered)
        / np.sqrt(original_energy * cleaned_energy)
    )
    change_rms = float(np.sqrt(np.mean((cleaned_values - original_values) ** 2)))
    original_rms = float(np.sqrt(np.mean(original_values**2)))
    if original_rms <= 0.0:
        raise ValueError("normalized change requires non-zero original data")

    original_spectra = _windowed_spectra(original_values, bounds)
    cleaned_spectra = _windowed_spectra(cleaned_values, bounds)
    frequencies_hz = np.fft.rfftfreq(
        window_samples,
        d=1.0 / sampling_frequency_hz,
    )
    band_metrics = tuple(
        _band_metric(
            name,
            low_hz,
            high_hz,
            frequencies_hz,
            original_spectra,
            cleaned_spectra,
        )
        for name, low_hz, high_hz in validated_bands
    )
    return PreservationMetrics(
        float(np.clip(signal_correlation, -1.0, 1.0)),
        change_rms / original_rms,
        band_metrics,
    )
