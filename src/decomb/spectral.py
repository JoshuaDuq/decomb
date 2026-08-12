"""Small deterministic spectral primitives used by the correction pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def hann_periodogram(data: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a one-sided Hann-windowed power spectral density."""
    values = np.asarray(data, dtype=float)
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("data must have at least two samples along the last axis.")
    if not np.isfinite(sfreq) or sfreq <= 0.0:
        raise ValueError("sfreq must be finite and positive.")
    if not np.all(np.isfinite(values)):
        raise ValueError("data must contain only finite values.")

    sample_count = values.shape[-1]
    window = np.hanning(sample_count)
    normalisation = sfreq * float(np.sum(window**2))
    spectrum = np.fft.rfft(values * window, axis=-1)
    power = np.abs(spectrum) ** 2 / normalisation
    if sample_count % 2 == 0:
        power[..., 1:-1] *= 2.0
    else:
        power[..., 1:] *= 2.0
    frequencies_hz = np.fft.rfftfreq(sample_count, d=1.0 / sfreq)
    return frequencies_hz, power


def to_db(power: np.ndarray) -> np.ndarray:
    """Convert non-negative power to decibels without taking log of zero."""
    values = np.asarray(power, dtype=float)
    if np.any(values < 0.0):
        raise ValueError("power must be non-negative.")
    return 10.0 * np.log10(np.maximum(values, np.finfo(float).tiny))


def refine_peak_frequency(
    frequencies_hz: Sequence[float],
    spectrum_db: Sequence[float],
    index: int,
) -> float:
    """Refine a local maximum with a three-point parabola in decibel space."""
    frequencies = np.asarray(frequencies_hz, dtype=float)
    spectrum = np.asarray(spectrum_db, dtype=float)
    if frequencies.shape != spectrum.shape:
        raise ValueError("frequencies_hz and spectrum_db must have the same shape.")
    if not 0 < index < spectrum.size - 1:
        raise ValueError("index must have a neighbour on either side.")

    left, centre, right = spectrum[index - 1 : index + 2]
    denominator = left - 2.0 * centre + right
    if denominator == 0.0:
        return float(frequencies[index])
    shift = 0.5 * (left - right) / denominator
    if not np.isfinite(shift) or abs(shift) > 0.5:
        return float(frequencies[index])
    bin_width_hz = float(frequencies[1] - frequencies[0])
    return float(frequencies[index] + shift * bin_width_hz)


def hann_resolution_hz(segment_seconds: float) -> float:
    """Return the Hann half-power width, the narrowest resolvable line."""
    if not np.isfinite(segment_seconds) or segment_seconds <= 0.0:
        raise ValueError("segment_seconds must be finite and positive.")
    return 1.4382 / float(segment_seconds)
