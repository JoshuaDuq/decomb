"""Small deterministic spectral primitives used by the correction pipeline."""

from __future__ import annotations

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


def hann_resolution_hz(segment_seconds: float) -> float:
    """Return the Hann half-power width, the narrowest resolvable line."""
    if not np.isfinite(segment_seconds) or segment_seconds <= 0.0:
        raise ValueError("segment_seconds must be finite and positive.")
    return 1.4382 / float(segment_seconds)
