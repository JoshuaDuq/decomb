"""Target-blind prediction of a periodic artifact from its higher harmonics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

LOWEST_REFERENCE_HZ = 20.0
HIGHEST_REFERENCE_HZ = 95.0


def high_harmonic_numbers(
    fundamental_hz: float,
    nyquist_hz: float,
) -> tuple[int, ...]:
    """Return consecutive harmonics inside the high-frequency reference band."""
    if not np.isfinite(fundamental_hz) or fundamental_hz <= 0.0:
        raise ValueError("fundamental_hz must be finite and positive")
    if not np.isfinite(nyquist_hz) or nyquist_hz <= HIGHEST_REFERENCE_HZ:
        raise ValueError("Nyquist must exceed the 95 Hz reference ceiling")

    first = int(np.ceil(LOWEST_REFERENCE_HZ / fundamental_hz))
    last = int(np.floor(HIGHEST_REFERENCE_HZ / fundamental_hz))
    harmonics = tuple(range(first, last + 1))
    if len(harmonics) < 2:
        raise ValueError("at least two adjacent high harmonics are required")
    return harmonics


def _validated_data(data: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("data must be a finite channels-by-samples array")
    return values


def harmonic_coefficients(
    data: NDArray[np.floating],
    sampling_frequency_hz: float,
    bounds: Sequence[tuple[int, int]],
    frequencies_hz: Sequence[float],
) -> NDArray[np.complex128]:
    """Project equal-length windows onto exact positive Fourier bins."""
    values = _validated_data(data)
    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0.0:
        raise ValueError("sampling_frequency_hz must be finite and positive")

    windows = tuple((int(start), int(stop)) for start, stop in bounds)
    if not windows or len({stop - start for start, stop in windows}) != 1:
        raise ValueError("bounds must contain equal-length windows")

    window_samples = windows[0][1] - windows[0][0]
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty sequence")
    bins = frequencies * window_samples / sampling_frequency_hz
    rounded_bins = np.rint(bins).astype(int)
    if not np.allclose(bins, rounded_bins, rtol=0.0, atol=1e-12):
        raise ValueError("frequencies must lie exactly on the window Fourier grid")
    if rounded_bins[0] <= 0 or rounded_bins[-1] >= window_samples // 2:
        raise ValueError("frequencies must lie strictly between DC and Nyquist")

    coefficients = []
    for start, stop in windows:
        if start < 0 or stop > values.shape[1] or stop <= start:
            raise ValueError("window bounds must lie inside data")
        spectrum = np.fft.rfft(values[:, start:stop], axis=-1)
        coefficients.append(2.0 * spectrum[:, rounded_bins] / window_samples)
    return np.stack(coefficients)


def extract_adjacent_features(
    data: NDArray[np.floating],
    sampling_frequency_hz: float,
    bounds: Sequence[tuple[int, int]],
    *,
    fundamental_hz: float,
) -> NDArray[np.complex128]:
    """Return high-only cross-products whose phase advances at the fundamental."""
    harmonics = high_harmonic_numbers(
        fundamental_hz,
        sampling_frequency_hz / 2.0,
    )
    frequencies = np.asarray(harmonics) * fundamental_hz
    coefficients = harmonic_coefficients(
        data,
        sampling_frequency_hz,
        bounds,
        frequencies,
    )
    products = coefficients[:, :, 1:] * np.conj(coefficients[:, :, :-1])
    return products.mean(axis=1)
