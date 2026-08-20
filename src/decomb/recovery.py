"""Subtract a fitted sinusoid at each authorized frequency.

Only the multitaper line fit ships. The spatial-subspace, trigger-locked and
trajectory-PCA candidates evaluated alongside it live on `archive/full-tree`;
`docs/rspca_validation.md` records why trajectory PCA is not a decomb.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from mne.filter import notch_filter
from numpy.typing import NDArray

from decomb import recordings

PARALLEL_BACKEND = "loky"

def _validated_frequency_tuple(
    frequencies_hz: Iterable[float],
) -> tuple[float, ...]:
    frequencies = tuple(float(frequency) for frequency in frequencies_hz)
    if not np.isfinite(frequencies).all():
        raise ValueError("frequencies must be finite")
    if any(frequency <= 0 for frequency in frequencies):
        raise ValueError("frequencies must be positive")
    if tuple(sorted(set(frequencies))) != frequencies:
        raise ValueError("frequencies must be sorted unique values")
    return frequencies

def validate_frequencies(
    frequencies_hz: Iterable[float],
    sampling_frequency_hz: float,
) -> tuple[float, ...]:
    """Validate target frequencies against the sampling frequency."""
    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0:
        raise ValueError("sampling frequency must be finite and positive")

    frequencies = _validated_frequency_tuple(frequencies_hz)
    nyquist_hz = sampling_frequency_hz / 2.0
    if frequencies and frequencies[-1] >= nyquist_hz:
        raise ValueError("frequencies must be strictly below Nyquist")
    return frequencies

@dataclass(frozen=True)
class SignalRecoveryResult:
    """Cleaned signal, reconstructed artifact, and targeted frequencies."""

    cleaned_data: NDArray[np.floating]
    artifact_data: NDArray[np.floating]
    frequencies_hz: tuple[float, ...]

    def __post_init__(self) -> None:
        cleaned_data = np.asarray(self.cleaned_data)
        artifact_data = np.asarray(self.artifact_data)
        if cleaned_data.ndim != 2 or artifact_data.shape != cleaned_data.shape:
            raise ValueError(
                "cleaned and artifact data must have the same two-dimensional "
                "shape"
            )
        if not np.isfinite(cleaned_data).all() or not np.isfinite(
            artifact_data
        ).all():
            raise ValueError("cleaned and artifact data must be finite")

        object.__setattr__(self, "cleaned_data", cleaned_data)
        object.__setattr__(self, "artifact_data", artifact_data)
        object.__setattr__(
            self,
            "frequencies_hz",
            _validated_frequency_tuple(self.frequencies_hz),
        )

def _validated_channel_data(data: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("channel data must be a two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("channel data must be finite")
    return values

def subtract_multitaper_sinusoids(
    data: NDArray[np.floating],
    sampling_frequency_hz: float,
    frequencies_hz: Iterable[float],
    *,
    window_s: float,
    n_jobs: int = 1,
) -> SignalRecoveryResult:
    """Subtract MNE multitaper sinusoid fits at authorized frequencies."""
    values = _validated_channel_data(data)
    frequencies = validate_frequencies(
        frequencies_hz,
        sampling_frequency_hz,
    )
    window_samples = recordings.estimation_window_samples(
        sampling_frequency_hz,
        window_s,
    )
    if values.shape[-1] < window_samples:
        raise ValueError("signal is shorter than one recovery window")

    cleaned_data = notch_filter(
        values,
        sampling_frequency_hz,
        np.asarray(frequencies),
        filter_length=window_samples,
        notch_widths=0.0,
        method="spectrum_fit",
        copy=True,
        n_jobs=recordings.validated_n_jobs(n_jobs),
        verbose=False,
    )
    return SignalRecoveryResult(
        cleaned_data,
        values - cleaned_data,
        frequencies,
    )
