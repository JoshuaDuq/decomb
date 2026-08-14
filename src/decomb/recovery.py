"""Signal-preserving removal before residual FIR notching."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from mne.filter import notch_filter
from numpy.typing import NDArray
from scipy.signal import find_peaks
from scipy.stats import kurtosis

from decomb import recordings


def _validated_frequency_tuple(
    frequencies_hz: Iterable[float],
) -> tuple[float, ...]:
    frequencies = tuple(float(frequency) for frequency in frequencies_hz)
    if not frequencies:
        raise ValueError("frequencies must not be empty")
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
    if frequencies[-1] >= nyquist_hz:
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


@dataclass(frozen=True)
class TrajectoryPCASettings:
    """Published rsPCA controls adapted to authorized artifact frequencies."""

    segment_s: float
    secondary_peak_ratio: float = 0.03
    maximum_excess_kurtosis: float = -0.5
    minimum_variance_fraction: float = 1e-5
    maximum_depth: int = 2

    def __post_init__(self) -> None:
        if not np.isfinite(self.segment_s) or self.segment_s <= 0.0:
            raise ValueError("segment_s must be finite and positive")
        if not np.isfinite(self.secondary_peak_ratio) or not (
            0.0 < self.secondary_peak_ratio < 1.0
        ):
            raise ValueError("secondary_peak_ratio must lie between zero and one")
        if not np.isfinite(self.maximum_excess_kurtosis):
            raise ValueError("maximum_excess_kurtosis must be finite")
        if not np.isfinite(self.minimum_variance_fraction) or not (
            0.0 < self.minimum_variance_fraction < 1.0
        ):
            raise ValueError("minimum_variance_fraction must lie between zero and one")
        if (
            not isinstance(self.maximum_depth, int)
            or isinstance(self.maximum_depth, bool)
            or self.maximum_depth < 1
        ):
            raise ValueError("maximum_depth must be a positive integer")


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
        verbose=False,
    )
    return SignalRecoveryResult(
        cleaned_data,
        values - cleaned_data,
        frequencies,
    )


def _validated_trigger_samples(
    trigger_samples: Iterable[int],
    *,
    n_times: int,
    cycle_samples: int,
) -> NDArray[np.int64]:
    values = np.asarray(tuple(trigger_samples), dtype=float)
    rounded = np.rint(values)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("trigger samples must contain at least two finite values")
    if not np.array_equal(values, rounded):
        raise ValueError("trigger samples must be integers")

    samples = rounded.astype(np.int64)
    if np.any(samples < 0) or np.any(samples >= n_times):
        raise ValueError("trigger samples must lie within the signal")
    if np.any(np.diff(samples) != cycle_samples):
        raise ValueError("trigger intervals must equal the configured repetition time")
    return samples


def _authorized_cycle_design(
    cycle_samples: int,
    sampling_frequency_hz: float,
    repetition_time_s: float,
    frequencies_hz: tuple[float, ...],
) -> NDArray[np.float64]:
    harmonic_numbers = np.asarray(frequencies_hz) * repetition_time_s
    if not np.allclose(
        harmonic_numbers,
        np.rint(harmonic_numbers),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "trigger-locked frequencies must be harmonics of the repetition time"
        )

    times_s = np.arange(cycle_samples) / sampling_frequency_hz
    phases = 2.0 * np.pi * np.outer(times_s, frequencies_hz)
    return np.column_stack((np.sin(phases), np.cos(phases)))


def _lattice_slices(
    n_times: int,
    origin_sample: int,
    cycle_samples: int,
) -> tuple[tuple[slice, slice], ...]:
    first_cycle = (-origin_sample) // cycle_samples
    last_cycle = (n_times - 1 - origin_sample) // cycle_samples
    slices = []
    for cycle_index in range(first_cycle, last_cycle + 1):
        cycle_start = origin_sample + cycle_index * cycle_samples
        data_start = max(0, cycle_start)
        data_stop = min(n_times, cycle_start + cycle_samples)
        phase_start = data_start - cycle_start
        phase_stop = phase_start + data_stop - data_start
        slices.append(
            (slice(data_start, data_stop), slice(phase_start, phase_stop))
        )
    return tuple(slices)


def subtract_trigger_locked_optimal_basis(
    data: NDArray[np.floating],
    sampling_frequency_hz: float,
    frequencies_hz: Iterable[float],
    trigger_samples: Iterable[int],
    *,
    repetition_time_s: float,
    component_count: int,
) -> SignalRecoveryResult:
    """Subtract a trigger template and its leading authorized basis variations."""
    values = _validated_channel_data(data)
    frequencies = validate_frequencies(
        frequencies_hz,
        sampling_frequency_hz,
    )
    cycle_samples = recordings.estimation_window_samples(
        sampling_frequency_hz,
        repetition_time_s,
    )
    triggers = _validated_trigger_samples(
        trigger_samples,
        n_times=values.shape[-1],
        cycle_samples=cycle_samples,
    )
    if (
        not isinstance(component_count, int)
        or isinstance(component_count, bool)
        or component_count < 1
    ):
        raise ValueError("component_count must be a positive integer")

    design = _authorized_cycle_design(
        cycle_samples,
        sampling_frequency_hz,
        repetition_time_s,
        frequencies,
    )
    complete_triggers = triggers[triggers + cycle_samples <= values.shape[-1]]
    maximum_components = min(complete_triggers.size - 1, design.shape[1])
    if component_count > maximum_components:
        raise ValueError(
            f"component_count exceeds the {maximum_components} estimable components"
        )

    coefficient_map = np.linalg.pinv(design)
    artifact_data = np.zeros_like(values)
    lattice_slices = _lattice_slices(
        values.shape[-1],
        int(triggers[0]),
        cycle_samples,
    )
    for channel_index, channel_data in enumerate(values):
        cycles = np.stack(
            [channel_data[start : start + cycle_samples] for start in complete_triggers]
        )
        channel_coefficients = cycles @ coefficient_map.T
        mean_coefficients = channel_coefficients.mean(axis=0)
        centered_coefficients = channel_coefficients - mean_coefficients
        _, _, principal_directions = np.linalg.svd(
            centered_coefficients,
            full_matrices=False,
        )
        basis = design @ principal_directions[:component_count].T
        template = design @ mean_coefficients

        for data_slice, phase_slice in lattice_slices:
            segment_basis = basis[phase_slice]
            segment_template = template[phase_slice]
            residual = values[channel_index, data_slice] - segment_template
            scores, _, _, _ = np.linalg.lstsq(
                segment_basis,
                residual,
                rcond=None,
            )
            artifact_data[channel_index, data_slice] = (
                segment_template + segment_basis @ scores
            )

    return SignalRecoveryResult(
        values - artifact_data,
        artifact_data,
        frequencies,
    )


def _significant_spectral_peaks(
    waveform: NDArray[np.float64],
    sampling_frequency_hz: float,
    relative_height: float,
) -> tuple[NDArray[np.float64], float]:
    windowed = waveform * np.hamming(waveform.size)
    amplitudes = np.abs(np.fft.rfft(windowed))
    frequencies_hz = np.fft.rfftfreq(
        waveform.size,
        d=1.0 / sampling_frequency_hz,
    )
    amplitudes[0] = 0.0
    peak_indices, _ = find_peaks(amplitudes)
    if peak_indices.size == 0:
        return np.empty(0), sampling_frequency_hz / waveform.size

    peak_amplitudes = amplitudes[peak_indices]
    significant = peak_amplitudes >= peak_amplitudes.max() * relative_height
    return (
        frequencies_hz[peak_indices[significant]],
        sampling_frequency_hz / waveform.size,
    )


def _contains_authorized_peak(
    peak_frequencies_hz: NDArray[np.float64],
    frequencies_hz: tuple[float, ...],
    resolution_hz: float,
) -> NDArray[np.bool_]:
    if peak_frequencies_hz.size == 0:
        return np.empty(0, dtype=bool)
    distances_hz = np.abs(
        peak_frequencies_hz[:, np.newaxis] - np.asarray(frequencies_hz)
    )
    return np.min(distances_hz, axis=1) <= resolution_hz / 2.0 + 1e-12


def _overlap_average(
    windows: NDArray[np.float64],
    bounds: tuple[tuple[int, int], ...],
    n_times: int,
) -> NDArray[np.float64]:
    values = np.zeros(n_times)
    counts = np.zeros(n_times)
    for window, (start, stop) in zip(windows, bounds, strict=True):
        values[start:stop] += window
        counts[start:stop] += 1.0
    if np.any(counts == 0.0):
        raise RuntimeError("trajectory windows did not cover the complete signal")
    return values / counts


def _trajectory_candidate_artifact(
    waveform: NDArray[np.float64],
    window_scores: NDArray[np.float64],
    component_energy: float,
    total_energy: float,
    bounds: tuple[tuple[int, int], ...],
    n_times: int,
    sampling_frequency_hz: float,
    frequencies_hz: tuple[float, ...],
    settings: TrajectoryPCASettings,
    depth: int,
) -> NDArray[np.float64] | None:
    variance_fraction = component_energy / total_energy
    if variance_fraction < settings.minimum_variance_fraction:
        return None
    peak_frequencies_hz, resolution_hz = _significant_spectral_peaks(
        waveform,
        sampling_frequency_hz,
        settings.secondary_peak_ratio,
    )
    authorized_peaks = _contains_authorized_peak(
        peak_frequencies_hz,
        frequencies_hz,
        resolution_hz,
    )
    if not authorized_peaks.any():
        return None

    component = _overlap_average(
        np.outer(window_scores, waveform),
        bounds,
        n_times,
    )
    is_single_authorized_peak = peak_frequencies_hz.size == 1 and authorized_peaks[0]
    excess_kurtosis = float(kurtosis(component, fisher=True, bias=False))
    if (
        is_single_authorized_peak
        and np.isfinite(excess_kurtosis)
        and excess_kurtosis <= settings.maximum_excess_kurtosis
    ):
        return component
    if depth == settings.maximum_depth:
        return None
    return _trajectory_component_artifact(
        component,
        sampling_frequency_hz,
        frequencies_hz,
        settings,
        depth + 1,
    )


def _trajectory_component_artifact(
    signal: NDArray[np.float64],
    sampling_frequency_hz: float,
    frequencies_hz: tuple[float, ...],
    settings: TrajectoryPCASettings,
    depth: int,
) -> NDArray[np.float64]:
    segment_samples = recordings.estimation_window_samples(
        sampling_frequency_hz,
        settings.segment_s,
    )
    bounds = recordings.overlapping_window_bounds(
        n_times=signal.size,
        window_samples=segment_samples,
        overlap=0.5,
    )
    windows = np.stack([signal[start:stop] for start, stop in bounds])
    mean_window = windows.mean(axis=0)
    centered_windows = windows - mean_window
    left_vectors, singular_values, directions = np.linalg.svd(
        centered_windows,
        full_matrices=False,
    )
    total_energy = float(np.sum(windows**2))
    artifact = np.zeros_like(signal)

    mean_scores = np.ones(windows.shape[0])
    mean_artifact = _trajectory_candidate_artifact(
        mean_window,
        mean_scores,
        float(mean_scores.size * np.sum(mean_window**2)),
        total_energy,
        bounds,
        signal.size,
        sampling_frequency_hz,
        frequencies_hz,
        settings,
        depth,
    )
    if mean_artifact is not None:
        artifact += mean_artifact

    for index, (singular_value, direction) in enumerate(
        zip(singular_values, directions, strict=True)
    ):
        component_artifact = _trajectory_candidate_artifact(
            direction,
            left_vectors[:, index] * singular_value,
            float(singular_value**2),
            total_energy,
            bounds,
            signal.size,
            sampling_frequency_hz,
            frequencies_hz,
            settings,
            depth,
        )
        if component_artifact is not None:
            artifact += component_artifact
    return artifact


def subtract_recursive_trajectory_pca(
    data: NDArray[np.floating],
    sampling_frequency_hz: float,
    frequencies_hz: Iterable[float],
    settings: TrajectoryPCASettings,
) -> SignalRecoveryResult:
    """Subtract clean-room rsPCA components tied to authorized frequencies."""
    values = _validated_channel_data(data)
    frequencies = validate_frequencies(
        frequencies_hz,
        sampling_frequency_hz,
    )
    segment_samples = recordings.estimation_window_samples(
        sampling_frequency_hz,
        settings.segment_s,
    )
    if values.shape[-1] < segment_samples:
        raise ValueError("signal is shorter than one trajectory segment")

    artifact_data = np.stack(
        [
            _trajectory_component_artifact(
                channel,
                sampling_frequency_hz,
                frequencies,
                settings,
                depth=1,
            )
            for channel in values
        ]
    )
    return SignalRecoveryResult(
        values - artifact_data,
        artifact_data,
        frequencies,
    )
