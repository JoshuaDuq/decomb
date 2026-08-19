"""Signal-preserving removal before residual FIR notching."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial

import numpy as np
from joblib import Parallel, delayed
from mne.filter import notch_filter
from numpy.typing import NDArray
from scipy.signal import fftconvolve, find_peaks

from decomb import recordings

#: Channel counts are small and the per-channel cost is large, so a process-based
#: backend earns its startup cost. ``n_jobs=1`` runs in-process with no pool at all.
PARALLEL_BACKEND = "loky"


def _map_channels(function, values, n_jobs: int):
    """Apply a per-channel function down the channel axis, optionally in parallel.

    Every recovery loop here reads one channel row and writes that row only, so the
    channel axis is the natural unit of work and the parallel result is identical to
    the sequential one.
    """
    if recordings.validated_n_jobs(n_jobs) == 1:
        return np.stack([function(channel) for channel in values])
    return np.stack(
        Parallel(n_jobs=n_jobs, backend=PARALLEL_BACKEND)(
            delayed(function)(channel) for channel in values
        )
    )


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


@dataclass(frozen=True)
class SpatialLineSubspaceModel:
    """Frozen spatial artifact basis for jointly fitted line frequencies."""

    frequencies_hz: tuple[float, ...]
    basis: NDArray[np.floating]

    def __post_init__(self) -> None:
        frequencies = _validated_frequency_tuple(self.frequencies_hz)
        basis = np.asarray(self.basis, dtype=np.float64)
        if basis.ndim != 2 or basis.shape[1] < 1:
            raise ValueError("spatial basis must have shape (channels, rank)")
        if not np.isfinite(basis).all():
            raise ValueError("spatial basis must be finite")
        if not np.allclose(
            basis.T @ basis,
            np.eye(basis.shape[1]),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("spatial basis columns must be orthonormal")

        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "basis", basis)

    @property
    def channel_count(self) -> int:
        """Number of channels represented by the spatial basis."""
        return self.basis.shape[0]

    @property
    def rank(self) -> int:
        """Number of removed line-artifact spatial directions."""
        return self.basis.shape[1]


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


def fit_spatial_line_subspace(
    data: NDArray[np.floating],
    sampling_frequency_hz: float,
    frequencies_hz: Iterable[float],
    *,
    window_s: float,
    rank: int,
    n_jobs: int = 1,
) -> SpatialLineSubspaceModel:
    """Learn fixed spatial bases from MNE line estimates of background data."""
    values = _validated_channel_data(data)
    frequencies = validate_frequencies(frequencies_hz, sampling_frequency_hz)
    if (
        not isinstance(rank, int)
        or isinstance(rank, bool)
        or rank < 1
        or rank > values.shape[0]
    ):
        raise ValueError("rank must be a positive integer no larger than channels")

    artifact = subtract_multitaper_sinusoids(
        values,
        sampling_frequency_hz,
        frequencies,
        window_s=window_s,
        n_jobs=n_jobs,
    ).artifact_data
    try:
        basis = _leading_spatial_basis(artifact, rank)
    except ValueError as error:
        raise ValueError(
            f"rank {rank} exceeds the joint line estimate's numerical rank"
        ) from error
    return SpatialLineSubspaceModel(frequencies, basis)


def _leading_spatial_basis(
    artifact_data: NDArray[np.floating],
    rank: int,
) -> NDArray[np.float64]:
    """Compute left singular vectors through the small channel covariance."""
    artifact = _validated_channel_data(artifact_data)
    eigenvalues, eigenvectors = np.linalg.eigh(artifact @ artifact.T)
    order = np.argsort(eigenvalues)[::-1]
    singular_values = np.sqrt(np.maximum(eigenvalues[order], 0.0))
    numerical_tolerance = (
        np.finfo(float).eps * max(artifact.shape) * singular_values[0]
    )
    if rank > artifact.shape[0] or singular_values[rank - 1] <= numerical_tolerance:
        raise ValueError("rank exceeds the line estimate's numerical rank")
    return eigenvectors[:, order[:rank]]


def subtract_spatial_line_subspace(
    data: NDArray[np.floating],
    sampling_frequency_hz: float,
    model: SpatialLineSubspaceModel,
    *,
    window_s: float,
    n_jobs: int = 1,
) -> SignalRecoveryResult:
    """Subtract only fitted line activity inside frozen spatial artifact bases."""
    values = _validated_channel_data(data)
    validate_frequencies(model.frequencies_hz, sampling_frequency_hz)
    if values.shape[0] != model.channel_count:
        raise ValueError("data channels must match the spatial line model")

    line_estimate = subtract_multitaper_sinusoids(
        values,
        sampling_frequency_hz,
        model.frequencies_hz,
        window_s=window_s,
        n_jobs=n_jobs,
    ).artifact_data
    artifact_data = model.basis @ (model.basis.T @ line_estimate)
    return SignalRecoveryResult(
        values - artifact_data,
        artifact_data,
        model.frequencies_hz,
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


def _trigger_locked_channel_artifact(
    channel_data: NDArray[np.float64],
    *,
    design: NDArray[np.float64],
    coefficient_map: NDArray[np.float64],
    complete_triggers: NDArray[np.int64],
    cycle_samples: int,
    lattice_slices: tuple[tuple[slice, slice], ...],
    component_count: int,
) -> NDArray[np.float64]:
    """Fit one channel's cycle template and its leading basis variations."""
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

    artifact = np.zeros_like(channel_data)
    for data_slice, phase_slice in lattice_slices:
        segment_basis = basis[phase_slice]
        segment_template = template[phase_slice]
        residual = channel_data[data_slice] - segment_template
        scores, _, _, _ = np.linalg.lstsq(
            segment_basis,
            residual,
            rcond=None,
        )
        artifact[data_slice] = segment_template + segment_basis @ scores
    return artifact


def subtract_trigger_locked_optimal_basis(
    data: NDArray[np.floating],
    sampling_frequency_hz: float,
    frequencies_hz: Iterable[float],
    trigger_samples: Iterable[int],
    *,
    repetition_time_s: float,
    component_count: int,
    n_jobs: int = 1,
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
    lattice_slices = _lattice_slices(
        values.shape[-1],
        int(triggers[0]),
        cycle_samples,
    )
    artifact_data = _map_channels(
        partial(
            _trigger_locked_channel_artifact,
            design=design,
            coefficient_map=coefficient_map,
            complete_triggers=complete_triggers,
            cycle_samples=cycle_samples,
            lattice_slices=lattice_slices,
            component_count=component_count,
        ),
        values,
        n_jobs,
    )
    return SignalRecoveryResult(
        values - artifact_data,
        artifact_data,
        frequencies,
    )


def _component_scores(
    signal: NDArray[np.float64],
    vector: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project the trajectory matrix onto one eigenvector.

    This is column ``i`` of ``X.T @ eigenvectors``, where ``X[k, t] == signal[t + k]``,
    so the projection is a plain cross-correlation of the signal with the eigenvector.
    Forming the whole score matrix instead costs ``n_samples * dim`` floats -- 1.2 GB
    for one EEG channel at 1 kHz and ``dim=300`` -- and materializes a contiguous copy
    of the strided trajectory view to hand BLAS, for a second 1.2 GB. Only the
    components that are actually reconstructed need their scores, so they are computed
    here on demand instead.
    """
    return fftconvolve(signal, vector[::-1], mode="valid")


def _diagonal_average(
    score: NDArray[np.float64],
    vector: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Reconstruct a signal from an SSA component by diagonal averaging."""
    dim = vector.size
    n_times = score.size + dim - 1
    weights = np.concatenate(
        [
            np.arange(1, dim),
            np.full(n_times - 2 * dim + 2, dim),
            np.arange(dim - 1, 0, -1),
        ]
    )
    return fftconvolve(score, vector, mode="full") / weights


def _rspca_component_spectral_peaks(
    vector: NDArray[np.float64],
    sampling_frequency_hz: float,
    relative_height: float,
) -> tuple[float, int]:
    """Find the max frequency and the number of significant peaks in the spectrum."""
    dim = vector.size
    fpt = 2 ** (int(np.log2(dim)) + 2)
    windowed = vector * np.hamming(dim)
    
    amplitudes = np.abs(np.fft.rfft(windowed, n=fpt))
    if fpt % 2 == 0:
        amplitudes = amplitudes[:-1]
        
    max_index = int(np.argmax(amplitudes))
    fc_hz = (max_index + 1) * sampling_frequency_hz / fpt
    
    peak_indices, _ = find_peaks(amplitudes)
    if peak_indices.size == 0:
        return fc_hz, 0
    
    peak_amplitudes = amplitudes[peak_indices]
    significant_count = int(np.sum(peak_amplitudes >= amplitudes.max() * relative_height))
    return fc_hz, significant_count


def _excess_kurtosis(values: NDArray[np.float64]) -> float:
    """Biased Fisher excess kurtosis, as `scipy.stats.kurtosis(..., bias=True)` defines it.

    Written out because the gate calls it once per component per recursion level -- 13,629
    times for one channel at dim=300 -- and scipy's dispatch wrapper costs far more than the
    two central moments it is wrapping. Returns 0.0 for a constant series, where the ratio
    is undefined, matching scipy's own degenerate handling rather than propagating a nan
    into the comparison.
    """
    centred = values - values.mean()
    second = float(np.mean(centred**2))
    if second <= 0.0:
        return 0.0
    fourth = float(np.mean(centred**4))
    return fourth / second**2 - 3.0


def _rs_pca_recursive(
    signal: NDArray[np.float64],
    dim: int,
    original_variance_fraction: float,
    sampling_frequency_hz: float,
    settings: TrajectoryPCASettings,
    depth: int,
) -> NDArray[np.float64]:
    """Recursively identify and remove single-peak components from the trajectory matrix."""
    n_times = signal.size
    n_samples = n_times - dim + 1
    
    X = np.lib.stride_tricks.sliding_window_view(signal, dim).T
    rmX = X - np.mean(X, axis=1, keepdims=True)
    
    covariance = (rmX @ rmX.T) / n_samples
    # 1.2 GB for one channel at 1 kHz and dim=300, and nothing below reads it again.
    # Holding it through the depth-2 recursion is what put a single channel at 6.8 GB.
    del rmX
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total_variance = np.sum(eigenvalues)
    normal_d = eigenvalues / total_variance if total_variance > 0 else np.zeros_like(eigenvalues)

    # Start from the whole signal and subtract only what is removed, which is what the
    # published MATLAB does: `rXbar = sig - sum(reconXbar(idx,:))`. Summing the keepers
    # instead is equivalent, because every component summed reconstructs the signal
    # exactly, but it costs a diagonal average for each of `dim` components at every
    # recursion level. That reconstruction is the dominant cost at EEG sampling rates,
    # and a component that is kept unchanged does not need reconstructing to be kept.
    result = signal.copy()

    for i in range(dim):
        variance_fraction = normal_d[i] * original_variance_fraction
        vector = eigenvectors[:, i]

        fc_hz, num_peaks = _rspca_component_spectral_peaks(
            vector,
            sampling_frequency_hz,
            settings.secondary_peak_ratio,
        )
        is_single_peak_gamma = fc_hz >= 20.0 and num_peaks <= 1
        may_recurse = (
            not is_single_peak_gamma
            and variance_fraction >= settings.minimum_variance_fraction
            and depth < settings.maximum_depth
        )
        # Neither removed nor descended into: it survives untouched, and it is already
        # present in `result` because that started as the signal itself.
        if not is_single_peak_gamma and not may_recurse:
            continue

        recon_component = _diagonal_average(
            _component_scores(signal, vector),
            vector,
        )
        middle_slice = recon_component[2 * dim - 1 : n_times - 2 * dim + 1]
        if middle_slice.size < 4:
            component_kurtosis = 0.0
        else:
            component_kurtosis = _excess_kurtosis(middle_slice)

        if is_single_peak_gamma:
            if component_kurtosis < settings.maximum_excess_kurtosis:
                result -= recon_component
            continue

        if component_kurtosis <= 100.0:
            cleaned_component = _rs_pca_recursive(
                recon_component,
                dim,
                variance_fraction,
                sampling_frequency_hz,
                settings,
                depth + 1,
            )
            result += cleaned_component - recon_component

    return result


def _clean_channel_with_trajectory_pca(
    channel_signal: NDArray[np.float64],
    *,
    dim: int,
    sampling_frequency_hz: float,
    settings: TrajectoryPCASettings,
) -> NDArray[np.float64]:
    """Normalize one channel, remove its rsPCA components, and restore its scale."""
    original_mean = np.mean(channel_signal)
    original_std = np.std(channel_signal, ddof=1)
    if original_std == 0.0:
        return channel_signal

    normalized_signal = (channel_signal - original_mean) / original_std
    cleaned_normalized = _rs_pca_recursive(
        normalized_signal,
        dim,
        1.0,
        sampling_frequency_hz,
        settings,
        depth=1,
    )
    return (cleaned_normalized * original_std) + original_mean


def subtract_recursive_trajectory_pca(
    data: NDArray[np.floating],
    sampling_frequency_hz: float,
    settings: TrajectoryPCASettings,
    *,
    n_jobs: int = 1,
) -> SignalRecoveryResult:
    """Subtract true rsPCA artifact components using recursive delay embedding."""
    values = _validated_channel_data(data)

    dim = int(np.round(sampling_frequency_hz * settings.segment_s))
    if values.shape[-1] < dim:
        raise ValueError("signal is shorter than one trajectory segment")

    cleaned_data = _map_channels(
        partial(
            _clean_channel_with_trajectory_pca,
            dim=dim,
            sampling_frequency_hz=sampling_frequency_hz,
            settings=settings,
        ),
        values,
        n_jobs,
    )
    return SignalRecoveryResult(cleaned_data, values - cleaned_data, tuple())
