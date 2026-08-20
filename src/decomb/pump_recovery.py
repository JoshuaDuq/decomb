"""Target-blind prediction of a periodic artifact from its higher harmonics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True)
class CrossHarmonicModel:
    """Frozen reduced-rank map from high-harmonic features to EEG coefficients."""

    channel_names: tuple[str, ...]
    feature_mean: NDArray[np.complex128]
    feature_scale: NDArray[np.float64]
    target_mean: NDArray[np.complex128]
    weights: NDArray[np.complex128]

    def predict(
        self,
        features: NDArray[np.complexfloating],
        channel_names: Sequence[str],
    ) -> NDArray[np.complex128]:
        """Predict target coefficients without refitting on target data."""
        if tuple(channel_names) != self.channel_names:
            raise ValueError("channel names do not match the fitted model")
        values = np.asarray(features, dtype=np.complex128)
        if values.ndim != 2 or values.shape[1] != self.weights.shape[0]:
            raise ValueError("features do not match the fitted model")
        if not np.isfinite(values).all():
            raise ValueError("features must be finite")
        standardized = (values - self.feature_mean) / self.feature_scale
        return self.target_mean + standardized @ self.weights


def fit_complex_model(
    features: NDArray[np.complexfloating],
    targets: NDArray[np.complexfloating],
    channel_names: Sequence[str],
    *,
    rank: int,
    penalty: float,
) -> CrossHarmonicModel:
    """Fit one deterministic reduced-rank complex ridge model."""
    predictors = np.asarray(features, dtype=np.complex128)
    responses = np.asarray(targets, dtype=np.complex128)
    names = tuple(str(name) for name in channel_names)
    expected_target_shape = (predictors.shape[0], len(names))
    if predictors.ndim != 2 or responses.shape != expected_target_shape:
        raise ValueError("features and targets have incompatible shapes")
    if not names or len(set(names)) != len(names):
        raise ValueError("channel names must be non-empty and unique")
    if not np.isfinite(predictors).all() or not np.isfinite(responses).all():
        raise ValueError("training arrays must be finite")
    if not isinstance(rank, int) or isinstance(rank, bool) or not (
        1 <= rank <= min(predictors.shape)
    ):
        raise ValueError("rank must fit inside the feature matrix")
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("penalty must be finite and positive")

    feature_mean = predictors.mean(axis=0)
    feature_scale = np.sqrt(
        np.mean(np.abs(predictors - feature_mean) ** 2, axis=0)
    )
    if np.any(feature_scale == 0.0):
        raise ValueError("constant cross-harmonic features cannot train a model")
    target_mean = responses.mean(axis=0)
    standardized = (predictors - feature_mean) / feature_scale
    centered_targets = responses - target_mean
    left, singular_values, right = np.linalg.svd(
        standardized,
        full_matrices=False,
    )
    left = left[:, :rank]
    singular_values = singular_values[:rank]
    right = right[:rank]
    shrinkage = singular_values / (singular_values**2 + penalty)
    weights = right.conj().T @ (
        shrinkage[:, None] * (left.conj().T @ centered_targets)
    )
    return CrossHarmonicModel(
        names,
        feature_mean,
        feature_scale,
        target_mean,
        weights,
    )


@dataclass(frozen=True)
class ModelSelection:
    """Inner-validation result and the model refitted on every training row."""

    model: CrossHarmonicModel
    rank: int
    penalty: float
    validation_error: float
    validation_participants: tuple[str, ...]


def select_model(
    features: NDArray[np.complexfloating],
    targets: NDArray[np.complexfloating],
    participants: Sequence[str],
    channel_names: Sequence[str],
    *,
    ranks: Sequence[int],
    penalties: Sequence[float],
) -> ModelSelection:
    """Select rank and penalty with participant-balanced inner validation."""
    predictors = np.asarray(features, dtype=np.complex128)
    responses = np.asarray(targets, dtype=np.complex128)
    groups = np.asarray(tuple(str(value) for value in participants))
    if groups.shape != (predictors.shape[0],):
        raise ValueError("participants must label every feature row")
    validation_participants = tuple(sorted(set(groups)))
    if len(validation_participants) < 2:
        raise ValueError("model selection requires at least two participants")
    candidate_ranks = tuple(int(rank) for rank in ranks)
    candidate_penalties = tuple(float(penalty) for penalty in penalties)
    if not candidate_ranks or not candidate_penalties:
        raise ValueError("model selection candidates must not be empty")

    candidates = []
    for rank in candidate_ranks:
        for penalty in candidate_penalties:
            participant_errors = []
            for participant in validation_participants:
                validation = groups == participant
                training = ~validation
                model = fit_complex_model(
                    predictors[training],
                    responses[training],
                    channel_names,
                    rank=rank,
                    penalty=penalty,
                )
                prediction = model.predict(
                    predictors[validation],
                    channel_names,
                )
                target_energy = float(np.sum(np.abs(responses[validation]) ** 2))
                if target_energy <= 0.0:
                    raise ValueError("validation targets must have positive energy")
                error_energy = float(
                    np.sum(np.abs(prediction - responses[validation]) ** 2)
                )
                participant_errors.append(error_energy / target_energy)
            candidates.append(
                (float(np.mean(participant_errors)), rank, penalty)
            )

    validation_error, rank, penalty = min(candidates)
    model = fit_complex_model(
        predictors,
        responses,
        channel_names,
        rank=rank,
        penalty=penalty,
    )
    return ModelSelection(
        model,
        rank,
        penalty,
        validation_error,
        validation_participants,
    )


@dataclass(frozen=True)
class PumpLockTest:
    """Recording-level maximum-coherence phase-randomization result."""

    maximum_coherence: float
    p_value: float
    surrogate_count: int


def _channel_coherence(
    observed: NDArray[np.complex128],
    predicted: NDArray[np.complex128],
) -> NDArray[np.float64]:
    numerator = np.abs(np.sum(observed * np.conj(predicted), axis=0)) ** 2
    denominator = np.sum(np.abs(observed) ** 2, axis=0) * np.sum(
        np.abs(predicted) ** 2,
        axis=0,
    )
    if np.any(denominator <= 0.0):
        raise ValueError("pump-lock coherence requires non-zero channel energy")
    return numerator / denominator


def pump_lock_test(
    observed: NDArray[np.complexfloating],
    predicted: NDArray[np.complexfloating],
    *,
    surrogate_count: int,
    seed: int,
) -> PumpLockTest:
    """Test maximum channel coherence against shared phase randomization."""
    values = np.asarray(observed, dtype=np.complex128)
    reference = np.asarray(predicted, dtype=np.complex128)
    if values.shape != reference.shape or values.ndim != 2:
        raise ValueError("observed and predicted coefficients must share a 2D shape")
    if not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise ValueError("pump-lock coefficients must be finite")
    if not isinstance(surrogate_count, int) or isinstance(surrogate_count, bool):
        raise ValueError("surrogate_count must be an integer")
    if surrogate_count < 99:
        raise ValueError("at least 99 surrogates are required")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    observed_maximum = float(_channel_coherence(values, reference).max())
    rng = np.random.default_rng(seed)
    surrogate_maxima = np.empty(surrogate_count)
    for index in range(surrogate_count):
        phases = np.exp(
            2j * np.pi * rng.random(reference.shape[0])
        )
        randomized = reference * phases[:, None]
        surrogate_maxima[index] = _channel_coherence(values, randomized).max()
    exceedances = int(np.sum(surrogate_maxima >= observed_maximum))
    p_value = (exceedances + 1.0) / (surrogate_count + 1.0)
    return PumpLockTest(observed_maximum, p_value, surrogate_count)


def reconstruct_artifact(
    n_times: int,
    bounds: Sequence[tuple[int, int]],
    coefficients: NDArray[np.complexfloating],
    sampling_frequency_hz: float,
    frequency_hz: float,
) -> NDArray[np.float64]:
    """Overlap-add local complex coefficients into a real artifact waveform."""
    if not isinstance(n_times, int) or isinstance(n_times, bool) or n_times < 1:
        raise ValueError("n_times must be a positive integer")
    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0.0:
        raise ValueError("sampling_frequency_hz must be finite and positive")
    if not np.isfinite(frequency_hz) or not (
        0.0 < frequency_hz < sampling_frequency_hz / 2.0
    ):
        raise ValueError("frequency_hz must lie strictly between DC and Nyquist")

    windows = tuple((int(start), int(stop)) for start, stop in bounds)
    values = np.asarray(coefficients, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] != len(windows):
        raise ValueError("one coefficient row is required per window")
    if not np.isfinite(values).all():
        raise ValueError("predicted coefficients must be finite")

    artifact = np.zeros((values.shape[1], n_times), dtype=float)
    weights = np.zeros(n_times, dtype=float)
    for row, (start, stop) in enumerate(windows):
        if start < 0 or stop > n_times or stop <= start:
            raise ValueError("window bounds must lie inside the output")
        sample_times_s = np.arange(start, stop) / sampling_frequency_hz
        carrier = np.exp(2j * np.pi * frequency_hz * sample_times_s)
        window = np.hamming(stop - start)
        artifact[:, start:stop] += (
            np.real(values[row, :, None] * carrier) * window
        )
        weights[start:stop] += window
    covered = weights > 0.0
    artifact[:, covered] /= weights[covered]
    return artifact


def subtract_predicted_artifact(
    data: NDArray[np.floating],
    bounds: Sequence[tuple[int, int]],
    predicted_coefficients: NDArray[np.complexfloating],
    sampling_frequency_hz: float,
    frequency_hz: float,
) -> NDArray[np.float64]:
    """Subtract one independently predicted local sinusoid from every channel."""
    values = _validated_data(data)
    artifact = reconstruct_artifact(
        values.shape[1],
        bounds,
        predicted_coefficients,
        sampling_frequency_hz,
        frequency_hz,
    )
    if artifact.shape != values.shape:
        raise ValueError("predicted coefficients must cover every data channel")
    return values - artifact


@dataclass(frozen=True)
class InjectionPreservation:
    """Strict paired recovery metrics for one known exact-frequency component."""

    relative_waveform_error: float
    amplitude_retention: float
    phase_error_degrees: float
    passes: bool


def measure_injection_preservation(
    injection: NDArray[np.floating],
    recovered: NDArray[np.floating],
    sampling_frequency_hz: float,
    frequency_hz: float,
) -> InjectionPreservation:
    """Measure waveform and complex-gain fidelity of a recovered injection."""
    reference = _validated_data(injection)
    candidate = _validated_data(recovered)
    if candidate.shape != reference.shape:
        raise ValueError("injection and recovered arrays must have the same shape")
    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0.0:
        raise ValueError("sampling_frequency_hz must be finite and positive")
    if not np.isfinite(frequency_hz) or not (
        0.0 < frequency_hz < sampling_frequency_hz / 2.0
    ):
        raise ValueError("frequency_hz must lie strictly between DC and Nyquist")

    reference_energy = float(np.sum(reference**2))
    if reference_energy <= 0.0:
        raise ValueError("injection must have positive energy")
    waveform_error = float(
        np.sqrt(np.sum((candidate - reference) ** 2) / reference_energy)
    )

    times_s = np.arange(reference.shape[1]) / sampling_frequency_hz
    carrier = np.exp(-2j * np.pi * frequency_hz * times_s)
    reference_coefficients = reference @ carrier
    candidate_coefficients = candidate @ carrier
    coefficient_energy = float(np.vdot(reference_coefficients, reference_coefficients).real)
    if coefficient_energy <= 0.0:
        raise ValueError("injection has no energy at frequency_hz")
    gain = (
        np.vdot(reference_coefficients, candidate_coefficients)
        / coefficient_energy
    )
    amplitude_retention = float(np.abs(gain))
    phase_error_degrees = float(np.abs(np.angle(gain, deg=True)))
    passes = (
        waveform_error <= 0.01
        and 0.99 <= amplitude_retention <= 1.01
        and phase_error_degrees <= 1.0
    )
    return InjectionPreservation(
        waveform_error,
        amplitude_retention,
        phase_error_degrees,
        passes,
    )
