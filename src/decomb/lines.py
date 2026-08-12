"""Multiplicity-controlled Thomson multitaper tests for sinusoidal lines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LineDetection:
    """One significant channel-window-frequency test."""

    frequency_hz: float
    raw_p_value: float
    corrected_p_value: float
    window_index: int
    channel_index: int


@dataclass(frozen=True)
class LineDetectionResult:
    """All detections after Holm correction within each EEG channel."""

    detections: tuple[LineDetection, ...]
    tested_frequencies_hz: tuple[float, ...]
    window_count: int
    channel_count: int

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.tested_frequencies_hz, dtype=float)
        if (
            frequencies.ndim != 1
            or frequencies.size < 1
            or not np.all(np.isfinite(frequencies))
            or not np.all(np.diff(frequencies) > 0.0)
            or frequencies[0] <= 0.0
        ):
            raise ValueError(
                "Tested frequencies must be finite, positive, and strictly increasing."
            )
        if self.window_count < 1 or self.channel_count < 1:
            raise ValueError("Detection results require positive window and channel counts.")
        tested = set(self.tested_frequencies_hz)
        for detection in self.detections:
            if detection.frequency_hz not in tested:
                raise ValueError("Every detection must belong to the tested frequency grid.")
            _p_value(detection.raw_p_value, "raw_p_value")
            _p_value(detection.corrected_p_value, "corrected_p_value")
            if not 0 <= detection.window_index < self.window_count:
                raise ValueError("Detection window indices must belong to the input data.")
            if not 0 <= detection.channel_index < self.channel_count:
                raise ValueError("Detection channel indices must belong to the input data.")

    @property
    def test_count_per_channel(self) -> int:
        """Number of window-frequency hypotheses in each channel family."""
        return self.window_count * len(self.tested_frequencies_hz)

    @property
    def total_test_count(self) -> int:
        """Total tests evaluated, without treating channels as one family."""
        return self.channel_count * self.test_count_per_channel


@dataclass(frozen=True)
class HarmonicClassification:
    """Harmonic labels for the unique significant frequencies."""

    frequencies_hz: tuple[float, ...]
    corrected_p_values: tuple[float, ...]
    harmonics: tuple[int | None, ...]
    fundamental_hz: float | None
    corrected_p_value: float | None


@dataclass(frozen=True)
class ArtifactLine:
    """One statistically supported frequency and its optional harmonic label."""

    position_hz: float
    raw_p_value: float
    corrected_p_value: float
    window_indices: tuple[int, ...]
    harmonic: int | None

    def __post_init__(self) -> None:
        if not np.isfinite(self.position_hz) or self.position_hz <= 0.0:
            raise ValueError("Artifact-line positions must be finite and positive.")
        _p_value(self.raw_p_value, "raw_p_value")
        _p_value(self.corrected_p_value, "corrected_p_value")
        if not self.window_indices:
            raise ValueError("An artifact line requires at least one supporting window.")
        if self.window_indices != tuple(sorted(set(self.window_indices))):
            raise ValueError("Supporting window indices must be sorted and unique.")
        if self.window_indices[0] < 0:
            raise ValueError("Supporting window indices must be non-negative.")
        if self.harmonic is not None and self.harmonic < 1:
            raise ValueError("Harmonic labels must be positive integers.")


@dataclass(frozen=True)
class ChannelArtifactModel:
    """Every supported line and descriptive comb model for one EEG channel."""

    channel_index: int
    channel_name: str
    lines: tuple[ArtifactLine, ...]
    fundamental_hz: float | None
    comb_corrected_p_value: float | None

    def __post_init__(self) -> None:
        if not self.lines:
            raise ValueError("A channel artifact model requires a supported line.")
        if self.channel_index < 0 or not self.channel_name.strip():
            raise ValueError("Channel artifact models require a valid channel identity.")
        positions = tuple(line.position_hz for line in self.lines)
        if positions != tuple(sorted(set(positions))):
            raise ValueError("Artifact-line positions must be sorted and unique.")
        has_harmonics = any(line.harmonic is not None for line in self.lines)
        if has_harmonics != (self.fundamental_hz is not None):
            raise ValueError("Comb fundamentals and harmonic labels must occur together.")
        if has_harmonics != (self.comb_corrected_p_value is not None):
            raise ValueError("A classified comb requires one corrected p-value.")
        if self.fundamental_hz is not None:
            _positive(self.fundamental_hz, "fundamental_hz")
            _p_value(self.comb_corrected_p_value, "comb_corrected_p_value")


@dataclass(frozen=True)
class ArtifactModel:
    """Every channel with a statistically supported line in one recording."""

    channels: tuple[ChannelArtifactModel, ...]
    window_count: int
    channel_count: int
    test_count_per_channel: int

    def __post_init__(self) -> None:
        indices = tuple(channel.channel_index for channel in self.channels)
        names = tuple(channel.channel_name for channel in self.channels)
        if indices != tuple(sorted(set(indices))):
            raise ValueError("Affected channel indices must be sorted and unique.")
        if len(names) != len(set(names)):
            raise ValueError("Affected channel names must be unique.")
        if indices and indices[-1] >= self.channel_count:
            raise ValueError("Affected channels must belong to the tested EEG channels.")
        if min(self.window_count, self.channel_count, self.test_count_per_channel) < 1:
            raise ValueError("Artifact models require positive test dimensions.")

    @property
    def line_count(self) -> int:
        return sum(len(channel.lines) for channel in self.channels)


def thomson_f_p_values(
    data: np.ndarray,
    sampling_frequency_hz: float,
    *,
    frequency_range_hz: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return Thomson F-test p-values for window-by-channel data.

    The calculation follows MNE's ``method="spectrum_fit"`` implementation:
    eight DPSS tapers with time-bandwidth product four, alternating tapers for
    the sinusoidal estimate and residual, and an F(2, 14) null distribution.
    """
    from scipy.fft import rfft
    from scipy.signal.windows import dpss
    from scipy.stats import f as f_distribution

    values = _validated_data(data)
    sampling_frequency = _positive(sampling_frequency_hz, "sampling_frequency_hz")
    minimum_hz, maximum_hz = _validated_frequency_range(
        frequency_range_hz,
        sampling_frequency,
    )

    sample_count = values.shape[-1]
    tapers = dpss(sample_count, NW=4.0, Kmax=8, sym=False)
    signal_tapers = tapers[::2]
    taper_sums = signal_tapers.sum(axis=1)
    taper_sum_squares = float(taper_sums @ taper_sums)

    frequencies_hz = np.fft.rfftfreq(sample_count, d=1.0 / sampling_frequency)
    inside = (
        (frequencies_hz > 0.0)
        & (frequencies_hz >= minimum_hz)
        & (frequencies_hz <= maximum_hz)
    )
    frequencies_hz = frequencies_hz[inside]
    p_values = np.empty((*values.shape[:-1], frequencies_hz.size), dtype=float)
    for window, window_p_values in zip(values, p_values, strict=True):
        centred = window - window.mean(axis=-1, keepdims=True)
        tapered_spectra = rfft(
            centred[:, np.newaxis, :] * tapers[np.newaxis, :, :],
            axis=-1,
        )[:, :, inside]
        signal_spectra = tapered_spectra[:, ::2, :]
        amplitudes = (
            np.sum(
                signal_spectra * taper_sums[np.newaxis, :, np.newaxis],
                axis=1,
            )
            / taper_sum_squares
        )
        fitted_spectra = (
            amplitudes[:, np.newaxis, :]
            * taper_sums[np.newaxis, :, np.newaxis]
        )
        numerator = 7.0 * np.abs(amplitudes) ** 2 * taper_sum_squares
        denominator = np.sum(
            np.abs(signal_spectra - fitted_spectra) ** 2,
            axis=1,
        )
        denominator += np.sum(np.abs(tapered_spectra[:, 1::2, :]) ** 2, axis=1)
        statistic = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator > 0.0,
        )
        window_p_values[:] = f_distribution.sf(statistic, 2, 14)
    return frequencies_hz, p_values


CORRECTIONS = ("holm", "bonferroni", "none")


def detect_lines_from_p_values(
    frequencies_hz: np.ndarray,
    p_values: np.ndarray,
    *,
    familywise_error_rate: float,
    correction: str = "holm",
) -> LineDetectionResult:
    """Threshold pre-computed Thomson F-test p-values with one correction procedure.

    Split out of :func:`detect_lines` so a caller comparing multiple corrections on the
    same data -- as the detection-procedure ablation does -- pays for the expensive
    Thomson F-test once rather than once per correction.
    """
    error_rate = float(familywise_error_rate)
    if not np.isfinite(error_rate) or not 0.0 < error_rate < 1.0:
        raise ValueError("familywise_error_rate must lie strictly between zero and one.")
    if correction not in CORRECTIONS:
        raise ValueError(f"correction must be one of {CORRECTIONS}, got {correction!r}.")

    window_count, channel_count, _ = p_values.shape
    adjust = {
        "holm": _holm_adjusted_p_values,
        "bonferroni": _bonferroni_adjusted_p_values,
        "none": lambda values: values,
    }[correction]
    corrected = np.empty_like(p_values)
    for channel_index in range(channel_count):
        channel_p_values = p_values[:, channel_index, :]
        corrected[:, channel_index, :] = adjust(channel_p_values)
    significant = np.argwhere(corrected < error_rate)
    detections = tuple(
        LineDetection(
            frequency_hz=float(frequencies_hz[frequency_index]),
            raw_p_value=float(p_values[window_index, channel_index, frequency_index]),
            corrected_p_value=float(corrected[window_index, channel_index, frequency_index]),
            window_index=int(window_index),
            channel_index=int(channel_index),
        )
        for window_index, channel_index, frequency_index in significant
    )
    return LineDetectionResult(
        detections,
        tuple(float(value) for value in frequencies_hz),
        window_count,
        channel_count,
    )


def detect_lines(
    data: np.ndarray,
    sampling_frequency_hz: float,
    *,
    frequency_range_hz: tuple[float, float],
    familywise_error_rate: float,
    correction: str = "holm",
) -> LineDetectionResult:
    """Detect lines within each EEG channel's window-frequency test family.

    ``correction`` selects the per-channel multiplicity procedure: Holm step-down
    (decomb's real pipeline), plain Bonferroni (an ablation isolating Holm's benefit
    over the same family), or no correction at all (matching MNE's uncorrected
    ``spectrum_fit`` threshold). All three compare the same Thomson F-test p-values.
    """
    frequencies_hz, p_values = thomson_f_p_values(
        data,
        sampling_frequency_hz,
        frequency_range_hz=frequency_range_hz,
    )
    return detect_lines_from_p_values(
        frequencies_hz,
        p_values,
        familywise_error_rate=familywise_error_rate,
        correction=correction,
    )


def _holm_adjusted_p_values(p_values: np.ndarray) -> np.ndarray:
    """Return Holm step-down adjusted p-values for one hypothesis family."""
    values = np.asarray(p_values, dtype=float)
    if values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("Holm correction requires finite p-values.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Holm correction requires p-values between zero and one.")

    flat = values.ravel()
    order = np.argsort(flat, kind="stable")
    multipliers = np.arange(flat.size, 0, -1, dtype=float)
    ordered = np.minimum(1.0, flat[order] * multipliers)
    ordered = np.maximum.accumulate(ordered)
    adjusted = np.empty_like(flat)
    adjusted[order] = ordered
    return adjusted.reshape(values.shape)


def _bonferroni_adjusted_p_values(p_values: np.ndarray) -> np.ndarray:
    """Return single-step Bonferroni adjusted p-values for one hypothesis family."""
    values = np.asarray(p_values, dtype=float)
    if values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("Bonferroni correction requires finite p-values.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Bonferroni correction requires p-values between zero and one.")
    return np.minimum(1.0, values * values.size)


def classify_harmonics(
    result: LineDetectionResult,
    *,
    frequency_bin_width_hz: float,
    spectral_resolution_hz: float,
    familywise_error_rate: float,
) -> HarmonicClassification:
    """Test whether significant lines share an integer fundamental.

    Candidate fundamentals are implied by detected frequencies. A partial-
    conjunction test requires at least two harmonic components and remains valid
    under arbitrary dependence. Bonferroni correction covers every fundamental
    implied by the complete tested grid. Classification is descriptive: it never
    creates an unobserved target.
    """
    bin_width_hz = _positive(frequency_bin_width_hz, "frequency_bin_width_hz")
    resolution_hz = _positive(spectral_resolution_hz, "spectral_resolution_hz")
    error_rate = _probability(familywise_error_rate, "familywise_error_rate")
    frequencies_hz, p_values = _unique_frequencies(result)
    raw_p_values = _unique_raw_p_values(result, frequencies_hz)
    isolated = HarmonicClassification(
        frequencies_hz,
        p_values,
        tuple(None for _ in frequencies_hz),
        None,
        None,
    )
    if len(frequencies_hz) < 2:
        return isolated

    minimum_fundamental_hz = resolution_hz
    candidate_count = sum(
        int(np.floor(frequency_hz / minimum_fundamental_hz))
        for frequency_hz in result.tested_frequencies_hz
    )
    candidates = np.unique(
        [
            frequency_hz / harmonic
            for frequency_hz in frequencies_hz
            for harmonic in range(
                1,
                int(np.floor(frequency_hz / minimum_fundamental_hz)) + 1,
            )
        ]
    )
    tolerance_hz = bin_width_hz / 2.0
    best: tuple[int, float, float, tuple[int | None, ...]] | None = None
    frequencies = np.asarray(frequencies_hz)
    probabilities = np.asarray(raw_p_values)
    tested_frequencies = np.asarray(result.tested_frequencies_hz)
    for candidate_hz in candidates:
        nearest_harmonics = np.rint(frequencies / candidate_hz).astype(int)
        residuals_hz = np.abs(frequencies - nearest_harmonics * candidate_hz)
        aligned = (nearest_harmonics >= 1) & (residuals_hz <= tolerance_hz)
        if np.unique(nearest_harmonics[aligned]).size < 2:
            continue
        first_testable_harmonic = max(
            1,
            int(np.ceil((tested_frequencies[0] - tolerance_hz) / candidate_hz)),
        )
        last_testable_harmonic = int(
            np.floor((tested_frequencies[-1] + tolerance_hz) / candidate_hz)
        )
        candidate_harmonic_count = (
            last_testable_harmonic - first_testable_harmonic + 1
        )
        harmonic_p_values = []
        for harmonic in np.unique(nearest_harmonics[aligned]):
            group = aligned & (nearest_harmonics == harmonic)
            frequency_p_values = np.minimum(
                1.0,
                probabilities[group] * result.window_count,
            )
            predicted_frequency_hz = harmonic * candidate_hz
            group_test_count = _frequency_group_test_count(
                tested_frequencies,
                predicted_frequency_hz,
                tolerance_hz,
            )
            harmonic_p_values.append(
                min(1.0, float(np.min(frequency_p_values)) * group_test_count)
            )
        second_smallest_p_value = sorted(harmonic_p_values)[1]
        partial_conjunction_p_value = min(
            1.0,
            (candidate_harmonic_count - 1) * second_smallest_p_value,
        )
        corrected_p_value = min(
            1.0,
            partial_conjunction_p_value * candidate_count,
        )
        if corrected_p_value >= error_rate:
            continue
        harmonics = tuple(
            int(harmonic) if is_aligned else None
            for harmonic, is_aligned in zip(
                nearest_harmonics,
                aligned,
                strict=True,
            )
        )
        score = (
            -int(np.count_nonzero(aligned)),
            corrected_p_value,
            -float(candidate_hz),
            harmonics,
        )
        if best is None or score[:3] < best[:3]:
            best = score

    if best is None:
        return isolated

    _, corrected_p_value, _, harmonics = best
    harmonic_numbers = np.asarray(
        [harmonic if harmonic is not None else 0 for harmonic in harmonics],
        dtype=float,
    )
    aligned = harmonic_numbers > 0
    refined_fundamental_hz = float(
        np.sum(harmonic_numbers[aligned] * frequencies[aligned])
        / np.sum(harmonic_numbers[aligned] ** 2)
    )
    return HarmonicClassification(
        frequencies_hz,
        p_values,
        harmonics,
        refined_fundamental_hz,
        corrected_p_value,
    )


def _frequency_group_test_count(
    tested_frequencies_hz: np.ndarray,
    predicted_frequency_hz: float,
    tolerance_hz: float,
) -> int:
    """Count sorted grid points in one closed harmonic-alignment interval."""
    lower = np.searchsorted(
        tested_frequencies_hz,
        predicted_frequency_hz - tolerance_hz,
        side="left",
    )
    upper = np.searchsorted(
        tested_frequencies_hz,
        predicted_frequency_hz + tolerance_hz,
        side="right",
    )
    return int(upper - lower)


def build_artifact_model(
    result: LineDetectionResult,
    *,
    channel_names: tuple[str, ...],
    frequency_bin_width_hz: float,
    spectral_resolution_hz: float,
    familywise_error_rate: float,
) -> ArtifactModel:
    """Build one independently corrected model per affected EEG channel."""
    names = tuple(channel_names)
    if len(names) != result.channel_count or len(names) != len(set(names)):
        raise ValueError("channel_names must identify every tested EEG channel once.")
    if any(not name.strip() for name in names):
        raise ValueError("EEG channel names must not be empty.")

    channel_models = []
    for channel_index, channel_name in enumerate(names):
        detections = tuple(
            detection
            for detection in result.detections
            if detection.channel_index == channel_index
        )
        if not detections:
            continue
        channel_result = LineDetectionResult(
            detections,
            result.tested_frequencies_hz,
            result.window_count,
            result.channel_count,
        )
        classification = classify_harmonics(
            channel_result,
            frequency_bin_width_hz=frequency_bin_width_hz,
            spectral_resolution_hz=spectral_resolution_hz,
            familywise_error_rate=familywise_error_rate,
        )
        artifact_lines = []
        for position_hz, harmonic in zip(
            classification.frequencies_hz,
            classification.harmonics,
            strict=True,
        ):
            evidence = tuple(
                detection
                for detection in detections
                if detection.frequency_hz == position_hz
            )
            artifact_lines.append(
                ArtifactLine(
                    position_hz=position_hz,
                    raw_p_value=min(item.raw_p_value for item in evidence),
                    corrected_p_value=min(
                        item.corrected_p_value for item in evidence
                    ),
                    window_indices=tuple(
                        sorted({item.window_index for item in evidence})
                    ),
                    harmonic=harmonic,
                )
            )
        channel_models.append(
            ChannelArtifactModel(
                channel_index=channel_index,
                channel_name=channel_name,
                lines=tuple(artifact_lines),
                fundamental_hz=classification.fundamental_hz,
                comb_corrected_p_value=classification.corrected_p_value,
            )
        )
    return ArtifactModel(
        channels=tuple(channel_models),
        window_count=result.window_count,
        channel_count=result.channel_count,
        test_count_per_channel=result.test_count_per_channel,
    )


def _validated_data(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.ndim != 3 or min(values.shape) < 1 or values.shape[-1] < 2:
        raise ValueError("data must be a non-empty window-by-channel-by-sample array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("data must contain only finite values.")
    return values


def _unique_frequencies(
    result: LineDetectionResult,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    probabilities: dict[float, float] = {}
    for detection in result.detections:
        probabilities[detection.frequency_hz] = min(
            probabilities.get(detection.frequency_hz, 1.0),
            detection.corrected_p_value,
        )
    frequencies_hz = tuple(sorted(probabilities))
    return frequencies_hz, tuple(probabilities[value] for value in frequencies_hz)


def _unique_raw_p_values(
    result: LineDetectionResult,
    frequencies_hz: tuple[float, ...],
) -> tuple[float, ...]:
    """Return the smallest raw window p-value at each detected frequency."""
    return tuple(
        min(
            detection.raw_p_value
            for detection in result.detections
            if detection.frequency_hz == frequency_hz
        )
        for frequency_hz in frequencies_hz
    )


def _validated_frequency_range(
    frequency_range_hz: tuple[float, float],
    sampling_frequency_hz: float,
) -> tuple[float, float]:
    values = np.asarray(frequency_range_hz, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("frequency_range_hz must contain two finite values.")
    minimum_hz, maximum_hz = (float(value) for value in values)
    if not 0.0 <= minimum_hz < maximum_hz < sampling_frequency_hz / 2.0:
        raise ValueError("frequency_range_hz must lie inside [0, Nyquist).")
    return minimum_hz, maximum_hz


def _positive(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _probability(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0.0 < number < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one.")
    return number


def _p_value(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie between zero and one, inclusive.")
    return number
