"""Multiplicity-controlled tests for coherent and persistent spectral lines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PERSISTENT_PEAK_SMOOTHING_HZ = 1.0


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
    """All detections after recording-family multiplicity correction."""

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
        """Number of window-frequency hypotheses contributed by each channel."""
        return self.window_count * len(self.tested_frequencies_hz)

    @property
    def total_test_count(self) -> int:
        """Number of hypotheses in the recording-wide decision family."""
        return self.channel_count * self.test_count_per_channel


@dataclass(frozen=True)
class SupportedLine:
    """One statistically supported frequency and its optional harmonic label."""

    position_hz: float
    raw_p_value: float
    corrected_p_value: float
    window_indices: tuple[int, ...]
    harmonic: int | None

    def __post_init__(self) -> None:
        if not np.isfinite(self.position_hz) or self.position_hz <= 0.0:
            raise ValueError("Supported-line positions must be finite and positive.")
        _p_value(self.raw_p_value, "raw_p_value")
        _p_value(self.corrected_p_value, "corrected_p_value")
        if not self.window_indices:
            raise ValueError("A supported line requires at least one supporting window.")
        if self.window_indices != tuple(sorted(set(self.window_indices))):
            raise ValueError("Supporting window indices must be sorted and unique.")
        if self.window_indices[0] < 0:
            raise ValueError("Supporting window indices must be non-negative.")
        if self.harmonic is not None and self.harmonic < 1:
            raise ValueError("Harmonic labels must be positive integers.")


@dataclass(frozen=True)
class ChannelLineModel:
    """Every supported line and descriptive comb model for one EEG channel."""

    channel_index: int
    channel_name: str
    lines: tuple[SupportedLine, ...]
    fundamental_hz: float | None
    comb_corrected_p_value: float | None

    def __post_init__(self) -> None:
        if not self.lines:
            raise ValueError("A channel line model requires a supported line.")
        if self.channel_index < 0 or not self.channel_name.strip():
            raise ValueError("Channel line models require a valid channel identity.")
        positions = tuple(line.position_hz for line in self.lines)
        if positions != tuple(sorted(set(positions))):
            raise ValueError("Supported-line positions must be sorted and unique.")
        has_harmonics = any(line.harmonic is not None for line in self.lines)
        if has_harmonics != (self.fundamental_hz is not None):
            raise ValueError("Comb fundamentals and harmonic labels must occur together.")
        if has_harmonics != (self.comb_corrected_p_value is not None):
            raise ValueError("A classified comb requires one corrected p-value.")
        if self.fundamental_hz is not None:
            _positive(self.fundamental_hz, "fundamental_hz")
            _p_value(self.comb_corrected_p_value, "comb_corrected_p_value")


@dataclass(frozen=True)
class LineModel:
    """Every channel with a statistically supported line in one recording."""

    channels: tuple[ChannelLineModel, ...]
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
            raise ValueError("Line models require positive test dimensions.")

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


def persistent_peak_p_values(
    data: np.ndarray,
    sampling_frequency_hz: float,
    *,
    frequency_range_hz: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Test narrowband power that persists without sinusoidal phase coherence.

    Production supplies 50%-overlapping windows. Every fourth window estimates a
    channel-specific smooth background; the intervening non-overlapping windows are a
    disjoint test sample. For each channel and frequency, a whitened rectangular
    periodogram compares three equal-width bands: the target and symmetric flankers.
    Under the local smooth-spectrum null the bands are exchangeable, so the target is
    uniquely largest with probability one third. An exact one-sided binomial test asks
    whether that event persists in the test windows.

    The recording-level p-value is placed only on windows that supported the peak. This
    retains auditable supporting-window indices and is conservative when the complete
    window-channel-frequency family is corrected downstream.
    """
    from scipy.fft import rfft
    from scipy.ndimage import median_filter
    from scipy.stats import binom

    values = _validated_data(data)
    sampling_frequency = _positive(
        sampling_frequency_hz,
        "sampling_frequency_hz",
    )
    minimum_hz, maximum_hz = _validated_frequency_range(
        frequency_range_hz,
        sampling_frequency,
    )

    sample_count = values.shape[-1]
    all_frequencies_hz = np.fft.rfftfreq(
        sample_count,
        d=1.0 / sampling_frequency,
    )
    inside = (
        (all_frequencies_hz > 0.0)
        & (all_frequencies_hz >= minimum_hz)
        & (all_frequencies_hz <= maximum_hz)
    )
    tested_indices = np.flatnonzero(inside)
    frequencies_hz = all_frequencies_hz[tested_indices]
    p_values = np.ones(
        (*values.shape[:-1], frequencies_hz.size),
        dtype=float,
    )

    if values.shape[0] < 3:
        raise ValueError(
            "Persistent-peak testing requires at least three windows for disjoint "
            "background and test samples."
        )
    background_indices = np.arange(0, values.shape[0], 4)
    test_indices = np.arange(2, values.shape[0], 4)
    background_windows = values[background_indices]
    test_windows = values[test_indices]
    background_windows = background_windows - background_windows.mean(
        axis=-1,
        keepdims=True,
    )
    test_windows = test_windows - test_windows.mean(axis=-1, keepdims=True)
    background_power = np.abs(rfft(background_windows, axis=-1)) ** 2
    power = np.abs(rfft(test_windows, axis=-1)) ** 2
    bin_width_hz = sampling_frequency / sample_count
    smoothing_bins = max(
        1,
        int(round(PERSISTENT_PEAK_SMOOTHING_HZ / bin_width_hz)) | 1,
    )
    background = median_filter(
        background_power.mean(axis=0),
        size=(1, smoothing_bins),
        mode="nearest",
    )
    power /= np.maximum(background, np.finfo(float).tiny)[np.newaxis, ...]
    for frequency_index, centre in enumerate(tested_indices):
        if centre < 5 or centre + 5 >= power.shape[-1]:
            continue
        target = power[..., centre - 1 : centre + 2].sum(axis=-1)
        lower = power[..., centre - 5 : centre - 2].sum(axis=-1)
        upper = power[..., centre + 3 : centre + 6].sum(axis=-1)
        supports = target > np.maximum(lower, upper)
        support_counts = supports.sum(axis=0)
        probabilities = binom.sf(
            support_counts - 1,
            test_windows.shape[0],
            1.0 / 3.0,
        )
        for local_index, window_index in enumerate(test_indices):
            p_values[window_index, :, frequency_index] = np.where(
                supports[local_index],
                probabilities,
                1.0,
            )
    return frequencies_hz, p_values


def line_test_p_values(
    data: np.ndarray,
    sampling_frequency_hz: float,
    *,
    frequency_range_hz: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Bonferroni-combine coherent-sinusoid and persistent-peak tests."""
    frequencies_hz, sinusoid_p_values = thomson_f_p_values(
        data,
        sampling_frequency_hz,
        frequency_range_hz=frequency_range_hz,
    )
    peak_frequencies_hz, peak_p_values = persistent_peak_p_values(
        data,
        sampling_frequency_hz,
        frequency_range_hz=frequency_range_hz,
    )
    if not np.array_equal(peak_frequencies_hz, frequencies_hz):
        raise ValueError("Line tests produced different frequency grids.")
    return frequencies_hz, np.minimum(
        1.0,
        2.0 * np.minimum(sinusoid_p_values, peak_p_values),
    )


def detect_lines_from_p_values(
    frequencies_hz: np.ndarray,
    p_values: np.ndarray,
    *,
    familywise_error_rate: float,
) -> LineDetectionResult:
    """Apply recording-family Holm correction to pre-computed input p-values."""
    error_rate = float(familywise_error_rate)
    if not np.isfinite(error_rate) or not 0.0 < error_rate < 1.0:
        raise ValueError("familywise_error_rate must lie strictly between zero and one.")

    window_count, channel_count, _ = p_values.shape
    corrected = _holm_adjusted_p_values(p_values)
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
) -> LineDetectionResult:
    """Detect lines with Holm control of the complete recording family."""
    frequencies_hz, p_values = line_test_p_values(
        data,
        sampling_frequency_hz,
        frequency_range_hz=frequency_range_hz,
    )
    return detect_lines_from_p_values(
        frequencies_hz,
        p_values,
        familywise_error_rate=familywise_error_rate,
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


def build_line_model(
    result: LineDetectionResult,
    *,
    channel_names: tuple[str, ...],
) -> LineModel:
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
        supported_lines = []
        for position_hz in sorted(
            {detection.frequency_hz for detection in detections}
        ):
            evidence = tuple(
                detection
                for detection in detections
                if detection.frequency_hz == position_hz
            )
            supported_lines.append(
                SupportedLine(
                    position_hz=position_hz,
                    raw_p_value=min(item.raw_p_value for item in evidence),
                    corrected_p_value=min(
                        item.corrected_p_value for item in evidence
                    ),
                    window_indices=tuple(
                        sorted({item.window_index for item in evidence})
                    ),
                    harmonic=None,
                )
            )
        channel_models.append(
            ChannelLineModel(
                channel_index=channel_index,
                channel_name=channel_name,
                lines=tuple(supported_lines),
                fundamental_hz=None,
                comb_corrected_p_value=None,
            )
        )
    return LineModel(
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


def _validated_frequency_range(
    frequency_range_hz: tuple[float, float],
    sampling_frequency_hz: float,
) -> tuple[float, float]:
    values = np.asarray(frequency_range_hz, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("frequency_range_hz must contain two finite values.")
    minimum_hz, maximum_hz = (float(value) for value in values)
    if not 0.0 <= minimum_hz < maximum_hz <= sampling_frequency_hz / 2.0:
        raise ValueError("frequency_range_hz must lie inside [0, Nyquist].")
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
