"""Sinusoid-free Gaussian surrogates matched to each real channel's own spectrum.

Used only for calibration studies: they answer whether a detection procedure controls
its declared false-positive rate on data known to contain no sinusoidal line, without
borrowing any of decomb's own detection code to build that ground truth.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SMOOTHING_HZ = 1.0


def background_power_spectrum(
    data: np.ndarray,
    sampling_frequency_hz: float,
    *,
    smoothing_hz: float = DEFAULT_SMOOTHING_HZ,
) -> np.ndarray:
    """Median-smoothed periodogram of one channel, with narrowband lines suppressed.

    A running median is robust to the positive spikes a sinusoidal line adds to a
    periodogram, so it recovers a channel's broadband envelope without them. A mean-based
    smoother would instead be dragged upward across a line's whole neighbourhood.
    """
    from scipy.signal import medfilt

    values = _validated_channel(data)
    sampling_frequency = _positive(sampling_frequency_hz, "sampling_frequency_hz")
    _positive(smoothing_hz, "smoothing_hz")

    spectrum = np.abs(np.fft.rfft(values)) ** 2
    bin_width_hz = sampling_frequency / values.size
    kernel_bins = int(round(smoothing_hz / bin_width_hz))
    kernel_bins = max(1, kernel_bins | 1)  # medfilt requires an odd kernel size
    kernel_bins = min(kernel_bins, spectrum.size - (1 - spectrum.size % 2))
    return medfilt(spectrum, kernel_size=max(1, kernel_bins))


def synthesize_gaussian_process(
    power_spectrum: np.ndarray,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One stationary Gaussian-process realization matching a target power spectrum.

    Each rfft coefficient is drawn as an independent Gaussian with variance set by the
    target power at that bin, so by construction no frequency carries a persistent line:
    every realization redraws its own random phase and magnitude.
    """
    spectrum = np.asarray(power_spectrum, dtype=float)
    if spectrum.ndim != 1 or spectrum.size < 2:
        raise ValueError("power_spectrum must be a one-dimensional rfft-length array.")
    if not np.all(np.isfinite(spectrum)) or np.any(spectrum < 0.0):
        raise ValueError("power_spectrum must contain finite, non-negative values.")
    expected_bins = sample_count // 2 + 1
    if spectrum.size != expected_bins:
        raise ValueError(
            f"power_spectrum has {spectrum.size} bins; {sample_count} samples require "
            f"{expected_bins}."
        )

    coefficients = np.empty(spectrum.size, dtype=complex)
    coefficients[0] = rng.normal(scale=np.sqrt(spectrum[0]))
    has_real_nyquist = sample_count % 2 == 0
    interior_stop = spectrum.size - 1 if has_real_nyquist else spectrum.size
    interior = slice(1, interior_stop)
    interior_scale = np.sqrt(spectrum[interior] / 2.0)
    coefficients[interior] = rng.normal(scale=interior_scale) + 1j * rng.normal(
        scale=interior_scale
    )
    if has_real_nyquist:
        coefficients[-1] = rng.normal(scale=np.sqrt(spectrum[-1]))
    return np.fft.irfft(coefficients, n=sample_count)


def surrogate_channel(
    data: np.ndarray,
    sampling_frequency_hz: float,
    rng: np.random.Generator,
    *,
    smoothing_hz: float = DEFAULT_SMOOTHING_HZ,
) -> np.ndarray:
    """One channel's sinusoid-free surrogate, matched to its own smoothed spectrum."""
    values = _validated_channel(data)
    envelope = background_power_spectrum(
        values,
        sampling_frequency_hz,
        smoothing_hz=smoothing_hz,
    )
    return synthesize_gaussian_process(envelope, values.size, rng)


def surrogate_eeg_data(
    data: np.ndarray,
    sampling_frequency_hz: float,
    rng: np.random.Generator,
    *,
    smoothing_hz: float = DEFAULT_SMOOTHING_HZ,
) -> np.ndarray:
    """Independent sinusoid-free surrogates for every channel in a channel-by-time array."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("data must be a non-empty channel-by-time array.")
    return np.stack(
        [
            surrogate_channel(
                channel_data,
                sampling_frequency_hz,
                rng,
                smoothing_hz=smoothing_hz,
            )
            for channel_data in values
        ],
        axis=0,
    )


def surrogate_raw(
    raw,
    rng: np.random.Generator,
    *,
    smoothing_hz: float = DEFAULT_SMOOTHING_HZ,
):
    """A same-length, same-boundary EEG-only surrogate of one real recording.

    Every EEG channel's data is replaced by a surrogate matched to that channel's own
    spectrum; annotations (including the acquisition-boundary markers decomb's windowing
    respects) and the sampling rate are copied unchanged, so the surrogate is windowed and
    tested exactly as the source recording would be.
    """
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(picks) == 0:
        raise ValueError("Surrogate generation requires at least one non-bad EEG channel.")
    sampling_frequency_hz = float(raw.info["sfreq"])
    surrogate_data = surrogate_eeg_data(
        raw.get_data(picks=picks),
        sampling_frequency_hz,
        rng,
        smoothing_hz=smoothing_hz,
    )
    surrogate = raw.copy().pick(picks).load_data()
    surrogate._data[:] = surrogate_data
    return surrogate


def _validated_channel(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("data must be a one-dimensional array of at least two samples.")
    if not np.all(np.isfinite(values)):
        raise ValueError("data must contain only finite values.")
    return values


def _positive(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number
