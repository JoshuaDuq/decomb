"""Stationary, drifting, and intermittent sinusoid injections for recovery trials.

Used to measure what a detection-and-notch procedure does to a known artifact of known
strength, added to a sinusoid-free surrogate background so no pre-existing line can bias
the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KINDS = ("stationary", "drifting", "intermittent")


@dataclass(frozen=True)
class SinusoidInjection:
    """One synthetic artifact: kind, frequency trajectory, and strength.

    Amplitudes are in the same volts MNE stores EEG data in, so they compare directly
    against the real cohort's own line strengths.
    """

    kind: str
    frequency_hz: float
    amplitude_v: float
    drift_hz: float = 0.0
    occupancy: float = 1.0
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}.")
        if not np.isfinite(self.frequency_hz) or self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be finite and positive.")
        if not np.isfinite(self.amplitude_v) or self.amplitude_v <= 0.0:
            raise ValueError("amplitude_v must be finite and positive.")
        if self.kind != "drifting" and self.drift_hz != 0.0:
            raise ValueError("drift_hz must be zero outside a drifting injection.")
        if self.kind != "intermittent" and self.occupancy != 1.0:
            raise ValueError("occupancy must be one outside an intermittent injection.")
        if not 0.0 < self.occupancy <= 1.0:
            raise ValueError("occupancy must lie in (0, 1].")
        if not np.isfinite(self.phase_rad):
            raise ValueError("phase_rad must be finite.")


def active_mask(
    occupancy: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One contiguous active span covering the requested share of the recording."""
    if not 0.0 < occupancy <= 1.0:
        raise ValueError("occupancy must lie in (0, 1].")
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    active_samples = max(1, int(round(occupancy * n_samples)))
    if active_samples >= n_samples:
        return np.ones(n_samples, dtype=bool)
    start = int(rng.integers(0, n_samples - active_samples + 1))
    mask = np.zeros(n_samples, dtype=bool)
    mask[start : start + active_samples] = True
    return mask


def synthesize_injection(
    spec: SinusoidInjection,
    n_samples: int,
    sampling_frequency_hz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """The injected waveform alone, in volts, ready to add to a background channel."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least two.")
    sampling_frequency = _positive(sampling_frequency_hz, "sampling_frequency_hz")
    times_s = np.arange(n_samples) / sampling_frequency
    duration_s = n_samples / sampling_frequency

    if spec.kind == "drifting":
        # Phase is the time integral of instantaneous frequency, which linearly ramps
        # from frequency_hz to frequency_hz + drift_hz across the recording.
        phase = 2.0 * np.pi * (
            spec.frequency_hz * times_s
            + spec.drift_hz * times_s**2 / (2.0 * duration_s)
        )
    else:
        phase = 2.0 * np.pi * spec.frequency_hz * times_s
    waveform = spec.amplitude_v * np.sin(phase + spec.phase_rad)

    if spec.kind == "intermittent":
        waveform = waveform * active_mask(spec.occupancy, n_samples, rng)
    return waveform


def injected_frequency_band_hz(
    spec: SinusoidInjection,
    *,
    half_width_hz: float,
) -> tuple[float, float]:
    """Frequency band the injection's trajectory occupies, expanded by half a resolution.

    Covers a drifting injection's full swept range; a stationary or intermittent
    injection occupies one frequency throughout its active span.
    """
    if not np.isfinite(half_width_hz) or half_width_hz < 0.0:
        raise ValueError("half_width_hz must be finite and non-negative.")
    low_hz = min(spec.frequency_hz, spec.frequency_hz + spec.drift_hz)
    high_hz = max(spec.frequency_hz, spec.frequency_hz + spec.drift_hz)
    return (low_hz - half_width_hz, high_hz + half_width_hz)


def inject_into_raw(raw, channel_name: str, spec: SinusoidInjection, rng: np.random.Generator):
    """Return a copy of ``raw`` with one injection added to a single EEG channel."""
    import mne

    if channel_name not in raw.ch_names:
        raise ValueError(f"Recording does not contain channel {channel_name!r}.")
    sampling_frequency_hz = float(raw.info["sfreq"])
    waveform = synthesize_injection(spec, raw.n_times, sampling_frequency_hz, rng)
    data = raw.get_data()
    data[raw.ch_names.index(channel_name)] += waveform
    injected = mne.io.RawArray(data, raw.info.copy(), verbose="ERROR")
    injected.set_annotations(raw.annotations.copy())
    return injected


def _positive(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number
