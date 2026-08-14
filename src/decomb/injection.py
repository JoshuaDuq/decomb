"""Stationary, drifting, and intermittent sinusoid injections for recovery trials.

Used to measure what a detection-and-notch procedure does to a known component of known
strength, added to a sinusoid-free surrogate background so no pre-existing line can bias
the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KINDS = ("stationary", "drifting", "intermittent")


@dataclass(frozen=True)
class SinusoidInjection:
    """One synthetic sinusoidal component: trajectory, occupancy, and strength.

    Amplitudes are in the same volts MNE stores EEG data in.
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
        if not np.isfinite(self.drift_hz):
            raise ValueError("drift_hz must be finite.")
        if self.kind != "drifting" and self.drift_hz != 0.0:
            raise ValueError("drift_hz must be zero outside a drifting injection.")
        if self.kind != "intermittent" and self.occupancy != 1.0:
            raise ValueError("occupancy must be one outside an intermittent injection.")
        if not 0.0 < self.occupancy <= 1.0:
            raise ValueError("occupancy must lie in (0, 1].")
        if not np.isfinite(self.phase_rad):
            raise ValueError("phase_rad must be finite.")


@dataclass(frozen=True)
class FactorialInjectionTarget:
    """Method-independent sinusoid defined by design factors and relative strength."""

    kind: str
    frequency_hz: float
    component_to_background_db: float
    drift_hz: float = 0.0
    occupancy: float = 1.0
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}.")
        if not np.isfinite(self.frequency_hz) or self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be finite and positive.")
        if not np.isfinite(self.component_to_background_db):
            raise ValueError("component_to_background_db must be finite.")
        if not np.isfinite(self.drift_hz):
            raise ValueError("drift_hz must be finite.")
        if self.kind != "drifting" and self.drift_hz != 0.0:
            raise ValueError("drift_hz must be zero outside a drifting target.")
        if self.kind != "intermittent" and self.occupancy != 1.0:
            raise ValueError("occupancy must be one outside an intermittent target.")
        if not 0.0 < self.occupancy <= 1.0:
            raise ValueError("occupancy must lie in (0, 1].")
        if not np.isfinite(self.phase_rad):
            raise ValueError("phase_rad must be finite.")

    def as_specification(self, amplitude_v: float) -> SinusoidInjection:
        """Attach a background-scaled voltage amplitude to this fixed design point."""
        return SinusoidInjection(
            kind=self.kind,
            frequency_hz=self.frequency_hz,
            amplitude_v=amplitude_v,
            drift_hz=self.drift_hz,
            occupancy=self.occupancy,
            phase_rad=self.phase_rad,
        )


@dataclass(frozen=True)
class InjectionRealization:
    """One waveform and the phase-independent temporal subspace that contains it."""

    waveform_v: np.ndarray
    temporal_basis: np.ndarray

    def __post_init__(self) -> None:
        waveform = np.asarray(self.waveform_v, dtype=float)
        basis = np.asarray(self.temporal_basis, dtype=float)
        if waveform.ndim != 1 or waveform.size < 2:
            raise ValueError("An injection waveform requires at least two samples.")
        if basis.shape != (2, waveform.size):
            raise ValueError("temporal_basis must have shape (2, n_samples).")
        if not np.all(np.isfinite(waveform)) or not np.all(np.isfinite(basis)):
            raise ValueError("Injection realizations must contain only finite values.")


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


def realize_injection(
    spec: SinusoidInjection,
    n_samples: int,
    sampling_frequency_hz: float,
    rng: np.random.Generator,
) -> InjectionRealization:
    """Realize a waveform and its sine-cosine subspace with one shared active mask."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least two.")
    sampling_frequency = _positive(sampling_frequency_hz, "sampling_frequency_hz")
    trajectory_edges_hz = (
        spec.frequency_hz,
        spec.frequency_hz + spec.drift_hz,
    )
    if min(trajectory_edges_hz) <= 0.0 or max(trajectory_edges_hz) >= (
        sampling_frequency / 2.0
    ):
        raise ValueError(
            "The injection frequency trajectory must lie strictly inside (0, Nyquist)."
        )
    times_s = np.arange(n_samples) / sampling_frequency
    duration_s = n_samples / sampling_frequency

    if spec.kind == "drifting":
        # Phase is the time integral of instantaneous frequency, which linearly ramps
        # from frequency_hz to frequency_hz + drift_hz across the recording.
        carrier_phase = 2.0 * np.pi * (
            spec.frequency_hz * times_s
            + spec.drift_hz * times_s**2 / (2.0 * duration_s)
        )
    else:
        carrier_phase = 2.0 * np.pi * spec.frequency_hz * times_s

    temporal_basis = np.stack(
        [np.sin(carrier_phase), np.cos(carrier_phase)],
        axis=0,
    )

    if spec.kind == "intermittent":
        temporal_basis *= active_mask(spec.occupancy, n_samples, rng)
    phase_weights = np.array([np.cos(spec.phase_rad), np.sin(spec.phase_rad)])
    waveform = spec.amplitude_v * phase_weights @ temporal_basis
    return InjectionRealization(waveform, temporal_basis)


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


def inject_into_raw(raw, channel_name: str, realization: InjectionRealization):
    """Return a copy of ``raw`` with one injection added to a single EEG channel."""
    if channel_name not in raw.ch_names:
        raise ValueError(f"Recording does not contain channel {channel_name!r}.")
    if realization.waveform_v.size != raw.n_times:
        raise ValueError("Injection and recording sample counts must match.")
    injected = raw.copy().load_data()
    injected._data[raw.ch_names.index(channel_name)] += realization.waveform_v
    return injected


def inject_spatially_balanced(
    raw,
    channel_name: str,
    realization: InjectionRealization,
):
    """Add a target waveform with an equal and opposite sum on other EEG channels."""
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    pick_names = tuple(raw.ch_names[index] for index in picks)
    if channel_name not in pick_names:
        raise ValueError(f"Recording has no non-bad EEG channel {channel_name!r}.")
    if len(picks) < 2:
        raise ValueError("Spatially balanced injection requires at least two EEG channels.")
    if realization.waveform_v.size != raw.n_times:
        raise ValueError("Injection and recording sample counts must match.")

    injected = raw.copy().load_data()
    target = raw.ch_names.index(channel_name)
    injected._data[target] += realization.waveform_v
    other_picks = picks[picks != target]
    injected._data[other_picks] -= realization.waveform_v / len(other_picks)
    return injected


def _positive(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number
