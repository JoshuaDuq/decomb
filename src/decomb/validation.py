"""Cohort-scale trials backing the flagship false-detection and recovery figure.

Pure per-recording and per-trial functions live here; the docs/ driver scripts loop over
the real cohort, call these, and cache results to disk so the figure itself can be
redrawn without recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from decomb import ablation, injection, notch, surrogates

METHOD_LABELS = {
    "holm": "decomb (Holm)",
    "bonferroni": "complete-family Bonferroni",
    "none": "MNE spectrum_fit",
}


@dataclass(frozen=True)
class FalseDetectionTrial:
    """One channel-recording's outcome under one detection procedure."""

    recording: str
    participant: str
    channel_name: str
    correction: str
    detected: bool


def false_detection_trials(
    raw,
    settings,
    rng: np.random.Generator,
    *,
    recording_name: str,
    participant: str,
) -> tuple[FalseDetectionTrial, ...]:
    """One sinusoid-free surrogate per real channel, tested under every correction.

    Every EEG channel of ``raw`` contributes its own surrogate (matched to its own
    empirical spectrum), and every correction procedure is evaluated on that same
    surrogate from one shared Thomson F-test pass.
    """
    surrogate = surrogates.surrogate_raw(raw, rng)
    channel_names = notch.eeg_channel_names(surrogate)
    models = ablation.fit_models_every_correction(surrogate, settings)
    trials = []
    for correction, model in models.items():
        detected_channels = {channel.channel_name for channel in model.channels}
        trials.extend(
            FalseDetectionTrial(
                recording=recording_name,
                participant=participant,
                channel_name=channel_name,
                correction=correction,
                detected=channel_name in detected_channels,
            )
            for channel_name in channel_names
        )
    return tuple(trials)


@dataclass(frozen=True)
class RecoveryTrial:
    """One paired background/injection trial's outcome under one detection procedure.

    ``remaining_fraction`` and ``collateral_energy_v2`` follow the paired-difference
    construction: what a correction procedure changes between the injected and
    background-only surrogate, never a single-arm measurement that an incidental
    background feature could bias.
    """

    recording: str
    channel_name: str
    correction: str
    kind: str
    frequency_hz: float
    amplitude_v: float
    drift_hz: float
    occupancy: float
    injected_energy_v2: float
    remaining_fraction: float
    collateral_energy_v2: float


@dataclass(frozen=True)
class _ChannelSpectrum:
    """One channel's PSD, computed once and reused for every band-power query."""

    frequencies_hz: np.ndarray
    psd: np.ndarray

    def band_power(self, low_hz: float, high_hz: float) -> float:
        return notch.band_power(self.frequencies_hz, self.psd, low_hz, high_hz)

    def total_power(self) -> float:
        return self.band_power(float(self.frequencies_hz[0]), float(self.frequencies_hz[-1]))


def _channel_spectrum(raw, channel_name: str, settings) -> _ChannelSpectrum:
    from decomb import recordings as recordings_module

    picks = [raw.ch_names.index(channel_name)]
    frequencies_hz, psd = recordings_module.psd(raw, picks, settings)
    return _ChannelSpectrum(frequencies_hz, psd[0])


def recovery_trial(
    background_raw,
    settings,
    spec: injection.SinusoidInjection,
    rng: np.random.Generator,
    *,
    recording_name: str,
    channel_name: str,
) -> tuple[RecoveryTrial, ...]:
    """Inject one artifact into a single-channel background and clean it three ways.

    ``background_raw`` must already be a sinusoid-free surrogate, built once and reused
    across every injection trial drawn against it so repeated calls do not resynthesize
    an expensive full-length Gaussian process per trial.
    """
    injected_raw = injection.inject_into_raw(background_raw, channel_name, spec, rng)
    half_width_hz = settings.spectral_resolution_hz / 2.0
    low_hz, high_hz = injection.injected_frequency_band_hz(spec, half_width_hz=half_width_hz)

    raw_background = _channel_spectrum(background_raw, channel_name, settings)
    raw_injected = _channel_spectrum(injected_raw, channel_name, settings)
    raw_background_power = raw_background.band_power(low_hz, high_hz)
    raw_injected_power = raw_injected.band_power(low_hz, high_hz)
    injected_energy_v2 = raw_injected_power - raw_background_power
    background_outside_power = raw_background.total_power() - raw_background_power
    injected_outside_power = raw_injected.total_power() - raw_injected_power

    background_models = ablation.fit_models_every_correction(background_raw, settings)
    injected_models = ablation.fit_models_every_correction(injected_raw, settings)

    trials = []
    for correction in background_models:
        cleaned_background = _cleaned_raw(background_raw, background_models[correction], settings)
        cleaned_injected = _cleaned_raw(injected_raw, injected_models[correction], settings)

        cleaned_background_spectrum = _channel_spectrum(cleaned_background, channel_name, settings)
        cleaned_injected_spectrum = _channel_spectrum(cleaned_injected, channel_name, settings)
        cleaned_background_power = cleaned_background_spectrum.band_power(low_hz, high_hz)
        cleaned_injected_power = cleaned_injected_spectrum.band_power(low_hz, high_hz)
        remaining_energy_v2 = cleaned_injected_power - cleaned_background_power
        remaining_fraction = (
            remaining_energy_v2 / injected_energy_v2
            if injected_energy_v2 > 0.0
            else float("nan")
        )

        cleaned_background_outside = (
            cleaned_background_spectrum.total_power() - cleaned_background_power
        )
        cleaned_injected_outside = (
            cleaned_injected_spectrum.total_power() - cleaned_injected_power
        )
        collateral_energy_v2 = (cleaned_injected_outside - cleaned_background_outside) - (
            injected_outside_power - background_outside_power
        )

        trials.append(
            RecoveryTrial(
                recording=recording_name,
                channel_name=channel_name,
                correction=correction,
                kind=spec.kind,
                frequency_hz=spec.frequency_hz,
                amplitude_v=spec.amplitude_v,
                drift_hz=spec.drift_hz,
                occupancy=spec.occupancy,
                injected_energy_v2=injected_energy_v2,
                remaining_fraction=remaining_fraction,
                collateral_energy_v2=collateral_energy_v2,
            )
        )
    return tuple(trials)


def _cleaned_raw(raw, model, settings):
    """Apply decomb's own stopband-width and merge rule to one ablation model's lines."""
    plans = notch.plan_channel_notches(model, settings)
    return notch.apply_channel_notches(raw, plans)
