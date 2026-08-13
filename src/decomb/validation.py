"""Scientifically matched surrogate calibration and paired injection trials."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from decomb import ablation, injection, lines, notch, recordings, surrogates

DECOMB_HOLM = "decomb_holm"
COMPLETE_BONFERRONI = "complete_family_bonferroni"
MNE_SPECTRUM_FIT_54S = "mne_spectrum_fit_54s"
MNE_SPECTRUM_FIT_10S = "mne_spectrum_fit_10s"

PRIMARY_METHODS = (
    DECOMB_HOLM,
    COMPLETE_BONFERRONI,
    MNE_SPECTRUM_FIT_54S,
)
ALL_METHODS = (*PRIMARY_METHODS, MNE_SPECTRUM_FIT_10S)
METHOD_LABELS = {
    DECOMB_HOLM: "Decomb Holm",
    COMPLETE_BONFERRONI: "Complete-family Bonferroni",
    MNE_SPECTRUM_FIT_54S: "MNE spectrum_fit (54 s)",
    MNE_SPECTRUM_FIT_10S: "MNE spectrum_fit (10 s)",
}


@dataclass(frozen=True)
class FalseDetectionTrial:
    """One channel-recording decision on a sinusoid-free surrogate."""

    recording: str
    participant: str
    channel_name: str
    method: str
    detected: bool


@dataclass(frozen=True)
class MneSpectrumFitResult:
    """Native MNE output and the channels with an F-test-authorized subtraction."""

    cleaned: object
    detected_channels: tuple[str, ...]


@dataclass(frozen=True)
class PairedEnergyMetrics:
    """Orthogonal decomposition of ``C(X + A) - C(X)``."""

    injected_energy_v2: float
    difference_energy_v2: float
    artifact_to_background_db: float
    remaining_fraction: float
    collateral_fraction: float


@dataclass(frozen=True)
class RecoveryTrial:
    """One paired injection result under one cleaning method."""

    recording: str
    participant: str
    channel_name: str
    method: str
    kind: str
    frequency_hz: float
    amplitude_v: float
    drift_hz: float
    occupancy: float
    injected_energy_v2: float
    difference_energy_v2: float
    artifact_to_background_db: float
    remaining_fraction: float
    collateral_fraction: float


def average_reference(raw):
    """Return data in the common-average subspace used by Decomb detection."""
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(picks) < 2:
        raise ValueError(
            "Average referencing requires at least two non-bad EEG channels."
        )
    referenced = raw.copy().load_data()
    data = referenced.get_data(picks=picks)
    referenced._data[picks] = data - data.mean(axis=0, keepdims=True)
    return referenced


def false_detection_trials(
    raw,
    settings,
    rng: np.random.Generator,
    *,
    recording_name: str,
    participant: str,
) -> tuple[FalseDetectionTrial, ...]:
    """Test one spectrum-matched surrogate under every validation method."""
    surrogate = average_reference(surrogates.surrogate_raw(raw, rng))
    channel_names = notch.eeg_channel_names(surrogate)
    models = ablation.fit_holm_and_bonferroni_models(surrogate, settings)
    detected_by_method = {
        DECOMB_HOLM: {channel.channel_name for channel in models["holm"].channels},
        COMPLETE_BONFERRONI: {
            channel.channel_name for channel in models["bonferroni"].channels
        },
        MNE_SPECTRUM_FIT_54S: set(
            mne_spectrum_fit_detected_channels(
                surrogate,
                window_s=settings.estimation_window_s,
                p_value=settings.familywise_error_rate,
            )
        ),
        MNE_SPECTRUM_FIT_10S: set(
            mne_spectrum_fit_detected_channels(
                surrogate,
                window_s=10.0,
                p_value=settings.familywise_error_rate,
            )
        ),
    }
    return tuple(
        FalseDetectionTrial(
            recording=recording_name,
            participant=participant,
            channel_name=channel_name,
            method=method,
            detected=channel_name in detected_by_method[method],
        )
        for method in ALL_METHODS
        for channel_name in channel_names
    )


def mne_spectrum_fit(raw, *, window_s: float, p_value: float) -> MneSpectrumFitResult:
    """Run MNE's public spectrum-fit subtraction with its native decision rule."""
    detected_channels = mne_spectrum_fit_detected_channels(
        raw,
        window_s=window_s,
        p_value=p_value,
    )
    cleaned = apply_mne_spectrum_fit(raw, window_s=window_s, p_value=p_value)
    return MneSpectrumFitResult(cleaned, detected_channels)


def apply_mne_spectrum_fit(raw, *, window_s: float, p_value: float):
    """Apply MNE's public spectrum-fit subtraction without duplicating its fit."""
    channel_names = notch.eeg_channel_names(raw)
    return raw.copy().notch_filter(
        freqs=None,
        picks=list(channel_names),
        filter_length=f"{window_s:.17g}s",
        method="spectrum_fit",
        p_value=p_value,
        skip_by_annotation=recordings.ACQUISITION_BOUNDARY_ANNOTATIONS,
        n_jobs=1,
        verbose="ERROR",
    )


def mne_spectrum_fit_detected_channels(
    raw,
    *,
    window_s: float,
    p_value: float,
) -> tuple[str, ...]:
    """Reproduce MNE's documented per-window Bonferroni F-test decisions.

    MNE's public filter returns only the cleaned samples. This evaluates the same Thomson
    statistic and ``p_value / n_times`` threshold used by MNE 1.12 so Panel A can report
    exact channel decisions without classifying overlap-add rounding noise as detection.
    """
    import mne

    if not np.isfinite(p_value) or not 0.0 < p_value < 1.0:
        raise ValueError("p_value must lie strictly between zero and one.")
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(picks) == 0:
        raise ValueError("MNE spectrum fitting requires a non-bad EEG channel.")
    sampling_frequency_hz = float(raw.info["sfreq"])
    requested_samples = int(np.ceil(window_s * sampling_frequency_hz))
    if requested_samples < 2:
        raise ValueError("MNE spectrum-fit windows require at least two samples.")

    data = raw.get_data(picks=picks)
    detected = np.zeros(len(picks), dtype=bool)
    for segment_start, segment_stop in recordings.acquisition_segments(raw):
        segment_samples = segment_stop - segment_start
        window_samples = min(requested_samples, segment_samples)
        for start, stop in _mne_spectrum_fit_window_bounds(
            segment_samples,
            window_samples,
        ):
            window = data[:, segment_start + start : segment_start + stop]
            _, window_p_values = lines.thomson_f_p_values(
                window[np.newaxis, :, :],
                sampling_frequency_hz,
                frequency_range_hz=(0.0, sampling_frequency_hz / 2.0),
            )
            detected |= np.any(
                window_p_values[0] < p_value / window.shape[-1],
                axis=1,
            )
    return tuple(
        raw.ch_names[pick]
        for pick, is_detected in zip(picks, detected, strict=True)
        if is_detected
    )


def _mne_spectrum_fit_window_bounds(
    sample_count: int,
    window_samples: int,
) -> tuple[tuple[int, int], ...]:
    """Window bounds used by MNE's constant-overlap-add spectrum fitting."""
    if sample_count < window_samples or window_samples < 2:
        raise ValueError("MNE spectrum-fit window geometry is invalid.")
    overlap_samples = (window_samples + 1) // 2
    step_samples = window_samples - overlap_samples
    starts = np.arange(0, sample_count - window_samples + 1, step_samples, dtype=int)
    stops = starts + window_samples
    stops[-1] = sample_count
    return tuple(zip(starts.tolist(), stops.tolist(), strict=True))


def recovery_trial(
    background_raw,
    settings,
    spec: injection.SinusoidInjection,
    rng: np.random.Generator,
    *,
    recording_name: str,
    participant: str,
    channel_name: str,
) -> tuple[RecoveryTrial, ...]:
    """Clean one paired average-referenced background/injection trial four ways."""
    if channel_name not in background_raw.ch_names:
        raise ValueError(f"Recording does not contain channel {channel_name!r}.")
    background = average_reference(background_raw)
    realization = injection.realize_injection(
        spec,
        background.n_times,
        float(background.info["sfreq"]),
        rng,
    )
    injected = injection.inject_into_average_reference(
        background,
        channel_name,
        realization,
    )
    valid_samples = _valid_sample_mask(background)
    temporal_basis = realization.temporal_basis * valid_samples

    cleaned_backgrounds = _clean_every_method(background, settings)
    cleaned_injections = _clean_every_method(injected, settings)
    trials = []
    for method in ALL_METHODS:
        metrics = paired_energy_metrics(
            background.get_data(),
            injected.get_data(),
            cleaned_backgrounds[method].get_data(),
            cleaned_injections[method].get_data(),
            temporal_basis,
            valid_samples,
        )
        trials.append(
            RecoveryTrial(
                recording=recording_name,
                participant=participant,
                channel_name=channel_name,
                method=method,
                kind=spec.kind,
                frequency_hz=spec.frequency_hz,
                amplitude_v=spec.amplitude_v,
                drift_hz=spec.drift_hz,
                occupancy=spec.occupancy,
                injected_energy_v2=metrics.injected_energy_v2,
                difference_energy_v2=metrics.difference_energy_v2,
                artifact_to_background_db=metrics.artifact_to_background_db,
                remaining_fraction=metrics.remaining_fraction,
                collateral_fraction=metrics.collateral_fraction,
            )
        )
    return tuple(trials)


def paired_energy_metrics(
    background: np.ndarray,
    injected: np.ndarray,
    cleaned_background: np.ndarray,
    cleaned_injected: np.ndarray,
    temporal_basis: np.ndarray,
    valid_samples: np.ndarray,
) -> PairedEnergyMetrics:
    """Decompose the paired cleaned difference into injected and orthogonal energy."""
    arrays = tuple(
        np.asarray(values, dtype=float)
        for values in (background, injected, cleaned_background, cleaned_injected)
    )
    if any(values.ndim != 2 for values in arrays):
        raise ValueError("Paired trial data must be channel-by-time arrays.")
    if len({values.shape for values in arrays}) != 1:
        raise ValueError("Every paired trial array must have the same shape.")
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("Paired trial data must contain only finite values.")
    mask = np.asarray(valid_samples, dtype=bool)
    if mask.shape != (arrays[0].shape[1],) or not np.any(mask):
        raise ValueError("valid_samples must select recording samples.")
    basis = np.asarray(temporal_basis, dtype=float)
    if basis.ndim != 2 or basis.shape[1] != mask.size:
        raise ValueError("temporal_basis must have shape (components, n_samples).")

    orthonormal_basis, triangular = np.linalg.qr(basis[:, mask].T, mode="reduced")
    if np.linalg.matrix_rank(triangular) != basis.shape[0]:
        raise ValueError("The injected temporal subspace must have full rank.")

    background_valid, injected_valid, cleaned_background_valid, cleaned_injected_valid = (
        values[:, mask] for values in arrays
    )
    artifact = injected_valid - background_valid
    cleaned_difference = cleaned_injected_valid - cleaned_background_valid
    projected_difference = (
        cleaned_difference @ orthonormal_basis
    ) @ orthonormal_basis.T
    collateral_difference = cleaned_difference - projected_difference
    projected_background = (
        background_valid @ orthonormal_basis
    ) @ orthonormal_basis.T

    injected_energy_v2 = float(np.sum(artifact**2))
    background_subspace_energy_v2 = float(np.sum(projected_background**2))
    if injected_energy_v2 <= 0.0 or background_subspace_energy_v2 <= 0.0:
        raise ValueError("Paired energy ratios require positive artifact and background energy.")
    difference_energy_v2 = float(np.sum(cleaned_difference**2))
    remaining_energy_v2 = float(np.sum(projected_difference**2))
    collateral_energy_v2 = float(np.sum(collateral_difference**2))
    return PairedEnergyMetrics(
        injected_energy_v2=injected_energy_v2,
        difference_energy_v2=difference_energy_v2,
        artifact_to_background_db=(
            10.0 * np.log10(injected_energy_v2 / background_subspace_energy_v2)
        ),
        remaining_fraction=remaining_energy_v2 / injected_energy_v2,
        collateral_fraction=collateral_energy_v2 / injected_energy_v2,
    )


def _valid_sample_mask(raw) -> np.ndarray:
    mask = np.zeros(raw.n_times, dtype=bool)
    for start, stop in recordings.acquisition_segments(raw):
        mask[start:stop] = True
    if not np.any(mask):
        raise ValueError("A validation recording requires valid acquisition samples.")
    return mask


def _clean_every_method(raw, settings) -> dict[str, object]:
    mne_54 = apply_mne_spectrum_fit(
        raw,
        window_s=settings.estimation_window_s,
        p_value=settings.familywise_error_rate,
    )
    mne_10 = apply_mne_spectrum_fit(
        raw,
        window_s=10.0,
        p_value=settings.familywise_error_rate,
    )
    return {
        DECOMB_HOLM: notch.clean_until_no_supported_lines(raw, settings).cleaned,
        COMPLETE_BONFERRONI: ablation.clean_until_no_bonferroni_lines(raw, settings).cleaned,
        MNE_SPECTRUM_FIT_54S: mne_54,
        MNE_SPECTRUM_FIT_10S: mne_10,
    }
