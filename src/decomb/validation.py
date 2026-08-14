"""Scientifically matched surrogate calibration and paired injection trials."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from decomb import injection, notch, recordings, surrogates


@dataclass(frozen=True)
class FalseDetectionTrial:
    """One channel-recording decision on a line-free surrogate."""

    recording: str
    participant: str
    channel_name: str
    line_detected: bool


@dataclass(frozen=True)
class PairedEnergyMetrics:
    """Orthogonal decomposition of ``C(X + A) - C(X)``."""

    injected_energy_v2: float
    difference_energy_v2: float
    component_to_background_db: float
    remaining_fraction: float
    collateral_fraction: float


@dataclass(frozen=True)
class RecoveryTrial:
    """One paired Decomb injection result."""

    recording: str
    participant: str
    channel_name: str
    kind: str
    frequency_hz: float
    amplitude_v: float
    drift_hz: float
    occupancy: float
    phase_rad: float
    injected_energy_v2: float
    difference_energy_v2: float
    component_to_background_db: float
    remaining_fraction: float
    collateral_fraction: float


@dataclass(frozen=True)
class SequentialAuthorizationTrial:
    """Full-sequence Decomb decisions for a persistent known injected line."""

    recording: str
    participant: str
    kind: str
    frequency_hz: float
    component_to_background_db: float
    drift_hz: float
    occupancy: float
    phase_rad: float
    injected_line_authorized: bool
    unsupported_line_authorized: bool
    removal_round_count: int


@dataclass(frozen=True)
class RecoveryResult:
    """Paired recovery result and applicable full-sequence calibration."""

    trial: RecoveryTrial
    sequential_authorization: SequentialAuthorizationTrial | None


@dataclass(frozen=True)
class DecombCleaningResult:
    """Cleaned samples and Decomb's auditable round sequence."""

    cleaned: object
    decomb: object


def false_detection_trials(
    raw,
    settings,
    rng: np.random.Generator,
    *,
    recording_name: str,
    participant: str,
) -> tuple[FalseDetectionTrial, ...]:
    """Test Decomb's joint authorization on one spectrum-matched surrogate."""
    surrogate = surrogates.surrogate_raw(raw, rng)
    channel_names = notch.eeg_channel_names(surrogate)
    decomb_evidence = notch.fit_harmonic_round(surrogate, settings)
    scanner_authorized = decomb_evidence.scanner_harmonics is not None
    detected_channels = (
        set(channel_names)
        if scanner_authorized
        else {
            channel.channel_name
            for channel in decomb_evidence.model.channels
        }
    )
    return tuple(
        FalseDetectionTrial(
            recording=recording_name,
            participant=participant,
            channel_name=channel_name,
            line_detected=channel_name in detected_channels,
        )
        for channel_name in channel_names
    )


def recovery_trial(
    background_raw,
    settings,
    target: injection.FactorialInjectionTarget,
    rng: np.random.Generator,
    *,
    recording_name: str,
    participant: str,
    channel_name: str,
) -> RecoveryResult:
    """Measure Decomb on one paired spatially balanced injection trial."""
    if channel_name not in background_raw.ch_names:
        raise ValueError(f"Recording does not contain channel {channel_name!r}.")
    background = background_raw.copy().load_data()
    spec, realization = realize_factorial_injection(
        background,
        channel_name,
        target,
        rng,
    )
    injected = injection.inject_spatially_balanced(
        background,
        channel_name,
        realization,
    )
    valid_samples = _valid_sample_mask(background)
    temporal_basis = realization.temporal_basis * valid_samples

    cleaned_background = _clean_decomb(background, settings)
    cleaned_injection = _clean_decomb(injected, settings)
    metrics = paired_energy_metrics(
        background.get_data(),
        injected.get_data(),
        cleaned_background.cleaned.get_data(),
        cleaned_injection.cleaned.get_data(),
        temporal_basis,
        valid_samples,
    )
    trial = RecoveryTrial(
        recording=recording_name,
        participant=participant,
        channel_name=channel_name,
        kind=spec.kind,
        frequency_hz=spec.frequency_hz,
        amplitude_v=spec.amplitude_v,
        drift_hz=spec.drift_hz,
        occupancy=spec.occupancy,
        phase_rad=spec.phase_rad,
        injected_energy_v2=metrics.injected_energy_v2,
        difference_energy_v2=metrics.difference_energy_v2,
        component_to_background_db=metrics.component_to_background_db,
        remaining_fraction=metrics.remaining_fraction,
        collateral_fraction=metrics.collateral_fraction,
    )
    sequential = None
    if spec.kind in {"stationary", "drifting"}:
        supported, unsupported = sequential_authorization_outcomes(
            cleaned_injection.decomb,
            spec,
            frequency_bin_width_hz=settings.frequency_bin_width_hz,
        )
        sequential = SequentialAuthorizationTrial(
            recording=recording_name,
            participant=participant,
            kind=spec.kind,
            frequency_hz=spec.frequency_hz,
            component_to_background_db=target.component_to_background_db,
            drift_hz=spec.drift_hz,
            occupancy=spec.occupancy,
            phase_rad=spec.phase_rad,
            injected_line_authorized=supported,
            unsupported_line_authorized=unsupported,
            removal_round_count=len(cleaned_injection.decomb.rounds),
        )
    return RecoveryResult(trial, sequential)


def realize_factorial_injection(
    background,
    channel_name: str,
    target: injection.FactorialInjectionTarget,
    rng: np.random.Generator,
) -> tuple[injection.SinusoidInjection, injection.InjectionRealization]:
    """Scale one fixed factorial target to its requested background-subspace SNR."""
    unit_spec = target.as_specification(1.0)
    unit = injection.realize_injection(
        unit_spec,
        background.n_times,
        float(background.info["sfreq"]),
        rng,
    )
    valid_samples = _valid_sample_mask(background)
    basis, triangular = np.linalg.qr(
        unit.temporal_basis[:, valid_samples].T,
        mode="reduced",
    )
    if np.linalg.matrix_rank(triangular) != unit.temporal_basis.shape[0]:
        raise ValueError("The factorial injection subspace must have full rank.")
    background_data = background.get_data()[:, valid_samples]
    projected_background = (background_data @ basis) @ basis.T
    background_energy_v2 = float(np.sum(projected_background**2))

    unit_injected = injection.inject_spatially_balanced(
        background,
        channel_name,
        unit,
    )
    unit_component = (
        unit_injected.get_data()[:, valid_samples] - background_data
    )
    unit_energy_v2 = float(np.sum(unit_component**2))
    if background_energy_v2 <= 0.0 or unit_energy_v2 <= 0.0:
        raise ValueError("Factorial injection scaling requires positive energies.")

    target_energy_v2 = background_energy_v2 * 10.0 ** (
        target.component_to_background_db / 10.0
    )
    amplitude_v = float(np.sqrt(target_energy_v2 / unit_energy_v2))
    spec = target.as_specification(amplitude_v)
    realization = injection.InjectionRealization(
        waveform_v=amplitude_v * unit.waveform_v,
        temporal_basis=unit.temporal_basis,
    )
    return spec, realization


def sequential_authorization_outcomes(
    cleaning,
    spec: injection.SinusoidInjection,
    *,
    frequency_bin_width_hz: float,
) -> tuple[bool, bool]:
    """Report injected-support and unsupported authorizations over every round."""
    half_width_hz = float(frequency_bin_width_hz) / 2.0
    support_low_hz, support_high_hz = injection.injected_frequency_band_hz(
        spec,
        half_width_hz=half_width_hz,
    )
    positions_hz = tuple(
        line.position_hz
        for round_ in cleaning.rounds
        for channel in round_.model.channels
        for line in channel.lines
    ) + tuple(
        harmonic * round_.scanner_harmonics.fundamental_hz
        for round_ in cleaning.rounds
        if round_.scanner_plan is not None
        for stopband in round_.scanner_plan.stopbands
        for harmonic in stopband.harmonics
    )
    supported = any(
        support_low_hz <= position_hz <= support_high_hz
        for position_hz in positions_hz
    )
    unsupported = any(
        position_hz < support_low_hz or position_hz > support_high_hz
        for position_hz in positions_hz
    )
    return supported, unsupported


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
    component = injected_valid - background_valid
    cleaned_difference = cleaned_injected_valid - cleaned_background_valid
    projected_difference = (
        cleaned_difference @ orthonormal_basis
    ) @ orthonormal_basis.T
    collateral_difference = cleaned_difference - projected_difference
    projected_background = (
        background_valid @ orthonormal_basis
    ) @ orthonormal_basis.T

    injected_energy_v2 = float(np.sum(component**2))
    background_subspace_energy_v2 = float(np.sum(projected_background**2))
    if injected_energy_v2 <= 0.0 or background_subspace_energy_v2 <= 0.0:
        raise ValueError("Paired energy ratios require positive component and background energy.")
    difference_energy_v2 = float(np.sum(cleaned_difference**2))
    remaining_energy_v2 = float(np.sum(projected_difference**2))
    collateral_energy_v2 = float(np.sum(collateral_difference**2))
    return PairedEnergyMetrics(
        injected_energy_v2=injected_energy_v2,
        difference_energy_v2=difference_energy_v2,
        component_to_background_db=(
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


def _clean_decomb(raw, settings) -> DecombCleaningResult:
    decomb = notch.clean_until_no_supported_lines(raw, settings)
    return DecombCleaningResult(cleaned=decomb.cleaned, decomb=decomb)
