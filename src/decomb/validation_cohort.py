"""Cohort-derived injection parameters and participant-level summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from decomb import injection, lines, notch, recordings, validation


@dataclass(frozen=True)
class ArtifactObservation:
    """Measured properties of one supported real-data stopband."""

    recording: str
    participant: str
    channel_name: str
    frequency_hz: float
    drift_hz: float
    occupancy: float
    amplitude_v: float


@dataclass(frozen=True)
class InjectionTarget:
    """One empirical injection specification before waveform realization."""

    kind: str
    frequency_hz: float
    amplitude_v: float
    drift_hz: float
    occupancy: float
    phase_rad: float

    def as_specification(self) -> injection.SinusoidInjection:
        return injection.SinusoidInjection(
            kind=self.kind,
            frequency_hz=self.frequency_hz,
            amplitude_v=self.amplitude_v,
            drift_hz=self.drift_hz,
            occupancy=self.occupancy,
            phase_rad=self.phase_rad,
        )


@dataclass(frozen=True)
class DetectionEstimate:
    """Detection proportion with a participant-cluster bootstrap interval."""

    method: str
    proportion: float
    lower: float
    upper: float
    participant_count: int
    channel_recording_count: int


@dataclass(frozen=True)
class RecordingPlan:
    """Cumulative authorized geometry and EEG-channel count for one recording."""

    recording: str
    eeg_channel_count: int
    plan: notch.HarmonicNotchPlan | None

    def __post_init__(self) -> None:
        if not self.recording.strip() or self.eeg_channel_count < 1:
            raise ValueError("Recording plans require an identity and EEG channels.")


@dataclass(frozen=True)
class LocalityBandwidth:
    """Actual recording-local and counterfactual cohort-global channel-Hz."""

    recording: str
    recording_local_channel_hz: float
    cohort_global_channel_hz: float


def observed_artifacts(
    raw,
    settings,
    model: lines.ArtifactModel,
    *,
    recording_name: str,
    participant: str,
) -> tuple[ArtifactObservation, ...]:
    """Measure injection parameters directly from supported real-data intervals."""
    referenced = validation.average_reference(raw)
    bounds = recordings.valid_window_bounds(
        referenced,
        window_s=settings.estimation_window_s,
        overlap=settings.estimation_overlap,
    )
    if len(bounds) != model.window_count:
        raise ValueError("The artifact model and recording use different windows.")

    observations = []
    for channel_plan in notch.plan_channel_notches(model, settings):
        channel = next(
            channel
            for channel in model.channels
            if channel.channel_name == channel_plan.channel_name
        )
        channel_data = referenced.get_data(picks=[channel.channel_name])[0]
        for stopband in channel_plan.geometry.stopbands:
            supported_lines = tuple(
                line
                for line in channel.lines
                if stopband.low_hz <= line.position_hz <= stopband.high_hz
            )
            if not supported_lines:
                raise ValueError("Every observed stopband requires supported lines.")
            positions_hz = np.array(
                [line.position_hz for line in supported_lines],
                dtype=float,
            )
            supporting_windows = tuple(
                sorted(
                    {
                        window_index
                        for line in supported_lines
                        for window_index in line.window_indices
                    }
                )
            )
            amplitudes_v = [
                _sinusoid_amplitude(
                    channel_data[start:stop],
                    float(referenced.info["sfreq"]),
                    line.position_hz,
                )
                for line in supported_lines
                for window_index in line.window_indices
                for start, stop in (bounds[window_index],)
            ]
            observations.append(
                ArtifactObservation(
                    recording=recording_name,
                    participant=participant,
                    channel_name=channel.channel_name,
                    frequency_hz=float((positions_hz.min() + positions_hz.max()) / 2.0),
                    drift_hz=float(positions_hz.max() - positions_hz.min()),
                    occupancy=len(supporting_windows) / model.window_count,
                    amplitude_v=float(np.median(amplitudes_v)),
                )
            )
    return tuple(observations)


def _sinusoid_amplitude(
    data: np.ndarray,
    sampling_frequency_hz: float,
    frequency_hz: float,
) -> float:
    """Least-squares peak amplitude at one known frequency."""
    values = np.asarray(data, dtype=float)
    times_s = np.arange(values.size) / sampling_frequency_hz
    design = np.column_stack(
        [
            np.sin(2.0 * np.pi * frequency_hz * times_s),
            np.cos(2.0 * np.pi * frequency_hz * times_s),
        ]
    )
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    amplitude_v = float(np.linalg.norm(coefficients))
    if not np.isfinite(amplitude_v) or amplitude_v <= 0.0:
        raise ValueError("Observed sinusoidal amplitudes must be finite and positive.")
    return amplitude_v


def sample_injection_targets(
    observations: tuple[ArtifactObservation, ...],
    *,
    count: int,
    frequency_range_hz: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[InjectionTarget, ...]:
    """Stratify injections over empirical frequency, drift, occupancy, and amplitude."""
    if count < 3:
        raise ValueError("At least three targets are required to represent every kind.")
    if not observations:
        raise ValueError("Injection targets require real artifact observations.")
    low_hz, high_hz = (float(value) for value in frequency_range_hz)
    if not 0.0 <= low_hz < high_hz:
        raise ValueError("frequency_range_hz must have increasing non-negative edges.")

    kinds = tuple(injection.KINDS[index % len(injection.KINDS)] for index in range(count))
    kind_counts = {kind: kinds.count(kind) for kind in injection.KINDS}
    frequencies = np.array([item.frequency_hz for item in observations])
    amplitudes = np.array([item.amplitude_v for item in observations])
    positive_drifts = np.array([item.drift_hz for item in observations if item.drift_hz > 0.0])
    partial_occupancies = np.array(
        [item.occupancy for item in observations if item.occupancy < 1.0]
    )
    if positive_drifts.size == 0:
        raise ValueError("Drifting injections require positive observed drift.")
    if partial_occupancies.size == 0:
        raise ValueError("Intermittent injections require observed partial occupancy.")

    values_by_kind = {}
    for kind, kind_count in kind_counts.items():
        quantiles = (np.arange(kind_count, dtype=float) + 0.5) / kind_count
        values_by_kind[kind] = {
            "frequency": rng.permutation(np.quantile(frequencies, quantiles)),
            "amplitude": rng.permutation(np.quantile(amplitudes, quantiles)),
            "drift": rng.permutation(np.quantile(positive_drifts, quantiles)),
            "occupancy": rng.permutation(np.quantile(partial_occupancies, quantiles)),
        }

    indices = {kind: 0 for kind in injection.KINDS}
    targets = []
    for kind in rng.permutation(kinds):
        index = indices[kind]
        indices[kind] += 1
        values = values_by_kind[kind]
        centre_hz = float(values["frequency"][index])
        amplitude_v = float(values["amplitude"][index])
        drift_hz = 0.0
        occupancy = 1.0
        frequency_hz = centre_hz
        if kind == "drifting":
            drift_magnitude_hz = float(values["drift"][index])
            drift_hz = drift_magnitude_hz * float(rng.choice((-1.0, 1.0)))
            frequency_hz = centre_hz - drift_hz / 2.0
        elif kind == "intermittent":
            occupancy = float(values["occupancy"][index])
        end_frequency_hz = frequency_hz + drift_hz
        if min(frequency_hz, end_frequency_hz) < low_hz or max(
            frequency_hz,
            end_frequency_hz,
        ) > high_hz:
            raise ValueError("An empirical injection trajectory leaves the analysis range.")
        targets.append(
            InjectionTarget(
                kind=kind,
                frequency_hz=frequency_hz,
                amplitude_v=amplitude_v,
                drift_hz=drift_hz,
                occupancy=occupancy,
                phase_rad=float(rng.uniform(0.0, 2.0 * np.pi)),
            )
        )
    return tuple(targets)


def detection_estimates(
    trials: tuple[validation.FalseDetectionTrial, ...],
    *,
    methods: tuple[str, ...] = validation.PRIMARY_METHODS,
    bootstrap_resamples: int,
    rng: np.random.Generator,
) -> tuple[DetectionEstimate, ...]:
    """Estimate detection rates with participant-cluster bootstrap intervals."""
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive.")
    estimates = []
    for method in methods:
        method_trials = tuple(trial for trial in trials if trial.method == method)
        participants = tuple(sorted({trial.participant for trial in method_trials}))
        if not participants:
            raise ValueError(f"No false-detection trials exist for {method!r}.")
        clusters = {
            participant: np.array(
                [
                    trial.detected
                    for trial in method_trials
                    if trial.participant == participant
                ],
                dtype=float,
            )
            for participant in participants
        }
        outcomes = np.concatenate(tuple(clusters.values()))
        bootstrap = np.empty(bootstrap_resamples, dtype=float)
        for index in range(bootstrap_resamples):
            selected = rng.choice(participants, size=len(participants), replace=True)
            bootstrap[index] = np.concatenate(
                [clusters[participant] for participant in selected]
            ).mean()
        lower, upper = np.quantile(bootstrap, (0.025, 0.975))
        estimates.append(
            DetectionEstimate(
                method=method,
                proportion=float(outcomes.mean()),
                lower=float(lower),
                upper=float(upper),
                participant_count=len(participants),
                channel_recording_count=outcomes.size,
            )
        )
    return tuple(estimates)


def locality_bandwidth(
    recording_plans: tuple[RecordingPlan, ...],
) -> tuple[LocalityBandwidth, ...]:
    """Compare actual recording-local geometry with one cohort-wide union."""
    if not recording_plans:
        raise ValueError("Locality bandwidth requires recording plans.")
    plans = tuple(item.plan for item in recording_plans if item.plan is not None)
    if not plans:
        raise ValueError("A cohort-wide union requires at least one authorized frequency.")
    cohort_plan = notch.merge_recording_plans(plans)
    cohort_width_hz = _unavailable_width_hz(cohort_plan)
    return tuple(
        LocalityBandwidth(
            recording=item.recording,
            recording_local_channel_hz=(
                0.0
                if item.plan is None
                else item.eeg_channel_count * _unavailable_width_hz(item.plan)
            ),
            cohort_global_channel_hz=item.eeg_channel_count * cohort_width_hz,
        )
        for item in recording_plans
    )


def _unavailable_width_hz(plan: notch.HarmonicNotchPlan) -> float:
    return float(sum(high_hz - low_hz for low_hz, high_hz in plan.unavailable_edges()))
