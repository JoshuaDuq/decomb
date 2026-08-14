"""Fixed factorial targets and participant-level validation summaries."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from decomb import injection, lines, notch, recordings, validation


@dataclass(frozen=True)
class LineObservation:
    """Measured properties of one supported real-data stopband."""

    recording: str
    participant: str
    channel_name: str
    frequency_hz: float
    drift_hz: float
    occupancy: float
    amplitude_v: float


@dataclass(frozen=True)
class DetectionEstimate:
    """Recording FWER estimate and secondary channel-level detection proportion."""

    recording_false_authorization_proportion: float
    lower: float
    upper: float
    participant_count: int
    recording_count: int
    channel_false_detection_proportion: float
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


def observed_lines(
    raw,
    settings,
    model: lines.LineModel,
    *,
    recording_name: str,
    participant: str,
) -> tuple[LineObservation, ...]:
    """Describe supported real-data intervals without defining benchmark targets."""
    bounds = recordings.valid_window_bounds(
        raw,
        window_s=settings.estimation_window_s,
        overlap=settings.estimation_overlap,
    )
    if len(bounds) != model.window_count:
        raise ValueError("The line model and recording use different windows.")

    observations = []
    for channel_plan in notch.plan_channel_notches(model, settings):
        channel = next(
            channel
            for channel in model.channels
            if channel.channel_name == channel_plan.channel_name
        )
        channel_data = raw.get_data(picks=[channel.channel_name])[0]
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
                    float(raw.info["sfreq"]),
                    line.position_hz,
                )
                for line in supported_lines
                for window_index in line.window_indices
                for start, stop in (bounds[window_index],)
            ]
            observations.append(
                LineObservation(
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


def factorial_injection_targets(
    *,
    frequency_range_hz: tuple[float, float],
) -> tuple[injection.FactorialInjectionTarget, ...]:
    """Return the fixed 90-point recovery design independent of Decomb settings."""
    low_hz, high_hz = (float(value) for value in frequency_range_hz)
    if not 0.0 <= low_hz < high_hz:
        raise ValueError("frequency_range_hz must have increasing non-negative edges.")

    frequencies_hz = tuple(
        low_hz + fraction * (high_hz - low_hz)
        for fraction in (0.25, 0.5, 0.75)
    )
    strengths_db = (-20.0, -10.0, 0.0)
    phases_rad = (0.0, np.pi / 2.0)
    drift_values_hz = (0.05, 0.2)
    occupancies = (0.25, 0.75)
    if frequencies_hz[-1] + max(drift_values_hz) >= high_hz:
        raise ValueError("The analysis range is too narrow for the factorial drift levels.")

    targets = [
        injection.FactorialInjectionTarget("stationary", frequency, strength, phase_rad=phase)
        for frequency, strength, phase in product(
            frequencies_hz,
            strengths_db,
            phases_rad,
        )
    ]
    targets.extend(
        injection.FactorialInjectionTarget(
            "drifting",
            frequency,
            strength,
            drift_hz=drift,
            phase_rad=phase,
        )
        for frequency, strength, drift, phase in product(
            frequencies_hz,
            strengths_db,
            drift_values_hz,
            phases_rad,
        )
    )
    targets.extend(
        injection.FactorialInjectionTarget(
            "intermittent",
            frequency,
            strength,
            occupancy=occupancy,
            phase_rad=phase,
        )
        for frequency, strength, occupancy, phase in product(
            frequencies_hz,
            strengths_db,
            occupancies,
            phases_rad,
        )
    )
    return tuple(targets)


def validate_factorial_targets_for_cohort(
    targets: tuple[injection.FactorialInjectionTarget, ...],
    *,
    sampling_frequencies_hz: tuple[float, ...],
) -> None:
    """Require every target trajectory to fit every recording in the cohort."""
    if not targets:
        raise ValueError("Factorial validation requires targets.")
    sampling_frequencies = np.asarray(sampling_frequencies_hz, dtype=float)
    if (
        sampling_frequencies.ndim != 1
        or sampling_frequencies.size == 0
        or not np.all(np.isfinite(sampling_frequencies))
        or np.any(sampling_frequencies <= 0.0)
    ):
        raise ValueError("Sampling frequencies must be finite and positive.")

    lowest_nyquist_hz = float(sampling_frequencies.min() / 2.0)
    highest_target_hz = max(
        target.frequency_hz + max(0.0, target.drift_hz)
        for target in targets
    )
    lowest_target_hz = min(
        target.frequency_hz + min(0.0, target.drift_hz)
        for target in targets
    )
    if lowest_target_hz <= 0.0 or highest_target_hz >= lowest_nyquist_hz:
        raise ValueError(
            "Every injection trajectory must lie strictly inside the lowest cohort "
            f"Nyquist frequency ({lowest_nyquist_hz:g} Hz); target range is "
            f"[{lowest_target_hz:g}, {highest_target_hz:g}] Hz."
        )


def detection_estimate(
    trials: tuple[validation.FalseDetectionTrial, ...],
    *,
    bootstrap_resamples: int,
    rng: np.random.Generator,
) -> DetectionEstimate:
    """Estimate detection rates with participant-cluster bootstrap intervals."""
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive.")
    participants = tuple(sorted({trial.participant for trial in trials}))
    if not participants:
        raise ValueError("False-detection trials are required.")
    recording_participants: dict[str, str] = {}
    recording_channels: dict[tuple[str, str], list[bool]] = {}
    for trial in trials:
        previous = recording_participants.setdefault(
            trial.recording,
            trial.participant,
        )
        if previous != trial.participant:
            raise ValueError("Each recording must belong to exactly one participant.")
        recording_channels.setdefault(
            (trial.participant, trial.recording),
            [],
        ).append(trial.line_detected)
    clusters = {
        participant: np.array(
            [
                any(outcomes)
                for (owner, _), outcomes in recording_channels.items()
                if owner == participant
            ],
            dtype=float,
        )
        for participant in participants
    }
    recording_outcomes = np.concatenate(tuple(clusters.values()))
    channel_outcomes = np.array(
        [trial.line_detected for trial in trials],
        dtype=float,
    )
    bootstrap = np.empty(bootstrap_resamples, dtype=float)
    for index in range(bootstrap_resamples):
        selected = rng.choice(participants, size=len(participants), replace=True)
        bootstrap[index] = np.concatenate(
            [clusters[participant] for participant in selected]
        ).mean()
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    return DetectionEstimate(
        recording_false_authorization_proportion=float(recording_outcomes.mean()),
        lower=float(lower),
        upper=float(upper),
        participant_count=len(participants),
        recording_count=recording_outcomes.size,
        channel_false_detection_proportion=float(channel_outcomes.mean()),
        channel_recording_count=channel_outcomes.size,
    )


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
