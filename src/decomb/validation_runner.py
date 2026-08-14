"""Reproducible orchestration and persistence for the 90-recording validation."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from decomb import notch, recordings, surrogates, validation, validation_cohort

FALSE_DETECTION_NAME = "false_detection_trials.tsv"
FALSE_AUTHORIZATION_ESTIMATE_NAME = "false_authorization_estimate.tsv"
RECOVERY_NAME = "recovery_trials.tsv"
SEQUENTIAL_NAME = "sequential_authorization_trials.tsv"
OBSERVATION_NAME = "line_observations.tsv"
LOCALITY_NAME = "locality_bandwidth.tsv"


@dataclass(frozen=True)
class CohortValidationResults:
    """All measurements needed to redraw the flagship figure."""

    false_detection_trials: tuple[validation.FalseDetectionTrial, ...]
    false_authorization_estimate: validation_cohort.DetectionEstimate
    recovery_trials: tuple[validation.RecoveryTrial, ...]
    sequential_authorization_trials: tuple[
        validation.SequentialAuthorizationTrial,
        ...,
    ]
    line_observations: tuple[validation_cohort.LineObservation, ...]
    locality_bandwidth: tuple[validation_cohort.LocalityBandwidth, ...]


@dataclass(frozen=True)
class VerificationSummary:
    """Exact refit, channel preservation, quantization, and null-copy counts."""

    recordings_exactly_refitted: int
    unchanged_channels: int
    derivatives_reproduced_exactly: int
    null_recordings_copied_unchanged: int


def run_cohort_validation(
    runs: tuple[Path, ...],
    settings,
    *,
    random_seed: int,
) -> CohortValidationResults:
    """Run one null surrogate and one factorial injection per real recording."""
    if not runs:
        raise ValueError("Cohort validation requires recordings.")
    targets = validation_cohort.factorial_injection_targets(
        frequency_range_hz=settings.frequency_range_hz,
    )
    if len(runs) != len(targets):
        raise ValueError(
            f"The complete factorial design requires {len(targets)} recordings; "
            f"received {len(runs)}."
        )
    sampling_frequencies_hz = tuple(
        float(recordings.read_bids_raw(vhdr).info["sfreq"])
        for vhdr in runs
    )
    validation_cohort.validate_factorial_targets_for_cohort(
        targets,
        sampling_frequencies_hz=sampling_frequencies_hz,
    )

    seed_sequences = np.random.SeedSequence(random_seed).spawn(2 * len(runs) + 2)
    false_trials = []
    observations = []
    recording_plans = []

    for index, (vhdr, seed_sequence) in enumerate(
        zip(runs, seed_sequences[: len(runs)], strict=True),
        start=1,
    ):
        started = time.time()
        raw = recordings.read_bids_raw(vhdr)
        cleaning = notch.clean_until_no_supported_lines(raw, settings)
        initial_model = (
            cleaning.rounds[0].model if cleaning.rounds else cleaning.residual_model
        )
        participant = recordings.subject_of(vhdr)
        observations.extend(
            validation_cohort.observed_lines(
                raw,
                settings,
                initial_model,
                recording_name=vhdr.stem,
                participant=participant,
            )
        )
        false_trials.extend(
            validation.false_detection_trials(
                raw,
                settings,
                np.random.default_rng(seed_sequence),
                recording_name=vhdr.stem,
                participant=participant,
            )
        )
        cumulative_plan = (
            None
            if not cleaning.rounds
            else notch.merge_recording_plans(
                tuple(round_.filter_plan for round_ in cleaning.rounds)
            )
        )
        recording_plans.append(
            validation_cohort.RecordingPlan(
                vhdr.stem,
                initial_model.channel_count,
                cumulative_plan,
            )
        )
        print(
            f"[{index}/{len(runs)}] calibration and locality {vhdr.stem} "
            f"({time.time() - started:.1f} s)"
        )

    design_rng = np.random.default_rng(seed_sequences[-2])
    targets = tuple(targets[index] for index in design_rng.permutation(len(targets)))
    recovery_trials = []
    sequential_trials = []
    recovery_seeds = seed_sequences[len(runs) : 2 * len(runs)]
    for index, (vhdr, target, seed_sequence) in enumerate(
        zip(runs, targets, recovery_seeds, strict=True),
        start=1,
    ):
        started = time.time()
        rng = np.random.default_rng(seed_sequence)
        raw = recordings.read_bids_raw(vhdr)
        background = surrogates.surrogate_raw(raw, rng)
        channel_names = notch.eeg_channel_names(background)
        channel_name = channel_names[int(rng.integers(0, len(channel_names)))]
        recovery = validation.recovery_trial(
            background,
            settings,
            target,
            rng,
            recording_name=vhdr.stem,
            participant=recordings.subject_of(vhdr),
            channel_name=channel_name,
        )
        recovery_trials.append(recovery.trial)
        if recovery.sequential_authorization is not None:
            sequential_trials.append(recovery.sequential_authorization)
        print(
            f"[{index}/{len(runs)}] paired {target.kind} injection {vhdr.stem} "
            f"({time.time() - started:.1f} s)"
        )

    return CohortValidationResults(
        false_detection_trials=tuple(false_trials),
        false_authorization_estimate=validation_cohort.detection_estimate(
            tuple(false_trials),
            bootstrap_resamples=10_000,
            rng=np.random.default_rng(seed_sequences[-1]),
        ),
        recovery_trials=tuple(recovery_trials),
        sequential_authorization_trials=tuple(sequential_trials),
        line_observations=tuple(observations),
        locality_bandwidth=validation_cohort.locality_bandwidth(tuple(recording_plans)),
    )


def write_results(results: CohortValidationResults, output_dir: Path) -> None:
    """Write every result table atomically with a fixed, auditable schema."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        FALSE_DETECTION_NAME: _dataclass_frame(results.false_detection_trials),
        FALSE_AUTHORIZATION_ESTIMATE_NAME: _dataclass_frame(
            (results.false_authorization_estimate,)
        ),
        RECOVERY_NAME: _dataclass_frame(results.recovery_trials),
        SEQUENTIAL_NAME: _dataclass_frame(results.sequential_authorization_trials),
        OBSERVATION_NAME: _dataclass_frame(results.line_observations),
        LOCALITY_NAME: _dataclass_frame(results.locality_bandwidth),
    }
    for name, table in tables.items():
        recordings.write_tsv_atomic(table, output / name)


def read_results(output_dir: Path) -> CohortValidationResults:
    """Load cached measurements and require their current exact schemas."""
    output = Path(output_dir)
    return CohortValidationResults(
        false_detection_trials=_read_dataclasses(
            output / FALSE_DETECTION_NAME,
            validation.FalseDetectionTrial,
        ),
        false_authorization_estimate=_read_single_dataclass(
            output / FALSE_AUTHORIZATION_ESTIMATE_NAME,
            validation_cohort.DetectionEstimate,
        ),
        recovery_trials=_read_dataclasses(
            output / RECOVERY_NAME,
            validation.RecoveryTrial,
        ),
        sequential_authorization_trials=_read_dataclasses(
            output / SEQUENTIAL_NAME,
            validation.SequentialAuthorizationTrial,
        ),
        line_observations=_read_dataclasses(
            output / OBSERVATION_NAME,
            validation_cohort.LineObservation,
        ),
        locality_bandwidth=_read_dataclasses(
            output / LOCALITY_NAME,
            validation_cohort.LocalityBandwidth,
        ),
    )


def audit_derivatives(
    runs: tuple[Path, ...],
    *,
    source_root: Path,
    derivative_root: Path,
    manifest_path: Path,
    verification_path: Path,
) -> VerificationSummary:
    """Read source and derivative data and count exact sample preservation."""
    manifest = pd.read_csv(manifest_path, sep="\t", float_precision="round_trip")
    verification = pd.read_csv(
        verification_path,
        sep="\t",
        float_precision="round_trip",
    )
    unchanged_channel_counts = {}
    recording_samples_equal = {}
    for source_vhdr in runs:
        derivative_vhdr = recordings.derivative_vhdr_path(
            source_vhdr,
            source_root,
            derivative_root,
        )
        source = recordings.read_bids_raw(source_vhdr)
        derivative = recordings.read_bids_raw(derivative_vhdr)
        if source.ch_names != derivative.ch_names or source.n_times != derivative.n_times:
            raise ValueError("Source and derivative recording geometry differs.")
        matching_channels = np.all(
            source.get_data() == derivative.get_data(),
            axis=1,
        )
        unchanged_channel_counts[source_vhdr.stem] = int(matching_channels.sum())
        recording_samples_equal[source_vhdr.stem] = bool(matching_channels.all())
    return verification_summary(
        manifest,
        verification,
        unchanged_channel_counts=unchanged_channel_counts,
        recording_samples_equal=recording_samples_equal,
    )


def verification_summary(
    manifest: pd.DataFrame,
    verification: pd.DataFrame,
    *,
    unchanged_channel_counts: dict[str, int],
    recording_samples_equal: dict[str, bool],
) -> VerificationSummary:
    """Require complete exact verification and summarize its audit counts."""
    required_manifest = {"recording", "outcome"}
    required_verification = {"recording", "maximum_sample_deviation_v"}
    if not required_manifest <= set(manifest) or not required_verification <= set(
        verification
    ):
        raise ValueError("Manifest or verification table lacks required audit columns.")
    manifest_recordings = set(manifest["recording"].astype(str))
    verification_recordings = set(verification["recording"].astype(str))
    if manifest_recordings != verification_recordings:
        raise ValueError("Manifest and verification must cover the same recordings.")
    if manifest_recordings != set(unchanged_channel_counts) or manifest_recordings != set(
        recording_samples_equal
    ):
        raise ValueError("Sample comparisons must cover the same recordings.")

    exact_derivatives = {
        recording
        for recording, block in verification.groupby("recording", sort=False)
        if np.all(block["maximum_sample_deviation_v"].to_numpy(dtype=float) == 0.0)
    }
    null_recordings = {
        recording
        for recording, block in manifest.groupby("recording", sort=False)
        if not np.any(block["outcome"].astype(str) == "line_detected")
    }
    copied_nulls = {
        recording
        for recording in null_recordings
        if recording_samples_equal[recording]
    }
    return VerificationSummary(
        recordings_exactly_refitted=len(verification_recordings),
        unchanged_channels=sum(unchanged_channel_counts.values()),
        derivatives_reproduced_exactly=len(exact_derivatives),
        null_recordings_copied_unchanged=len(copied_nulls),
    )


def _dataclass_frame(values: tuple[object, ...]) -> pd.DataFrame:
    if not values:
        raise ValueError("Validation result tables must not be empty.")
    return pd.DataFrame([asdict(value) for value in values])


def _read_dataclasses(path: Path, data_class) -> tuple[object, ...]:
    table = pd.read_csv(path, sep="\t", float_precision="round_trip")
    expected_columns = tuple(field.name for field in fields(data_class))
    if tuple(table.columns) != expected_columns:
        raise ValueError(
            f"{path.name} columns {tuple(table.columns)!r} do not match "
            f"{expected_columns!r}."
        )
    return tuple(data_class(**row) for row in table.to_dict(orient="records"))


def _read_single_dataclass(path: Path, data_class):
    values = _read_dataclasses(path, data_class)
    if len(values) != 1:
        raise ValueError(f"{path.name} must contain exactly one validation estimate.")
    return values[0]
