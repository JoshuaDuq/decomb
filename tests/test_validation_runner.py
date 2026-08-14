"""Cohort validation result persistence and sensitivity reporting."""

from __future__ import annotations

import pandas as pd
import pytest

from decomb import validation, validation_cohort, validation_runner


def _results() -> validation_runner.CohortValidationResults:
    false_trials = (
        validation.FalseDetectionTrial(
            "run-1",
            "sub-1",
            "C0",
            line_detected=False,
        ),
    )
    recovery_trials = (
        validation.RecoveryTrial(
            recording="run-1",
            participant="sub-1",
            channel_name="C0",
            kind="stationary",
            frequency_hz=10.0,
            amplitude_v=1e-6,
            drift_hz=0.0,
            occupancy=1.0,
            phase_rad=0.0,
            injected_energy_v2=1.0,
            difference_energy_v2=0.5,
            component_to_background_db=2.0,
            remaining_fraction=0.2,
            collateral_fraction=0.01,
        ),
    )
    observations = (
        validation_cohort.LineObservation(
            "run-1",
            "sub-1",
            "C0",
            10.0,
            0.1,
            0.5,
            1e-6,
        ),
    )
    locality = (validation_cohort.LocalityBandwidth("run-1", 1.0, 2.0),)
    sequential = (
        validation.SequentialAuthorizationTrial(
            recording="run-1",
            participant="sub-1",
            kind="stationary",
            frequency_hz=10.0,
            component_to_background_db=-10.0,
            drift_hz=0.0,
            occupancy=1.0,
            phase_rad=0.0,
            injected_line_authorized=True,
            unsupported_line_authorized=False,
            removal_round_count=1,
        ),
    )
    detection_estimate = validation_cohort.DetectionEstimate(
        recording_false_authorization_proportion=0.0,
        lower=0.0,
        upper=0.0,
        participant_count=1,
        recording_count=1,
        channel_false_detection_proportion=0.0,
        channel_recording_count=1,
    )
    return validation_runner.CohortValidationResults(
        false_trials,
        detection_estimate,
        recovery_trials,
        sequential,
        observations,
        locality,
    )


def test_validation_results_round_trip_without_schema_loss(tmp_path):
    expected = _results()

    validation_runner.write_results(expected, tmp_path)
    actual = validation_runner.read_results(tmp_path)

    assert actual == expected
    assert {path.name for path in tmp_path.iterdir()} == {
        "false_detection_trials.tsv",
        "false_authorization_estimate.tsv",
        "recovery_trials.tsv",
        "sequential_authorization_trials.tsv",
        "line_observations.tsv",
        "locality_bandwidth.tsv",
    }


def test_verification_summary_requires_and_counts_exact_recordings():
    manifest = pd.DataFrame(
        [
            {"recording": "line", "outcome": "line_detected"},
            {"recording": "line", "outcome": "no_line_detected"},
            {"recording": "null", "outcome": "no_line_detected"},
        ]
    )
    verification = pd.DataFrame(
        [
            {"recording": "line", "maximum_sample_deviation_v": 0.0},
            {"recording": "null", "maximum_sample_deviation_v": 0.0},
        ]
    )

    summary = validation_runner.verification_summary(
        manifest,
        verification,
        unchanged_channel_counts={"line": 2, "null": 4},
        recording_samples_equal={"line": False, "null": True},
    )

    assert summary.recordings_exactly_refitted == 2
    assert summary.unchanged_channels == 6
    assert summary.derivatives_reproduced_exactly == 2
    assert summary.null_recordings_copied_unchanged == 1


def test_verification_summary_rejects_incomplete_recording_coverage():
    manifest = pd.DataFrame(
        [{"recording": "run-1", "outcome": "no_line_detected"}]
    )
    verification = pd.DataFrame(
        [{"recording": "run-2", "maximum_sample_deviation_v": 0.0}]
    )

    with pytest.raises(ValueError, match="same recordings"):
        validation_runner.verification_summary(
            manifest,
            verification,
            unchanged_channel_counts={"run-1": 1},
            recording_samples_equal={"run-1": True},
        )
