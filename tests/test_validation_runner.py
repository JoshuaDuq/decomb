"""Cohort validation result persistence and sensitivity reporting."""

from __future__ import annotations

import pandas as pd
import pytest

from decomb import validation, validation_cohort, validation_runner


def _results() -> validation_runner.CohortValidationResults:
    false_trials = tuple(
        validation.FalseDetectionTrial(
            "run-1",
            "sub-1",
            "C0",
            method,
            detected=method == validation.MNE_SPECTRUM_FIT_10S,
        )
        for method in validation.ALL_METHODS
    )
    recovery_trials = tuple(
        validation.RecoveryTrial(
            recording="run-1",
            participant="sub-1",
            channel_name="C0",
            method=method,
            kind="stationary",
            frequency_hz=10.0,
            amplitude_v=1e-6,
            drift_hz=0.0,
            occupancy=1.0,
            injected_energy_v2=1.0,
            difference_energy_v2=0.5,
            artifact_to_background_db=2.0,
            remaining_fraction=0.2,
            collateral_fraction=0.01,
        )
        for method in validation.ALL_METHODS
    )
    observations = (
        validation_cohort.ArtifactObservation(
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
    return validation_runner.CohortValidationResults(
        false_trials,
        recovery_trials,
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
        "recovery_trials.tsv",
        "artifact_observations.tsv",
        "locality_bandwidth.tsv",
        "mne_window_sensitivity.tsv",
    }


def test_sensitivity_table_reports_both_mne_windows():
    table = validation_runner.mne_window_sensitivity(_results())

    assert isinstance(table, pd.DataFrame)
    assert set(table["window_s"]) == {10.0, 54.0}
    assert set(table["metric"]) == {
        "false_detection_proportion",
        "median_remaining_fraction",
        "median_collateral_fraction",
    }


def test_verification_summary_requires_and_counts_exact_recordings():
    manifest = pd.DataFrame(
        [
            {"recording": "line", "outcome": "artifact_detected"},
            {"recording": "line", "outcome": "no_artifact_detected"},
            {"recording": "null", "outcome": "no_artifact_detected"},
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
        [{"recording": "run-1", "outcome": "no_artifact_detected"}]
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
