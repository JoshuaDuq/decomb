"""Cohort sampling, uncertainty, and recording-local bandwidth summaries."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from decomb import lines, notch, validation, validation_cohort


def _observation(
    index: int,
    *,
    drift_hz: float,
    occupancy: float,
) -> validation_cohort.ArtifactObservation:
    return validation_cohort.ArtifactObservation(
        recording=f"run-{index}",
        participant=f"sub-{index}",
        channel_name="C0",
        frequency_hz=10.0 + index,
        drift_hz=drift_hz,
        occupancy=occupancy,
        amplitude_v=(index + 1) * 1e-6,
    )


def test_observed_artifacts_measure_frequency_drift_occupancy_and_amplitude():
    sampling_frequency_hz = 100.0
    times_s = np.arange(12_000) / sampling_frequency_hz
    data = np.stack(
        [
            4e-6 * np.sin(2.0 * np.pi * 10.0 * times_s),
            np.zeros(times_s.size),
        ]
    )
    raw = mne.io.RawArray(
        data,
        mne.create_info(["C0", "C1"], sampling_frequency_hz, "eeg"),
        verbose="ERROR",
    )
    model = lines.ArtifactModel(
        channels=(
            lines.ChannelArtifactModel(
                channel_index=0,
                channel_name="C0",
                lines=(
                    lines.ArtifactLine(10.0, 1e-9, 1e-6, (0, 1), None),
                    lines.ArtifactLine(10.05, 1e-9, 1e-6, (2,), None),
                ),
                fundamental_hz=None,
                comb_corrected_p_value=None,
            ),
        ),
        window_count=3,
        channel_count=2,
        test_count_per_channel=1_000,
    )
    settings = notch.HarmonicNotchSettings(54.0, 0.05, (1.0, 20.0))

    observations = validation_cohort.observed_artifacts(
        raw,
        settings,
        model,
        recording_name="run-1",
        participant="sub-1",
    )

    assert len(observations) == 1
    observation = observations[0]
    assert observation.frequency_hz == pytest.approx(10.025)
    assert observation.drift_hz == pytest.approx(0.05)
    assert observation.occupancy == pytest.approx(1.0)
    assert observation.amplitude_v > 0.0


def test_injection_targets_are_balanced_and_span_empirical_parameters():
    observations = tuple(
        _observation(index, drift_hz=0.1 + index / 10.0, occupancy=0.2 + index / 20.0)
        for index in range(6)
    )

    targets = validation_cohort.sample_injection_targets(
        observations,
        count=6,
        frequency_range_hz=(1.0, 30.0),
        rng=np.random.default_rng(0),
    )

    assert [target.kind for target in targets].count("stationary") == 2
    assert [target.kind for target in targets].count("drifting") == 2
    assert [target.kind for target in targets].count("intermittent") == 2
    assert all(target.amplitude_v > 0.0 for target in targets)
    assert all(target.drift_hz != 0.0 for target in targets if target.kind == "drifting")
    assert all(
        0.0 < target.occupancy < 1.0
        for target in targets
        if target.kind == "intermittent"
    )


def test_detection_estimates_resample_participants_as_clusters():
    trials = tuple(
        validation.FalseDetectionTrial(
            recording=f"{participant}-run-{run}",
            participant=participant,
            channel_name="C0",
            method=validation.DECOMB_HOLM,
            detected=participant == "sub-2",
        )
        for participant in ("sub-1", "sub-2")
        for run in range(3)
    )

    estimates = validation_cohort.detection_estimates(
        trials,
        methods=(validation.DECOMB_HOLM,),
        bootstrap_resamples=1_000,
        rng=np.random.default_rng(1),
    )

    assert len(estimates) == 1
    estimate = estimates[0]
    assert estimate.proportion == pytest.approx(0.5)
    assert estimate.lower <= estimate.proportion <= estimate.upper
    assert estimate.participant_count == 2


def test_locality_bandwidth_compares_recording_and_cohort_frequency_unions():
    first = notch.HarmonicNotchPlan(
        (notch.HarmonicStopband((), 9.9, 10.1, "isolated"),),
        transition_bandwidth_hz=0.2,
    )
    second = notch.HarmonicNotchPlan(
        (notch.HarmonicStopband((), 19.9, 20.1, "isolated"),),
        transition_bandwidth_hz=0.2,
    )
    recordings = (
        validation_cohort.RecordingPlan("run-1", 2, first),
        validation_cohort.RecordingPlan("run-2", 3, second),
    )

    bandwidth = validation_cohort.locality_bandwidth(recordings)

    by_recording = {row.recording: row for row in bandwidth}
    assert by_recording["run-1"].recording_local_channel_hz == pytest.approx(0.8)
    assert by_recording["run-1"].cohort_global_channel_hz == pytest.approx(1.6)
    assert by_recording["run-2"].recording_local_channel_hz == pytest.approx(1.2)
    assert by_recording["run-2"].cohort_global_channel_hz == pytest.approx(2.4)


def test_null_recording_has_zero_local_cost_but_nonzero_cohort_global_cost():
    plan = notch.HarmonicNotchPlan(
        (notch.HarmonicStopband((), 9.9, 10.1, "isolated"),),
        transition_bandwidth_hz=0.2,
    )
    recordings = (
        validation_cohort.RecordingPlan("null", 2, None),
        validation_cohort.RecordingPlan("line", 2, plan),
    )

    row = next(
        row
        for row in validation_cohort.locality_bandwidth(recordings)
        if row.recording == "null"
    )

    assert row.recording_local_channel_hz == 0.0
    assert row.cohort_global_channel_hz > 0.0
