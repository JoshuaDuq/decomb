"""Cohort sampling, uncertainty, and recording-local bandwidth summaries."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from decomb import lines, notch, validation, validation_cohort


def test_observed_lines_measure_frequency_drift_occupancy_and_amplitude():
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
    model = lines.LineModel(
        channels=(
            lines.ChannelLineModel(
                channel_index=0,
                channel_name="C0",
                lines=(
                    lines.SupportedLine(10.0, 1e-9, 1e-6, (0, 1), None),
                    lines.SupportedLine(10.05, 1e-9, 1e-6, (2,), None),
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

    observations = validation_cohort.observed_lines(
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


def test_factorial_injection_targets_cover_fixed_physical_design():
    targets = validation_cohort.factorial_injection_targets(
        frequency_range_hz=(1.0, 30.0),
    )

    assert len(targets) == 90
    assert {target.frequency_hz for target in targets} == {8.25, 15.5, 22.75}
    assert {target.component_to_background_db for target in targets} == {-20.0, -10.0, 0.0}
    assert {target.phase_rad for target in targets} == {0.0, np.pi / 2.0}
    assert sorted(
        {
            abs(target.drift_hz)
            for target in targets
            if target.kind == "drifting"
        }
    ) == pytest.approx([0.05, 0.2])
    assert {
        target.occupancy
        for target in targets
        if target.kind == "intermittent"
    } == {0.25, 0.75}


def test_factorial_targets_must_fit_every_cohort_nyquist_limit():
    targets = validation_cohort.factorial_injection_targets(
        frequency_range_hz=(1.0, 100.0),
    )

    with pytest.raises(ValueError, match="lowest cohort Nyquist frequency"):
        validation_cohort.validate_factorial_targets_for_cohort(
            targets,
            sampling_frequencies_hz=(250.0, 128.0),
        )


def test_factorial_targets_accept_a_cohort_with_sufficient_sampling_rates():
    targets = validation_cohort.factorial_injection_targets(
        frequency_range_hz=(1.0, 100.0),
    )

    validation_cohort.validate_factorial_targets_for_cohort(
        targets,
        sampling_frequencies_hz=(250.0, 200.0),
    )


def test_detection_estimate_resamples_participants_as_clusters():
    trials = tuple(
        validation.FalseDetectionTrial(
            recording=f"{participant}-run-1",
            participant=participant,
            channel_name=channel,
            line_detected=participant == "sub-2" and channel == "C0",
        )
        for participant in ("sub-1", "sub-2")
        for channel in ("C0", "C1")
    )

    estimate = validation_cohort.detection_estimate(
        trials,
        bootstrap_resamples=1_000,
        rng=np.random.default_rng(1),
    )

    assert not hasattr(estimate, "method")
    assert estimate.recording_false_authorization_proportion == pytest.approx(0.5)
    assert estimate.lower <= estimate.recording_false_authorization_proportion <= estimate.upper
    assert estimate.channel_false_detection_proportion == pytest.approx(0.25)
    assert estimate.participant_count == 2
    assert estimate.recording_count == 2
    assert estimate.channel_recording_count == 4


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
