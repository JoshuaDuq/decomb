"""Candidate orchestration before residual FIR notching."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
import pytest

from decomb import recovery, recovery_benchmark


def test_targets_use_only_first_authorized_round():
    rows = pd.DataFrame(
        {
            "recording": ["run", "run", "run", "run"],
            "removal_round": [1, 1, 2, 3],
            "outcome": [
                "line_detected",
                "scanner_harmonics_detected",
                "line_detected",
                "no_line_detected",
            ],
            "stopband_low_hz": [9.8, 19.8, 29.8, ""],
            "stopband_high_hz": [10.2, 20.2, 30.2, ""],
        }
    )

    targets = recovery_benchmark.targets_from_manifest(rows, "run")

    assert targets.ordinary_frequencies_hz == (10.0,)
    assert targets.scanner_frequencies_hz == (20.0,)
    assert targets.all_frequencies_hz == (10.0, 20.0)


def test_multitaper_candidate_changes_only_eeg_channels():
    sampling_frequency_hz = 100.0
    times_s = np.arange(1_200) / sampling_frequency_hz
    artifact = np.sin(2.0 * np.pi * 10.0 * times_s)
    data = np.vstack((artifact, -artifact, np.ones_like(artifact)))
    raw = mne.io.RawArray(
        data,
        mne.create_info(
            ["C3", "C4", "ECG"],
            sampling_frequency_hz,
            ["eeg", "eeg", "ecg"],
        ),
        verbose=False,
    )
    targets = recovery_benchmark.RecoveryTargets((10.0,), ())

    recovered = recovery_benchmark.recover_with_multitaper(
        raw,
        targets,
        window_s=4.0,
    )

    assert not np.array_equal(recovered.get_data(picks="eeg"), data[:2])
    np.testing.assert_array_equal(recovered.get_data(picks="ecg"), data[2:])


def test_trigger_and_trajectory_candidates_use_scanner_targets():
    sampling_frequency_hz = 100.0
    repetition_time_s = 1.0
    times_s = np.arange(4_000) / sampling_frequency_hz
    artifact = np.sin(2.0 * np.pi * 20.0 * times_s)
    raw = mne.io.RawArray(
        artifact[np.newaxis, :],
        mne.create_info(["Cz"], sampling_frequency_hz, "eeg"),
        verbose=False,
    )
    raw.set_annotations(
        mne.Annotations(
            onset=np.arange(40) * repetition_time_s,
            duration=0.0,
            description="Volume/V  1",
        )
    )
    targets = recovery_benchmark.RecoveryTargets((), (20.0,))

    trigger_recovered = recovery_benchmark.recover_with_trigger_basis(
        raw,
        targets,
        repetition_time_s=repetition_time_s,
        trigger_event_name="Volume/V  1",
        maximum_component_count=2,
        ordinary_window_s=4.0,
    )
    trajectory_recovered = recovery_benchmark.recover_with_trajectory_pca(
        raw,
        targets,
        recovery_settings=recovery.TrajectoryPCASettings(segment_s=2.0),
        ordinary_window_s=4.0,
    )

    assert np.linalg.norm(trigger_recovered.get_data()) < 0.1 * np.linalg.norm(
        raw.get_data()
    )
    assert np.linalg.norm(trajectory_recovered.get_data()) < 0.1 * np.linalg.norm(
        raw.get_data()
    )


def test_targeted_local_background_gate_detects_only_residual_excess():
    sampling_frequency_hz = 100.0
    times_s = np.arange(4_000) / sampling_frequency_hz
    line = np.sin(2.0 * np.pi * 20.0 * times_s)
    contaminated = mne.io.RawArray(
        np.vstack((line, -line)),
        mne.create_info(["C3", "C4"], sampling_frequency_hz, "eeg"),
        verbose=False,
    )
    cleaned = mne.io.RawArray(
        np.zeros((2, times_s.size)),
        contaminated.info.copy(),
        verbose=False,
    )
    targets = recovery_benchmark.RecoveryTargets((20.0,), ())

    assert not recovery_benchmark.targeted_local_background_is_null(
        contaminated,
        targets,
        window_s=2.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(0.0, 40.0),
    )
    assert recovery_benchmark.targeted_local_background_is_null(
        cleaned,
        targets,
        window_s=2.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(0.0, 40.0),
    )


def test_checkpoint_requires_both_stages_and_every_band():
    bands = (("theta", 4.0, 7.9), ("alpha", 8.0, 12.9))
    complete = pd.DataFrame(
        {
            "recording": ["run"] * 4,
            "candidate": ["multitaper"] * 4,
            "stage": ["recovery", "recovery", "final", "final"],
            "band": ["theta", "alpha", "theta", "alpha"],
        }
    )

    assert recovery_benchmark._completed_candidate_keys(complete, bands) == {
        ("run", "multitaper")
    }

    with pytest.raises(ValueError, match="incomplete final bands"):
        recovery_benchmark._completed_candidate_keys(complete.iloc[:-1], bands)
