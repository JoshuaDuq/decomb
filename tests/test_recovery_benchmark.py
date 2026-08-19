"""Candidate orchestration before residual FIR notching."""

from __future__ import annotations

from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd
import pytest

from decomb import recordings, recovery, recovery_benchmark


def test_joint_residual_filter_plans_preserve_actual_interleaved_order():
    harmonic_results = (
        SimpleNamespace(
            rounds=(
                SimpleNamespace(filter_plan="harmonic-1"),
                SimpleNamespace(filter_plan="harmonic-2"),
            )
        ),
        SimpleNamespace(
            rounds=(SimpleNamespace(filter_plan="harmonic-3"),)
        ),
    )
    result = recovery_benchmark.JointResidualCleaningResult(
        cleaned=object(),
        harmonic_results=harmonic_results,
        local_background_plans=("local-1",),
        terminal_local_background=SimpleNamespace(detections=()),
    )

    assert result.filter_plans == (
        "harmonic-1",
        "harmonic-2",
        "local-1",
        "harmonic-3",
    )


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


def test_spatial_candidate_freezes_background_model_and_changes_only_eeg():
    sampling_frequency_hz = 100.0
    times_s = np.arange(2_000) / sampling_frequency_hz
    artifact = np.sin(2.0 * np.pi * 10.0 * times_s)
    data = np.vstack((artifact, 0.5 * artifact, np.ones_like(artifact)))
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

    model = recovery_benchmark.fit_spatial_line_subspace(
        raw,
        targets,
        window_s=4.0,
        rank=1,
    )
    recovered = recovery_benchmark.recover_with_spatial_line_subspace(
        raw,
        model,
        window_s=4.0,
    )

    assert model.frequencies_hz == (10.0,)
    assert model.channel_count == 2
    assert np.linalg.norm(recovered.get_data(picks="eeg")) < 0.01 * np.linalg.norm(
        data[:2]
    )
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


def _null_harmonic_result(rounds=()):
    return SimpleNamespace(
        rounds=rounds,
        residual_model=SimpleNamespace(channels=()),
        residual_scanner_harmonics=None,
    )


def test_artifact_gate_reports_a_surviving_target_local_excess():
    result = recovery_benchmark.JointResidualCleaningResult(
        cleaned=object(),
        harmonic_results=(_null_harmonic_result(),),
        local_background_plans=(),
        terminal_local_background=SimpleNamespace(detections=("60 Hz",)),
    )

    assert result.terminal_residual_detector_null
    assert not result.targeted_local_background_excess_null
    assert not result.artifact_gate_passed


def test_artifact_gate_reports_a_surviving_coherent_line():
    surviving = SimpleNamespace(
        rounds=(),
        residual_model=SimpleNamespace(channels=("Cz",)),
        residual_scanner_harmonics=None,
    )
    result = recovery_benchmark.JointResidualCleaningResult(
        cleaned=object(),
        harmonic_results=(surviving,),
        local_background_plans=(),
        terminal_local_background=SimpleNamespace(detections=()),
    )

    assert not result.terminal_residual_detector_null
    assert result.targeted_local_background_excess_null
    assert not result.artifact_gate_passed


def test_artifact_gate_passes_only_on_both_terminal_nulls():
    result = recovery_benchmark.JointResidualCleaningResult(
        cleaned=object(),
        harmonic_results=(_null_harmonic_result(),),
        local_background_plans=(),
        terminal_local_background=SimpleNamespace(detections=()),
    )

    assert result.artifact_gate_passed


def test_trajectory_recovery_is_not_gated_on_scanner_targets(monkeypatch):
    """rsPCA takes no frequency list, so no target class may withhold it.

    It cannot remove the trigger-anchored comb -- comb teeth are not isolated single
    peaks -- so gating it on the comb targets skipped it entirely on a recording that
    had lines and no comb, which is the case it is actually for.
    """
    mne = pytest.importorskip("mne")
    mne.set_log_level("ERROR")
    rng = np.random.default_rng(3)
    raw = mne.io.RawArray(
        rng.normal(scale=1e-5, size=(2, 2_000)),
        mne.create_info(["Fp1", "Cz"], 1_000.0, "eeg"),
    )
    targets = recovery_benchmark.RecoveryTargets((57.2,), ())
    calls = []

    def trajectory(values, sampling_frequency_hz, settings, *, n_jobs=1):
        calls.append(settings.segment_s)
        return recovery.SignalRecoveryResult(values, np.zeros_like(values), ())

    monkeypatch.setattr(recovery, "subtract_recursive_trajectory_pca", trajectory)
    monkeypatch.setattr(
        recovery,
        "subtract_multitaper_sinusoids",
        lambda values, sfreq, freqs, *, window_s, n_jobs=1: (
            recovery.SignalRecoveryResult(values, np.zeros_like(values), tuple(freqs))
        ),
    )

    recovery_benchmark.recover_with_trajectory_pca(
        raw,
        targets,
        recovery_settings=recovery.TrajectoryPCASettings(segment_s=0.30),
        ordinary_window_s=1.0,
    )

    assert calls == [0.30]


def test_manifest_round_trip_preserves_target_frequencies(tmp_path):
    """A manifest read back must yield the frequencies that were written.

    `write_tsv_atomic` emits `%.17g`, which pandas' default float parser can land one
    ULP away from. That split single lines into pairs about 1e-14 Hz apart, inflating
    the target list, so manifest reads pass `float_precision="round_trip"`.
    """
    frame = pd.DataFrame(
        {
            "recording": ["run", "run"],
            "removal_round": [1, 1],
            "outcome": ["line_detected", "line_detected"],
            "stopband_low_hz": [72.97500000000001, 52.575000000000003],
            "stopband_high_hz": [73.425, 53.025000000000006],
        }
    )
    path = tmp_path / "manifest.tsv"
    recordings.write_tsv_atomic(frame, path)

    reread = pd.read_csv(
        path,
        sep="\t",
        keep_default_na=False,
        float_precision="round_trip",
    )

    assert (
        recovery_benchmark.targets_from_manifest(reread, "run").all_frequencies_hz
        == recovery_benchmark.targets_from_manifest(frame, "run").all_frequencies_hz
    )
