"""Paired neural-like injection design and recovery measurements."""

from __future__ import annotations

from types import SimpleNamespace

import mne
import numpy as np
import pytest

from decomb import (
    injection,
    lines,
    neural_recovery_validation,
    notch,
    recovery,
    recovery_benchmark,
)


def test_frequency_placements_cover_exact_near_and_between_lines():
    targets = recovery_benchmark.RecoveryTargets(
        ordinary_frequencies_hz=(10.0, 12.0),
        scanner_frequencies_hz=(),
    )

    placements = neural_recovery_validation.frequency_placements(
        targets,
        band=("alpha", 8.0, 14.0),
        frequency_bin_width_hz=0.1,
    )

    assert tuple(item.position for item in placements) == (
        "exact",
        "near",
        "between",
    )
    assert all(item.authorized_line_hz == 10.0 for item in placements)
    assert placements[0].centre_frequency_hz == pytest.approx(10.0)
    assert placements[1].centre_frequency_hz == pytest.approx(10.2)
    assert placements[2].centre_frequency_hz == pytest.approx(11.0)


def test_frequency_placements_require_two_resolved_authorized_lines():
    targets = recovery_benchmark.RecoveryTargets((10.0,), ())

    with pytest.raises(ValueError, match="two resolved authorized lines"):
        neural_recovery_validation.frequency_placements(
            targets,
            band=("alpha", 8.0, 14.0),
            frequency_bin_width_hz=0.1,
        )


def test_injection_design_centres_each_signal_kind_on_its_placement():
    placement = neural_recovery_validation.FrequencyPlacement(
        band_name="gamma",
        band_low_hz=30.0,
        band_high_hz=45.0,
        position="near",
        authorized_line_hz=35.0,
        neighbouring_line_hz=36.0,
        centre_frequency_hz=35.2,
    )

    stationary = neural_recovery_validation.injection_target(
        placement,
        "stationary",
        frequency_bin_width_hz=0.1,
        component_to_background_db=-10.0,
    )
    drifting = neural_recovery_validation.injection_target(
        placement,
        "drifting",
        frequency_bin_width_hz=0.1,
        component_to_background_db=-10.0,
    )
    intermittent = neural_recovery_validation.injection_target(
        placement,
        "intermittent",
        frequency_bin_width_hz=0.1,
        component_to_background_db=-10.0,
    )
    phase_modulated = neural_recovery_validation.injection_target(
        placement,
        "phase_modulated",
        frequency_bin_width_hz=0.1,
        component_to_background_db=-10.0,
    )

    assert stationary.frequency_hz == pytest.approx(35.2)
    assert stationary.drift_hz == 0.0
    assert drifting.frequency_hz == pytest.approx(35.0)
    assert drifting.drift_hz == pytest.approx(0.4)
    assert intermittent.frequency_hz == pytest.approx(35.2)
    assert intermittent.occupancy == pytest.approx(0.5)
    assert phase_modulated.frequency_hz == pytest.approx(35.2)
    assert phase_modulated.phase_modulation_hz == pytest.approx(0.1)
    assert phase_modulated.phase_deviation_rad == pytest.approx(1.0)


def test_completed_trial_keys_require_both_stages():
    rows = [
        {
            "recording": "run",
            "band": "gamma",
            "kind": "stationary",
            "position": "exact",
            "stage": "recovery",
        },
        {
            "recording": "run",
            "band": "gamma",
            "kind": "stationary",
            "position": "exact",
            "stage": "final",
        },
    ]

    completed = neural_recovery_validation.completed_trial_keys(rows)

    assert completed == {("run", "gamma", "stationary", "exact")}
    with pytest.raises(ValueError, match="requires recovery and final stages"):
        neural_recovery_validation.completed_trial_keys(rows[:1])


def test_paired_trial_reports_identity_recovery_without_preservation_gate(monkeypatch):
    sampling_frequency_hz = 100.0
    data = np.random.default_rng(4).normal(scale=1e-6, size=(3, 4_000))
    raw = mne.io.RawArray(
        data,
        mne.create_info(["Cz", "C3", "C4"], sampling_frequency_hz, "eeg"),
        verbose="ERROR",
    )
    targets = recovery_benchmark.RecoveryTargets((10.0, 12.0), ())
    placement = neural_recovery_validation.frequency_placements(
        targets,
        ("alpha", 8.0, 14.0),
        frequency_bin_width_hz=0.1,
    )[0]
    target = injection.FactorialInjectionTarget(
        "stationary",
        placement.centre_frequency_hz,
        -10.0,
        phase_rad=np.pi / 7.0,
    )
    background_cleaning = neural_recovery_validation.MultitaperCleaningResult(
        recovered=raw.copy(),
        cleaned=raw.copy(),
        residual_filter_plans=(),
        residual_round_count=0,
        terminal_residual_detector_null=True,
        targeted_local_background_excess_null=True,
        recovery_runtime_s=0.0,
        residual_runtime_s=0.0,
    )

    def identity_cleaning(injected, *args, **kwargs):
        return neural_recovery_validation.MultitaperCleaningResult(
            recovered=injected.copy(),
            cleaned=injected.copy(),
            residual_filter_plans=(),
            residual_round_count=0,
            terminal_residual_detector_null=True,
            targeted_local_background_excess_null=True,
            recovery_runtime_s=0.0,
            residual_runtime_s=0.0,
        )

    monkeypatch.setattr(
        neural_recovery_validation,
        "clean_with_multitaper",
        identity_cleaning,
    )

    rows = neural_recovery_validation.paired_trial_rows(
        raw,
        background_cleaning,
        targets,
        placement,
        target,
        np.random.default_rng(5),
        recording="run",
        participant="sub-test",
        channel_name="Cz",
        notch_settings=object(),
        recovery_window_s=2.0,
    )

    assert tuple(row["stage"] for row in rows) == ("recovery", "final")
    for row in rows:
        assert row["residual_protocol"] == "adaptive"
        assert row["remaining_fraction"] == pytest.approx(1.0)
        assert row["component_error_fraction"] == pytest.approx(0.0, abs=1e-20)
        assert row["component_correlation"] == pytest.approx(1.0)
        assert row["band_power_ratio"] == pytest.approx(1.0)
        assert row["preservation_gate_passed"] == ""
    final = rows[-1]
    assert final["artifact_gate_passed"]
    assert not final["filter_geometry_changed"]


def _plan(frequency_hz: float) -> notch.HarmonicNotchPlan:
    stopband = notch.HarmonicStopband(
        (),
        frequency_hz - 0.1,
        frequency_hz + 0.1,
        "isolated",
    )
    return notch.HarmonicNotchPlan((stopband,), 0.2)


def test_frozen_residual_cleaning_replays_plans_in_order_and_reruns_gates(
    monkeypatch,
):
    raw = mne.io.RawArray(
        np.zeros((2, 400)),
        mne.create_info(["Cz", "C3"], 100.0, "eeg"),
        verbose="ERROR",
    )
    targets = recovery_benchmark.RecoveryTargets((10.0,), ())
    plans = (_plan(10.0), _plan(12.0))
    applied = []
    null_model = lines.LineModel((), 1, 2, 20)
    settings = SimpleNamespace(
        estimation_window_s=2.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(1.0, 20.0),
    )
    monkeypatch.setattr(
        recovery_benchmark,
        "recover_with_multitaper",
        lambda raw, targets, window_s, n_jobs=-1: raw.copy(),
    )

    def apply_plan(cleaned, plan, *, n_jobs=-1):
        applied.append(plan)
        result = cleaned.copy().load_data()
        result._data += len(applied)
        return result

    monkeypatch.setattr(notch, "apply_harmonic_notches", apply_plan)
    monkeypatch.setattr(
        notch,
        "fit_harmonic_round",
        lambda cleaned, settings: SimpleNamespace(
            model=null_model,
            scanner_harmonics=None,
        ),
    )
    monkeypatch.setattr(
        recovery_benchmark,
        "targeted_local_background_is_null",
        lambda *args, **kwargs: True,
    )

    result = neural_recovery_validation.clean_with_frozen_residuals(
        raw,
        targets,
        settings,
        plans,
        recovery_window_s=2.0,
    )

    assert applied == list(plans)
    np.testing.assert_allclose(result.cleaned.get_data(), 3.0)
    assert result.residual_filter_plans == plans
    assert result.residual_round_count == 2
    assert result.artifact_gate_passed


def test_frozen_residual_cleaning_surfaces_failed_artifact_gates(monkeypatch):
    raw = mne.io.RawArray(
        np.zeros((2, 400)),
        mne.create_info(["Cz", "C3"], 100.0, "eeg"),
        verbose="ERROR",
    )
    targets = recovery_benchmark.RecoveryTargets((10.0,), ())
    settings = SimpleNamespace(
        estimation_window_s=2.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(1.0, 20.0),
    )
    monkeypatch.setattr(
        recovery_benchmark,
        "recover_with_multitaper",
        lambda raw, targets, window_s, n_jobs=-1: raw.copy(),
    )
    monkeypatch.setattr(
        notch,
        "fit_harmonic_round",
        lambda cleaned, settings: SimpleNamespace(
            model=SimpleNamespace(channels=(object(),)),
            scanner_harmonics=None,
        ),
    )
    monkeypatch.setattr(
        recovery_benchmark,
        "targeted_local_background_is_null",
        lambda *args, **kwargs: False,
    )

    result = neural_recovery_validation.clean_with_frozen_residuals(
        raw,
        targets,
        settings,
        (),
        recovery_window_s=2.0,
    )

    assert not result.terminal_residual_detector_null
    assert not result.targeted_local_background_excess_null
    assert not result.artifact_gate_passed


def test_frozen_paired_trial_uses_background_geometry(monkeypatch):
    data = np.random.default_rng(12).normal(scale=1e-6, size=(3, 4_000))
    raw = mne.io.RawArray(
        data,
        mne.create_info(["Cz", "C3", "C4"], 100.0, "eeg"),
        verbose="ERROR",
    )
    targets = recovery_benchmark.RecoveryTargets((10.0, 12.0), ())
    placement = neural_recovery_validation.frequency_placements(
        targets,
        ("alpha", 8.0, 14.0),
        frequency_bin_width_hz=0.1,
    )[2]
    target = injection.FactorialInjectionTarget(
        "intermittent",
        placement.centre_frequency_hz,
        -10.0,
        occupancy=0.5,
    )
    plans = (_plan(10.0),)
    background_cleaning = neural_recovery_validation.MultitaperCleaningResult(
        recovered=raw.copy(),
        cleaned=raw.copy(),
        residual_filter_plans=plans,
        residual_round_count=1,
        terminal_residual_detector_null=True,
        targeted_local_background_excess_null=True,
        recovery_runtime_s=0.0,
        residual_runtime_s=0.0,
    )

    def frozen_identity(injected, targets, settings, residual_plans, **kwargs):
        assert residual_plans == plans
        return neural_recovery_validation.MultitaperCleaningResult(
            recovered=injected.copy(),
            cleaned=injected.copy(),
            residual_filter_plans=residual_plans,
            residual_round_count=len(residual_plans),
            terminal_residual_detector_null=True,
            targeted_local_background_excess_null=True,
            recovery_runtime_s=0.0,
            residual_runtime_s=0.0,
        )

    monkeypatch.setattr(
        neural_recovery_validation,
        "clean_with_frozen_residuals",
        frozen_identity,
    )

    rows = neural_recovery_validation.frozen_paired_trial_rows(
        raw,
        background_cleaning,
        targets,
        placement,
        target,
        np.random.default_rng(13),
        recording="run",
        participant="sub-test",
        channel_name="Cz",
        notch_settings=object(),
        recovery_window_s=2.0,
    )

    assert {row["residual_protocol"] for row in rows} == {"frozen"}
    assert not rows[-1]["filter_geometry_changed"]
    assert rows[-1]["artifact_gate_passed"]


def test_spatial_paired_trial_freezes_background_model_and_residual_geometry(
    monkeypatch,
):
    data = np.random.default_rng(14).normal(scale=1e-6, size=(3, 4_000))
    raw = mne.io.RawArray(
        data,
        mne.create_info(["Cz", "C3", "C4"], 100.0, "eeg"),
        verbose="ERROR",
    )
    targets = recovery_benchmark.RecoveryTargets((10.0, 12.0), ())
    placement = neural_recovery_validation.frequency_placements(
        targets,
        ("alpha", 8.0, 14.0),
        frequency_bin_width_hz=0.1,
    )[0]
    target = injection.FactorialInjectionTarget(
        "stationary",
        placement.centre_frequency_hz,
        -10.0,
    )
    plans = (_plan(10.0),)
    basis = np.array([[1.0], [0.0], [0.0]])
    spatial_model = recovery.SpatialLineSubspaceModel(
        (10.0, 12.0),
        basis,
    )
    background_cleaning = neural_recovery_validation.MultitaperCleaningResult(
        recovered=raw.copy(),
        cleaned=raw.copy(),
        residual_filter_plans=plans,
        residual_round_count=1,
        terminal_residual_detector_null=True,
        targeted_local_background_excess_null=True,
        recovery_runtime_s=0.0,
        residual_runtime_s=0.0,
    )

    def frozen_spatial_identity(
        injected,
        targets,
        settings,
        model,
        residual_plans,
        **kwargs,
    ):
        assert model is spatial_model
        assert residual_plans == plans
        return neural_recovery_validation.MultitaperCleaningResult(
            recovered=injected.copy(),
            cleaned=injected.copy(),
            residual_filter_plans=residual_plans,
            residual_round_count=len(residual_plans),
            terminal_residual_detector_null=True,
            targeted_local_background_excess_null=True,
            recovery_runtime_s=0.0,
            residual_runtime_s=0.0,
        )

    monkeypatch.setattr(
        neural_recovery_validation,
        "clean_with_frozen_spatial_residuals",
        frozen_spatial_identity,
    )

    rows = neural_recovery_validation.spatial_paired_trial_rows(
        raw,
        background_cleaning,
        targets,
        placement,
        target,
        np.random.default_rng(15),
        spatial_model=spatial_model,
        recording="run",
        participant="sub-test",
        channel_name="Cz",
        notch_settings=object(),
        recovery_window_s=2.0,
    )

    assert {row["candidate"] for row in rows} == {"spatial_ssp"}
    assert {row["residual_protocol"] for row in rows} == {"spatial"}
    assert {row["spatial_rank"] for row in rows} == {1}
    assert not rows[-1]["filter_geometry_changed"]


def test_residual_protocol_selects_one_explicit_trial_function():
    assert (
        neural_recovery_validation.trial_function("adaptive")
        is neural_recovery_validation.paired_trial_rows
    )
    assert (
        neural_recovery_validation.trial_function("frozen")
        is neural_recovery_validation.frozen_paired_trial_rows
    )
    assert (
        neural_recovery_validation.trial_function("spatial")
        is neural_recovery_validation.spatial_paired_trial_rows
    )
    with pytest.raises(ValueError, match="residual protocol"):
        neural_recovery_validation.trial_function("unknown")


def test_spatial_rank_is_required_only_for_spatial_protocol():
    assert neural_recovery_validation.validated_spatial_rank("spatial", 1) == 1

    with pytest.raises(ValueError, match="requires --spatial-rank"):
        neural_recovery_validation.validated_spatial_rank("spatial", None)
    with pytest.raises(ValueError, match="only valid"):
        neural_recovery_validation.validated_spatial_rank("frozen", 1)


def test_spatial_background_preparation_fits_model_before_cleaning(monkeypatch):
    raw = object()
    targets = recovery_benchmark.RecoveryTargets((10.0,), ())
    settings = SimpleNamespace(estimation_window_s=4.0)
    spatial_model = object()
    background_cleaning = object()

    def fit_model(received_raw, received_targets, *, window_s, rank, n_jobs=-1):
        assert received_raw is raw
        assert received_targets is targets
        assert window_s == 4.0
        assert rank == 2
        return spatial_model

    def clean_background(
        received_raw,
        received_targets,
        received_settings,
        received_model,
        *,
        recovery_window_s,
        n_jobs=-1,
    ):
        assert received_raw is raw
        assert received_targets is targets
        assert received_settings is settings
        assert received_model is spatial_model
        assert recovery_window_s == 4.0
        return background_cleaning

    monkeypatch.setattr(
        recovery_benchmark,
        "fit_spatial_line_subspace",
        fit_model,
    )
    monkeypatch.setattr(
        neural_recovery_validation,
        "clean_background_with_spatial_subspace",
        clean_background,
    )

    cleaning, paired_trial = neural_recovery_validation.prepare_background(
        raw,
        targets,
        settings,
        residual_protocol="spatial",
        spatial_rank=2,
    )

    assert cleaning is background_cleaning
    assert paired_trial.func is neural_recovery_validation.spatial_paired_trial_rows
    assert paired_trial.keywords == {"spatial_model": spatial_model}


def test_notch_only_protocol_dispatches_to_its_paired_trial():
    assert (
        neural_recovery_validation.trial_function("notch_only")
        is neural_recovery_validation.notch_only_paired_trial_rows
    )


def test_notch_only_rejects_a_spatial_rank():
    assert neural_recovery_validation.validated_spatial_rank("notch_only", None) is None
    with pytest.raises(ValueError, match="only valid for the spatial protocol"):
        neural_recovery_validation.validated_spatial_rank("notch_only", 1)


def test_notch_only_cleaning_subtracts_nothing_before_the_fir(monkeypatch):
    """The arm exists to be the published pipeline, so recovery must never run."""
    raw = mne.io.RawArray(
        np.zeros((2, 400)),
        mne.create_info(["Cz", "C3"], 100.0, "eeg"),
        verbose="ERROR",
    )
    targets = recovery_benchmark.RecoveryTargets((10.0,), ())
    plans = (_plan(10.0), _plan(12.0))
    null_model = lines.LineModel((), 1, 2, 20)
    settings = SimpleNamespace(
        estimation_window_s=2.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(1.0, 20.0),
    )

    def refuse(*args, **kwargs):
        raise AssertionError("notch-only cleaning must not run recovery")

    monkeypatch.setattr(recovery_benchmark, "recover_with_multitaper", refuse)
    monkeypatch.setattr(
        notch,
        "apply_harmonic_notches",
        lambda cleaned, plan, *, n_jobs=-1: cleaned.copy(),
    )
    monkeypatch.setattr(
        notch,
        "fit_harmonic_round",
        lambda cleaned, settings: SimpleNamespace(
            model=null_model,
            scanner_harmonics=None,
        ),
    )
    monkeypatch.setattr(
        recovery_benchmark,
        "targeted_local_background_is_null",
        lambda *args, **kwargs: True,
    )

    result = neural_recovery_validation.clean_frozen_residuals_without_recovery(
        raw,
        targets,
        settings,
        plans,
    )

    assert result.recovered is raw
    assert result.recovery_runtime_s == 0.0
    assert result.residual_filter_plans == plans


def test_residual_stage_lines_uses_the_published_pipeline(monkeypatch):
    """`lines` must run main's converged line rounds and no target-local rounds."""
    raw = mne.io.RawArray(
        np.zeros((2, 400)),
        mne.create_info(["Cz", "C3"], 100.0, "eeg"),
        verbose="ERROR",
    )
    targets = recovery_benchmark.RecoveryTargets((10.0,), ())
    settings = SimpleNamespace(
        estimation_window_s=2.0, familywise_error_rate=0.05,
        frequency_range_hz=(1.0, 20.0),
    )
    null_model = lines.LineModel((), 1, 2, 20)
    calls = []

    def refuse_joint(*args, **kwargs):
        raise AssertionError("the `lines` stage must not run joint residual cleaning")

    monkeypatch.setattr(recovery_benchmark, "clean_joint_residuals", refuse_joint)
    monkeypatch.setattr(
        notch, "clean_until_no_supported_lines",
        lambda cleaned, settings, n_jobs=-1: (
            calls.append(n_jobs)
            or SimpleNamespace(cleaned=cleaned, rounds=(),
                               residual_model=null_model,
                               residual_scanner_harmonics=None)
        ),
    )
    monkeypatch.setattr(
        recovery_benchmark, "targeted_local_background_is_null",
        lambda *args, **kwargs: False,
    )

    result = neural_recovery_validation.clean_without_recovery(
        raw, targets, settings, residual_stage="lines")

    assert calls == [-1]
    assert result.residual_stage == "lines"
    # the local-background null is measured but not contracted for under `lines`
    assert result.targeted_local_background_excess_null is False
    assert result.artifact_gate_passed is True


def test_joint_stage_still_requires_the_local_background_null():
    result = neural_recovery_validation.MultitaperCleaningResult(
        recovered=object(), cleaned=object(), residual_filter_plans=(),
        residual_round_count=0, terminal_residual_detector_null=True,
        targeted_local_background_excess_null=False,
        recovery_runtime_s=0.0, residual_runtime_s=0.0, residual_stage="joint",
    )
    assert result.artifact_gate_passed is False


def test_unknown_residual_stage_is_refused():
    with pytest.raises(ValueError, match="residual stage must be one of"):
        neural_recovery_validation._converged_residual_result(
            object(), 0.0, recovery_benchmark.RecoveryTargets((10.0,), ()),
            SimpleNamespace(), residual_stage="nonsense")
