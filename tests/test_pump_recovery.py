"""Tests for target-blind recovery of the 1.2 Hz pump fundamental."""

import numpy as np
import pytest

from decomb import pump_recovery, pump_recovery_experiment


def test_high_harmonics_are_consecutive_teeth_between_20_and_95_hz():
    assert pump_recovery.high_harmonic_numbers(1.2, 500.0) == tuple(range(17, 80))


def test_adjacent_features_do_not_read_the_exact_fundamental():
    sampling_frequency_hz = 1000.0
    times_s = np.arange(40_000) / sampling_frequency_hz
    background = np.stack(
        [
            np.sin(2.0 * np.pi * 20.4 * times_s)
            + 0.6 * np.sin(2.0 * np.pi * 21.6 * times_s + 0.2),
            np.cos(2.0 * np.pi * 20.4 * times_s)
            + 0.8 * np.cos(2.0 * np.pi * 21.6 * times_s - 0.3),
        ]
    )
    injection = 7.0 * np.sin(2.0 * np.pi * 1.2 * times_s + 0.4)
    bounds = ((0, 20_000), (20_000, 40_000))

    original = pump_recovery.extract_adjacent_features(
        background,
        sampling_frequency_hz,
        bounds,
        fundamental_hz=1.2,
    )
    injected = pump_recovery.extract_adjacent_features(
        background + injection,
        sampling_frequency_hz,
        bounds,
        fundamental_hz=1.2,
    )

    np.testing.assert_allclose(injected, original, rtol=0.0, atol=1e-12)


def test_reduced_rank_model_predicts_a_held_out_fundamental():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(120, 6)) + 1j * rng.normal(size=(120, 6))
    mapping = rng.normal(size=(6, 3)) + 1j * rng.normal(size=(6, 3))
    targets = features @ mapping

    model = pump_recovery.fit_complex_model(
        features[:90],
        targets[:90],
        ("Fz", "Cz", "Pz"),
        rank=6,
        penalty=1e-8,
    )
    predicted = model.predict(features[90:], ("Fz", "Cz", "Pz"))

    np.testing.assert_allclose(predicted, targets[90:], rtol=1e-6, atol=1e-8)


def test_model_refuses_reordered_channels():
    features = np.eye(4, dtype=complex)
    targets = np.ones((4, 2), dtype=complex)
    model = pump_recovery.fit_complex_model(
        features,
        targets,
        ("Fz", "Cz"),
        rank=2,
        penalty=1.0,
    )

    with pytest.raises(ValueError, match="channel names"):
        model.predict(features, ("Cz", "Fz"))


def test_model_selection_holds_out_every_training_participant():
    rng = np.random.default_rng(5)
    features = rng.normal(size=(60, 4)) + 1j * rng.normal(size=(60, 4))
    mapping = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    targets = features @ mapping
    participants = np.repeat(("sub-01", "sub-02", "sub-03"), 20)

    selected = pump_recovery.select_model(
        features,
        targets,
        participants,
        ("Fz", "Cz"),
        ranks=(2, 4),
        penalties=(1e-8, 1.0),
    )
    repeated = pump_recovery.select_model(
        features,
        targets,
        participants,
        ("Fz", "Cz"),
        ranks=(2, 4),
        penalties=(1e-8, 1.0),
    )

    assert selected.validation_participants == ("sub-01", "sub-02", "sub-03")
    assert selected.rank == repeated.rank
    assert selected.penalty == repeated.penalty
    assert selected.validation_error == repeated.validation_error


def test_pump_lock_test_separates_phase_locked_and_balanced_null_coefficients():
    rng = np.random.default_rng(8)
    phases = rng.uniform(-np.pi, np.pi, size=(80, 4))
    prediction = np.exp(1j * phases)
    balanced_phases = np.exp(2j * np.pi * np.arange(80) / 80.0)
    null = prediction * balanced_phases[:, None]

    locked_result = pump_recovery.pump_lock_test(
        prediction,
        prediction,
        surrogate_count=999,
        seed=10,
    )
    null_result = pump_recovery.pump_lock_test(
        null,
        prediction,
        surrogate_count=999,
        seed=10,
    )

    assert locked_result.p_value <= 0.01
    assert null_result.p_value > 0.01


def test_overlap_add_reconstructs_one_fundamental_without_boundary_holes():
    sampling_frequency_hz = 1000.0
    bounds = ((0, 20_000), (10_000, 30_000), (20_000, 40_000))
    coefficient = 2.0 * np.exp(1j * 0.3)
    coefficients = np.full((3, 2), coefficient)

    artifact = pump_recovery.reconstruct_artifact(
        40_000,
        bounds,
        coefficients,
        sampling_frequency_hz,
        1.2,
    )
    times_s = np.arange(40_000) / sampling_frequency_hz
    expected = np.real(coefficient * np.exp(2j * np.pi * 1.2 * times_s))

    np.testing.assert_allclose(artifact[0], expected, rtol=0.0, atol=1e-10)


def test_target_blind_cleaner_preserves_an_exact_frequency_injection():
    sampling_frequency_hz = 1000.0
    times_s = np.arange(40_000) / sampling_frequency_hz
    bounds = ((0, 20_000), (10_000, 30_000), (20_000, 40_000))
    background = np.vstack(
        [
            np.sin(2.0 * np.pi * 20.4 * times_s),
            np.cos(2.0 * np.pi * 21.6 * times_s),
        ]
    )
    injection = np.vstack(
        [
            0.2 * np.sin(2.0 * np.pi * 1.2 * times_s),
            0.3 * np.cos(2.0 * np.pi * 1.2 * times_s),
        ]
    )
    predicted = np.full((len(bounds), 2), 0.4 + 0.2j)

    clean_background = pump_recovery.subtract_predicted_artifact(
        background,
        bounds,
        predicted,
        sampling_frequency_hz,
        1.2,
    )
    clean_injected = pump_recovery.subtract_predicted_artifact(
        background + injection,
        bounds,
        predicted,
        sampling_frequency_hz,
        1.2,
    )

    np.testing.assert_allclose(
        clean_injected - clean_background,
        injection,
        rtol=0.0,
        atol=1e-12,
    )


def test_injection_preservation_requires_waveform_amplitude_and_phase_fidelity():
    sampling_frequency_hz = 1000.0
    times_s = np.arange(20_000) / sampling_frequency_hz
    injection = np.vstack(
        [
            np.sin(2.0 * np.pi * 1.2 * times_s),
            np.cos(2.0 * np.pi * 1.2 * times_s),
        ]
    )

    exact = pump_recovery.measure_injection_preservation(
        injection,
        injection,
        sampling_frequency_hz,
        1.2,
    )
    attenuated = pump_recovery.measure_injection_preservation(
        injection,
        0.98 * injection,
        sampling_frequency_hz,
        1.2,
    )

    assert exact.passes
    assert exact.relative_waveform_error == pytest.approx(0.0)
    assert exact.amplitude_retention == pytest.approx(1.0)
    assert exact.phase_error_degrees == pytest.approx(0.0)
    assert not attenuated.passes


def test_injection_preservation_rejects_zero_energy_reference():
    with pytest.raises(ValueError, match="positive energy"):
        pump_recovery.measure_injection_preservation(
            np.zeros((2, 100)),
            np.zeros((2, 100)),
            100.0,
            1.2,
        )


def test_leave_one_participant_out_never_trains_on_the_test_participant():
    rng = np.random.default_rng(14)
    mapping = np.array([[1.2 + 0.3j], [-0.4 + 0.8j]])
    recordings = []
    expected = {}
    for participant in ("sub-01", "sub-02", "sub-03"):
        for run in (1, 2):
            features = rng.normal(size=(12, 2)) + 1j * rng.normal(size=(12, 2))
            targets = features @ mapping
            recording = f"{participant}_run-{run}"
            expected[recording] = targets
            recordings.append(
                pump_recovery.RecoveryRecording(
                    recording,
                    participant,
                    ("Cz",),
                    features,
                    targets,
                )
            )

    predictions = pump_recovery.leave_one_participant_out(
        recordings,
        ranks=(2,),
        penalties=(1e-8,),
    )

    assert {result.test_participant for result in predictions} == {
        "sub-01",
        "sub-02",
        "sub-03",
    }
    for result in predictions:
        assert result.test_participant not in result.training_participants
        np.testing.assert_allclose(
            result.predicted_coefficients,
            expected[result.recording],
            rtol=1e-6,
            atol=1e-8,
        )


def test_joint_extraction_keeps_target_out_of_predictors():
    sampling_frequency_hz = 1000.0
    times_s = np.arange(20_000) / sampling_frequency_hz
    data = np.vstack(
        [
            np.sin(2.0 * np.pi * 20.4 * times_s)
            + np.sin(2.0 * np.pi * 21.6 * times_s),
            np.cos(2.0 * np.pi * 20.4 * times_s)
            + np.cos(2.0 * np.pi * 21.6 * times_s),
        ]
    )
    injection = np.sin(2.0 * np.pi * 1.2 * times_s)

    original = pump_recovery.extract_recovery_coefficients(
        data,
        sampling_frequency_hz,
        ((0, 20_000),),
        (0, 1),
        fundamental_hz=1.2,
    )
    injected = pump_recovery.extract_recovery_coefficients(
        data + injection,
        sampling_frequency_hz,
        ((0, 20_000),),
        (0, 1),
        fundamental_hz=1.2,
    )

    np.testing.assert_allclose(injected.features, original.features, atol=1e-12)
    assert not np.allclose(
        injected.target_coefficients,
        original.target_coefficients,
    )


def test_pump_clock_reference_recovers_common_high_harmonic_phase():
    phase = np.linspace(0.0, 3.0 * np.pi, 80)
    clock = np.exp(1j * phase)
    pair_weights = np.array([0.3 + 0.2j, -0.6 + 0.5j, 1.1 - 0.4j])
    features = clock[:, None] * pair_weights

    recovered = pump_recovery.pump_clock_reference(features)
    coherence = np.abs(np.vdot(clock, recovered)) ** 2 / (
        np.vdot(clock, clock).real * np.vdot(recovered, recovered).real
    )

    assert coherence == pytest.approx(1.0)


def test_prediction_evaluation_uses_high_only_clock_for_presence_and_residual():
    window_count = 80
    clock = np.exp(1j * np.linspace(0.0, 4.0 * np.pi, window_count))
    features = clock[:, None] * np.array([0.5 + 0.2j, -0.3 + 0.8j])
    artifact = clock[:, None] * np.array([1.0 + 0.4j, -0.6 + 0.2j])
    balanced = np.exp(2j * np.pi * np.arange(window_count) / window_count)
    neural = 0.1 * artifact * balanced[:, None]
    prediction = pump_recovery.HeldOutPrediction(
        "sub-01_run-1",
        "sub-01",
        ("sub-02", "sub-03"),
        artifact + neural,
        artifact,
        2,
        1e-6,
        0.01,
    )

    result = pump_recovery_experiment.evaluate_prediction(
        prediction,
        features,
        surrogate_count=999,
    )

    assert result.source_p_value <= 0.01
    assert result.residual_p_value > 0.01
    assert result.normalized_prediction_error < 0.02


def test_experiment_seed_is_stable_and_recording_specific():
    first = pump_recovery_experiment.stable_seed("sub-01_run-1", "source")
    repeated = pump_recovery_experiment.stable_seed("sub-01_run-1", "source")
    other = pump_recovery_experiment.stable_seed("sub-01_run-2", "source")

    assert first == repeated
    assert first != other
