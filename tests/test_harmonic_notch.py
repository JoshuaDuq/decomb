"""Automatic comb and isolated-line notch planning."""

from __future__ import annotations

import json

import mne
import numpy as np
import pandas as pd
import pytest

from decomb import lines, notch, recordings
from decomb.config import load_config


def _settings() -> notch.HarmonicNotchSettings:
    return notch.HarmonicNotchSettings.from_config(load_config())


def _supported_model(
    *line_definitions: tuple[float, int | None],
) -> lines.LineModel:
    supported_lines = tuple(
        lines.SupportedLine(
            position_hz=position_hz,
            raw_p_value=1e-15,
            corrected_p_value=1e-12,
            window_indices=(0,),
            harmonic=harmonic,
        )
        for position_hz, harmonic in line_definitions
    )
    harmonics = [harmonic for _, harmonic in line_definitions if harmonic is not None]
    return lines.LineModel(
        channels=(
            lines.ChannelLineModel(
                channel_index=0,
                channel_name="Cz",
                lines=supported_lines,
                fundamental_hz=10.0 if harmonics else None,
                comb_corrected_p_value=1e-10 if harmonics else None,
            ),
        ),
        window_count=3,
        channel_count=1,
        test_count_per_channel=100_000,
    )


def _with_scanner_triggers(raw, settings=None):
    settings = _settings() if settings is None else settings
    raw.set_annotations(
        mne.Annotations(
            onset=np.arange(
                0.0,
                raw.times[-1],
                settings.scanner_repetition_time_s,
            ),
            duration=0.0,
            description=settings.scanner_trigger_event_name,
        )
    )
    return raw


def test_plan_never_adds_an_unsupported_harmonic():
    model = _supported_model((20.0, 2), (40.0, 4))

    plan = notch.plan_harmonic_stopbands(model.channels[0], _settings())

    assert {harmonic for band in plan.stopbands for harmonic in band.harmonics} == {2, 4}
    assert not any(band.low_hz <= 30.0 <= band.high_hz for band in plan.stopbands)


def test_independently_detected_drift_defines_the_stopband_envelope():
    model = _supported_model((96.0, 80), (96.1, 80))

    plan = notch.plan_harmonic_stopbands(model.channels[0], _settings())

    assert len(plan.stopbands) == 1
    assert plan.stopbands[0].low_hz < 96.0
    assert plan.stopbands[0].high_hz > 96.1


def test_isolated_line_detection_does_not_require_a_comb():
    sampling_frequency_hz = 250.0
    times_s = np.arange(int(120.0 * sampling_frequency_hz)) / sampling_frequency_hz
    data = np.random.default_rng(7).normal(scale=1e-6, size=(4, times_s.size))
    data[0] += 20e-6 * np.sin(2.0 * np.pi * 60.0 * times_s)
    raw = mne.io.RawArray(
        data,
        mne.create_info(["C3", "C4", "P3", "P4"], sampling_frequency_hz, "eeg"),
        verbose="ERROR",
    )

    model = notch.fit_harmonic_model(raw, _settings())

    assert all(channel.fundamental_hz is None for channel in model.channels)
    assert all(
        any(abs(line.position_hz - 60.0) < 0.02 for line in channel.lines)
        for channel in model.channels
    )


def test_fitted_model_and_plans_preserve_channel_specific_evidence():
    sampling_frequency_hz = 250.0
    times_s = np.arange(int(120.0 * sampling_frequency_hz)) / sampling_frequency_hz
    data = np.random.default_rng(19).normal(scale=1e-6, size=(2, times_s.size))
    data[0] += 20e-6 * np.sin(2.0 * np.pi * 60.0 * times_s)
    raw = mne.io.RawArray(
        data,
        mne.create_info(["C3", "C4"], sampling_frequency_hz, "eeg"),
        verbose="ERROR",
    )

    model = notch.fit_harmonic_model(raw, _settings())
    plans = notch.plan_channel_notches(model, _settings())

    assert [channel.channel_name for channel in model.channels] == ["C3"]
    assert model.channel_count == 2
    assert model.test_count_per_channel > 0
    assert [plan.channel_name for plan in plans] == ["C3"]
    assert all(
        any(
            stopband.low_hz <= 60.0 <= stopband.high_hz
            for stopband in plan.geometry.stopbands
        )
        for plan in plans
    )


def test_detection_includes_a_visible_acquisition_common_line():
    sampling_frequency_hz = 250.0
    times_s = np.arange(int(120.0 * sampling_frequency_hz)) / sampling_frequency_hz
    rng = np.random.default_rng(29)
    data = rng.normal(scale=1e-6, size=(3, times_s.size))
    data[0] += 20e-6 * np.sin(2.0 * np.pi * 60.0 * times_s)
    common_reference = 100e-6 * np.sin(2.0 * np.pi * 37.0 * times_s)
    info = mne.create_info(["C3", "C4", "Pz"], sampling_frequency_hz, "eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    rereferenced = mne.io.RawArray(
        data + common_reference,
        info,
        verbose="ERROR",
    )

    original_model = notch.fit_harmonic_model(raw, _settings())
    rereferenced_model = notch.fit_harmonic_model(rereferenced, _settings())

    assert not any(
        abs(line.position_hz - 37.0) < 0.02
        for channel in original_model.channels
        for line in channel.lines
    )
    assert all(
        any(abs(line.position_hz - 37.0) < 0.02 for line in channel.lines)
        for channel in rereferenced_model.channels
    )


def test_detection_requires_two_non_bad_eeg_channels():
    raw = mne.io.RawArray(
        np.zeros((1, 12_000)),
        mne.create_info(["Cz"], 200.0, "eeg"),
        verbose="ERROR",
    )

    with pytest.raises(ValueError, match="at least two non-bad EEG channels"):
        notch.fit_harmonic_model(raw, _settings())


def test_scanner_fundamental_is_fixed_by_the_configured_trigger_sequence():
    raw = mne.io.RawArray(
        np.zeros((2, 5_000)),
        mne.create_info(["C3", "C4"], 1_000.0, "eeg"),
        verbose="ERROR",
    )
    raw.set_annotations(
        mne.Annotations(
            onset=np.arange(5) * 0.9,
            duration=0.0,
            description="Volume/V  1",
        )
    )

    fundamental_hz = notch.scanner_fundamental_hz(raw, _settings())

    assert fundamental_hz == pytest.approx(1.0 / 0.9)


def test_scanner_trigger_name_must_match_an_annotation_exactly():
    raw = mne.io.RawArray(
        np.zeros((2, 5_000)),
        mne.create_info(["C3", "C4"], 1_000.0, "eeg"),
        verbose="ERROR",
    )
    raw.set_annotations(
        mne.Annotations(
            onset=np.arange(5) * 0.9,
            duration=0.0,
            description="Volume/V 1",
        )
    )

    with pytest.raises(ValueError, match="Volume/V  1.*not present"):
        notch.scanner_fundamental_hz(raw, _settings())


def test_scanner_trigger_intervals_must_equal_the_configured_tr():
    raw = mne.io.RawArray(
        np.zeros((2, 5_000)),
        mne.create_info(["C3", "C4"], 1_000.0, "eeg"),
        verbose="ERROR",
    )
    raw.set_annotations(
        mne.Annotations(
            onset=[0.0, 0.9, 1.8, 2.71, 3.6],
            duration=0.0,
            description="Volume/V  1",
        )
    )

    with pytest.raises(ValueError, match="configured 0.9 s TR"):
        notch.scanner_fundamental_hz(raw, _settings())


def test_two_supported_scanner_harmonics_authorize_the_complete_comb():
    frequencies_hz = np.arange(1.0, 5.01, 0.25)
    p_values = np.ones((2, 2, frequencies_hz.size))
    p_values[0, 0, np.flatnonzero(frequencies_hz == 2.0)[0]] = 1e-12

    local_evidence = notch.detect_scanner_harmonics(
        frequencies_hz,
        p_values,
        fundamental_hz=1.0,
        familywise_error_rate=0.025,
    )
    p_values[1, 1, np.flatnonzero(frequencies_hz == 4.0)[0]] = 1e-12
    supported = notch.detect_scanner_harmonics(
        frequencies_hz,
        p_values,
        fundamental_hz=1.0,
        familywise_error_rate=0.025,
    )

    assert local_evidence is not None
    assert local_evidence.supporting_harmonics == (2,)
    assert not local_evidence.authorizes_complete_comb
    assert supported is not None
    assert supported.fundamental_hz == 1.0
    assert supported.supporting_harmonics == (2, 4)
    assert supported.authorizes_complete_comb
    assert supported.corrected_p_value < 0.025
    assert supported.frequency_count == 5


def test_one_trigger_anchored_harmonic_authorizes_only_its_local_notch():
    frequencies_hz = np.arange(1.0, 5.01, 0.25)
    p_values = np.ones((2, 2, frequencies_hz.size))
    p_values[0, 0, np.flatnonzero(frequencies_hz == 2.0)[0]] = 1e-12

    evidence = notch.detect_scanner_harmonics(
        frequencies_hz,
        p_values,
        fundamental_hz=1.0,
        familywise_error_rate=0.025,
    )
    plan = notch.plan_scanner_harmonic_notches(
        evidence,
        _settings(),
        maximum_hz=5.0,
    )

    assert evidence.supporting_harmonics == (2,)
    assert not evidence.authorizes_complete_comb
    assert tuple(
        harmonic
        for stopband in plan.stopbands
        for harmonic in stopband.harmonics
    ) == (2,)
    assert plan.stopbands[0].width_hz == pytest.approx(
        _settings().supported_scanner_harmonic_stopband_width_hz
    )


def test_scanner_harmonics_plan_contains_every_harmonic_in_the_study_range():
    evidence = notch.ScannerHarmonicEvidence(
        fundamental_hz=10.0,
        corrected_p_value=1e-12,
        supporting_harmonics=(2, 4),
    )

    plan = notch.plan_scanner_harmonic_notches(
        evidence,
        _settings(),
        maximum_hz=49.0,
    )

    assert tuple(
        harmonic
        for stopband in plan.stopbands
        for harmonic in stopband.harmonics
    ) == (1, 2, 3, 4)
    widths_by_harmonic = {
        stopband.harmonics[0]: stopband.width_hz for stopband in plan.stopbands
    }
    assert widths_by_harmonic[1] == pytest.approx(0.25)
    assert widths_by_harmonic[2] == pytest.approx(
        _settings().supported_scanner_harmonic_stopband_width_hz
    )
    assert widths_by_harmonic[3] == pytest.approx(0.25)
    assert widths_by_harmonic[4] == pytest.approx(
        _settings().supported_scanner_harmonic_stopband_width_hz
    )


def test_cleaning_applies_the_complete_supported_scanner_harmonics(monkeypatch):
    raw = mne.io.RawArray(
        np.zeros((2, 12_000)),
        mne.create_info(["C3", "C4"], 1_000.0, "eeg"),
        verbose="ERROR",
    )
    raw.set_annotations(
        mne.Annotations(
            onset=np.arange(12, dtype=float),
            duration=0.0,
            description="Scanner/Volume",
        )
    )
    settings = notch.HarmonicNotchSettings(
        estimation_window_s=10.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(1.0, 5.0),
        scanner_repetition_time_s=1.0,
        scanner_trigger_event_name="Scanner/Volume",
    )
    scanner_passes = 0

    def p_values(raw, pass_settings):
        nonlocal scanner_passes
        frequencies_hz = np.arange(1.0, 6.0)
        probabilities = np.ones((1, 2, frequencies_hz.size))
        if pass_settings.estimation_window_s == 4.0:
            scanner_passes += 1
            if scanner_passes == 1:
                probabilities[0, 0, 1] = 1e-12
                probabilities[0, 1, 3] = 1e-12
        return frequencies_hz, probabilities

    def apply(raw, plan, *, n_jobs=-1):
        filtered = raw.copy()
        filtered._data[0, 0] += 1.0
        return filtered

    monkeypatch.setattr(notch, "_line_test_p_values", p_values)
    monkeypatch.setattr(notch, "_thomson_f_p_values", p_values)
    monkeypatch.setattr(notch, "apply_harmonic_notches", apply)
    monkeypatch.setattr(
        notch,
        "_measure_scanner_stopband_changes",
        lambda raw_before, raw_after, plan, settings: (
            (-20.0,) * len(plan.stopbands)
        ),
        raising=False,
    )

    result = notch.clean_until_no_supported_lines(raw, settings)

    assert len(result.rounds) == 1
    removal_round = result.rounds[0]
    assert removal_round.model.channels == ()
    assert removal_round.scanner_harmonics is not None
    assert tuple(
        harmonic
        for stopband in removal_round.scanner_plan.stopbands
        for harmonic in stopband.harmonics
    ) == (1, 2, 3, 4, 5)
    assert result.residual_model.channels == ()
    assert result.residual_scanner_harmonics is None


def test_first_round_diagnosis_matches_the_joint_cleaning_evidence(monkeypatch):
    settings = notch.HarmonicNotchSettings(
        estimation_window_s=10.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(1.0, 5.0),
        scanner_repetition_time_s=1.0,
        scanner_trigger_event_name="Scanner/Volume",
    )
    raw = _with_scanner_triggers(
        mne.io.RawArray(
            np.zeros((2, 12_000)),
            mne.create_info(["C3", "C4"], 1_000.0, "eeg"),
            verbose="ERROR",
        ),
        settings,
    )

    def p_values(raw, pass_settings):
        frequencies_hz = np.arange(1.0, 6.0)
        probabilities = np.ones((1, 2, frequencies_hz.size))
        if pass_settings.estimation_window_s == 4.0:
            probabilities[0, 0, 1] = 1e-12
            probabilities[0, 1, 3] = 1e-12
        return frequencies_hz, probabilities

    monkeypatch.setattr(notch, "_line_test_p_values", p_values)
    monkeypatch.setattr(notch, "_thomson_f_p_values", p_values)

    evidence = notch.fit_harmonic_round(raw, settings)

    assert evidence.model.channels == ()
    assert evidence.scanner_harmonics is not None
    assert evidence.scanner_harmonics.supporting_harmonics == (2, 4)
    assert tuple(
        harmonic
        for stopband in evidence.scanner_plan.stopbands
        for harmonic in stopband.harmonics
    ) == (1, 2, 3, 4, 5)
    assert evidence.filter_plan == evidence.scanner_plan


def test_bad_eeg_channels_cannot_authorize_a_recording_notch():
    sampling_frequency_hz = 250.0
    times_s = np.arange(int(120.0 * sampling_frequency_hz)) / sampling_frequency_hz
    data = np.random.default_rng(31).normal(
        scale=1e-6,
        size=(3, times_s.size),
    )
    data[0] += 20e-6 * np.sin(2.0 * np.pi * 60.0 * times_s)
    raw = mne.io.RawArray(
        data,
        mne.create_info(["C3", "C4", "Pz"], sampling_frequency_hz, "eeg"),
        verbose="ERROR",
    )
    raw.info["bads"] = ["C3"]

    model = notch.fit_harmonic_model(raw, _settings())

    assert model.channel_count == 2
    assert "C3" not in {channel.channel_name for channel in model.channels}


def test_recording_plan_filters_every_eeg_channel():
    sfreq = 200.0
    times = np.arange(int(120.0 * sfreq)) / sfreq
    data = np.vstack(
        (
            np.sin(2.0 * np.pi * 20.0 * times),
            np.sin(2.0 * np.pi * 20.0 * times),
        )
    )
    raw = mne.io.RawArray(
        data,
        mne.create_info(["C3", "C4"], sfreq, ch_types="eeg"),
        verbose="ERROR",
    )
    model = _supported_model((20.0, 2))

    plan = notch.plan_recording_notches(model, _settings())
    filtered = notch.apply_harmonic_notches(raw, plan)

    interior = slice(int(30.0 * sfreq), int(90.0 * sfreq))
    for channel_index in range(2):
        assert _tone_amplitude(
            filtered.get_data()[channel_index, interior],
            20.0,
            sfreq,
        ) < _tone_amplitude(
            raw.get_data()[channel_index, interior],
            20.0,
            sfreq,
        ) / 50.0


def test_fit_harmonic_model_detects_a_strong_line():
    narrow_settings = notch.HarmonicNotchSettings(
        estimation_window_s=54.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(1.0, 10.0),
    )
    sampling_frequency_hz = 250.0
    times_s = np.arange(int(120.0 * sampling_frequency_hz)) / sampling_frequency_hz
    data = np.random.default_rng(19).normal(scale=1e-6, size=(2, times_s.size))
    data[0] += 20e-6 * np.sin(2.0 * np.pi * 5.0 * times_s)
    raw = mne.io.RawArray(
        data,
        mne.create_info(["C3", "C4"], sampling_frequency_hz, "eeg"),
        verbose="ERROR",
    )

    model = notch.fit_harmonic_model(raw, narrow_settings)

    assert "C3" in [channel.channel_name for channel in model.channels]
    c3 = next(channel for channel in model.channels if channel.channel_name == "C3")
    assert any(abs(line.position_hz - 5.0) < 0.02 for line in c3.lines)
def test_manifest_records_channel_local_holm_evidence():
    sampling_frequency_hz = 250.0
    times_s = np.arange(int(120.0 * sampling_frequency_hz)) / sampling_frequency_hz
    data = np.random.default_rng(23).normal(scale=1e-6, size=(2, times_s.size))
    data[0] += 20e-6 * np.sin(2.0 * np.pi * 60.0 * times_s)
    raw = mne.io.RawArray(
        data,
        mne.create_info(["C3", "C4"], sampling_frequency_hz, "eeg"),
        verbose="ERROR",
    )
    model = notch.fit_harmonic_model(raw, _settings())
    plans = notch.plan_channel_notches(model, _settings())

    rows = notch.line_manifest_rows(
        "recording",
        model,
        plans,
        (),
        _settings(),
    )

    assert {row["channel"] for row in rows} == {"C3"}
    assert all(
        row["multiple_testing_method"]
        == "bonferroni_two_shape_union_then_holm"
        for row in rows
    )
    assert all(
        row["multiple_testing_scope"]
        == "as_recorded_non_bad_eeg_recording_removal_sequence"
        for row in rows
    )
    assert all(row["round_familywise_error_rate"] == 0.0125 for row in rows)
    assert all(row["detection_test_count_per_channel"] > 0 for row in rows)
    assert all(row["detected_line_input_p_values"] for row in rows)


def test_plan_contains_every_authorized_harmonic_and_isolated_line():
    model = _supported_model((20.0, 2), (30.0, 3), (35.5, None), (40.0, 4))

    plan = notch.plan_harmonic_stopbands(model.channels[0], _settings())

    planned_harmonics = {
        harmonic for stopband in plan.stopbands for harmonic in stopband.harmonics
    }
    assert planned_harmonics == {2, 3, 4}
    assert sum(stopband.kind == "isolated" for stopband in plan.stopbands) == 1
    assert _settings().filter_resolution_window_s == 54.0
    assert _settings().ordinary_line_stopband_width_hz == 0.25
    assert all(stopband.width_hz >= 0.25 for stopband in plan.stopbands)
    assert plan.transition_bandwidth_hz == pytest.approx(3.3 / 54.0)
    assert _settings().per_edge_transition_bandwidth_hz == pytest.approx(3.3 / 108.0)


def test_mne_filter_design_reports_its_exact_length_and_response():
    settings = _settings()
    plan = notch.HarmonicNotchPlan(
        (notch.HarmonicStopband((2,), 19.95, 20.05),),
        transition_bandwidth_hz=settings.transition_bandwidth_hz,
    )

    design = notch.characterize_harmonic_filter(500.0, plan)

    assert design.length_samples == 54_001
    assert design.length_s == pytest.approx(108.002)
    assert np.all(
        np.isfinite(
            (
                design.minimum_stopband_attenuation_db,
                design.maximum_passband_deviation_db,
            )
        )
    )
    assert design.minimum_stopband_attenuation_db >= 0.0
    assert design.maximum_passband_deviation_db >= 0.0
    assert design.manifest_fields() == {
        "fir_filter_length_samples": 54_001,
        "fir_filter_length_s": pytest.approx(108.002),
        "fir_minimum_stopband_attenuation_db": pytest.approx(
            design.minimum_stopband_attenuation_db
        ),
        "fir_maximum_passband_deviation_db": pytest.approx(
            design.maximum_passband_deviation_db
        ),
    }


def test_stopband_covers_every_observed_position_and_bin_uncertainty():
    model = _supported_model((19.98, 2), (20.03, 2), (30.0, 3), (40.0, 4))

    plan = notch.plan_harmonic_stopbands(model.channels[0], _settings())
    second = next(stopband for stopband in plan.stopbands if stopband.harmonics == (2,))

    assert second.low_hz < 19.98
    assert second.high_hz > 20.03


def test_stationary_interval_covers_the_minimum_visible_line_width():
    model = _supported_model((20.0, 2))

    plan = notch.plan_harmonic_stopbands(model.channels[0], _settings())

    assert plan.stopbands[0].width_hz == pytest.approx(
        _settings().ordinary_line_stopband_width_hz
    )


def test_intervals_without_enough_transition_passband_are_merged():
    first = notch.HarmonicStopband((2,), 20.00, 20.03)
    second = notch.HarmonicStopband((), 20.05, 20.08, kind="isolated")

    merged = notch._merge_stopbands([first, second], minimum_gap_hz=0.03)

    assert len(merged) == 1
    assert merged[0].kind == "mixed"
    assert merged[0].harmonics == (2,)


def _tone_amplitude(data: np.ndarray, frequency_hz: float, sfreq: float) -> float:
    times = np.arange(data.size) / sfreq
    basis = np.column_stack(
        (np.sin(2.0 * np.pi * frequency_hz * times), np.cos(2.0 * np.pi * frequency_hz * times))
    )
    coefficients, *_ = np.linalg.lstsq(basis, data, rcond=None)
    return float(np.linalg.norm(coefficients))


def test_apply_suppresses_only_the_planned_interval():
    sfreq = 200.0
    times = np.arange(int(80.0 * sfreq)) / sfreq
    data = np.sin(2.0 * np.pi * 20.0 * times)
    data += 0.5 * np.sin(2.0 * np.pi * 17.0 * times)
    raw = mne.io.RawArray(
        data[np.newaxis, :],
        mne.create_info(["EEG 001"], sfreq, ch_types="eeg"),
        verbose="ERROR",
    )
    plan = notch.HarmonicNotchPlan(
        (notch.HarmonicStopband((2,), 19.95, 20.05),),
        transition_bandwidth_hz=0.2,
    )

    filtered = notch.apply_harmonic_notches(raw, plan)

    interior = slice(int(20.0 * sfreq), int(60.0 * sfreq))
    before = raw.get_data()[0, interior]
    after = filtered.get_data()[0, interior]
    assert _tone_amplitude(after, 20.0, sfreq) < _tone_amplitude(before, 20.0, sfreq) / 100.0
    assert _tone_amplitude(after, 17.0, sfreq) == pytest.approx(
        _tone_amplitude(before, 17.0, sfreq),
        rel=0.01,
    )


def test_apply_filters_only_the_channel_with_statistical_evidence():
    sfreq = 200.0
    times = np.arange(int(80.0 * sfreq)) / sfreq
    data = np.vstack(
        (
            np.sin(2.0 * np.pi * 20.0 * times),
            np.sin(2.0 * np.pi * 20.0 * times),
        )
    )
    raw = mne.io.RawArray(
        data,
        mne.create_info(["C3", "C4"], sfreq, ch_types="eeg"),
        verbose="ERROR",
    )
    geometry = notch.HarmonicNotchPlan(
        (notch.HarmonicStopband((2,), 19.95, 20.05),),
        transition_bandwidth_hz=0.2,
    )
    plans = (notch.ChannelNotchPlan("C3", geometry),)

    filtered = notch.apply_channel_notches(raw, plans)

    interior = slice(int(20.0 * sfreq), int(60.0 * sfreq))
    assert _tone_amplitude(filtered.get_data()[0, interior], 20.0, sfreq) < (
        _tone_amplitude(raw.get_data()[0, interior], 20.0, sfreq) / 100.0
    )
    np.testing.assert_array_equal(filtered.get_data()[1], raw.get_data()[1])


def test_no_supported_line_produces_an_unchanged_copy_and_null_manifest():
    raw = mne.io.RawArray(
        np.random.default_rng(41).normal(size=(2, 1_000)),
        mne.create_info(["C3", "C4"], 200.0, ch_types="eeg"),
        verbose="ERROR",
    )
    model = lines.LineModel(
        channels=(),
        window_count=3,
        channel_count=2,
        test_count_per_channel=1_000,
    )

    plans = notch.plan_channel_notches(model, _settings())
    filtered = notch.apply_channel_notches(raw, plans)
    rows = notch.line_manifest_rows(
        "recording",
        model,
        plans,
        (),
        _settings(),
    )

    assert plans == ()
    assert filtered is not raw
    np.testing.assert_array_equal(filtered.get_data(), raw.get_data())
    assert len(rows) == 1
    assert rows[0]["outcome"] == "no_line_detected"
    assert notch.channel_plans_from_rows(rows) == ()
    notch._validate_manifest_evidence(rows, _settings())


def test_cleaning_continues_until_a_fresh_holm_fit_is_null(monkeypatch):
    raw = _with_scanner_triggers(mne.io.RawArray(
        np.ones((1, 1_000)),
        mne.create_info(["Cz"], 200.0, ch_types="eeg"),
        verbose="ERROR",
    ))
    frequencies_hz = np.array([20.0, 20.1])
    p_value_rounds = iter(
        (
            np.array([[[0.001, 1.0]]]),
            np.array([[[1.0, 0.001]]]),
            np.array([[[1.0, 1.0]]]),
        )
    )
    def p_values(raw, settings):
        if settings.estimation_window_s == 4.0:
            return frequencies_hz, np.ones((1, 1, frequencies_hz.size))
        return frequencies_hz, next(p_value_rounds)

    monkeypatch.setattr(notch, "_line_test_p_values", p_values)
    monkeypatch.setattr(notch, "_thomson_f_p_values", p_values)

    def attenuate(raw, plan, *, n_jobs=-1):
        filtered = raw.copy()
        filtered._data *= 0.5
        return filtered

    monkeypatch.setattr(notch, "apply_harmonic_notches", attenuate)
    monkeypatch.setattr(
        notch,
        "_measure_channel_stopband_changes",
        lambda before, after, plans, settings: tuple(0.0 for _ in plans),
    )

    result = notch.clean_until_no_supported_lines(raw, _settings())

    assert len(result.rounds) == 2
    assert [round_.model.line_count for round_ in result.rounds] == [1, 1]
    assert result.residual_model.channels == ()
    np.testing.assert_array_equal(result.cleaned.get_data(), raw.get_data() / 4.0)


def test_persistent_peak_family_is_fitted_only_on_the_unfiltered_source(
    monkeypatch,
):
    raw = _with_scanner_triggers(mne.io.RawArray(
        np.ones((2, 3_000)),
        mne.create_info(["C3", "C4"], 200.0, ch_types="eeg"),
        verbose="ERROR",
    ))
    frequencies_hz = np.array([20.0, 20.1])
    initial_calls = 0

    def initial_p_values(raw, settings):
        nonlocal initial_calls
        initial_calls += 1
        probabilities = np.ones((1, 2, 2))
        if initial_calls == 1:
            probabilities[0, 0, 0] = 1e-12
        return frequencies_hz, probabilities

    def residual_p_values(raw, settings):
        channel_count = 2
        return frequencies_hz, np.ones((1, channel_count, 2))

    monkeypatch.setattr(notch, "_line_test_p_values", initial_p_values)
    monkeypatch.setattr(notch, "_thomson_f_p_values", residual_p_values)
    monkeypatch.setattr(
        notch,
        "apply_harmonic_notches",
        lambda raw, plan, *, n_jobs=-1: raw.copy().apply_function(
            lambda data: data * 0.5
        ),
    )
    monkeypatch.setattr(
        notch,
        "_measure_channel_stopband_changes",
        lambda before, after, plans, settings: (0.0,),
    )

    result = notch.clean_until_no_supported_lines(raw, _settings())

    assert initial_calls == 1
    assert len(result.rounds) == 1
    assert result.residual_model.channels == ()
    np.testing.assert_array_equal(result.cleaned.get_data(), raw.get_data() / 2.0)


def test_cleaning_spends_the_recording_error_rate_across_rounds(monkeypatch):
    raw = _with_scanner_triggers(mne.io.RawArray(
        np.ones((2, 1_000)),
        mne.create_info(["C3", "C4"], 200.0, ch_types="eeg"),
        verbose="ERROR",
    ))
    frequencies_hz = np.array([20.0])
    p_value_rounds = iter(
        (
            np.array([[[0.001], [1.0]]]),
            np.array([[[0.02], [1.0]]]),
        )
    )
    def p_values(raw, settings):
        if settings.estimation_window_s == 4.0:
            scanner_frequencies_hz = np.array([20.0, 20.1])
            return scanner_frequencies_hz, np.ones((1, 2, 2))
        return frequencies_hz, next(p_value_rounds)

    monkeypatch.setattr(notch, "_line_test_p_values", p_values)
    monkeypatch.setattr(notch, "_thomson_f_p_values", p_values)

    def attenuate(raw, plan, *, n_jobs=-1):
        filtered = raw.copy()
        filtered._data *= 0.5
        return filtered

    monkeypatch.setattr(notch, "apply_harmonic_notches", attenuate)
    monkeypatch.setattr(
        notch,
        "_measure_channel_stopband_changes",
        lambda before, after, plans, settings: tuple(0.0 for _ in plans),
    )

    result = notch.clean_until_no_supported_lines(raw, _settings())

    assert len(result.rounds) == 1


def test_clean_recording_satisfies_the_residual_postcondition_without_filtering(
    monkeypatch,
):
    raw = _with_scanner_triggers(mne.io.RawArray(
        np.ones((1, 1_000)),
        mne.create_info(["Cz"], 200.0, ch_types="eeg"),
        verbose="ERROR",
    ))
    def null_p_values(raw, settings):
        return np.array([20.0, 20.1]), np.array([[[1.0, 1.0]]])

    monkeypatch.setattr(notch, "_line_test_p_values", null_p_values)
    monkeypatch.setattr(notch, "_thomson_f_p_values", null_p_values)

    result = notch.clean_until_no_supported_lines(raw, _settings())

    assert result.rounds == ()
    assert result.residual_model.channels == ()
    assert result.cleaned is not raw
    np.testing.assert_array_equal(result.cleaned.get_data(), raw.get_data())


def test_manifest_records_every_removal_round_and_the_terminal_null():
    model = _supported_model((20.0, 2))
    plans = notch.plan_channel_notches(model, _settings())
    residual_model = lines.LineModel(
        channels=(),
        window_count=model.window_count,
        channel_count=model.channel_count,
        test_count_per_channel=model.test_count_per_channel,
    )
    cleaned = mne.io.RawArray(
        np.zeros((1, 1_000)),
        mne.create_info(["Cz"], 500.0, ch_types="eeg"),
        verbose="ERROR",
    )
    result = notch.HarmonicCleaningResult(
        cleaned,
        (
            notch.HarmonicRemovalRound(
                model,
                plans,
                notch.plan_recording_notches(model, _settings()),
                (-20.0,),
            ),
            notch.HarmonicRemovalRound(
                model,
                plans,
                notch.plan_recording_notches(model, _settings()),
                (-10.0,),
            ),
        ),
        residual_model,
    )

    rows = notch.cleaning_manifest_rows(
        "recording",
        result,
        (),
        _settings(),
    )

    assert [row["removal_round"] for row in rows] == [1, 2, 3]
    assert [row["outcome"] for row in rows] == [
        "line_detected",
        "line_detected",
        "no_line_detected",
    ]
    assert rows[-1]["multiple_testing_method"] == "holm_and_scanner_bonferroni"
    assert rows[-1]["multiple_testing_scope"] == (
        "joint_as_recorded_line_and_trigger_anchored_scanner_families"
    )
    assert len(notch.removal_rounds_from_rows(rows)) == 2
    notch._validate_manifest_evidence(rows, _settings())

    with pytest.raises(ValueError, match="terminal null"):
        notch._validate_manifest_evidence(rows[:-1], _settings())

    fractional_round = [dict(row) for row in rows]
    fractional_round[0]["removal_round"] = 1.5
    with pytest.raises(ValueError, match="integer removal_round"):
        notch._validate_manifest_evidence(fractional_round, _settings())


def test_manifest_records_trigger_anchored_comb_evidence():
    settings = notch.HarmonicNotchSettings(
        estimation_window_s=10.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(1.0, 5.0),
        scanner_repetition_time_s=1.0,
        scanner_trigger_event_name="Scanner/Volume",
    )
    model = lines.LineModel((), 3, 2, 15)
    evidence = notch.ScannerHarmonicEvidence(1.0, 1e-10, (2, 4))
    scanner_plan = notch.plan_scanner_harmonic_notches(
        evidence,
        settings,
        maximum_hz=5.0,
    )
    cleaned = mne.io.RawArray(
        np.zeros((2, 12_000)),
        mne.create_info(["C3", "C4"], 1_000.0, "eeg"),
        verbose="ERROR",
    )
    result = notch.HarmonicCleaningResult(
        cleaned,
        (
            notch.HarmonicRemovalRound(
                model,
                (),
                scanner_plan,
                (-20.0,) * len(scanner_plan.stopbands),
                evidence,
                scanner_plan,
            ),
        ),
        model,
    )

    rows = notch.cleaning_manifest_rows("recording", result, (), settings)

    detected = rows[:-1]
    assert len(detected) == len(scanner_plan.stopbands)
    assert {row["harmonics"] for row in detected} == {"1;2;3;4;5"}
    assert {row["outcome"] for row in detected} == {"scanner_harmonics_detected"}
    assert {row["fundamental_hz"] for row in detected} == {1.0}
    assert {row["scanner_family_corrected_p_value"] for row in detected} == {
        1e-10
    }
    assert {row["scanner_supporting_harmonics"] for row in detected} == {
        "2;4"
    }
    assert {row["scanner_repetition_time_s"] for row in detected} == {1.0}
    assert {row["scanner_trigger_event_name"] for row in detected} == {
        "Scanner/Volume"
    }
    assert rows[-1]["outcome"] == "no_line_detected"
    assert len(notch.removal_rounds_from_rows(rows)) == 1
    notch._validate_manifest_evidence(rows, settings)


def test_manifest_records_one_authorized_scanner_harmonic_without_a_comb():
    settings = notch.HarmonicNotchSettings(
        estimation_window_s=10.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(1.0, 5.0),
        scanner_repetition_time_s=1.0,
        scanner_trigger_event_name="Scanner/Volume",
    )
    evidence = notch.ScannerHarmonicEvidence(1.0, 1e-10, (2,))
    plan = notch.plan_scanner_harmonic_notches(
        evidence,
        settings,
        maximum_hz=5.0,
    )

    rows = notch.scanner_harmonic_manifest_rows(
        "recording",
        evidence,
        plan,
        (),
        settings,
        round_index=1,
    )

    assert [row["harmonics"] for row in rows] == ["2"]
    assert {row["multiple_testing_method"] for row in rows} == {"bonferroni"}
    notch._validate_round_manifest_evidence(rows, settings)


def test_verification_refits_and_replays_scanner_harmonics_evidence(
    tmp_path,
    monkeypatch,
):
    settings = notch.HarmonicNotchSettings(
        estimation_window_s=10.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(1.0, 5.0),
        scanner_repetition_time_s=1.0,
        scanner_trigger_event_name="Scanner/Volume",
    )
    raw = _with_scanner_triggers(
        mne.io.RawArray(
            np.zeros((2, 12_000)),
            mne.create_info(["C3", "C4"], 1_000.0, ch_types="eeg"),
            verbose="ERROR",
        ),
        settings,
    )
    residual_model = lines.LineModel((), 1, 2, 5)
    evidence = notch.ScannerHarmonicEvidence(
        1.0,
        1e-11,
        (2, 4),
        window_count=1,
        channel_count=2,
        frequency_count=5,
    )
    scanner_plan = notch.plan_scanner_harmonic_notches(
        evidence,
        settings,
        maximum_hz=5.0,
    )
    result = notch.HarmonicCleaningResult(
        raw,
        (
            notch.HarmonicRemovalRound(
                residual_model,
                (),
                scanner_plan,
                (0.0,) * len(scanner_plan.stopbands),
                evidence,
                scanner_plan,
            ),
        ),
        residual_model,
    )
    manifest_rows = notch.cleaning_manifest_rows(
        "recording",
        result,
        (),
        settings,
    )
    scanner_passes = 0

    def p_values(raw, pass_settings):
        nonlocal scanner_passes
        frequencies_hz = np.arange(1.0, 6.0)
        probabilities = np.ones((1, 2, frequencies_hz.size))
        if pass_settings.estimation_window_s == 4.0:
            scanner_passes += 1
            if scanner_passes == 1:
                probabilities[0, 0, 1] = 1e-12
                probabilities[0, 1, 3] = 1e-12
        return frequencies_hz, probabilities

    def apply(raw, plan, *, n_jobs=-1):
        filtered = raw.copy()
        filtered._data[:, 0] += 1.0
        return filtered

    monkeypatch.setattr(recordings, "read_bids_raw", lambda path: raw.copy())
    monkeypatch.setattr(notch, "_line_test_p_values", p_values)
    monkeypatch.setattr(notch, "_thomson_f_p_values", p_values)
    monkeypatch.setattr(notch, "apply_harmonic_notches", apply)
    monkeypatch.setattr(
        notch,
        "_measure_scanner_stopband_changes",
        lambda *args: (0.0,) * len(scanner_plan.stopbands),
    )
    monkeypatch.setattr(notch, "_validate_exact_derivative", lambda *args: 0.0)

    rows = notch.verify_harmonic_run(
        tmp_path / "recording.vhdr",
        tmp_path / "cleaned.vhdr",
        manifest_rows,
        settings,
    )

    assert {row["outcome"] for row in rows} == {
        "scanner_harmonics_detected",
        "no_line_detected",
    }
    assert scanner_passes == 3


def test_residual_postcondition_rejects_a_statistically_dirty_derivative(
    monkeypatch,
):
    raw = _with_scanner_triggers(mne.io.RawArray(
        np.zeros((2, 1_000)),
        mne.create_info(["C3", "C4"], 500.0, ch_types="eeg"),
        verbose="ERROR",
    ))

    def p_values(raw, pass_settings):
        frequencies_hz = np.array([20.0, 20.1])
        probabilities = np.ones((1, 2, 2))
        if pass_settings.estimation_window_s != 4.0:
            probabilities[0, 0, 0] = 1e-12
        return frequencies_hz, probabilities

    monkeypatch.setattr(notch, "_line_test_p_values", p_values)
    monkeypatch.setattr(notch, "_thomson_f_p_values", p_values)

    with pytest.raises(RuntimeError, match="Holm-significant residual"):
        notch.validate_residual_postcondition(raw, _settings())


def test_residual_postcondition_rejects_an_authorized_scanner_harmonics(monkeypatch):
    settings = notch.HarmonicNotchSettings(
        estimation_window_s=10.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(1.0, 5.0),
        scanner_repetition_time_s=1.0,
        scanner_trigger_event_name="Scanner/Volume",
    )
    raw = _with_scanner_triggers(
        mne.io.RawArray(
            np.zeros((2, 12_000)),
            mne.create_info(["C3", "C4"], 1_000.0, ch_types="eeg"),
            verbose="ERROR",
        ),
        settings,
    )

    def p_values(raw, pass_settings):
        frequencies_hz = np.arange(1.0, 6.0)
        probabilities = np.ones((1, 2, frequencies_hz.size))
        if pass_settings.estimation_window_s == 4.0:
            probabilities[0, 0, 1] = 1e-12
            probabilities[0, 1, 3] = 1e-12
        return frequencies_hz, probabilities

    monkeypatch.setattr(notch, "_line_test_p_values", p_values)
    monkeypatch.setattr(notch, "_thomson_f_p_values", p_values)

    with pytest.raises(RuntimeError, match="scanner-comb"):
        notch.validate_residual_postcondition(raw, settings)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("detected_line_corrected_p_values", "0"),
        ("tested_eeg_channel_count", 2),
        ("harmonics", "999"),
    ),
)
def test_refitted_evidence_rejects_tampered_authorization(field, value):
    model = _supported_model((20.0, 2))
    plans = notch.plan_channel_notches(model, _settings())
    expected = notch.line_manifest_rows(
        "recording",
        model,
        plans,
        (),
        _settings(),
    )
    tampered = [dict(row) for row in expected]
    tampered[0][field] = value

    with pytest.raises(ValueError, match="refitted statistical evidence"):
        notch._validate_refitted_evidence(tampered, expected)


def test_refitted_evidence_rejects_tampered_scanner_support():
    settings = notch.HarmonicNotchSettings(
        estimation_window_s=10.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(1.0, 5.0),
        scanner_repetition_time_s=1.0,
        scanner_trigger_event_name="Scanner/Volume",
    )
    evidence = notch.ScannerHarmonicEvidence(1.0, 1e-10, (2, 4))
    plan = notch.plan_scanner_harmonic_notches(
        evidence,
        settings,
        maximum_hz=5.0,
    )
    expected = notch.scanner_harmonic_manifest_rows(
        "recording",
        evidence,
        plan,
        (),
        settings,
        round_index=1,
    )
    tampered = [dict(row) for row in expected]
    for row in tampered:
        row["scanner_supporting_harmonics"] = "2;3"

    with pytest.raises(ValueError, match="refitted statistical evidence"):
        notch._validate_refitted_evidence(tampered, expected)


def test_apply_declares_the_same_acquisition_boundaries_used_for_estimation(
    monkeypatch,
):
    sfreq = 200.0
    raw = mne.io.RawArray(
        np.zeros((1, int(80.0 * sfreq))),
        mne.create_info(["Cz"], sfreq, "eeg"),
        verbose="ERROR",
    )
    plan = notch.HarmonicNotchPlan(
        (notch.HarmonicStopband((2,), 19.95, 20.05),),
        transition_bandwidth_hz=0.2,
    )
    captured = {}

    def record_filter_call(self, **kwargs):
        captured.update(kwargs)
        return self

    monkeypatch.setattr(mne.io.BaseRaw, "notch_filter", record_filter_call)

    notch.apply_harmonic_notches(raw, plan)

    assert captured["skip_by_annotation"] == recordings.ACQUISITION_BOUNDARY_ANNOTATIONS


def test_apply_refuses_a_continuous_span_shorter_than_the_fir():
    sfreq = 200.0
    raw = mne.io.RawArray(
        np.zeros((2, int(10.0 * sfreq))),
        mne.create_info(["C3", "C4"], sfreq, "eeg"),
        verbose="ERROR",
    )
    plan = notch.HarmonicNotchPlan(
        (notch.HarmonicStopband((2,), 19.95, 20.05),),
        transition_bandwidth_hz=0.2,
    )

    with pytest.raises(ValueError, match="shorter than the 6601-sample FIR"):
        notch.apply_harmonic_notches(raw, plan)


def test_exclusion_rows_report_kind_transitions_and_band_availability():
    plan = notch.HarmonicNotchPlan(
        (
            notch.HarmonicStopband((2,), 19.95, 20.05),
            notch.HarmonicStopband((), 29.95, 30.05, kind="isolated"),
        ),
        transition_bandwidth_hz=0.2,
    )

    rows = notch.harmonic_exclusion_rows(
        "recording",
        plan,
        (("beta", 13.0, 30.0),),
    )

    assert rows[0]["kind"] == "comb"
    assert rows[1]["kind"] == "isolated"
    assert rows[1]["harmonics"] == ""
    assert rows[0]["unavailable_low_hz"] == pytest.approx(19.85)
    assert rows[0]["beta_retained_share"] < 1.0


def test_manifest_evidence_contains_only_supported_lines():
    model = _supported_model((20.0, 2), (35.5, None), (40.0, 4))
    plans = notch.plan_channel_notches(model, _settings())

    rows = notch.line_manifest_rows("recording", model, plans, (), _settings())

    recorded_frequencies = {
        float(value)
        for row in rows
        for value in str(row["detected_line_frequencies_hz"]).split(";")
    }
    assert recorded_frequencies == {20.0, 35.5, 40.0}
    assert all(row["familywise_error_rate"] == 0.05 for row in rows)
    assert all(row["detection_test_count_per_channel"] == 100_000 for row in rows)


def test_manifest_tsv_round_trips_comb_and_isolated_plans_exactly(tmp_path):
    plan = notch.HarmonicNotchPlan(
        (
            notch.HarmonicStopband((2,), np.nextafter(19.95, np.inf), 20.05),
            notch.HarmonicStopband((), 35.49, 35.51, kind="isolated"),
        ),
        transition_bandwidth_hz=np.nextafter(0.2, np.inf),
    )
    path = tmp_path / "manifest.tsv"
    recordings.write_tsv_atomic(
        pd.DataFrame(notch.harmonic_exclusion_rows("recording", plan, ())),
        path,
    )

    stored = pd.read_csv(path, sep="\t", float_precision="round_trip")

    assert notch.harmonic_plan_from_rows(stored.to_dict("records")) == plan


def test_manifest_tsv_preserves_recording_level_scanner_evidence(tmp_path):
    settings = notch.HarmonicNotchSettings(
        estimation_window_s=10.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(1.0, 5.0),
        scanner_repetition_time_s=1.0,
        scanner_trigger_event_name="Scanner/Volume",
    )
    evidence = notch.ScannerHarmonicEvidence(1.0, 1e-10, (2, 4))
    plan = notch.plan_scanner_harmonic_notches(
        evidence,
        settings,
        maximum_hz=5.0,
    )
    path = tmp_path / "manifest.tsv"
    recordings.write_tsv_atomic(
        pd.DataFrame(
            notch.scanner_harmonic_manifest_rows(
                "recording",
                evidence,
                plan,
                (),
                settings,
                round_index=1,
            )
        ),
        path,
    )

    stored = notch._read_manifest(path)

    assert set(stored["channel"]) == {""}
    notch._validate_round_manifest_evidence(
        stored.to_dict("records"),
        settings,
    )


def test_source_dataset_url_is_relative_to_the_published_derivative(tmp_path):
    source_root = tmp_path / "inputs" / "study" / "bids"
    derivative_root = tmp_path / "published" / "derivatives" / "decomb"

    assert notch.relative_source_dataset_url(source_root, derivative_root) == (
        "../../../inputs/study/bids"
    )


def test_verification_rejects_an_unfiltered_derivative(brainvision_run):
    vhdr, raw = brainvision_run
    geometry = notch.HarmonicNotchPlan(
        (notch.HarmonicStopband((2,), 19.95, 20.05),),
        transition_bandwidth_hz=2.0,
    )
    with pytest.raises(RuntimeError, match="does not equal the declared FIR derivative"):
        notch._validate_exact_derivative(raw, raw.copy(), vhdr, (geometry,))


def test_verification_accepts_the_exact_quantized_filter_result(brainvision_run):
    vhdr, raw = brainvision_run
    geometry = notch.HarmonicNotchPlan(
        (notch.HarmonicStopband((2,), 19.95, 20.05),),
        transition_bandwidth_hz=2.0,
    )
    filtered = notch.apply_harmonic_notches(raw, geometry)
    quantized = recordings.quantized_eeg_data(
        vhdr,
        filtered.get_data(),
        filtered.ch_names,
    )
    written = filtered.copy()
    written._data = quantized

    assert notch._validate_exact_derivative(raw, written, vhdr, (geometry,)) == 0.0


def test_verification_replays_every_removal_round(brainvision_run):
    vhdr, raw = brainvision_run
    geometry = notch.HarmonicNotchPlan(
        (notch.HarmonicStopband((2,), 19.95, 20.05),),
        transition_bandwidth_hz=2.0,
    )
    filtered_once = notch.apply_harmonic_notches(raw, geometry)
    filtered_twice = notch.apply_harmonic_notches(filtered_once, geometry)
    quantized = recordings.quantized_eeg_data(
        vhdr,
        filtered_twice.get_data(),
        filtered_twice.ch_names,
    )
    written = filtered_twice.copy()
    written._data = quantized

    assert notch._validate_exact_derivative(
        raw,
        written,
        vhdr,
        (geometry, geometry),
    ) == 0.0


def test_verification_rejects_filter_provenance_that_cannot_be_reproduced():
    plan = notch.HarmonicNotchPlan(
        (notch.HarmonicStopband((2,), 19.95, 20.05),),
        transition_bandwidth_hz=0.2,
    )
    design = notch.characterize_harmonic_filter(500.0, plan)
    row = {**notch.harmonic_exclusion_rows("recording", plan, ())[0]}
    row.update(design.manifest_fields())
    row["fir_filter_length_samples"] = int(row["fir_filter_length_samples"]) + 2

    with pytest.raises(ValueError, match="filter design"):
        notch._validate_filter_design((row,), design)


def test_verification_rejects_a_line_without_channel_level_significance():
    model = _supported_model((20.0, 2), (40.0, 4))
    plans = notch.plan_channel_notches(model, _settings())
    rows = notch.line_manifest_rows(
        "recording",
        model,
        plans,
        (),
        _settings(),
    )
    rows[0]["detected_line_corrected_p_values"] = "0.2"

    with pytest.raises(ValueError, match="statistically supported"):
        notch._validate_manifest_evidence(rows, _settings())


def test_derivative_description_records_computed_source_and_derived_method(tmp_path):
    staging = tmp_path / ".staging"
    staging.mkdir()
    description = staging / "dataset_description.json"
    description.write_text(
        json.dumps({"Name": "source", "BIDSVersion": "1.10.0"}),
        encoding="utf-8",
    )

    notch.write_harmonic_derivative_description(staging, "../source", _settings())

    written = json.loads(description.read_text(encoding="utf-8"))
    assert written["SourceDatasets"] == [{"URL": "../source"}]
    generated_description = written["GeneratedBy"][-1]["Description"]
    assert "pre-allocated" in generated_description
    assert "controlled adaptive removal rounds" not in generated_description
    parameters = written["GeneratedBy"][-1]["Parameters"]
    assert parameters["multiple_testing_method"] == notch.MULTIPLE_TESTING_METHOD
    assert parameters["familywise_error_unit"] == (
        "as_recorded_non_bad_eeg_recording_removal_sequence"
    )
    assert parameters["detection_reference"] == (
        "as_recorded_non_bad_eeg_channels"
    )
    assert parameters["filter_scope"] == "all_eeg_channels"
    assert parameters["convergence_rule"] == (
        "fresh_joint_line_and_scanner_harmonics_null"
    )
    assert parameters["multiple_testing_scope"] == (
        "recording_wide_alpha_spending_split_equally_between_test_families"
    )
    assert parameters["alpha_spending_rule"] == "alpha / (round * (round + 1))"
    assert parameters["estimation_window_s"] == 10.0
    assert parameters["scanner_harmonics_estimation_window_s"] == 4.0
    assert parameters["scanner_harmonics_local_supporting_harmonics"] == 1
    assert parameters["scanner_harmonics_complete_comb_supporting_harmonics"] == 2
    assert parameters["scanner_harmonics_target_rule"] == (
        "one_supported_harmonic_targets_its_tooth_two_target_complete_comb"
    )
    assert parameters["supported_scanner_harmonic_stopband_width_hz"] == 2.25
    assert parameters["scanner_repetition_time_s"] == 0.9
    assert parameters["scanner_trigger_event_name"] == "Volume/V  1"
    assert parameters["ordinary_line_stopband_width_hz"] == 0.25
    assert parameters["filter_resolution_window_s"] == 54.0
    assert parameters["transition_bandwidth_hz"] == pytest.approx(3.3 / 54.0)
    assert parameters["per_edge_transition_bandwidth_hz"] == pytest.approx(3.3 / 108.0)
    assert parameters["filter_length"] == "auto"
    assert parameters["fir_window"] == "hamming"
    assert parameters["fir_design"] == "firwin"
    assert parameters["pad"] == "reflect_limited"


def test_verification_accepts_the_settings_recorded_during_apply(tmp_path):
    derivative_root = tmp_path / "derivative"
    derivative_root.mkdir()
    (derivative_root / "dataset_description.json").write_text(
        json.dumps({"Name": "source", "BIDSVersion": "1.10.0"}),
        encoding="utf-8",
    )
    applied = _settings()
    notch.write_harmonic_derivative_description(derivative_root, "../source", applied)

    verified = notch.settings_for_verification(derivative_root, applied)

    assert verified == applied


def test_verification_reconstructs_nondefault_scanner_settings(tmp_path):
    derivative_root = tmp_path / "derivative"
    derivative_root.mkdir()
    (derivative_root / "dataset_description.json").write_text(
        json.dumps({"Name": "source", "BIDSVersion": "1.10.0"}),
        encoding="utf-8",
    )
    applied = notch.HarmonicNotchSettings(
        estimation_window_s=10.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(0.0, 100.0),
        scanner_repetition_time_s=1.2,
        scanner_trigger_event_name="MRI/Volume",
    )
    notch.write_harmonic_derivative_description(derivative_root, "../source", applied)

    verified = notch.settings_for_verification(derivative_root, applied)

    assert verified.scanner_repetition_time_s == 1.2
    assert verified.scanner_trigger_event_name == "MRI/Volume"


def test_verification_refuses_settings_changed_after_apply(tmp_path):
    derivative_root = tmp_path / "derivative"
    derivative_root.mkdir()
    (derivative_root / "dataset_description.json").write_text(
        json.dumps({"Name": "source", "BIDSVersion": "1.10.0"}),
        encoding="utf-8",
    )
    applied = _settings()
    current = notch.HarmonicNotchSettings(
        estimation_window_s=108.0,
        familywise_error_rate=applied.familywise_error_rate,
        frequency_range_hz=applied.frequency_range_hz,
    )
    notch.write_harmonic_derivative_description(derivative_root, "../source", applied)

    with pytest.raises(ValueError, match="estimation_window_s.*apply"):
        notch.settings_for_verification(derivative_root, current)


def test_verification_refuses_changed_filter_provenance(tmp_path):
    derivative_root = tmp_path / "derivative"
    derivative_root.mkdir()
    description_path = derivative_root / "dataset_description.json"
    description_path.write_text(
        json.dumps({"Name": "source", "BIDSVersion": "1.10.0"}),
        encoding="utf-8",
    )
    settings = _settings()
    notch.write_harmonic_derivative_description(
        derivative_root,
        "../source",
        settings,
    )
    description = json.loads(description_path.read_text(encoding="utf-8"))
    description["GeneratedBy"][-1]["Parameters"]["fir_window"] = "hann"
    description_path.write_text(json.dumps(description), encoding="utf-8")

    with pytest.raises(ValueError, match="filter provenance"):
        notch.settings_for_verification(derivative_root, settings)


def test_public_pipeline_has_one_automatic_correction_stage():
    from decomb import cli

    assert cli.STAGES == ("diagnose", "apply", "verify", "psd")


def test_comb_fundamental_defaults_to_one_over_the_tr():
    settings = notch.HarmonicNotchSettings(
        estimation_window_s=10.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(0.0, 100.0),
        scanner_repetition_time_s=0.9,
    )

    assert settings.comb_fundamental_hz is None
    assert settings.comb_fundamental == pytest.approx(1.0 / 0.9)


def test_declared_comb_fundamental_replaces_the_tr_derived_grid():
    """A cold head at 72 cycles per minute is 1.2 Hz, not the 1.1111 Hz volume rate."""
    settings = notch.HarmonicNotchSettings(
        estimation_window_s=10.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(0.0, 100.0),
        scanner_repetition_time_s=0.9,
        comb_fundamental_hz=1.2,
    )

    assert settings.comb_fundamental == pytest.approx(1.2)


@pytest.mark.parametrize("value", [0.0, -1.2, float("nan"), float("inf")])
def test_invalid_declared_comb_fundamental_is_refused(value):
    with pytest.raises(ValueError, match="comb_fundamental_hz"):
        notch.HarmonicNotchSettings(
            estimation_window_s=10.0,
            familywise_error_rate=0.05,
            frequency_range_hz=(0.0, 100.0),
            comb_fundamental_hz=value,
        )


def test_declared_fundamental_is_used_after_the_trigger_check_still_passes(monkeypatch):
    """The declared grid replaces the TR grid; the trigger timing check still runs."""
    import mne

    raw = mne.io.RawArray(
        np.zeros((2, 4_000)),
        mne.create_info(["Cz", "C3"], 1_000.0, "eeg"),
        verbose="ERROR",
    )
    raw.set_annotations(
        mne.Annotations(onset=[0.0, 0.9, 1.8], duration=[0.0] * 3,
                        description=["Volume/V  1"] * 3)
    )
    settings = notch.HarmonicNotchSettings(
        estimation_window_s=10.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(0.0, 100.0),
        scanner_repetition_time_s=0.9,
        comb_fundamental_hz=1.2,
    )

    assert notch.scanner_fundamental_hz(raw, settings) == pytest.approx(1.2)

    mistimed = notch.HarmonicNotchSettings(
        estimation_window_s=10.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(0.0, 100.0),
        scanner_repetition_time_s=0.8,
        comb_fundamental_hz=1.2,
    )
    with pytest.raises(ValueError, match="TR within half a sample"):
        notch.scanner_fundamental_hz(raw, mistimed)
