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
) -> lines.ArtifactModel:
    artifact_lines = tuple(
        lines.ArtifactLine(
            position_hz=position_hz,
            raw_p_value=1e-15,
            corrected_p_value=1e-12,
            window_indices=(0,),
            harmonic=harmonic,
        )
        for position_hz, harmonic in line_definitions
    )
    harmonics = [harmonic for _, harmonic in line_definitions if harmonic is not None]
    return lines.ArtifactModel(
        channels=(
            lines.ChannelArtifactModel(
                channel_index=0,
                channel_name="Cz",
                lines=artifact_lines,
                fundamental_hz=10.0 if harmonics else None,
                comb_corrected_p_value=1e-10 if harmonics else None,
            ),
        ),
        window_count=3,
        channel_count=1,
        test_count_per_channel=100_000,
    )


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
    data += 20e-6 * np.sin(2.0 * np.pi * 60.0 * times_s)
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
    assert any(
        stopband.low_hz <= 60.0 <= stopband.high_hz
        for stopband in plans[0].geometry.stopbands
    )


@pytest.mark.parametrize("correction", ["holm", "bonferroni", "none"])
def test_fit_harmonic_model_accepts_every_correction_procedure(correction):
    # A narrow tested band keeps uncorrected detection's nominal false-positive count
    # (no family-wise control by design) small enough for classify_harmonics's
    # candidate-fundamental search to stay fast; see the comment on the uncorrected
    # permissiveness test below for why a wide band is not safe here.
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

    model = notch.fit_harmonic_model(raw, narrow_settings, correction=correction)

    assert "C3" in [channel.channel_name for channel in model.channels]
    c3 = next(channel for channel in model.channels if channel.channel_name == "C3")
    assert any(abs(line.position_hz - 5.0) < 0.02 for line in c3.lines)


def test_uncorrected_fit_is_at_least_as_permissive_as_holm():
    # A narrow tested band, not the packaged 0-100 Hz default: uncorrected detection has
    # no false-positive control, so a wide band with many tested bins gives it thousands
    # of nominal detections and drives classify_harmonics's candidate-fundamental search
    # (which scales with detection count) into minutes-long runtime for pure noise. That
    # blowup is real and belongs to the ablation study, not to this plumbing test.
    narrow_settings = notch.HarmonicNotchSettings(
        estimation_window_s=54.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(1.0, 10.0),
    )
    sampling_frequency_hz = 250.0
    times_s = np.arange(int(120.0 * sampling_frequency_hz)) / sampling_frequency_hz
    data = np.random.default_rng(23).normal(scale=1e-6, size=(4, times_s.size))
    raw = mne.io.RawArray(
        data,
        mne.create_info(["C3", "C4", "P3", "P4"], sampling_frequency_hz, "eeg"),
        verbose="ERROR",
    )

    holm_model = notch.fit_harmonic_model(raw, narrow_settings, correction="holm")
    uncorrected_model = notch.fit_harmonic_model(raw, narrow_settings, correction="none")

    holm_lines = sum(len(channel.lines) for channel in holm_model.channels)
    uncorrected_lines = sum(len(channel.lines) for channel in uncorrected_model.channels)
    assert uncorrected_lines >= holm_lines


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

    rows = notch.artifact_manifest_rows(
        "recording",
        model,
        plans,
        (),
        _settings(),
    )

    assert {row["channel"] for row in rows} == {"C3"}
    assert all(row["multiple_testing_method"] == "holm" for row in rows)
    assert all(row["detection_test_count_per_channel"] > 0 for row in rows)
    assert all(row["detected_line_raw_p_values"] for row in rows)


def test_plan_contains_every_authorized_harmonic_and_isolated_line():
    model = _supported_model((20.0, 2), (30.0, 3), (35.5, None), (40.0, 4))

    plan = notch.plan_harmonic_stopbands(model.channels[0], _settings())

    planned_harmonics = {
        harmonic for stopband in plan.stopbands for harmonic in stopband.harmonics
    }
    assert planned_harmonics == {2, 3, 4}
    assert sum(stopband.kind == "isolated" for stopband in plan.stopbands) == 1
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


def test_stationary_interval_has_the_hann_half_power_width():
    model = _supported_model((20.0, 2))

    plan = notch.plan_harmonic_stopbands(model.channels[0], _settings())

    assert plan.stopbands[0].width_hz == pytest.approx(_settings().spectral_resolution_hz)


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
    model = lines.ArtifactModel(
        channels=(),
        window_count=3,
        channel_count=2,
        test_count_per_channel=1_000,
    )

    plans = notch.plan_channel_notches(model, _settings())
    filtered = notch.apply_channel_notches(raw, plans)
    rows = notch.artifact_manifest_rows(
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
    assert rows[0]["outcome"] == "no_artifact_detected"
    assert notch.channel_plans_from_rows(rows) == ()
    notch._validate_manifest_evidence(rows, _settings())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("detected_line_corrected_p_values", "0"),
        ("tested_eeg_channel_count", 2),
        ("detected_line_harmonics", "999"),
        ("harmonics", "999"),
    ),
)
def test_refitted_evidence_rejects_tampered_authorization(field, value):
    model = _supported_model((20.0, 2))
    plans = notch.plan_channel_notches(model, _settings())
    expected = notch.artifact_manifest_rows(
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

    rows = notch.artifact_manifest_rows("recording", model, plans, (), _settings())

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
        transition_bandwidth_hz=0.2,
    )
    plans = (notch.ChannelNotchPlan("Cz", geometry),)

    with pytest.raises(RuntimeError, match="does not equal the declared FIR derivative"):
        notch._validate_exact_derivative(raw, raw.copy(), vhdr, plans)


def test_verification_accepts_the_exact_quantized_filter_result(brainvision_run):
    vhdr, raw = brainvision_run
    geometry = notch.HarmonicNotchPlan(
        (notch.HarmonicStopband((2,), 19.95, 20.05),),
        transition_bandwidth_hz=0.2,
    )
    plans = (notch.ChannelNotchPlan("Cz", geometry),)
    filtered = notch.apply_channel_notches(raw, plans)
    quantized = recordings.quantized_eeg_data(
        vhdr,
        filtered.get_data(),
        filtered.ch_names,
    )
    written = filtered.copy()
    written._data = quantized

    assert notch._validate_exact_derivative(raw, written, vhdr, plans) == 0.0


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
    rows = notch.artifact_manifest_rows(
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
    parameters = written["GeneratedBy"][-1]["Parameters"]
    assert parameters["multiple_testing_method"] == "holm"
    assert parameters["familywise_error_unit"] == "eeg_channel"
    assert parameters["filter_scope"] == "statistically_supported_channels"
    assert parameters["estimation_window_s"] == 54.0
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
