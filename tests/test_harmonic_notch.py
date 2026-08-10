"""Automatic, evidence-bounded harmonic notch planning."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd
import pytest

from decomb import harmonics, notch, recordings
from decomb.config import load_config


def _settings(**overrides) -> notch.HarmonicNotchSettings:
    return replace(
        notch.HarmonicNotchSettings.from_config(load_config()),
        **overrides,
    )


def _estimate(
    positions: dict[int, float],
    *,
    fundamental_hz: float = 10.0,
    standard_error_hz: float = 0.001,
) -> harmonics.CombEstimate:
    harmonic_numbers = tuple(sorted(positions))
    return harmonics.CombEstimate(
        fundamental_hz=fundamental_hz,
        fitted_harmonics=harmonic_numbers,
        fitted_positions_hz=tuple(positions[harmonic] for harmonic in harmonic_numbers),
        supported_harmonics=harmonic_numbers,
        supported_positions_hz=tuple(positions[harmonic] for harmonic in harmonic_numbers),
        residual_rms_hz=0.0,
        max_abs_residual_hz=0.0,
        fundamental_jackknife_se_hz=standard_error_hz,
    )


def _model() -> harmonics.AdaptiveCombModel:
    whole = _estimate({2: 20.00, 3: 30.00, 4: 40.00})
    windows = (
        harmonics.HarmonicEvidence(
            (2, 3, 4),
            (19.98, 30.01, 40.01),
        ),
        harmonics.HarmonicEvidence((2, 3), (20.03, 29.99)),
    )
    return harmonics.AdaptiveCombModel(whole_estimate=whole, window_evidence=windows)


def test_plan_notches_only_supported_harmonics_in_the_requested_range():
    settings = _settings(
        estimation_window_s=54.0,
        harmonic_range=(2, 4),
        removal_harmonic_range=(2, 3),
        low_hz=1.0,
        high_hz=45.0,
    )

    plan = notch.plan_harmonic_stopbands(_model(), settings)

    assert [band.harmonics for band in plan.stopbands] == [(2,), (3,)]
    assert plan.transition_bandwidth_hz == pytest.approx(
        settings.transition_bandwidth_resolutions * settings.spectral_resolution_hz
    )


def test_stopband_covers_every_observed_position_and_its_uncertainty():
    settings = _settings(
        estimation_window_s=54.0,
        harmonic_range=(2, 4),
        removal_harmonic_range=(2, 3),
        low_hz=1.0,
        high_hz=45.0,
        uncertainty_confidence_z=2.0,
    )

    plan = notch.plan_harmonic_stopbands(_model(), settings)
    second = plan.stopbands[0]

    assert second.low_hz <= 19.98
    assert second.low_hz <= 20.00 - 2.0 * 2 * 0.001
    assert second.high_hz >= 20.03


def test_stationary_stopband_has_a_resolution_defined_minimum_width():
    settings = _settings(
        estimation_window_s=54.0,
        harmonic_range=(2, 4),
        removal_harmonic_range=(2, 3),
        low_hz=1.0,
        high_hz=45.0,
    )

    plan = notch.plan_harmonic_stopbands(_model(), settings)
    third = plan.stopbands[1]

    assert third.width_hz == pytest.approx(settings.minimum_stopband_width_hz)


def test_overlapping_harmonic_intervals_are_merged():
    whole = _estimate({2: 20.00, 3: 20.06})
    model = harmonics.AdaptiveCombModel(
        whole_estimate=whole,
        window_evidence=(harmonics.HarmonicEvidence((2, 3), (20.00, 20.06)),),
    )
    settings = _settings(
        estimation_window_s=54.0,
        harmonic_range=(2, 3),
        removal_harmonic_range=(2, 3),
        low_hz=1.0,
        high_hz=45.0,
    )

    plan = notch.plan_harmonic_stopbands(model, settings)

    assert len(plan.stopbands) == 1
    assert plan.stopbands[0].harmonics == (2, 3)


def _tone_amplitude(data: np.ndarray, frequency_hz: float, sfreq: float) -> float:
    times = np.arange(data.size) / sfreq
    basis = np.column_stack(
        (
            np.sin(2.0 * np.pi * frequency_hz * times),
            np.cos(2.0 * np.pi * frequency_hz * times),
        )
    )
    coefficients, *_ = np.linalg.lstsq(basis, data, rcond=None)
    return float(np.linalg.norm(coefficients))


def test_apply_harmonic_notches_suppresses_only_the_planned_interval():
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
        stopbands=(notch.HarmonicStopband((2,), 19.95, 20.05),),
        transition_bandwidth_hz=0.2,
    )

    filtered = notch.apply_harmonic_notches(raw, plan, filter_jobs=1)

    interior = slice(int(20.0 * sfreq), int(60.0 * sfreq))
    before = raw.get_data()[0, interior]
    after = filtered.get_data()[0, interior]
    assert _tone_amplitude(after, 20.0, sfreq) < _tone_amplitude(before, 20.0, sfreq) / 100.0
    assert _tone_amplitude(after, 17.0, sfreq) == pytest.approx(
        _tone_amplitude(before, 17.0, sfreq),
        rel=0.01,
    )


def test_exclusion_rows_report_transitions_and_total_band_availability():
    plan = notch.HarmonicNotchPlan(
        stopbands=(
            notch.HarmonicStopband((2,), 19.95, 20.05),
            notch.HarmonicStopband((3,), 29.95, 30.05),
        ),
        transition_bandwidth_hz=0.2,
    )
    analysed_bands = (("beta", 13.0, 30.0), ("gamma", 30.0, 45.0))

    rows = notch.harmonic_exclusion_rows("sub-01_task-rest", plan, analysed_bands)

    assert rows[0]["unavailable_low_hz"] == pytest.approx(19.85)
    assert rows[0]["unavailable_high_hz"] == pytest.approx(20.15)
    assert rows[0]["harmonics"] == "2"
    assert rows[1]["harmonics"] == "3"
    expected_beta_share = (0.30 + 0.15) / 17.0
    expected_gamma_share = 0.15 / 15.0
    assert rows[0]["beta_unavailable_share"] == pytest.approx(expected_beta_share)
    assert rows[0]["beta_retained_share"] == pytest.approx(1.0 - expected_beta_share)
    assert rows[0]["gamma_unavailable_share"] == pytest.approx(expected_gamma_share)
    assert rows[0]["gamma_retained_share"] == pytest.approx(1.0 - expected_gamma_share)


def test_only_the_whole_recording_authorizes_the_comb(monkeypatch):
    whole = object()
    windows = (object(), object())
    spectra = SimpleNamespace(whole=whole, windows=windows)
    whole_estimate = _estimate({2: 20.0, 3: 30.0, 4: 40.0})
    evidence = {
        id(windows[0]): harmonics.HarmonicEvidence(
            (2, 3, 4),
            (19.99, 29.99, 39.99),
        ),
        id(windows[1]): harmonics.HarmonicEvidence((2, 3), (20.01, 30.01)),
    }
    fit_calls = []
    localization_calls = []

    monkeypatch.setattr(
        notch.recordings,
        "session_run_spectra",
        lambda raw, settings: spectra,
    )

    def estimate(spectrum, settings):
        fit_calls.append(spectrum)
        return whole_estimate

    def localize(spectrum, settings, supported_harmonics, fundamental_hz):
        localization_calls.append((spectrum, supported_harmonics, fundamental_hz))
        return evidence[id(spectrum)]

    monkeypatch.setattr(notch, "_estimate_comb_spectrum", estimate)
    monkeypatch.setattr(notch, "_localize_window_evidence", localize)

    model = notch.fit_harmonic_model(
        object(),
        _settings(
            harmonic_range=(2, 4),
            removal_harmonic_range=(2, 4),
            min_harmonics_for_fit=3,
        ),
    )

    assert model.whole_estimate is whole_estimate
    assert model.window_evidence == tuple(evidence[id(window)] for window in windows)
    assert fit_calls == [whole]
    assert localization_calls == [
        (window, whole_estimate.supported_harmonics, whole_estimate.fundamental_hz)
        for window in windows
    ]


def test_public_pipeline_has_one_automatic_correction_stage():
    from decomb import cli

    assert cli.STAGES == ("diagnose", "apply", "verify", "psd")


def test_packaged_notch_settings_use_the_validated_54_second_geometry():
    settings = notch.HarmonicNotchSettings.from_config(load_config())

    assert settings.estimation_window_s == 54.0
    assert settings.spectral_resolution_hz == pytest.approx(1.4382 / 54.0)
    assert settings.transition_bandwidth_hz == pytest.approx(
        settings.transition_bandwidth_resolutions * 1.4382 / 54.0
    )


def test_apply_routes_to_automatic_harmonic_notching(monkeypatch):
    import argparse

    from decomb import cli

    called = []
    monkeypatch.setattr(notch, "run", called.append)
    args = argparse.Namespace(stage="apply", subjects=None)

    cli.run_stage(args)

    assert called == [args]


def test_verify_routes_to_harmonic_notch_audit(monkeypatch):
    import argparse

    from decomb import cli

    called = []
    monkeypatch.setattr(notch, "run_verify", called.append, raising=False)
    args = argparse.Namespace(stage="verify", subjects=None)

    cli.run_stage(args)

    assert called == [args]


def test_manifest_rows_reconstruct_the_exact_filter_plan():
    plan = notch.HarmonicNotchPlan(
        stopbands=(
            notch.HarmonicStopband((2,), 19.95, 20.05),
            notch.HarmonicStopband((3, 4), 29.95, 40.05),
        ),
        transition_bandwidth_hz=0.2,
    )
    rows = notch.harmonic_exclusion_rows("recording", plan, ())

    reconstructed = notch.harmonic_plan_from_rows(rows)

    assert reconstructed == plan


def test_manifest_tsv_round_trips_the_exact_filter_plan(tmp_path):
    low_hz = np.nextafter(19.95, np.inf)
    high_hz = np.nextafter(20.05, -np.inf)
    transition_hz = np.nextafter(0.2, np.inf)
    plan = notch.HarmonicNotchPlan(
        stopbands=(notch.HarmonicStopband((2,), low_hz, high_hz),),
        transition_bandwidth_hz=transition_hz,
    )
    path = tmp_path / "harmonic_notch_manifest.tsv"
    recordings.write_tsv_atomic(
        pd.DataFrame(notch.harmonic_exclusion_rows("recording", plan, ())),
        path,
    )

    stored = pd.read_csv(path, sep="\t", float_precision="round_trip")
    reconstructed = notch.harmonic_plan_from_rows(stored.to_dict("records"))

    assert reconstructed == plan


def test_source_dataset_url_is_relative_to_the_published_derivative(tmp_path):
    source_root = tmp_path / "inputs" / "study" / "bids"
    derivative_root = tmp_path / "published" / "derivatives" / "decomb"

    url = notch.relative_source_dataset_url(source_root, derivative_root)

    assert url == "../../../inputs/study/bids"


def test_derivative_description_records_the_computed_source_url(tmp_path):
    staging_root = tmp_path / ".decomb.staging"
    staging_root.mkdir()
    description = staging_root / "dataset_description.json"
    description.write_text(
        json.dumps({"Name": "source", "BIDSVersion": "1.10.0"}),
        encoding="utf-8",
    )

    notch.write_harmonic_derivative_description(
        staging_root,
        "../../../inputs/study/bids",
        _settings(),
    )

    written = json.loads(description.read_text(encoding="utf-8"))
    assert written["SourceDatasets"] == [{"URL": "../../../inputs/study/bids"}]


def test_clean_run_writes_the_filtered_binary_and_exclusion_rows(tmp_path, monkeypatch):
    sfreq = 200.0
    times = np.arange(int(80.0 * sfreq)) / sfreq
    data = np.sin(2.0 * np.pi * 20.0 * times)[np.newaxis, :]
    raw = mne.io.RawArray(
        data,
        mne.create_info(["EEG 001"], sfreq, ch_types="eeg"),
        verbose="ERROR",
    )
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    vhdr = source_root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.vhdr"
    written = []

    def read(path):
        if path == vhdr:
            return raw
        return mne.io.RawArray(
            written[0],
            raw.info.copy(),
            verbose="ERROR",
        )

    def write(_source, destination, values):
        destination.parent.mkdir(parents=True, exist_ok=True)
        written.append(np.asarray(values, dtype=float))

    monkeypatch.setattr(notch.recordings, "read_bids_raw", read)
    monkeypatch.setattr(notch.recordings, "write_eeg_binary", write)
    monkeypatch.setattr(notch, "fit_harmonic_model", lambda _raw, _settings: _model())
    settings = _settings(
        harmonic_range=(2, 4),
        removal_harmonic_range=(2, 2),
        low_hz=1.0,
        high_hz=45.0,
        filter_jobs=1,
    )

    rows = notch.clean_harmonic_run(
        vhdr,
        output_root,
        source_root,
        settings,
        (("beta", 13.0, 30.0),),
    )

    assert len(written) == 1
    assert rows[0]["recording"] == vhdr.stem
    assert rows[0]["harmonics"] == "2"
    assert rows[0]["in_stopband_change_db"] < -20.0
    assert rows[0]["fundamental_hz"] == pytest.approx(10.0)


def test_verify_computes_the_cleaned_spectrum_once_per_recording(monkeypatch):
    original = mne.io.RawArray(
        np.zeros((1, 1000)),
        mne.create_info(["EEG 001"], 100.0, ch_types="eeg"),
        verbose="ERROR",
    )
    cleaned = original.copy()
    plan = notch.HarmonicNotchPlan(
        stopbands=(
            notch.HarmonicStopband((2,), 19.95, 20.05),
            notch.HarmonicStopband((3,), 29.95, 30.05),
        ),
        transition_bandwidth_hz=0.2,
    )
    rows = notch.harmonic_exclusion_rows("recording", plan, ())
    calls = []

    monkeypatch.setattr(
        notch.recordings,
        "read_bids_raw",
        lambda path: original if path.name == "source.vhdr" else cleaned,
    )
    monkeypatch.setattr(notch, "_measure_stopband_changes", lambda *args: (-30.0, -31.0))

    def spectrum(_raw, _settings):
        calls.append(_raw)
        frequencies_hz = np.arange(0.0, 50.0, 0.01)
        return frequencies_hz, np.zeros_like(frequencies_hz), np.zeros_like(frequencies_hz)

    monkeypatch.setattr(notch.recordings, "run_spectrum", spectrum)

    notch.verify_harmonic_run(
        Path("source.vhdr"),
        Path("cleaned.vhdr"),
        rows,
        _settings(),
    )

    assert calls == [original, cleaned]
