"""Automatic comb and isolated-line notch planning."""

from __future__ import annotations

import json

import mne
import numpy as np
import pandas as pd
import pytest

from decomb import harmonics, notch, recordings
from decomb.config import load_config


def _settings() -> notch.HarmonicNotchSettings:
    return notch.HarmonicNotchSettings.from_config(load_config())


def _estimate(positions: dict[int, float], *, fundamental_hz: float = 10.0):
    harmonic_numbers = tuple(sorted(positions))
    return harmonics.CombEstimate(
        fundamental_hz=fundamental_hz,
        harmonics=harmonic_numbers,
        positions_hz=tuple(positions[number] for number in harmonic_numbers),
        evidence_bic=-20.0,
    )


def _model(*, isolated_positions=()) -> harmonics.AdaptiveCombModel:
    whole = _estimate({2: 20.00, 3: 30.00, 4: 40.00})
    windows = (
        harmonics.HarmonicEvidence((2, 3, 4), (19.98, 30.01, 40.01)),
        harmonics.HarmonicEvidence((2, 3, 4), (20.03, 29.99, 40.00)),
    )
    isolated = harmonics.IsolatedLineModel(
        tuple(isolated_positions),
        tuple(-10.0 for _ in isolated_positions),
        tuple(tuple(isolated_positions) for _ in windows),
    )
    return harmonics.AdaptiveCombModel(whole, windows, isolated)


def test_plan_contains_every_authorized_harmonic_and_isolated_line():
    plan = notch.plan_harmonic_stopbands(_model(isolated_positions=(35.5,)), _settings())

    planned_harmonics = {
        harmonic for stopband in plan.stopbands for harmonic in stopband.harmonics
    }
    assert planned_harmonics == {2, 3, 4}
    assert sum(stopband.kind == "isolated" for stopband in plan.stopbands) == 1
    assert plan.transition_bandwidth_hz == pytest.approx(3.3 / 54.0)


def test_stopband_covers_every_observed_position_and_bin_uncertainty():
    plan = notch.plan_harmonic_stopbands(_model(), _settings())
    second = next(stopband for stopband in plan.stopbands if stopband.harmonics == (2,))

    assert second.low_hz < 19.98
    assert second.high_hz > 20.03


def test_stationary_interval_has_the_hann_half_power_width():
    whole = _estimate({2: 20.0})
    evidence = (harmonics.HarmonicEvidence((2,), (20.0,)),)
    model = harmonics.AdaptiveCombModel(
        whole,
        evidence,
        harmonics.IsolatedLineModel((), (), ((),)),
    )

    plan = notch.plan_harmonic_stopbands(model, _settings())

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
    assert parameters["estimation_window_s"] == 54.0
    assert parameters["transition_bandwidth_hz"] == pytest.approx(3.3 / 54.0)


def test_public_pipeline_has_one_automatic_correction_stage():
    from decomb import cli

    assert cli.STAGES == ("diagnose", "apply", "verify", "psd")
