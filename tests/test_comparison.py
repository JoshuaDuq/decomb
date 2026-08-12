"""Shared-input ablation of two MNE FIR notch geometries."""

from __future__ import annotations

import numpy as np
import pytest

from decomb import comparison, harmonics, notch
from decomb.config import load_config


def _model() -> harmonics.AdaptiveCombModel:
    whole = harmonics.CombEstimate(
        fundamental_hz=10.0,
        harmonics=(2, 3, 4),
        positions_hz=(20.0, 30.0, 40.0),
        evidence_bic=-20.0,
    )
    windows = (
        harmonics.HarmonicEvidence((2, 3, 4), (19.98, 30.01, 40.01)),
        harmonics.HarmonicEvidence((2, 3, 4), (20.03, 29.99, 40.00)),
    )
    isolated = harmonics.IsolatedLineModel(
        (35.5,),
        (-10.0,),
        ((35.49,), (35.52,)),
    )
    return harmonics.AdaptiveCombModel(whole, windows, isolated)


def _settings() -> notch.HarmonicNotchSettings:
    return notch.HarmonicNotchSettings.from_config(load_config())


def test_merged_mne_default_plan_uses_the_same_centres_and_default_parameters():
    model = _model()
    settings = _settings()
    measured_intervals = notch.observed_line_intervals(model, settings)
    plan = comparison.merged_mne_default_plan(model, settings)

    measured_centres_hz = sorted(band.centre_hz for band in measured_intervals)
    assert [band.centre_hz for band in plan.stopbands] == pytest.approx(measured_centres_hz)
    assert [band.width_hz for band in plan.stopbands] == pytest.approx(
        [frequency_hz / 200.0 for frequency_hz in measured_centres_hz]
    )
    assert plan.transition_bandwidth_hz == 1.0


def test_merged_mne_default_plan_combines_overlapping_transitions():
    whole = harmonics.CombEstimate(
        fundamental_hz=0.5,
        harmonics=(40, 41),
        positions_hz=(20.0, 20.5),
        evidence_bic=-20.0,
    )
    windows = (harmonics.HarmonicEvidence((40, 41), (20.0, 20.5)),) * 2
    model = harmonics.AdaptiveCombModel(
        whole,
        windows,
        harmonics.IsolatedLineModel((), (), ((), ())),
    )

    plan = comparison.merged_mne_default_plan(model, _settings())

    assert len(plan.stopbands) == 1
    assert plan.stopbands[0].harmonics == (40, 41)


def test_sparse_mne_defaults_equal_the_overlap_merged_geometry():
    import mne

    sampling_frequency_hz = 250.0
    data = np.random.default_rng(7).normal(size=(1, 5000))
    raw = mne.io.RawArray(
        data,
        mne.create_info(["Cz"], sampling_frequency_hz, "eeg"),
        verbose="ERROR",
    )

    literal = comparison.apply_literal_mne_defaults(raw, _model(), _settings())
    merged_plan = comparison.merged_mne_default_plan(_model(), _settings())
    merged = notch.apply_harmonic_notches(raw, merged_plan)

    np.testing.assert_allclose(literal.get_data(), merged.get_data())


def test_literal_dense_mne_defaults_surface_the_design_error():
    import mne

    whole = harmonics.CombEstimate(
        fundamental_hz=0.5,
        harmonics=(40, 41),
        positions_hz=(20.0, 20.5),
        evidence_bic=-20.0,
    )
    windows = (harmonics.HarmonicEvidence((40, 41), (20.0, 20.5)),) * 2
    model = harmonics.AdaptiveCombModel(
        whole,
        windows,
        harmonics.IsolatedLineModel((), (), ((), ())),
    )
    raw = mne.io.RawArray(
        np.zeros((1, 5000)),
        mne.create_info(["Cz"], 250.0, "eeg"),
        verbose="ERROR",
    )

    stopbands = comparison.mne_default_parameter_stopbands(model, _settings())
    half_transition_hz = comparison.MNE_DEFAULT_TRANSITION_BANDWIDTH_HZ / 2.0
    assert stopbands[0].high_hz + half_transition_hz >= (
        stopbands[1].low_hz - half_transition_hz
    )
    with pytest.raises(ValueError):
        comparison.apply_literal_mne_defaults(raw, model, _settings())


def test_measured_geometry_preserves_more_band_than_merged_mne_defaults():
    measured = notch.plan_harmonic_stopbands(_model(), _settings())
    merged_default = comparison.merged_mne_default_plan(_model(), _settings())

    assert comparison.unavailable_width_hz(measured, (1.0, 100.0)) < (
        comparison.unavailable_width_hz(merged_default, (1.0, 100.0))
    )


def test_arm_summary_reports_frequency_and_fir_duration():
    summary = comparison._arm_summary(
        available_width_hz=89.5,
        analysed_width_hz=100.0,
        filter_design=notch.HarmonicFilterDesign(27001, 108.004, 50.0, 0.02),
    )

    assert summary == "89.5 Hz of 100 Hz available · 108.0 s FIR"


def test_geometry_description_names_stopbands_plus_transitions_as_unavailable():
    plan = notch.HarmonicNotchPlan(
        (notch.HarmonicStopband((2,), 19.95, 20.05),),
        transition_bandwidth_hz=1.0,
    )

    _, description = comparison._window_geometry(plan, (19.0, 21.0))

    assert description == "1 unavailable interval of 1.10 Hz"


def test_real_geometry_ablation_figure_reports_both_geometries(tmp_path):
    measured = notch.plan_harmonic_stopbands(_model(), _settings())
    merged_default = comparison.merged_mne_default_plan(_model(), _settings())
    frequencies_hz = np.arange(1.0, 100.0, 0.02)
    source = 1e-12 / frequencies_hz
    result = comparison.MneFirGeometryAblation(
        frequencies_hz=frequencies_hz,
        source_psd=source,
        decomb_psd=source.copy(),
        merged_mne_default_psd=source.copy(),
        decomb_plan=measured,
        merged_mne_default_plan=merged_default,
        decomb_filter=notch.characterize_harmonic_filter(250.0, measured),
        merged_mne_default_filter=notch.characterize_harmonic_filter(
            250.0,
            merged_default,
        ),
        duration_s=120.0,
    )
    output = tmp_path / "notch-comparison.png"

    comparison.figure_mne_fir_geometry_ablation(
        result,
        output,
        recording_description="one real recording",
    )

    assert output.is_file() and output.stat().st_size > 0


def test_both_arms_are_measured_on_the_same_real_samples(monkeypatch):
    import mne

    sampling_frequency_hz = 250.0
    times = np.arange(int(120.0 * sampling_frequency_hz)) / sampling_frequency_hz
    data = np.random.default_rng(4).normal(scale=1e-6, size=(2, times.size))
    data += 4e-6 * np.sin(2.0 * np.pi * 20.0 * times)
    raw = mne.io.RawArray(
        data,
        mne.create_info(["Cz", "Pz"], sampling_frequency_hz, "eeg"),
        verbose="ERROR",
    )
    monkeypatch.setattr(comparison.notch, "fit_harmonic_model", lambda raw, settings: _model())

    result = comparison.measure_mne_fir_geometry_ablation(raw, _settings())

    assert result.source_psd.shape == result.decomb_psd.shape
    assert result.source_psd.shape == result.merged_mne_default_psd.shape
    assert result.source_psd.shape == result.frequencies_hz.shape
    assert result.duration_s == pytest.approx(120.0)
    assert result.decomb_filter.length_s > result.merged_mne_default_filter.length_s
    assert result.decomb_filter.length_samples > 0
    assert result.merged_mne_default_filter.length_samples > 0
