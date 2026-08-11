"""Real-data comparison with conventional MNE notch defaults."""

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


def test_traditional_plan_uses_the_same_detected_centres_and_mne_defaults():
    model = _model()
    settings = _settings()
    measured_intervals = notch.observed_line_intervals(model, settings)
    plan = comparison.traditional_notch_plan(model, settings)

    measured_centres_hz = sorted(band.centre_hz for band in measured_intervals)
    assert [band.centre_hz for band in plan.stopbands] == pytest.approx(measured_centres_hz)
    assert [band.width_hz for band in plan.stopbands] == pytest.approx(
        [frequency_hz / 200.0 for frequency_hz in measured_centres_hz]
    )
    assert plan.transition_bandwidth_hz == 1.0


def test_traditional_defaults_merge_targets_whose_transitions_overlap():
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

    plan = comparison.traditional_notch_plan(model, _settings())

    assert len(plan.stopbands) == 1
    assert plan.stopbands[0].harmonics == (40, 41)


def test_measured_geometry_preserves_more_band_than_traditional_defaults():
    measured = notch.plan_harmonic_stopbands(_model(), _settings())
    traditional = comparison.traditional_notch_plan(_model(), _settings())

    assert comparison.unavailable_width_hz(measured, (1.0, 100.0)) < (
        comparison.unavailable_width_hz(traditional, (1.0, 100.0))
    )


def test_real_comparison_figure_reports_both_geometries(tmp_path):
    measured = notch.plan_harmonic_stopbands(_model(), _settings())
    traditional = comparison.traditional_notch_plan(_model(), _settings())
    frequencies_hz = np.arange(1.0, 100.0, 0.02)
    source = 1e-12 / frequencies_hz
    result = comparison.NotchComparison(
        frequencies_hz=frequencies_hz,
        source_psd=source,
        decomb_psd=source.copy(),
        traditional_psd=source.copy(),
        decomb_plan=measured,
        traditional_plan=traditional,
        duration_s=120.0,
    )
    output = tmp_path / "notch-comparison.png"

    comparison.figure_notch_comparison(
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

    result = comparison.measure_notch_comparison(raw, _settings())

    assert result.source_psd.shape == result.decomb_psd.shape
    assert result.source_psd.shape == result.traditional_psd.shape
    assert result.source_psd.shape == result.frequencies_hz.shape
    assert result.duration_s == pytest.approx(120.0)
