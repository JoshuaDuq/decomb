import numpy as np
import pytest

from decomb import notch, residual


def _settings(**overrides) -> notch.HarmonicNotchSettings:
    from dataclasses import replace

    from decomb.config import load_config

    base = notch.HarmonicNotchSettings.from_config(load_config())
    return replace(base, **overrides) if overrides else base


def test_bins_are_measured_in_the_estimation_window():
    settings = _settings()

    assert residual.cluster_gap_hz(settings) == pytest.approx(0.30)
    assert residual.stopband_margin_hz(settings) == pytest.approx(0.125)


def test_bin_geometry_scales_with_the_estimation_window():
    settings = _settings(estimation_window_s=20.0)

    assert residual.cluster_gap_hz(settings) == pytest.approx(0.15)
    assert residual.stopband_margin_hz(settings) == pytest.approx(0.0625)


def test_bins_closer_than_the_gap_are_one_cluster():
    groups = residual.cluster((40.0, 40.2, 40.5, 42.0), gap_hz=0.3)

    assert groups == [[40.0, 40.2, 40.5], [42.0]]


def test_prominence_is_the_peak_over_the_local_median():
    freqs = np.arange(1.0, 100.0, 0.1)
    db = np.zeros_like(freqs)
    db[np.argmin(np.abs(freqs - 40.0))] = 7.0

    assert residual.prominence_db(db, freqs, 40.0) == pytest.approx(7.0)
    assert residual.prominence_db(db, freqs, 60.0) == pytest.approx(0.0)


def test_teeth_are_multiples_of_the_declared_fundamental_above_20_hz():
    teeth = residual.comb_teeth(_settings(comb_fundamental_hz=1.2), sampling_frequency_hz=250.0)

    assert teeth[0] == pytest.approx(20.4)
    assert teeth[-1] <= 95.0
    assert np.allclose(np.diff(teeth), 1.2)


def test_comb_teeth_fall_back_to_the_trigger_derived_fundamental():
    settings = _settings()
    assert settings.comb_fundamental_hz is None

    teeth = residual.comb_teeth(settings, sampling_frequency_hz=250.0)

    assert np.allclose(np.diff(teeth), 1.0 / settings.scanner_repetition_time_s)


def test_a_cluster_over_the_floor_becomes_a_stopband_with_margin():
    settings = _settings()
    freqs = np.arange(1.0, 100.0, 0.1)
    db = np.zeros_like(freqs)
    db[np.argmin(np.abs(freqs - 40.0))] = 5.0

    spans = residual.threshold_stopbands(db, freqs, (40.0, 60.0), settings)

    assert spans == ((40.0 - 0.125, 40.0 + 0.125),)


def test_clusters_under_the_floor_are_not_notched():
    settings = _settings()
    freqs = np.arange(1.0, 100.0, 0.1)
    db = np.zeros_like(freqs)
    db[np.argmin(np.abs(freqs - 40.0))] = 1.5

    assert residual.threshold_stopbands(db, freqs, (40.0,), settings) == ()


def test_adjacent_stopbands_merge():
    settings = _settings()
    freqs = np.arange(1.0, 100.0, 0.1)
    db = np.zeros_like(freqs)
    for centre in (40.0, 40.2):
        db[np.argmin(np.abs(freqs - centre))] = 5.0

    spans = residual.threshold_stopbands(db, freqs, (40.0, 40.2), settings)

    assert spans == ((40.0 - 0.125, 40.2 + 0.125),)


def _plan(*spans):
    from types import SimpleNamespace

    return SimpleNamespace(
        geometry=notch.HarmonicNotchPlan(
            tuple(notch.HarmonicStopband((), lo, hi, "isolated") for lo, hi in spans),
            0.061,
        )
    )


def test_ordinary_targets_are_round_one_stopband_centres_not_line_positions():
    from types import SimpleNamespace

    evidence = SimpleNamespace(plans=(_plan((39.5, 40.5)), _plan((59.0, 61.0))))

    assert residual.ordinary_line_frequencies(evidence) == (40.0, 60.0)


def test_a_declared_threshold_plan_carries_the_stopbands():
    settings = _settings(comb_fundamental_hz=1.2)
    record = residual.ThresholdRecord(((40.0, 40.5),), (5.0,))

    plan = record.plan(settings)

    assert plan is not None
    assert [(s.low_hz, s.high_hz) for s in plan.stopbands] == [(40.0, 40.5)]


def test_an_empty_threshold_record_has_no_plan():
    assert residual.ThresholdRecord((), ()).plan(_settings()) is None


def test_threshold_manifest_rows_declare_their_unavailable_bandwidth():
    settings = _settings(comb_fundamental_hz=1.2)
    bands = (("beta", 13.0, 30.0), ("gamma", 30.1, 80.0))
    record = residual.ThresholdRecord(((40.0, 40.5),), (5.0,))

    rows = record.manifest_rows("sub-0001_run-1_eeg", bands, settings)

    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "threshold_notched"
    assert row["removal_round"] == ""
    assert row["stopband_low_hz"] == 40.0
    assert row["authorizing_prominence_db"] == 5.0
    assert row["unavailable_low_hz"] < 40.0
    assert row["gamma_unavailable_share"] > 0.0
    assert notch.MANIFEST_REQUIRED_COLUMNS <= set(row)
