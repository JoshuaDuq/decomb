from types import SimpleNamespace

import numpy as np
import pytest

from decomb import lines, notch, subtraction

BANDS = (("beta", 13.0, 30.0), ("gamma", 30.1, 80.0))


def _settings() -> notch.HarmonicNotchSettings:
    from decomb.config import load_config

    return notch.HarmonicNotchSettings.from_config(load_config())


def test_band_availability_counts_bare_intervals():
    shares = notch.band_availability_from_intervals(((20.0, 21.7),), BANDS)

    assert shares["beta_unavailable_share"] == pytest.approx(1.7 / 17.0)
    assert shares["beta_retained_share"] == pytest.approx(1.0 - 1.7 / 17.0)
    assert shares["gamma_unavailable_share"] == 0.0


def test_authorized_frequencies_are_the_supported_scanner_harmonics():
    evidence = notch.ScannerHarmonicEvidence(
        fundamental_hz=10.0,
        corrected_p_value=1e-12,
        supporting_harmonics=(2, 4),
    )
    round_evidence = SimpleNamespace(
        model=lines.LineModel((), 1, 2, 5),
        scanner_harmonics=evidence,
    )

    assert subtraction.authorized_frequencies(round_evidence, _settings()) == (20.0, 40.0)


def test_authorized_frequencies_match_what_the_planner_would_notch():
    evidence = notch.ScannerHarmonicEvidence(
        fundamental_hz=10.0,
        corrected_p_value=1e-12,
        supporting_harmonics=(2, 4),
    )
    plan = notch.plan_scanner_harmonic_notches(evidence, _settings(), maximum_hz=49.0)
    planned = tuple(
        harmonic * evidence.fundamental_hz
        for stopband in plan.stopbands
        for harmonic in stopband.harmonics
    )
    round_evidence = SimpleNamespace(
        model=lines.LineModel((), 1, 2, 5), scanner_harmonics=evidence
    )

    assert subtraction.authorized_frequencies(round_evidence, _settings()) == planned


def test_damage_interval_spans_two_frequency_bins_each_side():
    settings = _settings()
    half = 2.0 * settings.frequency_bin_width_hz

    assert subtraction.damage_intervals((40.0,), settings) == ((40.0 - half, 40.0 + half),)


def test_overlapping_damage_intervals_merge():
    settings = _settings()
    half = 2.0 * settings.frequency_bin_width_hz
    close = 40.0 + half

    merged = subtraction.damage_intervals((40.0, close), settings)

    assert merged == ((40.0 - half, close + half),)


def test_separated_damage_intervals_do_not_merge():
    settings = _settings()
    half = 2.0 * settings.frequency_bin_width_hz

    intervals = subtraction.damage_intervals((40.0, 40.0 + 5.0 * half), settings)

    assert len(intervals) == 2


def test_subtraction_removes_the_authorized_frequency_from_the_data():
    import mne

    settings = _settings()
    sfreq = 200.0
    times = np.arange(0, 60.0, 1.0 / sfreq)
    data = np.vstack([np.sin(2 * np.pi * 40.0 * times)] * 2) * 1e-6
    raw = mne.io.RawArray(
        data, mne.create_info(["C3", "C4"], sfreq, "eeg"), verbose="ERROR"
    )
    round_evidence = SimpleNamespace(
        model=lines.LineModel((), 1, 2, 5),
        scanner_harmonics=notch.ScannerHarmonicEvidence(
            fundamental_hz=40.0, corrected_p_value=1e-12, supporting_harmonics=(1,)
        ),
    )

    cleaned, record = subtraction.subtract_authorized(
        raw, round_evidence, settings, n_jobs=1
    )

    assert record.frequencies_hz == (40.0,)
    assert record.window_s == settings.estimation_window_s
    assert np.abs(cleaned.get_data()).max() < 0.2 * np.abs(raw.get_data()).max()


def test_subtraction_manifest_rows_declare_unavailable_bandwidth():
    settings = _settings()
    half = 2.0 * settings.frequency_bin_width_hz
    record = subtraction.SubtractionRecord((40.0,), settings.estimation_window_s)

    rows = record.manifest_rows("sub-0001_run-1_eeg", BANDS, settings)

    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "subtracted"
    assert row["recording"] == "sub-0001_run-1_eeg"
    assert row["subtracted_frequencies_hz"] == "40.0"
    assert row["unavailable_low_hz"] == pytest.approx(40.0 - half)
    assert row["unavailable_high_hz"] == pytest.approx(40.0 + half)
    assert row["gamma_unavailable_share"] == pytest.approx(2.0 * half / (80.0 - 30.1))
    assert row["beta_unavailable_share"] == 0.0


def test_subtraction_manifest_rows_carry_every_required_manifest_column():
    settings = _settings()
    record = subtraction.SubtractionRecord((40.0,), settings.estimation_window_s)

    row = record.manifest_rows("sub-0001_run-1_eeg", BANDS, settings)[0]

    assert notch.MANIFEST_REQUIRED_COLUMNS <= set(row)
