from types import SimpleNamespace

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
