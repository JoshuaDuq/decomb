"""Comb fitting and exact harmonic support localization."""

from __future__ import annotations

import numpy as np
import pytest

from decomb import harmonics


def _spectrum(
    peaks_hz: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies_hz = np.arange(0.0, 101.0, 0.01)
    spectrum_db = np.zeros_like(frequencies_hz)
    prominence_db = np.zeros_like(frequencies_hz)
    for peak_hz in peaks_hz:
        index = int(np.argmin(np.abs(frequencies_hz - peak_hz)))
        spectrum_db[index - 1 : index + 2] = (2.0, 12.0, 2.0)
        prominence_db[index] = 12.0
    return frequencies_hz, spectrum_db, prominence_db


def _fit(spectrum):
    return harmonics.estimate_comb(
        *spectrum,
        nominal_fundamental_hz=10.0,
        fit_harmonic_range=(2, 8),
        supported_harmonic_range=(1, 9),
        search_hz=0.2,
        min_prominence_db=1.0,
        min_harmonics=5,
        max_harmonic_residual_hz=0.015,
        max_residual_rms_hz=0.01,
    )


def test_fit_uses_the_reliable_range_then_localizes_outer_supported_harmonics():
    estimate = _fit(_spectrum(tuple(10.0 * harmonic for harmonic in range(1, 10))))

    assert estimate.fitted_harmonics == tuple(range(2, 9))
    assert estimate.supported_harmonics == tuple(range(1, 10))
    assert estimate.fundamental_hz == pytest.approx(10.0)


def test_an_off_grid_peak_cannot_become_a_supported_harmonic():
    peaks = tuple(10.0 * harmonic for harmonic in range(1, 10) if harmonic != 5)
    estimate = _fit(_spectrum((*peaks, 50.04)))

    assert 5 not in estimate.fitted_harmonics
    assert 5 not in estimate.supported_harmonics


def test_an_unidentifiable_comb_raises_instead_of_inheriting_a_grid():
    with pytest.raises(ValueError, match="at least 5"):
        _fit(_spectrum((20.0, 30.0, 40.0)))


def test_window_evidence_can_only_localize_whole_recording_targets():
    spectrum = _spectrum((20.0, 30.0, 40.0, 55.0))

    evidence = harmonics.localize_supported_harmonics(
        *spectrum,
        supported_harmonics=(2, 3, 4),
        fundamental_hz=10.0,
        search_hz=0.015,
        min_prominence_db=1.0,
    )

    assert evidence.harmonics == (2, 3, 4)
    assert 55.0 not in evidence.positions_hz


def test_a_window_with_no_visible_harmonic_is_explicitly_empty():
    evidence = harmonics.localize_supported_harmonics(
        *_spectrum(()),
        supported_harmonics=(2, 3, 4),
        fundamental_hz=10.0,
        search_hz=0.015,
        min_prominence_db=1.0,
    )

    assert evidence == harmonics.HarmonicEvidence((), ())
