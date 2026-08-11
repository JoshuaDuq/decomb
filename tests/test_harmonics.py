"""Threshold-free comb discovery and complete harmonic enumeration."""

from __future__ import annotations

import numpy as np
import pytest

from decomb import harmonics, spectral


def _comb_spectrum(
    *,
    fundamental_hz: float = 1.2,
    duration_s: float = 54.0,
    maximum_hz: float = 100.0,
    present_harmonics=range(8, 84),
) -> tuple[np.ndarray, np.ndarray]:
    frequencies_hz = np.fft.rfftfreq(
        int(duration_s * 1000.0),
        d=1.0 / 1000.0,
    )
    frequencies_hz = frequencies_hz[frequencies_hz <= maximum_hz]
    rng = np.random.default_rng(20260810)
    spectrum_db = -20.0 * np.log10(np.maximum(frequencies_hz, frequencies_hz[1]))
    spectrum_db += rng.normal(scale=0.15, size=frequencies_hz.size)
    for harmonic in present_harmonics:
        frequency_hz = harmonic * fundamental_hz
        if frequency_hz > maximum_hz:
            continue
        index = int(np.argmin(np.abs(frequencies_hz - frequency_hz)))
        spectrum_db[index] += 12.0 / harmonic**0.1
    return frequencies_hz, spectrum_db


def _add_hann_line(
    frequencies_hz: np.ndarray,
    background_db: np.ndarray,
    *,
    position_hz: float,
    contrast_db: float,
) -> np.ndarray:
    """Add one sinusoidal Hann response in linear power."""
    bin_width_hz = float(frequencies_hz[1] - frequencies_hz[0])
    bin_offset = (frequencies_hz - position_hz) / bin_width_hz
    hann_amplitude = (
        0.5 * np.sinc(bin_offset)
        + 0.25 * np.sinc(bin_offset - 1.0)
        + 0.25 * np.sinc(bin_offset + 1.0)
    )
    hann_power = (hann_amplitude / 0.5) ** 2
    background_power = 10.0 ** (background_db / 10.0)
    local_background = np.interp(position_hz, frequencies_hz, background_power)
    line_power = local_background * (10.0 ** (contrast_db / 10.0) - 1.0)
    return spectral.to_db(background_power + line_power * hann_power)


def test_model_selection_recovers_the_comb_without_a_nominal_frequency():
    frequencies_hz, spectrum_db = _comb_spectrum()

    estimate = harmonics.estimate_comb(
        frequencies_hz,
        spectrum_db,
        spectral_resolution_hz=spectral.hann_resolution_hz(54.0),
    )

    assert estimate.fundamental_hz == pytest.approx(1.2, abs=5e-4)
    assert estimate.evidence_bic < 0.0


def test_dense_weak_ripple_does_not_displace_a_sparse_strong_comb():
    frequencies_hz, spectrum_db = _comb_spectrum()
    spectrum_db += 0.8 * np.cos(2.0 * np.pi * frequencies_hz / 0.123561)

    estimate = harmonics.estimate_comb(
        frequencies_hz,
        spectrum_db,
        spectral_resolution_hz=spectral.hann_resolution_hz(54.0),
    )

    assert estimate.fundamental_hz == pytest.approx(1.2, abs=5e-4)


def test_every_harmonic_through_100_hz_is_authorized_even_when_weak_or_absent():
    frequencies_hz, spectrum_db = _comb_spectrum(
        present_harmonics=range(24, 84, 3),
    )

    estimate = harmonics.estimate_comb(
        frequencies_hz,
        spectrum_db,
        spectral_resolution_hz=spectral.hann_resolution_hz(54.0),
    )

    expected = tuple(range(1, int(100.0 / estimate.fundamental_hz) + 1))
    assert estimate.harmonics == expected
    assert len(estimate.positions_hz) == len(expected)


def test_configured_frequency_range_limits_authorized_harmonics():
    frequencies_hz, spectrum_db = _comb_spectrum()

    estimate = harmonics.estimate_comb(
        frequencies_hz,
        spectrum_db,
        spectral_resolution_hz=spectral.hann_resolution_hz(54.0),
        frequency_range_hz=(20.0, 80.0),
    )

    expected = tuple(
        harmonic
        for harmonic in range(1, int(80.0 / estimate.fundamental_hz) + 1)
        if harmonic * estimate.fundamental_hz >= 20.0
    )
    assert estimate.harmonics == expected
    assert all(20.0 <= position <= 80.0 for position in estimate.positions_hz)


def test_no_comb_surfaces_as_an_error_instead_of_inventing_targets():
    frequencies_hz = np.linspace(0.0, 100.0, 5401)
    spectrum_db = np.zeros_like(frequencies_hz)

    with pytest.raises(harmonics.NoCombDetected, match="no supported harmonic comb"):
        harmonics.estimate_comb(
            frequencies_hz,
            spectrum_db,
            spectral_resolution_hz=spectral.hann_resolution_hz(54.0),
        )


def test_window_localization_never_drops_an_authorized_harmonic():
    frequencies_hz, spectrum_db = _comb_spectrum(present_harmonics=range(24, 84, 4))
    harmonics_to_localize = tuple(range(1, 84))

    evidence = harmonics.localize_harmonics(
        frequencies_hz,
        spectrum_db,
        harmonics=harmonics_to_localize,
        fundamental_hz=1.2,
        spectral_resolution_hz=spectral.hann_resolution_hz(54.0),
    )

    assert evidence.harmonics == harmonics_to_localize
    assert len(evidence.positions_hz) == len(harmonics_to_localize)


def test_stable_off_comb_line_is_selected_without_a_prominence_threshold():
    frequencies_hz, spectrum_db = _comb_spectrum()
    isolated_hz = 42.35
    windows = []
    for seed in range(12):
        background_db = spectrum_db + np.random.default_rng(seed).normal(
            scale=0.1,
            size=spectrum_db.size,
        )
        windows.append(
            _add_hann_line(
                frequencies_hz,
                background_db,
                position_hz=isolated_hz,
                contrast_db=10.0,
            )
        )
    whole = spectral.to_db(
        np.mean(10.0 ** (np.asarray(windows) / 10.0), axis=0)
    )
    comb = harmonics.estimate_comb(
        frequencies_hz,
        whole,
        spectral_resolution_hz=spectral.hann_resolution_hz(54.0),
    )

    isolated = harmonics.detect_isolated_lines(
        frequencies_hz,
        whole,
        windows,
        comb=comb,
        spectral_resolution_hz=spectral.hann_resolution_hz(54.0),
        independent_window_indices=tuple(range(0, len(windows), 2)),
    )

    assert any(abs(position - isolated_hz) < 0.03 for position in isolated.positions_hz)
    assert all(value < 0.0 for value in isolated.evidence_bic)


def test_drifting_off_comb_line_is_selected_and_its_trajectory_is_localized():
    frequencies_hz, background_db = _comb_spectrum()
    line_positions_hz = np.linspace(42.30, 42.37, 12)
    windows = [
        _add_hann_line(
            frequencies_hz,
            background_db,
            position_hz=position_hz,
            contrast_db=18.0,
        )
        for position_hz in line_positions_hz
    ]
    whole_power = np.mean(10.0 ** (np.asarray(windows) / 10.0), axis=0)
    whole_db = spectral.to_db(whole_power)
    comb = harmonics.estimate_comb(
        frequencies_hz,
        whole_db,
        spectral_resolution_hz=spectral.hann_resolution_hz(54.0),
    )

    isolated = harmonics.detect_isolated_lines(
        frequencies_hz,
        whole_db,
        windows,
        comb=comb,
        spectral_resolution_hz=spectral.hann_resolution_hz(54.0),
        independent_window_indices=tuple(range(0, len(windows), 2)),
    )

    assert any(abs(position_hz - 42.335) < 0.04 for position_hz in isolated.positions_hz)
    line_index = int(np.argmin(np.abs(np.asarray(isolated.positions_hz) - 42.335)))
    trajectory_hz = np.asarray(isolated.window_positions_hz)[:, line_index]
    assert isolated.positions_hz[line_index] == pytest.approx(42.335, abs=0.04)
    assert trajectory_hz[0] == pytest.approx(line_positions_hz[0], abs=0.02)
    assert trajectory_hz[-1] == pytest.approx(line_positions_hz[-1], abs=0.02)


def test_irregular_line_trajectory_is_not_split_at_a_shallow_spectral_valley():
    frequencies_hz, background_db = _comb_spectrum()
    line_positions_hz = np.array(
        [42.35, 42.31, 42.37, 42.33, 42.39, 42.32, 42.36, 42.30, 42.38, 42.34, 42.37, 42.31]
    )
    windows = []
    for seed, position_hz in enumerate(line_positions_hz):
        noisy_background_db = background_db + np.random.default_rng(seed).normal(
            scale=0.1,
            size=background_db.size,
        )
        windows.append(
            _add_hann_line(
                frequencies_hz,
                noisy_background_db,
                position_hz=position_hz,
                contrast_db=18.0,
            )
        )
    whole_db = spectral.to_db(
        np.mean(10.0 ** (np.asarray(windows) / 10.0), axis=0)
    )
    comb = harmonics.estimate_comb(
        frequencies_hz,
        whole_db,
        spectral_resolution_hz=spectral.hann_resolution_hz(54.0),
    )

    isolated = harmonics.detect_isolated_lines(
        frequencies_hz,
        whole_db,
        windows,
        comb=comb,
        spectral_resolution_hz=spectral.hann_resolution_hz(54.0),
        independent_window_indices=tuple(range(0, len(windows), 2)),
    )

    assert any(abs(position_hz - 42.35) < 0.08 for position_hz in isolated.positions_hz)
    line_index = int(np.argmin(np.abs(np.asarray(isolated.positions_hz) - 42.35)))
    trajectory_hz = np.asarray(isolated.window_positions_hz)[:, line_index]
    assert np.min(trajectory_hz) == pytest.approx(np.min(line_positions_hz), abs=0.02)
    assert np.max(trajectory_hz) == pytest.approx(np.max(line_positions_hz), abs=0.02)


def test_isolated_lines_outside_configured_frequency_range_are_ignored():
    frequencies_hz, spectrum_db = _comb_spectrum()
    isolated_hz = 42.35
    windows = []
    for seed in range(12):
        background_db = spectrum_db + np.random.default_rng(seed).normal(
            scale=0.1,
            size=spectrum_db.size,
        )
        windows.append(
            _add_hann_line(
                frequencies_hz,
                background_db,
                position_hz=isolated_hz,
                contrast_db=10.0,
            )
        )
    whole = spectral.to_db(
        np.mean(10.0 ** (np.asarray(windows) / 10.0), axis=0)
    )
    comb = harmonics.estimate_comb(
        frequencies_hz,
        whole,
        spectral_resolution_hz=spectral.hann_resolution_hz(54.0),
        frequency_range_hz=(50.0, 80.0),
    )

    isolated = harmonics.detect_isolated_lines(
        frequencies_hz,
        whole,
        windows,
        comb=comb,
        spectral_resolution_hz=spectral.hann_resolution_hz(54.0),
        frequency_range_hz=(50.0, 80.0),
        independent_window_indices=tuple(range(0, len(windows), 2)),
    )

    assert all(50.0 <= position <= 80.0 for position in isolated.positions_hz)
    assert not any(abs(position - isolated_hz) < 0.03 for position in isolated.positions_hz)


def test_broad_stable_spectral_peak_is_not_an_isolated_line():
    frequencies_hz, spectrum_db = _comb_spectrum()
    broad_peak_hz = 10.0
    spectrum_db += 12.0 * np.exp(
        -0.5 * ((frequencies_hz - broad_peak_hz) / 0.85) ** 2
    )
    windows = [
        spectrum_db
        + np.random.default_rng(seed).normal(scale=0.03, size=spectrum_db.size)
        for seed in range(12)
    ]
    whole = np.mean(windows, axis=0)
    comb = harmonics.estimate_comb(
        frequencies_hz,
        whole,
        spectral_resolution_hz=spectral.hann_resolution_hz(54.0),
    )

    isolated = harmonics.detect_isolated_lines(
        frequencies_hz,
        whole,
        windows,
        comb=comb,
        spectral_resolution_hz=spectral.hann_resolution_hz(54.0),
        independent_window_indices=tuple(range(0, len(windows), 2)),
    )

    assert not any(8.0 <= position <= 12.0 for position in isolated.positions_hz)


def test_isolated_line_search_handles_a_peak_at_the_upper_analysis_edge():
    frequencies_hz, spectrum_db = _comb_spectrum()
    edge_peak_db = 8.0 * np.exp(-0.5 * ((frequencies_hz - 100.0) / 0.03) ** 2)
    windows = [
        spectrum_db
        + edge_peak_db
        + np.random.default_rng(seed).normal(scale=0.03, size=spectrum_db.size)
        for seed in range(12)
    ]
    whole = np.mean(windows, axis=0)
    comb = harmonics.estimate_comb(
        frequencies_hz,
        whole,
        spectral_resolution_hz=spectral.hann_resolution_hz(54.0),
    )

    isolated = harmonics.detect_isolated_lines(
        frequencies_hz,
        whole,
        windows,
        comb=comb,
        spectral_resolution_hz=spectral.hann_resolution_hz(54.0),
        independent_window_indices=tuple(range(0, len(windows), 2)),
    )

    assert all(position_hz <= 100.0 for position_hz in isolated.positions_hz)
