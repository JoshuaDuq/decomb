"""A window that cannot be fitted must not end the run for every other recording.

The comb fit is run per adaptive window, inside a loop over every recording in a dataset,
and a window that could not establish a grid used to raise. On a 15-participant cohort one
54-second window of one recording -- a six-second movement burst raised its broadband floor
by 13 dB and buried the comb -- ended a benchmark that had already measured 54 recordings.
Its other sixteen windows were fine, and so were the other 89 recordings.

A window with no grid of its own can take the one the whole recording confirmed. That has
to stay the exception, so the case where most windows cannot fit is still refused.
"""

from __future__ import annotations

import numpy as np
import pytest

from decomb import estimators


def _spectrum(positions, df=0.002, high=110.0, height=20.0):
    freqs = np.arange(1.0, high, df)
    spectrum = np.zeros_like(freqs)
    sigma = 0.109 / 2.355
    for centre in positions:
        spectrum[:] = np.maximum(spectrum, height * np.exp(-0.5 * ((freqs - centre) / sigma) ** 2))
    return freqs, spectrum, spectrum.copy()


def _run_estimate() -> estimators.CombEstimate:
    """A grid confirmed over a whole recording, for a window to inherit."""
    freqs, spectrum, prominence = _spectrum([k * 1.2 for k in range(24, 80)])
    return estimators.estimate_comb(freqs, spectrum, prominence, isolated_nominal_hz=())


class TestAWindowInheritsRatherThanRaising:
    def test_a_window_too_sparse_to_fit_takes_the_run_grid(self):
        run = _run_estimate()
        # Three harmonics: far below the floor, and the shape of a window whose comb is
        # buried rather than absent.
        freqs, spectrum, prominence = _spectrum([24 * 1.2, 40 * 1.2, 60 * 1.2])

        with pytest.raises(ValueError, match="harmonics"):
            estimators.estimate_comb(freqs, spectrum, prominence, isolated_nominal_hz=())

        inherited = estimators.estimate_comb(
            freqs, spectrum, prominence, isolated_nominal_hz=(), fallback=run
        )
        assert inherited.inherited_fundamental
        assert inherited.fundamental_hz == run.fundamental_hz
        assert inherited.fundamental_jackknife_se_hz == run.fundamental_jackknife_se_hz

    def test_an_inherited_window_carries_no_harmonics_of_its_own(self):
        """So every comb target is read off the inherited grid, not a peak measured here.

        ``removal_frequencies`` prefers a measured position over ``fundamental * k``. A
        window that could not establish a grid has no position worth preferring.
        """
        run = _run_estimate()
        freqs, spectrum, prominence = _spectrum([24 * 1.2, 40 * 1.2, 60 * 1.2])

        inherited = estimators.estimate_comb(
            freqs, spectrum, prominence, isolated_nominal_hz=(), fallback=run
        )
        assert inherited.harmonics_used == ()
        assert inherited.harmonic_positions_hz == ()

    def test_scattered_peaks_inherit_too(self):
        """The refusals differ but say one thing: these peaks do not establish a grid."""
        run = _run_estimate()
        positions = [k * 1.2 + (0.24 if k % 2 else -0.24) for k in range(24, 80)]
        freqs, spectrum, prominence = _spectrum(positions)

        with pytest.raises(ValueError, match="scatter"):
            estimators.estimate_comb(freqs, spectrum, prominence, isolated_nominal_hz=())

        inherited = estimators.estimate_comb(
            freqs, spectrum, prominence, isolated_nominal_hz=(), fallback=run
        )
        assert inherited.inherited_fundamental

    def test_an_inherited_window_still_finds_its_own_isolated_lines(self):
        """Inheriting a comb must not import the run's isolated lines into this window.

        Those are removed per window, and a window that did not show a line should not have
        it notched out on the strength of another window's evidence.
        """
        run = _run_estimate()
        freqs, spectrum, prominence = _spectrum([24 * 1.2, 40 * 1.2, 60 * 1.2, 57.25])

        inherited = estimators.estimate_comb(
            freqs, spectrum, prominence, isolated_nominal_hz=(57.25,), fallback=run
        )
        assert inherited.isolated_hz == (57.25,)

    def test_without_a_fallback_the_refusal_still_stands(self):
        """The whole-run fit is made with no fallback, and must keep refusing."""
        freqs, spectrum, prominence = _spectrum([24 * 1.2, 40 * 1.2, 60 * 1.2])
        with pytest.raises(ValueError, match="harmonics"):
            estimators.estimate_comb(freqs, spectrum, prominence, isolated_nominal_hz=())


class TestInheritingStaysTheException:
    def _estimate(self, *, inherited: bool) -> estimators.CombEstimate:
        return estimators.CombEstimate(
            fundamental_hz=1.2,
            harmonics_used=() if inherited else tuple(range(24, 80)),
            harmonic_positions_hz=(
                () if inherited else tuple(1.2 * k for k in range(24, 80))
            ),
            residual_rms_hz=0.01,
            max_abs_residual_hz=0.02,
            fundamental_jackknife_se_hz=1e-5,
            isolated_hz=(),
            isolated_prominence_db=(),
            inherited_fundamental=inherited,
        )

    def test_a_minority_of_inherited_windows_is_accepted(self):
        estimates = [self._estimate(inherited=False) for _ in range(4)]
        estimates[0] = self._estimate(inherited=True)
        model = estimators.build_adaptive_comb_model(estimates[0], estimates)
        assert len(model.window_estimates) == 4

    def test_inherited_windows_may_not_outnumber_supported_ones(self):
        estimates = [self._estimate(inherited=True) for _ in range(3)]
        estimates[0] = self._estimate(inherited=False)
        with pytest.raises(ValueError, match="outnumbering"):
            estimators.build_adaptive_comb_model(estimates[0], estimates)

    def test_an_inherited_window_is_not_judged_on_harmonics_it_cannot_have(self):
        """It carries none by construction; the count floor would refuse it every time."""
        estimates = [self._estimate(inherited=False), self._estimate(inherited=True)]
        model = estimators.build_adaptive_comb_model(estimates[0], estimates)
        assert model.window_estimates[1].inherited_fundamental
