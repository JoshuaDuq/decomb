"""The fit's tolerances are widths in a spectrum, so they follow that spectrum's resolution.

They used to be in hertz, and hertz is only the right unit at one window length. The comb
is fitted from Hann periodograms over ``estimation_window_s``, whose narrowest possible
peak is ``1.4382 / T`` -- 26.6 mHz at the shipped 54 s. Doubling the window to 108 s halves
that while a fixed ``max_harmonic_residual_hz: 0.06`` stays put, so the same setting becomes
twice as permissive in the only units that describe it.

That was not hypothetical. Sweeping the window over a 15-participant cohort, one recording's
supported harmonic count fell from 24 to 19 while another's rose from 19 to 26, because
lengthening the window improved the SNR of every line and loosened these tolerances at the
same time. Separating the two is what makes a window sweep interpretable.
"""

from __future__ import annotations

import pytest

from decomb import remove, spectral

SHIPPED_WINDOW_S = 54.0


class TestTheShippedBehaviourIsUnchanged:
    """The multipliers were calibrated to reproduce the hertz values they replaced."""

    @pytest.mark.parametrize(
        "attribute,previous_hz",
        (
            ("max_harmonic_residual_hz", 0.06),
            ("max_fit_residual_rms_hz", 0.04),
            ("max_line_width_hz", 0.25),
        ),
    )
    def test_the_default_matches_what_it_replaced(self, attribute, previous_hz):
        settings = remove.RemovalSettings(estimation_window_s=SHIPPED_WINDOW_S)
        assert getattr(settings, attribute) == pytest.approx(previous_hz, rel=2e-3)


class TestTheyFollowTheWindow:
    @pytest.mark.parametrize(
        "attribute",
        ("max_harmonic_residual_hz", "max_fit_residual_rms_hz", "max_line_width_hz"),
    )
    def test_doubling_the_window_halves_the_tolerance(self, attribute):
        short = remove.RemovalSettings(estimation_window_s=SHIPPED_WINDOW_S)
        long = remove.RemovalSettings(estimation_window_s=2 * SHIPPED_WINDOW_S)
        assert getattr(long, attribute) == pytest.approx(getattr(short, attribute) / 2.0)

    @pytest.mark.parametrize("window_s", (27.0, 54.0, 81.0, 108.0, 135.0))
    def test_the_tolerance_stays_a_fixed_number_of_resolutions(self, window_s):
        settings = remove.RemovalSettings(estimation_window_s=window_s)
        resolution = spectral.hann_resolution_hz(window_s)
        assert settings.max_harmonic_residual_hz / resolution == pytest.approx(2.25)
        assert settings.max_fit_residual_rms_hz / resolution == pytest.approx(1.5)
        assert settings.max_line_width_hz / resolution == pytest.approx(9.4)

    def test_the_resolution_is_the_one_the_fit_actually_reads(self):
        """The window spectra are Hann periodograms over the estimation window."""
        settings = remove.RemovalSettings(estimation_window_s=81.0)
        assert settings.spectral_resolution_hz == spectral.hann_resolution_hz(81.0)


class TestTheBoundsStayCoherent:
    def test_a_scatter_bound_wider_than_the_membership_bound_is_refused(self):
        """Harmonics beyond the membership bound are dropped before the RMS is taken.

        A scatter bound larger than that can never bind, so accepting one would leave a
        criterion in the config that reads as though it were in force and is not.
        """
        with pytest.raises(ValueError, match="can never bind"):
            remove.RemovalSettings(
                max_harmonic_residual_resolutions=1.0,
                max_fit_residual_rms_resolutions=2.0,
            )

    def test_equal_bounds_are_allowed(self):
        settings = remove.RemovalSettings(
            max_harmonic_residual_resolutions=2.0,
            max_fit_residual_rms_resolutions=2.0,
        )
        assert settings.max_fit_residual_rms_hz == settings.max_harmonic_residual_hz

    @pytest.mark.parametrize(
        "field",
        (
            "max_harmonic_residual_resolutions",
            "max_fit_residual_rms_resolutions",
            "max_line_width_resolutions",
        ),
    )
    def test_a_non_positive_multiplier_is_refused(self, field):
        with pytest.raises(ValueError, match="finite and positive"):
            remove.RemovalSettings(**{field: 0.0})
