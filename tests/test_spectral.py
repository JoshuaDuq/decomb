"""Tests for the spectral estimators."""

from __future__ import annotations

import numpy as np
import pytest

from decomb import spectral

TR = 0.9
"""An arbitrary acquisition period. These functions have no default: a k/TR grid
only exists when something periodic was running, so the caller must say so."""


def _tone(frequency_hz: float, sfreq: float, n_times: int, amplitude: float, phase: float = 0.0):
    times = np.arange(n_times) / sfreq
    return amplitude * np.sin(2 * np.pi * frequency_hz * times + phase)


class TestHannPeriodogram:
    def test_recovers_tone_frequency(self):
        sfreq, n_times = 500.0, 10800
        signal = _tone(57.1759, sfreq, n_times, amplitude=3.0)
        freqs, psd = spectral.hann_periodogram(signal, sfreq)
        assert freqs[int(np.argmax(psd))] == pytest.approx(57.1759, abs=freqs[1])

    def test_parseval_scaling_matches_windowed_mean_square(self):
        rng = np.random.default_rng(0)
        sfreq, n_times = 500.0, 10800
        signal = rng.normal(size=n_times)
        freqs, psd = spectral.hann_periodogram(signal, sfreq)
        window = np.hanning(n_times)
        expected = np.sum((signal * window) ** 2) / (sfreq * np.sum(window**2))
        assert np.sum(psd) * freqs[1] == pytest.approx(
            expected * sfreq * freqs[1] / freqs[1], rel=0.02
        )

    def test_handles_leading_dimensions(self):
        sfreq, n_times = 500.0, 900
        data = np.stack([_tone(20.0, sfreq, n_times, 1.0), _tone(40.0, sfreq, n_times, 1.0)])
        freqs, psd = spectral.hann_periodogram(data, sfreq)
        assert psd.shape == (2, freqs.size)
        assert freqs[int(np.argmax(psd[0]))] == pytest.approx(20.0, abs=freqs[1])
        assert freqs[int(np.argmax(psd[1]))] == pytest.approx(40.0, abs=freqs[1])

    def test_rejects_non_finite(self):
        with pytest.raises(ValueError, match="finite"):
            spectral.hann_periodogram(np.array([1.0, np.nan, 2.0]), 500.0)


class TestLocalBackground:
    def test_flat_spectrum_gives_zero_prominence(self):
        spectrum = np.full(500, -110.0)
        prom = spectral.prominence_db(spectrum, half_width_bins=50)
        assert np.allclose(prom[50:-50], 0.0)

    def test_edges_are_nan(self):
        spectrum = np.full(500, -110.0)
        background = spectral.local_background_db(spectrum, half_width_bins=50)
        assert np.all(np.isnan(background[:50]))
        assert np.all(np.isnan(background[-50:]))

    def test_isolated_line_keeps_its_prominence(self):
        spectrum = np.full(500, -110.0)
        spectrum[250] = -95.0
        prom = spectral.prominence_db(spectrum, half_width_bins=50)
        assert prom[250] == pytest.approx(15.0)

    def test_core_exclusion_keeps_line_out_of_its_own_background(self):
        spectrum = np.full(41, -110.0)
        spectrum[18:23] = -90.0
        with_core = spectral.prominence_db(spectrum, half_width_bins=20, core_bins=2)
        assert with_core[20] == pytest.approx(20.0)

    def test_survives_several_lines_inside_the_window(self):
        spectrum = np.full(500, -110.0)
        for index in range(200, 300, 24):
            spectrum[index] = -90.0
        prom = spectral.prominence_db(spectrum, half_width_bins=50)
        assert prom[248] == pytest.approx(20.0)

    def test_rejects_core_wider_than_window(self):
        with pytest.raises(ValueError, match="core_bins must be smaller"):
            spectral.local_background_db(np.zeros(100), half_width_bins=5, core_bins=5)


class TestRobustNull:
    def test_recovers_gaussian_scale(self):
        rng = np.random.default_rng(1)
        values = rng.normal(loc=0.0, scale=2.0, size=20000)
        location, scale = spectral.robust_null(values)
        assert location == pytest.approx(0.0, abs=0.1)
        assert scale == pytest.approx(2.0, rel=0.05)

    def test_upper_tail_contamination_does_not_inflate_scale(self):
        rng = np.random.default_rng(2)
        values = np.concatenate([rng.normal(scale=1.0, size=5000), rng.normal(loc=30, size=200)])
        _, scale = spectral.robust_null(values)
        assert scale == pytest.approx(1.0, rel=0.1)

    def test_rejects_tiny_sample(self):
        with pytest.raises(ValueError, match="at least 32"):
            spectral.robust_null(np.zeros(10))


class TestFdr:
    def test_uniform_pvalues_yield_few_discoveries(self):
        rng = np.random.default_rng(3)
        qvalues = spectral.fdr_bh(rng.uniform(size=5000))
        assert np.sum(qvalues < 0.05) == 0

    def test_strong_signal_survives(self):
        pvalues = np.concatenate([np.full(10, 1e-12), np.linspace(0.05, 1.0, 990)])
        qvalues = spectral.fdr_bh(pvalues)
        assert np.sum(qvalues < 0.05) == 10

    def test_monotone_in_the_sorted_order(self):
        rng = np.random.default_rng(4)
        pvalues = rng.uniform(size=200)
        qvalues = spectral.fdr_bh(pvalues)
        order = np.argsort(pvalues)
        assert np.all(np.diff(qvalues[order]) >= -1e-12)

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="inside"):
            spectral.fdr_bh([0.5, 1.5])


class TestClusterPeaks:
    def test_one_cluster_yields_its_largest_bin(self):
        flags = np.zeros(20, dtype=bool)
        flags[5:9] = True
        values = np.zeros(20)
        values[5:9] = [1.0, 4.0, 2.0, 1.0]
        assert spectral.cluster_peaks(flags, values) == [6]

    def test_separated_clusters_stay_separate(self):
        flags = np.zeros(30, dtype=bool)
        flags[3:5] = True
        flags[20:22] = True
        values = np.arange(30, dtype=float)
        assert spectral.cluster_peaks(flags, values) == [4, 21]

    def test_single_quiet_bin_does_not_split_a_line(self):
        flags = np.array([False, True, False, True, False])
        values = np.array([0.0, 3.0, 0.0, 9.0, 0.0])
        assert spectral.cluster_peaks(flags, values, join_gap_bins=1) == [3]

    def test_zero_gap_keeps_them_apart(self):
        flags = np.array([False, True, False, True, False])
        values = np.array([0.0, 3.0, 0.0, 9.0, 0.0])
        assert spectral.cluster_peaks(flags, values, join_gap_bins=0) == [1, 3]

    def test_nothing_significant_gives_no_lines(self):
        assert spectral.cluster_peaks(np.zeros(10, dtype=bool), np.zeros(10)) == []


class TestRefinePeakFrequency:
    def test_recovers_off_bin_tone(self):
        sfreq, n_times = 500.0, 10800
        true_hz = 57.1759 + 0.017
        freqs, psd = spectral.hann_periodogram(_tone(true_hz, sfreq, n_times, 1.0), sfreq)
        db = spectral.to_db(psd)
        peak = int(np.argmax(db))
        refined = spectral.refine_peak_frequency(freqs, db, peak)
        assert abs(refined - true_hz) < abs(freqs[peak] - true_hz)
        assert refined == pytest.approx(true_hz, abs=0.005)

    def test_on_bin_tone_is_left_alone(self):
        sfreq, n_times = 500.0, 10800
        true_hz = 24 / 0.9  # exactly on a bin of the commensurate grid
        freqs, psd = spectral.hann_periodogram(_tone(true_hz, sfreq, n_times, 1.0), sfreq)
        db = spectral.to_db(psd)
        refined = spectral.refine_peak_frequency(freqs, db, int(np.argmax(db)))
        assert refined == pytest.approx(true_hz, abs=1e-3)


class TestSpectralLinewidth:
    def test_pure_tone_is_as_narrow_as_the_window_allows(self):
        sfreq, n_times = 500.0, 10800
        freqs, psd = spectral.hann_periodogram(_tone(57.2, sfreq, n_times, 1.0), sfreq)
        db = spectral.to_db(psd)
        width = spectral.spectral_linewidth_hz(freqs, db, int(np.argmax(db)))
        assert width == pytest.approx(spectral.hann_resolution_hz(n_times / sfreq), rel=0.15)

    def test_a_broad_resonance_is_much_wider(self):
        # A 2 Hz-wide Gaussian bump standing in for an alpha peak.
        freqs = np.arange(0, 100, 1 / 21.6)
        db = -120 + 12 * np.exp(-0.5 * ((freqs - 10.0) / 0.85) ** 2)
        width = spectral.spectral_linewidth_hz(freqs, db, int(np.argmin(np.abs(freqs - 10.0))))
        # 2 * sigma * sqrt(2 ln(4/3)) for a 12 dB Gaussian, and 19 window widths across.
        assert width == pytest.approx(1.289, abs=0.02)
        assert width > 15 * spectral.hann_resolution_hz(21.6)

    def test_returns_nan_when_the_peak_never_falls_away(self):
        freqs = np.arange(0, 100, 1 / 21.6)
        db = np.full(freqs.size, -110.0)
        assert np.isnan(spectral.spectral_linewidth_hz(freqs, db, 500))

    def test_hann_resolution_scales_inversely_with_duration(self):
        assert spectral.hann_resolution_hz(21.6) == pytest.approx(
            spectral.hann_resolution_hz(10.8) / 2
        )


class TestCombIndex:
    def test_acquisition_harmonic_is_on_the_grid(self):
        position = spectral.comb_index(55 / TR, TR)
        assert position.harmonic_index == 55
        assert position.on_comb

    def test_off_comb_line_reports_its_offset(self):
        position = spectral.comb_index(57.1759, TR)
        assert position.harmonic_index == 51
        assert not position.on_comb
        assert position.offset_hz == pytest.approx(57.1759 - 51 / TR, abs=1e-9)

    def test_sixty_hertz_coincides_with_harmonic_54(self):
        position = spectral.comb_index(60.0, TR)
        assert position.harmonic_index == 54
        assert position.on_comb


class TestCombPhaseAndUniformity:
    def test_comb_lines_cluster_at_zero(self):
        freqs = np.array([18, 27, 37, 45, 55, 62, 74, 80, 88, 95]) / TR
        assert np.allclose(spectral.comb_phase(freqs, TR), 0.0, atol=1e-9)
        assert spectral.comb_uniformity_pvalue(freqs, TR) < 1e-3

    def test_four_perfect_comb_lines_are_only_weak_evidence(self):
        # Directional statistics on a handful of lines cannot be decisive, and the test
        # should not pretend otherwise.
        freqs = np.array([18, 37, 55, 74]) / TR
        assert 1e-3 < spectral.comb_uniformity_pvalue(freqs, TR) < 0.05

    def test_unrelated_frequencies_do_not_cluster(self):
        rng = np.random.default_rng(5)
        freqs = rng.uniform(30, 90, size=40)
        assert spectral.comb_uniformity_pvalue(freqs, TR) > 0.05


class TestFitArithmeticComb:
    def test_recovers_known_spacing(self):
        spacing = 2.40741
        freqs = 51.574 + spacing * np.arange(5)
        fit = spectral.fit_arithmetic_comb(freqs)
        assert fit.spacing_hz == pytest.approx(spacing, abs=1e-6)
        assert fit.intercept_hz == pytest.approx(51.574, abs=1e-6)
        assert fit.rmse_hz < 1e-9

    def test_tolerates_a_missing_member(self):
        spacing = 2.40741
        freqs = 51.574 + spacing * np.array([0, 1, 3, 4])
        fit = spectral.fit_arithmetic_comb(freqs)
        assert fit.indices == (0, 1, 3, 4)
        assert fit.spacing_hz == pytest.approx(spacing, abs=1e-6)

    def test_reports_residuals_for_an_imperfect_family(self):
        freqs = np.array([51.574, 53.981, 56.389, 58.796])
        fit = spectral.fit_arithmetic_comb(freqs)
        assert fit.spacing_hz == pytest.approx(2.4074, abs=1e-3)
        assert fit.max_abs_residual_hz < 1e-3

    def test_rejects_too_few_lines(self):
        with pytest.raises(ValueError, match="at least three"):
            spectral.fit_arithmetic_comb([1.0, 2.0])


class TestDominantSpacing:
    def test_finds_the_repeated_gap(self):
        freqs = np.concatenate([51.574 + 2.4074 * np.arange(5), [37.1, 47.0]])
        spacing, support = spectral.dominant_spacing(
            freqs, max_difference_hz=12.0, tolerance_hz=0.05
        )
        assert spacing == pytest.approx(2.4074, abs=0.01)
        assert support >= 4


class TestRefineCombFundamental:
    def test_recovers_the_fundamental_when_the_commonest_gap_is_a_multiple(self):
        # A 1.2 Hz comb with every third member missing: two-apart pairs outnumber
        # adjacent ones, so the most-supported gap is 2.4 Hz.
        freqs = np.array([1.2 * k for k in range(22, 60) if k % 3])
        spacing, _ = spectral.dominant_spacing(freqs, max_difference_hz=12.0, tolerance_hz=0.06)
        fundamental, members = spectral.refine_comb_fundamental(freqs, spacing, tolerance_hz=0.06)
        assert fundamental == pytest.approx(1.2, abs=1e-6)
        assert members.all()

    def test_leaves_a_true_fundamental_alone(self):
        freqs = np.array([1.2 * k for k in range(22, 50)])
        fundamental, members = spectral.refine_comb_fundamental(freqs, 1.2, tolerance_hz=0.06)
        assert fundamental == pytest.approx(1.2, abs=1e-9)
        assert members.all()

    def test_does_not_subdivide_for_one_stray_line(self):
        freqs = np.concatenate([[1.2 * k for k in range(22, 50)], [58.2]])
        fundamental, _ = spectral.refine_comb_fundamental(freqs, 1.2, tolerance_hz=0.06)
        assert fundamental == pytest.approx(1.2, abs=1e-9)

    def test_excludes_lines_that_are_not_multiples(self):
        freqs = np.array([1.2 * k for k in range(22, 50)] + [57.2247, 47.0362])
        fundamental, members = spectral.refine_comb_fundamental(freqs, 2.4, tolerance_hz=0.06)
        assert fundamental == pytest.approx(1.2, abs=1e-6)
        assert not members[-1] and not members[-2]

    def test_rejects_a_non_positive_spacing(self):
        with pytest.raises(ValueError, match="finite positive"):
            spectral.refine_comb_fundamental([1.0, 2.0], 0.0, tolerance_hz=0.1)


class TestBootstrapCi:
    def test_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(6)
        values = rng.normal(loc=10.0, scale=2.0, size=15)
        point, low, high = spectral.bootstrap_ci(values, n_resamples=2000, seed=7)
        assert low <= point <= high

    def test_is_deterministic_for_a_fixed_seed(self):
        values = np.arange(15, dtype=float)
        assert spectral.bootstrap_ci(values, seed=11) == spectral.bootstrap_ci(values, seed=11)

    def test_mean_statistic_is_available(self):
        values = np.array([1.0, 2.0, 30.0])
        assert spectral.bootstrap_ci(values, statistic="mean")[0] == pytest.approx(11.0)


class TestRayleigh:
    def test_uniform_phases_are_not_significant(self):
        phases = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        resultant, pvalue = spectral.rayleigh_test(phases)
        assert resultant < 1e-9
        assert pvalue > 0.9

    def test_concentrated_phases_are_significant(self):
        rng = np.random.default_rng(8)
        phases = rng.normal(loc=1.0, scale=0.2, size=66)
        resultant, pvalue = spectral.rayleigh_test(phases)
        assert resultant > 0.9
        assert pvalue < 1e-10


class TestLineExcessFraction:
    def _spectrum(self, line_gain_db=0.0, line_hz=(50.0,)):
        freqs = np.arange(0, 110, 1 / 21.6)
        psd = np.full(freqs.size, 1e-12)
        for frequency in line_hz:
            psd[int(np.argmin(np.abs(freqs - frequency)))] *= 10 ** (line_gain_db / 10)
        return freqs, psd

    def test_a_flat_band_with_no_line_is_zero(self):
        freqs, psd = self._spectrum(line_gain_db=0.0)
        fraction = spectral.line_excess_fraction(
            freqs,
            psd,
            low_hz=30.0,
            high_hz=80.0,
            line_freqs=[1.2 * k for k in range(26, 66)],
            half_width_bins=100,
        )
        assert fraction == pytest.approx(0.0, abs=1e-9)

    def test_masking_the_line_bins_instead_would_invent_a_large_share(self):
        """The alternative estimator this one replaces, on a spectrum with no line in it.

        Dropping the line bins and comparing band powers removes their background along
        with their line, so ordinary spectrum is counted as contamination.
        """
        freqs, psd = self._spectrum(line_gain_db=0.0)
        lines = np.array([1.2 * k for k in range(26, 66)])
        band = (freqs >= 30.0) & (freqs <= 80.0)
        at_a_line = np.abs(freqs[:, None] - lines[None, :]).min(axis=1) <= 0.15
        masked_share = 1.0 - psd[band & ~at_a_line].sum() / psd[band].sum()
        assert masked_share > 0.15

    def test_a_known_line_is_recovered(self):
        freqs, psd = self._spectrum(line_gain_db=30.0, line_hz=(50.0,))
        bin_width = freqs[1] - freqs[0]
        band = (freqs >= 30.0) & (freqs <= 80.0)
        expected = (1e-12 * (10**3 - 1)) * bin_width / (float(np.sum(psd[band])) * bin_width)
        fraction = spectral.line_excess_fraction(
            freqs, psd, low_hz=30.0, high_hz=80.0, line_freqs=[50.0], half_width_bins=100
        )
        assert fraction == pytest.approx(expected, rel=0.02)

    def test_a_bin_dug_below_background_does_not_count_as_negative(self):
        freqs, psd = self._spectrum()
        psd[int(np.argmin(np.abs(freqs - 50.0)))] = 1e-18  # a hole, as removal leaves
        fraction = spectral.line_excess_fraction(
            freqs, psd, low_hz=30.0, high_hz=80.0, line_freqs=[50.0], half_width_bins=100
        )
        assert fraction == pytest.approx(0.0, abs=1e-12)

    def test_no_line_inside_the_band_gives_zero(self):
        freqs, psd = self._spectrum(line_gain_db=30.0, line_hz=(50.0,))
        assert (
            spectral.line_excess_fraction(
                freqs, psd, low_hz=8.0, high_hz=12.9, line_freqs=[50.0], half_width_bins=100
            )
            == 0.0
        )

    def test_rejects_an_inverted_band(self):
        freqs, psd = self._spectrum()
        with pytest.raises(ValueError, match="low_hz must be below"):
            spectral.line_excess_fraction(
                freqs, psd, low_hz=80.0, high_hz=30.0, line_freqs=[50.0], half_width_bins=100
            )
