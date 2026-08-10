"""Tests for the line catalogue: detection, comb fitting, and classification.

Synthetic spectra with lines planted at known frequencies stand in for real data, so
every assertion is against a ground truth the test itself set.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.configured_catalogue import catalogue

TR = 0.9  # an arbitrary acquisition period, supplied explicitly below

BANDS = {
    "delta": (1.0, 3.9),
    "theta": (4.0, 7.9),
    "alpha": (8.0, 12.9),
    "beta": (13.0, 30.0),
    "gamma": (30.1, 80.0),
}


def synthetic_cohort(line_freqs, *, n_subjects=15, amplitude_db=15.0, seed=0):
    """A 1/f cohort with narrow lines planted at the given frequencies."""
    freqs = np.arange(0, 110.0 + 1e-9, 1 / 21.6)
    rng = np.random.default_rng(seed)
    background = 10 ** ((-100.0 - 12.0 * np.log10(np.maximum(freqs, 0.5))) / 10.0)
    spectra = []
    for _ in range(n_subjects):
        spectrum = background * 10 ** (rng.normal(0, 0.05, freqs.size))
        for frequency in line_freqs:
            index = int(np.argmin(np.abs(freqs - frequency)))
            spectrum[index] *= 10 ** (amplitude_db / 10.0)
        spectra.append(spectrum)
    return freqs, np.stack(spectra)


class TestHalfWidthBins:
    def test_high_resolution_grid_gives_one_hundred_bins(self):
        freqs = np.arange(0, 110, 1 / 21.6)
        assert catalogue.half_width_bins(freqs) == 100

    def test_matched_grid_gives_a_proportionally_smaller_window(self):
        freqs = np.arange(0, 110, 1 / 3.6)
        assert catalogue.half_width_bins(freqs) == 17


class TestDetectionMask:
    def test_covers_the_analysis_range(self):
        freqs = np.arange(0, 110, 1 / 21.6)
        mask = catalogue.detection_mask(freqs)
        assert not mask[freqs < 3.0].any()
        assert not mask[freqs > 95.0].any()
        assert mask[np.argmin(np.abs(freqs - 45.0))]

    def test_excludes_a_band_when_asked_to(self):
        freqs = np.arange(0, 110, 1 / 21.6)
        mask = catalogue.detection_mask(freqs, exclude_hz=(59.5, 60.5))
        assert not mask[np.argmin(np.abs(freqs - 60.0))]
        assert mask[np.argmin(np.abs(freqs - 58.0))]

    def test_excludes_nothing_by_default(self):
        """No band is a blind spot unless the caller made it one."""
        freqs = np.arange(0, 110, 1 / 21.6)
        assert catalogue.detection_mask(freqs)[np.argmin(np.abs(freqs - 60.0))]


class TestDetectCohortLines:
    def test_recovers_planted_lines(self):
        planted = [37.0, 51.5740, 57.1759, 74.0]
        freqs, spectra = synthetic_cohort(planted)
        lines = catalogue.detect_cohort_lines(catalogue.build_grid(freqs, spectra))
        found = lines["refined_hz"].to_numpy()
        assert len(found) == len(planted)
        for frequency in planted:
            assert np.min(np.abs(found - frequency)) < 0.05

    def test_reports_no_lines_when_the_spectrum_is_smooth(self):
        freqs, spectra = synthetic_cohort([])
        with pytest.raises(RuntimeError, match="No line survived"):
            catalogue.detect_cohort_lines(catalogue.build_grid(freqs, spectra))

    def test_flags_position_on_the_acquisition_grid_when_a_tr_is_given(self):
        on_comb = 55 / TR
        off_comb = 57.1759
        freqs, spectra = synthetic_cohort([on_comb, off_comb])
        lines = catalogue.detect_cohort_lines(catalogue.build_grid(freqs, spectra), tr_seconds=TR)
        flags = dict(zip(np.round(lines["refined_hz"], 2), lines["on_tr_comb"]))
        assert flags[round(on_comb, 2)]
        assert not flags[round(off_comb, 2)]

    def test_reports_no_grid_position_without_a_tr(self):
        """The k/TR question is optional, and unanswered rather than guessed at."""
        freqs, spectra = synthetic_cohort([55 / TR])
        lines = catalogue.detect_cohort_lines(catalogue.build_grid(freqs, spectra))
        assert not lines["on_tr_comb"].any()
        assert lines["tr_offset_hz"].isna().all()

    def test_ignores_a_line_inside_an_excluded_band(self):
        freqs, spectra = synthetic_cohort([60.0, 51.574])
        lines = catalogue.detect_cohort_lines(
            catalogue.build_grid(freqs, spectra), exclude_hz=(59.5, 60.5)
        )
        assert np.min(np.abs(lines["refined_hz"].to_numpy() - 60.0)) > 0.5

    def test_finds_that_same_line_when_the_band_is_not_excluded(self):
        """The exclusion is a choice, not a blind spot baked into the detector."""
        freqs, spectra = synthetic_cohort([60.0, 51.574])
        lines = catalogue.detect_cohort_lines(catalogue.build_grid(freqs, spectra))
        assert np.min(np.abs(lines["refined_hz"].to_numpy() - 60.0)) < 0.1

    def test_prevalence_counts_every_participant_when_the_line_is_universal(self):
        freqs, spectra = synthetic_cohort([57.1759], amplitude_db=25.0)
        lines = catalogue.detect_cohort_lines(catalogue.build_grid(freqs, spectra))
        assert int(lines["n_subjects_detected"].iloc[0]) == 15

    def test_confidence_interval_brackets_the_point_estimate(self):
        freqs, spectra = synthetic_cohort([57.1759])
        lines = catalogue.detect_cohort_lines(catalogue.build_grid(freqs, spectra))
        row = lines.iloc[0]
        assert row["ci_low_db"] <= row["cohort_median_prominence_db"] <= row["ci_high_db"]


class TestBandImpact:
    def test_line_inside_a_band_inflates_it(self):
        freqs, spectra = synthetic_cohort([51.574], amplitude_db=30.0)
        grid = catalogue.build_grid(freqs, spectra)
        lines = catalogue.detect_cohort_lines(grid)
        classified = catalogue.classify_lines(lines, catalogue.comb_structure(lines))
        impact = catalogue.band_impact(grid, [f"sub-{i:04d}" for i in range(15)], classified, BANDS)
        mid = impact.loc[impact["band"] == "gamma", "artifact_share_percent"]
        assert mid.min() > 5.0

    def test_band_without_lines_is_untouched(self):
        freqs, spectra = synthetic_cohort([51.574])
        grid = catalogue.build_grid(freqs, spectra)
        lines = catalogue.detect_cohort_lines(grid)
        classified = catalogue.classify_lines(lines, catalogue.comb_structure(lines))
        impact = catalogue.band_impact(grid, [f"sub-{i:04d}" for i in range(15)], classified, BANDS)
        alpha = impact.loc[impact["band"] == "alpha"]
        assert (alpha["n_lines_inside"] == 0).all()
        assert np.allclose(alpha["artifact_share"], 0.0)

    def test_a_band_of_pure_background_reports_no_artifact(self):
        # The estimator error this replaced: dropping line bins made an empty band look
        # contaminated in proportion to how many bins were dropped.
        freqs, spectra = synthetic_cohort([51.574])
        grid = catalogue.build_grid(freqs, spectra)
        lines = catalogue.detect_cohort_lines(grid)
        classified = catalogue.classify_lines(lines, catalogue.comb_structure(lines))
        impact = catalogue.band_impact(grid, [f"sub-{i:04d}" for i in range(15)], classified, BANDS)
        theta = impact.loc[impact["band"] == "theta", "artifact_share"]
        assert np.allclose(theta, 0.0)


class TestCombStructure:
    def test_recovers_a_planted_fundamental(self):
        fundamental = 1.2
        planted = [fundamental * k for k in range(22, 40)]
        freqs, spectra = synthetic_cohort(planted)
        lines = catalogue.detect_cohort_lines(catalogue.build_grid(freqs, spectra))
        row = catalogue.comb_structure(lines).set_index("family").loc["narrow_comb"]
        assert row["fundamental_hz"] == pytest.approx(fundamental, abs=1e-3)
        assert row["harmonic_min"] == 22
        assert row["harmonic_max"] == 39
        assert row["rmse_hz"] < 0.02

    def test_separates_lines_that_do_not_join_the_comb(self):
        planted = [1.2 * k for k in range(22, 40)] + [57.2247]
        freqs, spectra = synthetic_cohort(planted)
        lines = catalogue.detect_cohort_lines(catalogue.build_grid(freqs, spectra))
        structure = catalogue.comb_structure(lines).set_index("family")
        assert int(structure.loc["narrow_comb", "n_lines"]) == 18
        assert int(structure.loc["narrow_off_comb", "n_lines"]) == 1

    def test_widely_spaced_lines_report_no_comb_rather_than_failing(self):
        freqs, spectra = synthetic_cohort([20.0, 45.0, 88.0])
        lines = catalogue.detect_cohort_lines(catalogue.build_grid(freqs, spectra))
        structure = catalogue.comb_structure(lines)
        assert "narrow_comb" not in set(structure["family"])
        assert int(structure.loc[structure["family"] == "narrow_off_comb", "n_lines"].iloc[0]) == 3


class TestClassifyLines:
    def test_comb_members_are_labelled_regardless_of_measured_width(self):
        planted = [1.2 * k for k in range(24, 60)]
        freqs, spectra = synthetic_cohort(planted)
        lines = catalogue.detect_cohort_lines(catalogue.build_grid(freqs, spectra))
        classified = catalogue.classify_lines(lines, catalogue.comb_structure(lines))
        assert (classified["kind"] == "comb").sum() >= 30
        assert classified.loc[classified["kind"] == "comb", "comb_harmonic"].min() >= 24

    def test_a_narrow_line_off_the_comb_is_isolated(self):
        planted = [1.2 * k for k in range(24, 60)] + [57.2247]
        freqs, spectra = synthetic_cohort(planted)
        lines = catalogue.detect_cohort_lines(catalogue.build_grid(freqs, spectra))
        classified = catalogue.classify_lines(lines, catalogue.comb_structure(lines))
        row = classified.loc[np.isclose(classified["refined_hz"], 57.2247, atol=0.05)].iloc[0]
        assert row["kind"] == "isolated"
        assert row["comb_harmonic"] == -1
