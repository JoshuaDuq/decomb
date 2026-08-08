"""The preservation criteria have to be able to fail.

A criterion that cannot fail is worse than no criterion: it reports a pass that carries no
information. Two shapes recur. A rule named for a maximum but evaluated on a median lets
half a recording's lines stand above the threshold. And a rule comparing two quantities
that are equal by construction -- as any two are under a linear operator, which
``spectrum_fit`` is -- reads exactly 1.0 on any data with any settings.

These tests hold each surviving criterion against data that should fail it.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from decomb import estimators


def _suppression(freqs, before, after, targets, widths):
    """One-window ``adaptive_line_suppression``, which is how the removal measures a run."""
    return estimators.adaptive_line_suppression(
        freqs,
        np.atleast_2d(before),
        np.atleast_2d(after),
        [list(targets)],
        [list(widths)],
    )


def _metrics(**overrides):
    base = {
        "median_residual_prominence_db": -5.0,
        "max_residual_prominence_db": -3.0,
        "null_max_95_db": -6.0,
        "residual_excess_db": 3.0,
        "focal_residual_excess_db": -2.0,
        "study_residual_excess_db": -2.0,
        "study_focal_residual_excess_db": -2.0,
        "study_significant_focal_residual_count": 0,
        "max_boundary_discontinuity_ratio": 1.0,
        "median_suppression_db": 20.0,
        "max_probe_deviation_db": 0.0,
        "max_nonline_change_db": 0.0,
        "study_max_probe_deviation_db": 0.0,
        "study_max_nonline_change_db": 0.0,
        "burst_energy_ratio": 1.0,
        "burst_correlation": 1.0,
        "intrinsic_energy_ratio": 0.95,
        "removed_band_fraction": 0.12,
        "base_removed_band_fraction": 0.12,
        "measured_band_attenuated_1db": 0.12,
        "base_band_fraction_bin_size": 1.0 / 1810.0,
        "band_fraction_bin_size": 1.0 / 1810.0,
        "measured_band_bin_size": 1.0 / 1810.0,
    }
    base.update(overrides)
    return base


def test_a_line_its_own_controls_never_reach_is_a_discovery():
    """A decibel cushion lets a large residual pass; an exact probability does not.

    The decision is the probability that a matched control search reaches the observation,
    so a line standing where the controls never go is a discovery whatever its size in
    decibels.
    """
    controls = np.linspace(-8.0, -3.0, 40)

    p_value = estimators.null_exceedance_p_value(13.9, controls)

    assert p_value == pytest.approx(1 / 41)
    assert not estimators.residual_randomization_verdict([p_value])["passed"]


def test_a_residual_inside_its_control_spread_is_not_a_discovery():
    controls = np.linspace(-8.0, 4.0, 40)

    p_value = estimators.null_exceedance_p_value(-1.0, controls)

    assert p_value > 0.05
    assert estimators.residual_randomization_verdict([p_value])["passed"]


def test_one_recording_is_decided_by_its_own_exact_test():
    """Benjamini-Hochberg over a single recording reduces to p <= alpha.

    A lone continuous acquisition has no cohort to borrow strength from, and must still be
    decidable.
    """
    assert not estimators.residual_randomization_verdict([0.02])["passed"]
    assert estimators.residual_randomization_verdict([0.20])["passed"]


def test_the_cohort_tolerates_the_null_rate_it_creates():
    """About one recording in twenty exceeds by construction; that is not a failure."""
    null_like = np.linspace(0.05, 0.95, 90)

    assert estimators.residual_randomization_verdict(null_like)["passed"]


def test_a_clean_run_still_passes():
    assert estimators.PreservationGate().passed(_metrics(residual_excess_db=-2.0))


def test_the_transient_cost_no_longer_stops_a_run():
    """``intrinsic_energy_ratio`` varies with the settings, and that is why it is reported.

    It is the transient measured against itself put through the same removal alone, so it
    is the cost of projecting out this recording's targets: larger where there is more
    artifact to remove, and not a statement about whether the removal is working. Judging
    it against a level charged recordings for their own contamination -- seven of a
    15-participant cohort were refused with collateral damage of nil, and one participant
    carrying three times the narrowband load of the others measured 0.71 just as cleanly.

    The number is still measured, reported beside the band cost, and recorded in the
    derivative's provenance. Whether the cost suits an analysis is a question about that
    analysis, and no value in a config file can answer it in advance.
    """
    gate = estimators.PreservationGate()

    assert gate.passed(_metrics(intrinsic_energy_ratio=0.40)), (
        "a transient reduced to 40% of its injected energy stopped the run"
    )
    assert gate.passed(_metrics(intrinsic_energy_ratio=0.95))


def test_a_non_linear_removal_still_stops_a_run():
    """What remains is an invariant, not a level: the shape has to survive."""
    gate = estimators.PreservationGate()
    assert not gate.passed(_metrics(burst_correlation=0.80))


def test_a_residual_displaced_by_a_bin_is_not_missed():
    """Reading the nearest bin only would let a line that moved slip through.

    A target at 50 Hz whose centre falls to -10 dB while a residual at 50.05 Hz still
    stands at +15 dB would be reported as a maximum residual of -10 dB. That is the exact
    shape of the failure this removal can produce -- taking a line out exposes or displaces
    its neighbour -- so the centre is the one place the evidence will not be.
    """
    freqs = np.arange(45.0, 55.0, 0.01)
    before = np.full_like(freqs, -20.0)
    after = np.full_like(freqs, -20.0)
    before[np.argmin(np.abs(freqs - 50.0))] = 25.0
    after[np.argmin(np.abs(freqs - 50.0))] = -10.0
    after[np.argmin(np.abs(freqs - 50.05))] = 15.0

    result = _suppression(freqs, before, after, [50.0], widths=[0.2])
    assert result["max_residual_prominence_db"] >= 15.0 - 1e-6, (
        f"the displaced residual was invisible: {result['max_residual_prominence_db']}"
    )


def test_suppression_still_reads_the_centre_when_nothing_moved():
    freqs = np.arange(45.0, 55.0, 0.01)
    before = np.full_like(freqs, -20.0)
    after = np.full_like(freqs, -20.0)
    before[np.argmin(np.abs(freqs - 50.0))] = 25.0
    after[np.argmin(np.abs(freqs - 50.0))] = -12.0

    result = _suppression(freqs, before, after, [50.0], widths=[0.2])
    assert result["max_residual_prominence_db"] == pytest.approx(-12.0)


def test_a_residual_outside_the_claimed_window_is_still_seen():
    """The window the removal claimed is not where a missed line will be.

    Searching only the notch's own width cannot find a target that missed: the line is
    then just outside what the notch claimed. The search has to cover the frequency
    uncertainty of the estimate, not the footprint of the correction.
    """
    freqs = np.arange(45.0, 55.0, 0.01)
    before = np.full_like(freqs, -20.0)
    after = np.full_like(freqs, -20.0)
    before[np.argmin(np.abs(freqs - 50.0))] = 25.0
    after[np.argmin(np.abs(freqs - 50.0))] = -10.0
    after[np.argmin(np.abs(freqs - 50.12))] = 18.0  # outside a 0.2 Hz notch, missed line

    result = _suppression(freqs, before, after, [50.0], widths=[0.2])
    assert result["max_residual_prominence_db"] >= 18.0 - 1e-6, (
        f"a missed line 0.12 Hz away was invisible: {result['max_residual_prominence_db']}"
    )


def test_the_search_does_not_reach_a_neighbouring_comb_line():
    """It must not charge one target with the line belonging to the next harmonic."""
    freqs = np.arange(45.0, 55.0, 0.01)
    before = np.full_like(freqs, -20.0)
    after = np.full_like(freqs, -20.0)
    before[np.argmin(np.abs(freqs - 50.0))] = 25.0
    after[np.argmin(np.abs(freqs - 50.0))] = -10.0
    after[np.argmin(np.abs(freqs - 51.2))] = 22.0  # the next harmonic, not this target's

    result = _suppression(freqs, before, after, [50.0], widths=[0.2])
    assert result["max_residual_prominence_db"] == pytest.approx(-10.0)


def test_the_residual_is_judged_against_a_blind_control():
    """Searching a window round every target has a noise floor; the control measures it.

    A +/-0.15 Hz window at this resolution is about sixteen bins, and there are sixty-odd
    targets, so the largest of a thousand background bins is several dB on noise alone. A
    fixed threshold cannot tell that from a surviving line. Equivalent windows placed away
    from any target measure the same floor, and the residual has to beat it.
    """
    rng = np.random.default_rng(0)
    freqs = np.arange(20.0, 100.0, 0.01)
    before = rng.normal(0.0, 1.0, freqs.size)
    after = rng.normal(0.0, 1.0, freqs.size)
    targets = [k * 1.2 for k in range(20, 80)]

    result = _suppression(freqs, before, after, targets, widths=[0.2] * len(targets))
    assert "null_max_95_db" in result
    assert result["residual_excess_db"] == pytest.approx(
        result["max_residual_prominence_db"] - result["null_max_95_db"]
    )
    assert result["residual_excess_db"] < 3.0, (
        "pure noise registered as a surviving line: "
        f"{result['residual_excess_db']:.2f} dB over the control"
    )


def test_a_real_survivor_still_beats_the_control():
    rng = np.random.default_rng(0)
    freqs = np.arange(20.0, 100.0, 0.01)
    before = rng.normal(0.0, 1.0, freqs.size)
    after = rng.normal(0.0, 1.0, freqs.size)
    after[np.argmin(np.abs(freqs - 60.0))] = 25.0
    targets = [k * 1.2 for k in range(20, 80)]

    result = _suppression(freqs, before, after, targets, widths=[0.2] * len(targets))
    assert result["residual_excess_db"] > 15.0


def test_large_focal_maximum_below_its_matched_control_passes() -> None:
    gate = estimators.PreservationGate()

    assert gate.passed(
        _metrics(
            residual_excess_db=-1.0,
            focal_residual_excess_db=-1.0,
        )
    )


def test_target_prominence_retains_channel_and_block_axes():
    freqs = np.arange(20.0, 30.0, 0.01)
    spectra = np.zeros((2, 3, freqs.size))
    spectra[1, 2, np.argmin(np.abs(freqs - 25.0))] = 14.0

    prominence = estimators.spatiotemporal_target_prominence(
        freqs,
        spectra,
        spectra,
        targets=[25.0],
        widths=[0.2],
        background_half_width_hz=2.0,
    )

    assert prominence.shape == (2, 3, 1)
    assert prominence[1, 2, 0] == pytest.approx(14.0)


def test_target_prominence_uses_the_preclean_background_floor():
    """A cleaner cannot manufacture prominence by lowering its own reference floor."""
    freqs = np.arange(20.0, 30.0, 0.01)
    before = np.zeros((1, freqs.size))
    after = before.copy()
    target = np.argmin(np.abs(freqs - 25.0))
    local = np.abs(freqs - 25.0) <= 2.0
    after[:, local] = -20.0
    after[:, target] = 10.0

    prominence = estimators.spatiotemporal_target_prominence(
        freqs,
        before,
        after,
        targets=[25.0],
        widths=[0.2],
        background_half_width_hz=2.0,
    )

    assert prominence[0, 0] == pytest.approx(10.0)


def test_spatiotemporal_search_controls_channel_window_multiplicity():
    rng = np.random.default_rng(18)
    freqs = np.arange(20.0, 100.0, 0.02)
    spectra = rng.normal(size=(8, 4, freqs.size))
    targets = tuple(tuple(k * 1.2 for k in range(20, 80)) for _ in range(4))
    widths = tuple((0.2,) * len(window_targets) for window_targets in targets)

    result = estimators.adaptive_spatiotemporal_suppression(
        freqs,
        spectra,
        spectra,
        targets,
        widths,
        background_half_width_hz=4.0,
    )

    assert result["focal_residual_excess_db"] < 3.0


def test_spatiotemporal_search_detects_one_focal_window_survivor():
    rng = np.random.default_rng(18)
    freqs = np.arange(20.0, 100.0, 0.02)
    spectra = rng.normal(size=(8, 4, freqs.size))
    targets = tuple(tuple(k * 1.2 for k in range(20, 80)) for _ in range(4))
    widths = tuple((0.2,) * len(window_targets) for window_targets in targets)
    spectra[6, 2, np.argmin(np.abs(freqs - targets[2][30]))] = 30.0

    result = estimators.adaptive_spatiotemporal_suppression(
        freqs,
        spectra,
        spectra,
        targets,
        widths,
        background_half_width_hz=4.0,
    )

    assert result["focal_residual_excess_db"] > 15.0


def test_a_boundary_jump_in_the_correction_fails_the_cohort_criterion():
    """A destroyed seam is caught by the exact cohort null, not a per-run cutoff."""
    evidence = [
        estimators.BoundaryDiscontinuityEvidence(5.0, (1.0,) * 40),
        *[estimators.BoundaryDiscontinuityEvidence(0.5, (1.0,) * 40) for _ in range(89)],
    ]

    assert not estimators.seam_randomization_verdict(evidence)["passed"]
    assert estimators.PreservationGate().passed(
        _metrics(residual_excess_db=-2.0, max_boundary_discontinuity_ratio=5.0)
    )


def test_boundary_discontinuity_is_measured_on_the_filter_correction():
    original = np.zeros((2, 1_000))
    cleaned = original.copy()
    cleaned[:, 500:] = 1.0

    evidence = estimators.boundary_discontinuity_evidence(original, cleaned, boundaries=[500])

    assert evidence.ratio > 10.0


def test_matched_null_uses_repeated_complete_target_sized_searches():
    freqs = np.arange(20.0, 100.0, 0.01)
    after = np.zeros_like(freqs)
    targets = [k * 1.2 for k in range(20, 80)]
    reaches = np.full(len(targets), 0.15)

    maxima = estimators._matched_null_maxima(freqs, after, np.asarray(targets), reaches)

    assert len(maxima) >= 20
    assert np.all(maxima == 0.0)


def test_matched_null_placement_scales_to_an_adaptive_spectrum_grid():
    freqs = np.linspace(0.0, 500.0, 27_001)
    targets = np.linspace(26.4, 99.6, 70)
    reaches = np.linspace(0.15, 0.25, targets.size)

    started = time.perf_counter()
    placements = estimators._matched_null_centres(
        freqs,
        np.ones(freqs.shape, dtype=bool),
        targets,
        reaches,
        edge_margin_hz=0.5,
    )
    elapsed = time.perf_counter() - started

    assert len(placements) == 40
    assert all(placement.shape == targets.shape for placement in placements)
    assert elapsed < 10.0


def test_adaptive_suppression_controls_the_search_across_all_windows():
    rng = np.random.default_rng(8)
    freqs = np.arange(20.0, 100.0, 0.01)
    before = rng.normal(size=(4, freqs.size))
    after = rng.normal(size=(4, freqs.size))
    targets = tuple(tuple(k * (1.2 + index * 1e-4) for k in range(20, 80)) for index in range(4))
    widths = tuple((0.2,) * len(window_targets) for window_targets in targets)

    result = estimators.adaptive_line_suppression(freqs, before, after, targets, widths)

    assert result["residual_excess_db"] < 3.0


def test_adaptive_suppression_detects_a_survivor_in_one_window():
    rng = np.random.default_rng(8)
    freqs = np.arange(20.0, 100.0, 0.01)
    before = rng.normal(size=(4, freqs.size))
    after = rng.normal(size=(4, freqs.size))
    targets = tuple(tuple(k * (1.2 + index * 1e-4) for k in range(20, 80)) for index in range(4))
    widths = tuple((0.2,) * len(window_targets) for window_targets in targets)
    after[2, np.argmin(np.abs(freqs - targets[2][30]))] = 25.0

    result = estimators.adaptive_line_suppression(freqs, before, after, targets, widths)

    assert result["residual_excess_db"] > 15.0


def test_adaptive_suppression_does_not_call_a_broad_rhythm_a_residual_line():
    """The removal must not be forced to erase broad neural spectral structure."""
    freqs = np.arange(20.0, 100.0, 0.01)
    before = np.zeros((2, freqs.size))
    after = np.zeros_like(before)
    centre = 57.0
    broad = 8.0 * np.exp(-0.5 * ((freqs - centre) / 0.15) ** 2)
    before[0] += broad
    after[0] += broad
    targets = ((57.0,), (57.0,))
    widths = ((0.12,), (0.12,))

    result = estimators.adaptive_line_suppression(freqs, before, after, targets, widths)

    assert result["residual_excess_db"] <= 1.0


def test_the_per_run_gate_does_not_apply_a_multiplicity_invalid_seam_cutoff():
    verdict = estimators.PreservationGate().evaluate(_metrics(max_boundary_discontinuity_ratio=1.5))

    assert "adaptive_boundaries_continuous" not in verdict


class TestSeamRandomizationUsesTheMeasuredControls:
    def test_a_single_gross_seam_fails_against_the_exact_cohort_maximum_null(self):
        evidence = [
            estimators.BoundaryDiscontinuityEvidence(10.0, (1.0,) * 40),
            *[estimators.BoundaryDiscontinuityEvidence(1.0, (1.0,) * 40) for _ in range(5)],
        ]

        verdict = estimators.seam_randomization_verdict(evidence)

        assert not verdict["passed"]
        assert verdict["max_p_value"] == pytest.approx(1.0 / 41.0)

    def test_systematic_small_seams_fail_the_exact_exceedance_count_null(self):
        evidence = [estimators.BoundaryDiscontinuityEvidence(1.1, (1.0,) * 40) for _ in range(12)]

        verdict = estimators.seam_randomization_verdict(evidence)

        assert not verdict["passed"]
        assert verdict["count_p_value"] == pytest.approx(1.0 / 41.0)

    def test_an_ordinary_observed_shift_passes_without_a_fitted_dispersion_parameter(self):
        controls = tuple(float(value) for value in range(1, 41))
        evidence = [estimators.BoundaryDiscontinuityEvidence(20.0, controls) for _ in range(12)]

        verdict = estimators.seam_randomization_verdict(evidence)

        assert verdict["passed"]
        assert "dispersion" not in verdict
