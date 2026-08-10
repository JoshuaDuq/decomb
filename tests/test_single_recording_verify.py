"""A one-participant dataset must be verifiable, because the workflow accepts one.

A lone continuous acquisition -- a resting or baseline recording -- is a valid input.

``verify`` did not. It characterises the lines it finds with a percentile bootstrap over
the sampling unit, and a bootstrap over one participant has no interval to report, so the
stage aborted with "bootstrap_ci needs at least two finite values" after ``apply`` had
already written the derivative.

A between-participant interval that a one-participant dataset cannot supply is a missing
measurement, not a failed analysis. The line is still detected and its position, width and
prominence are all still measured; only the interval is undefined.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.configured_catalogue import catalogue


def _grid(n_subjects: int) -> catalogue.Grid:
    """A noisy 1/f background carrying one narrow line, per participant.

    The noise is not decoration: the background null is estimated from the lower tail of
    the prominence distribution, so a perfectly flat spectrum has no scale to estimate.
    """
    freqs = np.arange(20.0, 100.0, 0.05)
    rng = np.random.default_rng(11)
    spectra = []
    for _ in range(n_subjects):
        background = (freqs / 20.0) ** -1.5 * rng.lognormal(0.0, 0.35, size=freqs.size)
        line = 40.0 * background.mean() * np.exp(-0.5 * ((freqs - 47.0) / 0.05) ** 2)
        spectra.append(background + line)
    return catalogue.build_grid(freqs, np.stack(spectra))


def test_a_single_participant_still_yields_its_lines():
    lines = catalogue.detect_cohort_lines(_grid(1))

    assert len(lines) >= 1
    assert lines.frequency_hz.between(46.5, 47.5).any()


def test_the_interval_a_single_participant_cannot_supply_is_reported_as_undefined():
    """Not zero, and not the point value: those would read as a measured interval."""
    lines = catalogue.detect_cohort_lines(_grid(1))

    assert lines.ci_low_db.isna().all()
    assert lines.ci_high_db.isna().all()


def test_the_point_estimate_is_still_measured():
    lines = catalogue.detect_cohort_lines(_grid(1))

    assert lines.cohort_median_prominence_db.notna().all()


def test_two_participants_still_get_a_real_interval():
    """The fallback must not swallow the interval wherever one is actually available."""
    lines = catalogue.detect_cohort_lines(_grid(2))

    assert lines.ci_low_db.notna().all()
    assert lines.ci_high_db.notna().all()
    assert (lines.ci_low_db <= lines.cohort_median_prominence_db).all()
    assert (lines.cohort_median_prominence_db <= lines.ci_high_db).all()


def test_a_bootstrap_over_one_value_is_still_refused_on_its_own():
    """The fallback belongs to the caller. The estimator keeps saying no, because a
    percentile interval over a single observation would be a fabricated one."""
    from decomb import spectral

    with pytest.raises(ValueError, match="at least two finite values"):
        spectral.bootstrap_ci([3.0])
