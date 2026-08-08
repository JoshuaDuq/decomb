"""``removal.min_harmonics_for_fit`` has to mean the same thing everywhere it is checked.

It was threaded into ``estimate_comb`` but ``build_adaptive_comb_model`` compared against
the module constant instead, so a floor set in a config file was honoured by one and
silently overridden by the other, and the refusal quoted a number the user had not chosen.
``test_everything_is_configurable`` did not catch it because it exercises only the first
path.

This matters beyond tidiness: the packaged default of 20 is the reason a benchmark of 90
recordings ended on a window that supported 19, and lowering it in a config would not have
changed that.
"""

from __future__ import annotations

import pytest

from decomb import estimators


def _estimate(n_harmonics: int) -> estimators.CombEstimate:
    harmonics = tuple(range(24, 24 + n_harmonics))
    return estimators.CombEstimate(
        fundamental_hz=1.2,
        harmonics_used=harmonics,
        harmonic_positions_hz=tuple(1.2 * harmonic for harmonic in harmonics),
        residual_rms_hz=0.01,
        max_abs_residual_hz=0.02,
        fundamental_jackknife_se_hz=1e-5,
        isolated_hz=(),
        isolated_prominence_db=(),
    )


def test_a_floor_below_the_default_is_honoured():
    """19 harmonics passes when the config asked for 15, rather than refusing at 20."""
    estimates = [_estimate(19), _estimate(19)]
    model = estimators.build_adaptive_comb_model(estimates[0], estimates, min_harmonics=15)
    assert model.window_estimates[0].n_harmonics == 19


def test_a_floor_above_the_default_is_honoured():
    """The check must not be dead code in the other direction either."""
    estimates = [_estimate(25), _estimate(25)]
    with pytest.raises(ValueError, match="at least 30 are required"):
        estimators.build_adaptive_comb_model(estimates[0], estimates, min_harmonics=30)


def test_the_refusal_quotes_the_floor_the_user_set():
    estimates = [_estimate(10), _estimate(10)]
    with pytest.raises(ValueError) as excinfo:
        estimators.build_adaptive_comb_model(estimates[0], estimates, min_harmonics=12)
    assert "12" in str(excinfo.value)
    assert "20" not in str(excinfo.value)


def test_the_default_is_still_the_module_constant():
    estimates = [_estimate(19), _estimate(19)]
    with pytest.raises(ValueError, match=f"at least {estimators.MIN_HARMONICS_FOR_FIT}"):
        estimators.build_adaptive_comb_model(estimates[0], estimates)
