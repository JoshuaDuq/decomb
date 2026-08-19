import pytest

from decomb import notch

BANDS = (("beta", 13.0, 30.0), ("gamma", 30.1, 80.0))


def test_band_availability_counts_bare_intervals():
    shares = notch.band_availability_from_intervals(((20.0, 21.7),), BANDS)

    assert shares["beta_unavailable_share"] == pytest.approx(1.7 / 17.0)
    assert shares["beta_retained_share"] == pytest.approx(1.0 - 1.7 / 17.0)
    assert shares["gamma_unavailable_share"] == 0.0
