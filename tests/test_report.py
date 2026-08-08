"""The outcome report must follow the participant-specific transform provenance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from decomb import report
from decomb.remove import RemovalSettings


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "recording": "sub-0001_task-x_run-1_eeg",
                "fundamental_hz": 1.2,
                "isolated_hz": "47.04;57.22",
                "adjacent_hz": "27.72",
            },
            {
                "recording": "sub-0002_task-x_run-1_eeg",
                "fundamental_hz": 1.199,
                "isolated_hz": "94.31",
                "adjacent_hz": "81.50",
            },
        ]
    )


def test_report_targets_are_resolved_per_subject_from_the_manifest():
    targets = report.subject_artifact_targets(
        _manifest(),
        ("sub-0001", "sub-0002"),
        RemovalSettings(removal_harmonic_range=(22, 83), high_hz=99.8),
    )

    assert any(abs(value - 47.04) < 1e-9 for value in targets["sub-0001"])
    assert any(abs(value - 27.72) < 1e-9 for value in targets["sub-0001"])
    assert not any(abs(value - 47.04) < 1e-9 for value in targets["sub-0002"])
    assert any(abs(value - 94.31) < 1e-9 for value in targets["sub-0002"])
    assert any(abs(value - 81.50) < 1e-9 for value in targets["sub-0002"])
    assert not any(59.5 <= value <= 60.5 for values in targets.values() for value in values)


def test_the_mains_band_comes_from_the_settings_not_from_this_module():
    """A 50 Hz site must not have a 60 Hz band excluded from its own report."""
    targets = report.subject_artifact_targets(
        _manifest(),
        ("sub-0001", "sub-0002"),
        RemovalSettings(mains_notch_hz=(49.5, 50.5), removal_harmonic_range=(22, 83), high_hz=99.8),
    )

    values = [value for row in targets.values() for value in row]
    assert not any(49.5 <= value <= 50.5 for value in values)
    assert any(59.5 <= value <= 60.5 for value in values)


def test_a_notch_band_is_left_out_of_the_report_as_well():
    """`apply` stays out of every `notch_bands` band, so the report must not charge it."""
    targets = report.subject_artifact_targets(
        _manifest(),
        ("sub-0001",),
        RemovalSettings(excluded_bands_hz=((56.8, 57.7),), high_hz=99.8),
    )

    assert not any(56.8 <= value <= 57.7 for value in targets["sub-0001"])


def test_report_refuses_missing_subject_provenance():
    with pytest.raises(ValueError, match="no manifest rows for sub-0003"):
        report.subject_artifact_targets(
            _manifest(),
            ("sub-0001", "sub-0003"),
            RemovalSettings(),
        )


def test_artifact_share_uses_each_subjects_own_targets():
    freqs = np.arange(1.0, 100.0, 0.1)
    psd = np.ones((2, freqs.size))
    psd[0, np.argmin(abs(freqs - 47.0))] = 100.0
    psd[1, np.argmin(abs(freqs - 94.0))] = 100.0

    shares, counts = report.artifact_share_by_subject(
        freqs,
        psd,
        (45.0, 95.0),
        {"sub-0001": (47.0,), "sub-0002": (94.0,)},
        ("sub-0001", "sub-0002"),
        half_width_bins=10,
        line_half_width_hz=0.15,
    )

    assert np.all(shares > 0.0)
    assert counts.tolist() == [1, 1]
