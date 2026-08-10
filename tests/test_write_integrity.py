"""A non-finite sample must never reach disk.

The round-trip check compares ``max|written - intended|`` against a tolerance. A NaN makes
that deviation NaN, and ``NaN > tolerance`` is False, so the one check standing between a
numerical failure and 11 GB of derived data passes it. The removal also runs with numpy
warnings suppressed, and the focused tests raise divide-by-zero, overflow and invalid-value
warnings from the same matmul the cleaning uses.
"""

from __future__ import annotations

import numpy as np
import pytest

from decomb import recordings


def test_writing_a_non_finite_array_raises(brainvision_run, tmp_path):
    path, raw = brainvision_run
    data = raw.get_data()
    data[0, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        recordings.write_eeg_binary(path, tmp_path / "out.eeg", data)


def test_writing_an_infinite_array_raises(brainvision_run, tmp_path):
    path, raw = brainvision_run
    data = raw.get_data()
    data[1, 3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        recordings.write_eeg_binary(path, tmp_path / "out.eeg", data)


def test_a_finite_array_still_writes(brainvision_run, tmp_path):
    path, raw = brainvision_run
    recordings.write_eeg_binary(path, tmp_path / "out.eeg", raw.get_data())
    assert (tmp_path / "out.eeg").exists()
