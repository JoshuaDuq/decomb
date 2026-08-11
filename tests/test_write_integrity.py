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


def test_quantization_uses_the_same_calibration_order_as_mne(brainvision_run):
    path, raw = brainvision_run
    values = raw.get_data() + np.finfo(np.float32).eps
    _, resolutions = recordings.parse_channel_scaling(path)
    stored = ((values * 1e6) / resolutions[:, np.newaxis]).astype("<f4")
    expected = stored.astype(float) * (resolutions[:, np.newaxis] * 1e-6)

    quantized = recordings.quantized_eeg_data(path, values)

    assert np.array_equal(quantized, expected)


def test_sidecar_mirror_excludes_hidden_backups_and_every_eeg_binary(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "sub-0001" / "eeg").mkdir(parents=True)
    kept = source / "sub-0001" / "eeg" / "sub-0001_task-rest_eeg.vhdr"
    kept.write_text("header", encoding="utf-8")
    for name in (
        "sub-0001_task-rest_eeg.eeg",
        "sub-0001_task-rest_eeg.eeg.precrop.bak",
        "sub-0001_task-rest_eeg.vhdr.bak",
        ".DS_Store",
        "._sub-0001_task-rest_eeg.vhdr",
        "temporary.tmp",
    ):
        (kept.parent / name).write_text("not a sidecar", encoding="utf-8")

    copied = recordings.mirror_sidecars(source, output)

    assert copied == 1
    assert [path.name for path in output.rglob("*") if path.is_file()] == [kept.name]
