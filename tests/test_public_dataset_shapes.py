"""BIDS layouts accepted by the recording discovery boundary."""

from __future__ import annotations

import numpy as np
import pytest

from decomb import recordings


def _write_run(
    root,
    subject: str,
    task: str,
    *,
    run: str | None,
    session: str | None = None,
) -> None:
    import mne
    from mne_bids import BIDSPath, write_raw_bids

    sampling_frequency_hz = 250.0
    times = np.arange(int(sampling_frequency_hz * 60.0)) / sampling_frequency_hz
    data = np.random.default_rng(0).normal(scale=1e-6, size=(2, times.size))
    raw = mne.io.RawArray(
        data,
        mne.create_info(["EEG 001", "EEG 002"], sampling_frequency_hz, "eeg"),
        verbose="ERROR",
    )
    path = BIDSPath(
        subject=subject,
        task=task,
        run=run,
        session=session,
        datatype="eeg",
        root=root,
        extension=".vhdr",
    )
    write_raw_bids(
        raw,
        path,
        format="BrainVision",
        allow_preload=True,
        verbose="ERROR",
    )


def test_a_single_recording_without_a_run_entity_is_found(tmp_path):
    _write_run(tmp_path, "0001", "rest", run=None)

    found = recordings.discover_runs(tmp_path, subjects=None, task="rest")

    assert len(found) == 1
    assert "_run-" not in found[0].name


def test_a_session_hierarchy_and_subject_filter_are_supported(tmp_path):
    _write_run(tmp_path, "0001", "rest", run=None, session="01")
    _write_run(tmp_path, "0002", "rest", run=None, session="01")

    found = recordings.discover_runs(
        tmp_path,
        subjects=["sub-0002"],
        task="rest",
    )

    assert [recordings.subject_of(path) for path in found] == ["sub-0002"]
    assert "ses-01" in str(found[0])


def test_task_filtering_is_not_paradigm_specific(tmp_path):
    _write_run(tmp_path, "0001", "rest", run=None)
    _write_run(tmp_path, "0001", "oddball", run="1")

    assert len(recordings.discover_runs(tmp_path, None, task="rest")) == 1
    assert len(recordings.discover_runs(tmp_path, None, task="oddball")) == 1
    assert len(recordings.discover_runs(tmp_path, None, task="*")) == 2


def test_an_absent_task_reports_the_requested_task(tmp_path):
    _write_run(tmp_path, "0001", "rest", run=None)

    with pytest.raises(FileNotFoundError, match="not-a-task"):
        recordings.discover_runs(tmp_path, subjects=None, task="not-a-task")


def test_desc_brainvision_derivative_is_readable_by_mne_bids(tmp_path):
    source_root = tmp_path / "source"
    derivative_root = tmp_path / "derivative"
    _write_run(source_root, "0001", "rest", run="1")
    source_vhdr = recordings.discover_runs(source_root, None, task="rest")[0]
    source = recordings.read_bids_raw(source_vhdr)
    recordings.mirror_sidecars(source_root, derivative_root)
    destination_vhdr = recordings.derivative_vhdr_path(
        source_vhdr,
        source_root,
        derivative_root,
    )
    recordings.write_brainvision_sidecars(source_vhdr, destination_vhdr)
    recordings.write_eeg_binary(
        destination_vhdr,
        destination_vhdr.with_suffix(".eeg"),
        source.get_data(),
        source.ch_names,
    )

    derivative = recordings.read_bids_raw(destination_vhdr)

    assert derivative.ch_names == source.ch_names
    assert np.array_equal(
        derivative.get_data(),
        recordings.quantized_eeg_data(
            destination_vhdr,
            source.get_data(),
            source.ch_names,
        ),
    )


def test_independent_windows_never_overlap_after_tail_alignment():
    bounds = ((0, 10), (5, 15), (10, 20), (13, 23), (20, 30))

    indices = recordings.non_overlapping_window_indices(bounds)

    assert indices == (0, 2, 4)
