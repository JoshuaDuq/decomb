"""BrainVision output integrity: finite values, scaling, naming, and references."""

from __future__ import annotations

import numpy as np
import pytest

from decomb import recordings


def _write_scaling_header(path):
    path.write_text(
        """Brain Vision Data Exchange Header File Version 1.0
[Common Infos]
DataOrientation=MULTIPLEXED

[Binary Infos]
BinaryFormat=IEEE_FLOAT_32

[Channel Infos]
Ch3=Micro,,3,µV
Ch1=Volt,,0.5,V
Ch5=Nano,,5,nV
Ch2=Milli,,2,mV
Ch4=Ascii micro,,4,uV
""",
        encoding="utf-8",
    )


def test_writing_a_non_finite_array_raises(brainvision_run, tmp_path):
    path, raw = brainvision_run
    data = raw.get_data()
    data[0, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        recordings.write_eeg_binary(path, tmp_path / "out.eeg", data, raw.ch_names)


def test_writing_an_infinite_array_raises(brainvision_run, tmp_path):
    path, raw = brainvision_run
    data = raw.get_data()
    data[1, 3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        recordings.write_eeg_binary(path, tmp_path / "out.eeg", data, raw.ch_names)


def test_a_finite_array_still_writes(brainvision_run, tmp_path):
    path, raw = brainvision_run
    recordings.write_eeg_binary(
        path,
        tmp_path / "out.eeg",
        raw.get_data(),
        raw.ch_names,
    )
    assert (tmp_path / "out.eeg").exists()


def test_quantization_uses_the_same_calibration_order_as_mne(brainvision_run):
    path, raw = brainvision_run
    values = raw.get_data() + np.finfo(np.float32).eps
    scaling = recordings.parse_channel_scaling(path)
    calibrations = np.asarray(scaling.calibrations)[:, np.newaxis]
    stored = (values / calibrations).astype("<f4")
    expected = stored.astype(float) * calibrations

    quantized = recordings.quantized_eeg_data(path, values, raw.ch_names)

    assert np.array_equal(quantized, expected)


def test_channel_scaling_respects_declared_units_and_channel_indices(tmp_path):
    path = tmp_path / "recording.vhdr"
    _write_scaling_header(path)

    scaling = recordings.parse_channel_scaling(path)

    assert scaling.channel_names == (
        "Volt",
        "Milli",
        "Micro",
        "Ascii micro",
        "Nano",
    )
    assert scaling.calibrations == pytest.approx(
        (0.5, 2e-3, 3e-6, 4e-6, 5e-9)
    )


def test_binary_writing_uses_declared_units(tmp_path):
    path = tmp_path / "recording.vhdr"
    destination = tmp_path / "recording.eeg"
    _write_scaling_header(path)
    scaling = recordings.parse_channel_scaling(path)
    calibrations = np.asarray(scaling.calibrations)
    stored = np.arange(1.0, 11.0).reshape(5, 2)
    values = stored * calibrations[:, np.newaxis]

    recordings.write_eeg_binary(
        path,
        destination,
        values,
        scaling.channel_names,
    )

    assert np.array_equal(
        np.fromfile(destination, dtype="<f4").reshape(2, 5).T,
        stored,
    )


def test_binary_writing_rejects_a_channel_order_mismatch(brainvision_run, tmp_path):
    path, raw = brainvision_run

    with pytest.raises(ValueError, match="channel order"):
        recordings.write_eeg_binary(
            path,
            tmp_path / "out.eeg",
            raw.get_data(),
            tuple(reversed(raw.ch_names)),
        )


def test_channel_scaling_rejects_an_unknown_unit(tmp_path):
    path = tmp_path / "recording.vhdr"
    _write_scaling_header(path)
    path.write_text(
        path.read_text(encoding="utf-8").replace("5,nV", "5,kV"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported BrainVision unit"):
        recordings.parse_channel_scaling(path)


def test_channel_scaling_uses_the_declared_codepage(tmp_path):
    path = tmp_path / "recording.vhdr"
    path.write_bytes(
        """Brain Vision Data Exchange Header File Version 1.0
[Common Infos]
Codepage=ANSI
DataOrientation=MULTIPLEXED
[Binary Infos]
BinaryFormat=IEEE_FLOAT_32
[Channel Infos]
Ch1=Température,,1,µV
""".encode("cp1252")
    )

    scaling = recordings.parse_channel_scaling(path)

    assert scaling.channel_names == ("Température",)


def test_sidecar_mirror_excludes_hidden_backups_and_every_eeg_binary(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "sub-0001" / "eeg").mkdir(parents=True)
    kept = source / "sub-0001" / "eeg" / "sub-0001_task-rest_eeg.json"
    kept.write_text("{}", encoding="utf-8")
    for name in (
        "sub-0001_task-rest_eeg.vhdr",
        "sub-0001_task-rest_eeg.vmrk",
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


def test_derivative_header_path_adds_a_description_entity(tmp_path):
    source_root = tmp_path / "source"
    derivative_root = tmp_path / "derivative"
    source = (
        source_root
        / "sub-0001"
        / "eeg"
        / "sub-0001_task-rest_run-1_eeg.vhdr"
    )

    destination = recordings.derivative_vhdr_path(
        source,
        source_root,
        derivative_root,
    )

    assert destination == (
        derivative_root
        / "sub-0001"
        / "eeg"
        / "sub-0001_task-rest_run-1_desc-decomb_eeg.vhdr"
    )


def test_derivative_brainvision_triplet_updates_internal_references(
    brainvision_run,
    tmp_path,
):
    import mne

    source, raw = brainvision_run
    destination = tmp_path / "sub-0001_task-rest_run-1_desc-decomb_eeg.vhdr"

    recordings.write_brainvision_sidecars(source, destination)
    recordings.write_eeg_binary(
        destination,
        destination.with_suffix(".eeg"),
        raw.get_data(),
        raw.ch_names,
    )

    header = destination.read_text(encoding="utf-8")
    marker = destination.with_suffix(".vmrk").read_text(encoding="utf-8")
    assert "DataFile=sub-0001_task-rest_run-1_desc-decomb_eeg.eeg" in header
    assert "MarkerFile=sub-0001_task-rest_run-1_desc-decomb_eeg.vmrk" in header
    assert "DataFile=sub-0001_task-rest_run-1_desc-decomb_eeg.eeg" in marker
    written = mne.io.read_raw_brainvision(destination, preload=True, verbose="ERROR")
    assert written.ch_names == raw.ch_names
    assert np.array_equal(
        written.get_data(),
        recordings.quantized_eeg_data(destination, raw.get_data(), raw.ch_names),
    )
