from types import SimpleNamespace

import numpy as np
import pytest

from decomb import lines, notch, residual, subtraction

BANDS = (("beta", 13.0, 30.0), ("gamma", 30.1, 80.0))


def _settings() -> notch.HarmonicNotchSettings:
    from decomb.config import load_config

    return notch.HarmonicNotchSettings.from_config(load_config())


def test_band_availability_counts_bare_intervals():
    shares = notch.band_availability_from_intervals(((20.0, 21.7),), BANDS)

    assert shares["beta_unavailable_share"] == pytest.approx(1.7 / 17.0)
    assert shares["beta_retained_share"] == pytest.approx(1.0 - 1.7 / 17.0)
    assert shares["gamma_unavailable_share"] == 0.0


def test_authorized_frequencies_are_the_supported_scanner_harmonics():
    evidence = notch.ScannerHarmonicEvidence(
        fundamental_hz=10.0,
        corrected_p_value=1e-12,
        supporting_harmonics=(2, 4),
    )
    round_evidence = SimpleNamespace(
        model=lines.LineModel((), 1, 2, 5),
        scanner_harmonics=evidence,
    )

    assert subtraction.authorized_frequencies(round_evidence, _settings()) == (20.0, 40.0)


def test_authorized_frequencies_match_what_the_planner_would_notch():
    evidence = notch.ScannerHarmonicEvidence(
        fundamental_hz=10.0,
        corrected_p_value=1e-12,
        supporting_harmonics=(2, 4),
    )
    plan = notch.plan_scanner_harmonic_notches(evidence, _settings(), maximum_hz=49.0)
    planned = tuple(
        harmonic * evidence.fundamental_hz
        for stopband in plan.stopbands
        for harmonic in stopband.harmonics
    )
    round_evidence = SimpleNamespace(
        model=lines.LineModel((), 1, 2, 5), scanner_harmonics=evidence
    )

    assert subtraction.authorized_frequencies(round_evidence, _settings()) == planned


def test_the_fit_window_is_twice_the_detection_window():
    settings = _settings()

    assert subtraction.fit_window_s(settings) == 2.0 * settings.estimation_window_s


def test_damage_halves_because_the_fit_window_doubles():
    settings = _settings()
    narrow = subtraction.damage_intervals((40.0,), subtraction.fit_window_s(settings))
    wide = subtraction.damage_intervals((40.0,), settings.estimation_window_s)

    assert narrow == ((40.0 - 0.1, 40.0 + 0.1),)
    assert wide == ((40.0 - 0.2, 40.0 + 0.2),)


def test_damage_interval_spans_two_frequency_bins_each_side():
    half = 2.0 / 20.0

    assert subtraction.damage_intervals((40.0,), 20.0) == ((40.0 - half, 40.0 + half),)


def test_overlapping_damage_intervals_merge():
    half = 2.0 / 20.0
    close = 40.0 + half

    merged = subtraction.damage_intervals((40.0, close), 20.0)

    assert merged == ((40.0 - half, close + half),)


def test_separated_damage_intervals_do_not_merge():
    half = 2.0 / 20.0

    intervals = subtraction.damage_intervals((40.0, 40.0 + 5.0 * half), 20.0)

    assert len(intervals) == 2


def test_subtraction_removes_the_authorized_frequency_from_the_data():
    import mne

    settings = _settings()
    sfreq = 200.0
    times = np.arange(0, 60.0, 1.0 / sfreq)
    data = np.vstack([np.sin(2 * np.pi * 40.0 * times)] * 2) * 1e-6
    data += np.random.default_rng(0).normal(scale=1e-8, size=data.shape)
    raw = mne.io.RawArray(
        data, mne.create_info(["C3", "C4"], sfreq, "eeg"), verbose="ERROR"
    )
    round_evidence = SimpleNamespace(
        model=lines.LineModel((), 1, 2, 5),
        plans=(
            SimpleNamespace(
                geometry=notch.HarmonicNotchPlan(
                    (notch.HarmonicStopband((), 39.5, 40.5, "isolated"),), 0.061
                )
            ),
        ),
        scanner_harmonics=None,
    )

    cleaned, record = subtraction.subtract_authorized(
        raw, round_evidence, settings, n_jobs=1
    )

    assert 40.0 in record.frequencies_hz
    assert record.window_s == 2.0 * settings.estimation_window_s
    assert np.abs(cleaned.get_data()).max() < 0.2 * np.abs(raw.get_data()).max()


def test_subtraction_manifest_rows_declare_unavailable_bandwidth():
    settings = _settings()
    half = 2.0 * settings.frequency_bin_width_hz
    record = subtraction.SubtractionRecord((40.0,), settings.estimation_window_s)

    rows = record.manifest_rows("sub-0001_run-1_eeg", BANDS, settings)

    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "subtracted"
    assert row["recording"] == "sub-0001_run-1_eeg"
    assert row["subtracted_frequencies_hz"] == "40.0"
    assert row["unavailable_low_hz"] == pytest.approx(40.0 - half)
    assert row["unavailable_high_hz"] == pytest.approx(40.0 + half)
    assert row["gamma_unavailable_share"] == pytest.approx(2.0 * half / (80.0 - 30.1))
    assert row["beta_unavailable_share"] == 0.0


def test_subtraction_manifest_rows_carry_every_required_manifest_column():
    settings = _settings()
    record = subtraction.SubtractionRecord((40.0,), settings.estimation_window_s)

    row = record.manifest_rows("sub-0001_run-1_eeg", BANDS, settings)[0]

    assert notch.MANIFEST_REQUIRED_COLUMNS <= set(row)


def _drifting_line(times):
    """A line that sweeps a whole bin, so subtraction at one bin leaves a residual."""
    phase = 2 * np.pi * (60.0 * times + 0.5 * times**2 / times[-1])
    return np.vstack([np.sin(phase)] * 2) * 1e-6


def _stationary_line(times):
    """A line that holds still, which subtraction removes outright."""
    return np.vstack([np.sin(2 * np.pi * 40.0 * times)] * 2) * 1e-6


def _synthetic_bids_recording(tmp_path, make_line=_drifting_line):
    import mne
    from mne_bids import BIDSPath, write_raw_bids

    sfreq = 200.0
    duration_s = 180.0
    times = np.arange(0, duration_s, 1.0 / sfreq)
    data = make_line(times)
    data += np.random.default_rng(0).normal(scale=1e-7, size=data.shape)
    raw = mne.io.RawArray(
        data, mne.create_info(["C3", "C4"], sfreq, "eeg"), verbose="ERROR"
    )
    raw.set_annotations(
        mne.Annotations(
            onset=np.arange(0.0, duration_s, 0.9),
            duration=0.0,
            description="Volume/V  1",
        )
    )
    root = tmp_path / "bids"
    path = BIDSPath(
        subject="0001",
        task="thermalactive",
        run="1",
        datatype="eeg",
        root=root,
        extension=".vhdr",
    )
    write_raw_bids(raw, path, format="BrainVision", allow_preload=True, verbose="ERROR")
    return root, path.fpath


def _fir_round_indices(rows) -> set[int]:
    return {
        int(row["removal_round"])
        for row in subtraction.cascade_rows(rows)
        if str(row["outcome"]) != "no_line_detected"
    }


def test_clean_harmonic_run_emits_subtracted_rows(monkeypatch, tmp_path):
    captured = {}

    def fake_subtract(raw, evidence, settings, *, n_jobs=-1):
        captured["called"] = True
        return raw.copy(), subtraction.SubtractionRecord(
            (40.0,), settings.estimation_window_s
        )

    monkeypatch.setattr(subtraction, "subtract_authorized", fake_subtract)
    from decomb import recordings

    source_root, vhdr = _synthetic_bids_recording(tmp_path)
    recordings.mirror_sidecars(source_root, tmp_path / "out")
    rows = notch.clean_harmonic_run(
        vhdr, tmp_path / "out", source_root, _settings(), BANDS, n_jobs=1
    )

    assert captured["called"]
    assert any(row["kind"] == "subtracted" for row in rows)


def test_manifest_without_subtracted_rows_still_verifies():
    rows = [
        {"recording": "sub-0001_run-1_eeg", "kind": "isolated", "harmonics": "2"},
    ]

    assert subtraction.subtraction_rows(rows) == []


def test_replayed_subtraction_reproduces_the_recorded_frequencies():
    settings = _settings()
    record = subtraction.SubtractionRecord((40.0, 60.0), settings.estimation_window_s)
    rows = record.manifest_rows("sub-0001_run-1_eeg", BANDS, settings)

    assert subtraction.recorded_frequencies(subtraction.subtraction_rows(rows)) == (
        40.0,
        60.0,
    )


def _apply_to_synthetic_recording(tmp_path, make_line=_drifting_line):
    from decomb import recordings

    source_root, vhdr = _synthetic_bids_recording(tmp_path, make_line)
    output_root = tmp_path / "out"
    recordings.mirror_sidecars(source_root, output_root)
    rows = notch.clean_harmonic_run(
        vhdr, output_root, source_root, _settings(), BANDS, n_jobs=1
    )
    cleaned_vhdr = recordings.derivative_vhdr_path(vhdr, source_root, output_root)
    return vhdr, cleaned_vhdr, rows


def test_verify_replays_subtraction_and_the_fir_cascade(tmp_path):
    vhdr, cleaned_vhdr, rows = _apply_to_synthetic_recording(tmp_path)

    assert subtraction.subtraction_rows(rows)
    assert residual.threshold_rows(rows), "fixture must exercise the residual stage"
    assert _fir_round_indices(rows), "fixture must leave a residual line for the FIR"

    verified = notch.verify_harmonic_run(vhdr, cleaned_vhdr, rows, _settings())

    assert verified
    assert any(row["kind"] == "threshold_notched" for row in verified)


def test_verify_replays_a_subtraction_that_leaves_no_residual_line(tmp_path):
    vhdr, cleaned_vhdr, rows = _apply_to_synthetic_recording(tmp_path, _stationary_line)

    assert subtraction.subtraction_rows(rows)
    assert not _fir_round_indices(rows)

    assert notch.verify_harmonic_run(vhdr, cleaned_vhdr, rows, _settings())


def test_verify_rejects_a_derivative_the_replayed_chain_does_not_reproduce(tmp_path):
    vhdr, cleaned_vhdr, rows = _apply_to_synthetic_recording(tmp_path)
    from decomb import recordings

    written = recordings.read_bids_raw(cleaned_vhdr)
    tampered_data = written.get_data()
    tampered_data[0, :100] += 5e-6
    recordings.write_eeg_binary(
        cleaned_vhdr, cleaned_vhdr.with_suffix(".eeg"), tampered_data, written.ch_names
    )

    with pytest.raises(RuntimeError, match="does not equal the declared"):
        notch.verify_harmonic_run(vhdr, cleaned_vhdr, rows, _settings())


def test_verify_rejects_a_manifest_whose_subtraction_is_unauthorized(tmp_path):
    vhdr, cleaned_vhdr, rows = _apply_to_synthetic_recording(tmp_path)
    tampered = [dict(row) for row in rows]
    for row in tampered:
        if row["kind"] == "subtracted":
            row["subtracted_frequencies_hz"] = "17.0"

    with pytest.raises(ValueError, match="subtract"):
        notch.verify_harmonic_run(vhdr, cleaned_vhdr, tampered, _settings())


def test_apply_then_verify_round_trips_subtraction_through_the_manifest(tmp_path):
    import argparse

    source_root, _ = _synthetic_bids_recording(tmp_path)
    args = argparse.Namespace(
        config=None,
        bids_root=source_root,
        output_root=tmp_path / "derivative",
        report_dir=tmp_path / "reports",
        n_jobs=1,
    )

    notch.run(args)

    manifest = notch._read_manifest(args.output_root / notch.MANIFEST_NAME)
    assert notch.MANIFEST_REQUIRED_COLUMNS <= set(manifest.columns)
    rows = manifest.to_dict("records")
    assert subtraction.subtraction_rows(rows)
    assert _fir_round_indices(rows)

    notch.run_verify(args)

    verification = notch._read_manifest(args.report_dir / notch.VERIFICATION_NAME)
    assert (verification["kind"] == "subtracted").any()


def test_verify_rejects_a_manifest_whose_residual_notches_are_unauthorized(tmp_path):
    vhdr, cleaned_vhdr, rows = _apply_to_synthetic_recording(tmp_path)
    tampered = [dict(row) for row in rows]
    for row in tampered:
        if row["kind"] == "threshold_notched":
            row["stopband_low_hz"] = 17.0
            row["stopband_high_hz"] = 17.5

    with pytest.raises(ValueError, match="residual stopbands"):
        notch.verify_harmonic_run(vhdr, cleaned_vhdr, tampered, _settings())


def test_a_manifest_without_residual_rows_still_verifies():
    rows = [{"recording": "sub-0001_run-1_eeg", "kind": "isolated", "harmonics": "2"}]

    assert residual.threshold_rows(rows) == []
    assert subtraction.cascade_rows(rows) == rows


def test_every_row_declares_the_recording_wide_availability_of_all_three_stages(tmp_path):
    _, _, rows = _apply_to_synthetic_recording(tmp_path)

    assert subtraction.subtraction_rows(rows)
    assert residual.threshold_rows(rows)

    declared = {float(row["gamma_retained_share"]) for row in rows}
    assert len(declared) == 1, "every row must declare the same recording-wide share"

    intervals = [
        (float(row["unavailable_low_hz"]), float(row["unavailable_high_hz"]))
        for row in rows
        if row.get("unavailable_low_hz", "") != ""
    ]
    expected = notch.band_availability_from_intervals(intervals, BANDS)

    assert declared.pop() == pytest.approx(expected["gamma_retained_share"])
