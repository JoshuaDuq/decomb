"""Read-only diagnosis must match the first production removal round."""

from pathlib import Path

import mne
import numpy as np

from decomb import diagnose, notch, recordings


def test_diagnosis_reports_trigger_anchored_scanner_harmonics(monkeypatch):
    settings = notch.HarmonicNotchSettings(
        estimation_window_s=10.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(1.0, 5.0),
        scanner_repetition_time_s=1.0,
        scanner_trigger_event_name="Scanner/Volume",
    )
    raw = mne.io.RawArray(
        np.zeros((2, 12_000)),
        mne.create_info(["C3", "C4"], 1_000.0, "eeg"),
        verbose="ERROR",
    )
    raw.set_annotations(
        mne.Annotations(
            onset=np.arange(12, dtype=float),
            duration=0.0,
            description="Scanner/Volume",
        )
    )

    def p_values(raw, pass_settings):
        frequencies_hz = np.arange(1.0, 6.0)
        probabilities = np.ones((1, 2, frequencies_hz.size))
        if pass_settings.estimation_window_s == 4.0:
            probabilities[0, 0, 1] = 1e-12
            probabilities[0, 1, 3] = 1e-12
        return frequencies_hz, probabilities

    monkeypatch.setattr(recordings, "read_bids_raw", lambda path: raw)
    monkeypatch.setattr(notch, "_line_test_p_values", p_values)
    monkeypatch.setattr(notch, "_thomson_f_p_values", p_values)

    result = diagnose._diagnose_recording(
        Path("recording.vhdr"),
        settings,
        (),
    )

    assert result.model_row["scanner_harmonics_authorized"] == 1
    assert result.model_row["scanner_supporting_harmonics"] == "2;4"
    assert result.model_row["n_scanner_plan_harmonics"] == 5
    assert {row["outcome"] for row in result.stopband_rows} == {
        "scanner_harmonics_detected"
    }
