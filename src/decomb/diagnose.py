"""Diagnose the exact automatic line-notch plan without changing data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from decomb import notch, recordings


@dataclass(frozen=True)
class RecordingDiagnosis:
    """The complete diagnostic output for one independently fitted recording."""

    model_row: dict[str, float | int | str]
    detected_line_rows: tuple[dict[str, float | int | str], ...]
    stopband_rows: tuple[dict[str, float | int | str], ...]


def _diagnose_recording(
    vhdr: Path,
    settings: notch.HarmonicNotchSettings,
    bands: tuple[tuple[str, float, float], ...],
) -> RecordingDiagnosis:
    """Fit one recording without sharing data or statistical evidence."""
    raw = recordings.read_bids_raw(vhdr)
    model = notch.fit_harmonic_model(raw, settings)
    plans = notch.plan_channel_notches(model, settings)
    unavailable_channel_bandwidth_hz = sum(
        high_hz - low_hz
        for plan in plans
        for low_hz, high_hz in plan.geometry.unavailable_edges()
    )
    harmonic_line_count = sum(
        line.harmonic is not None
        for channel in model.channels
        for line in channel.lines
    )
    model_row = {
        "recording": vhdr.stem,
        "n_tested_eeg_channels": model.channel_count,
        "n_affected_eeg_channels": len(model.channels),
        "n_comb_channels": sum(
            channel.fundamental_hz is not None for channel in model.channels
        ),
        "n_detected_channel_lines": model.line_count,
        "n_harmonic_lines": harmonic_line_count,
        "n_isolated_lines": model.line_count - harmonic_line_count,
        "n_channel_stopbands": sum(
            len(plan.geometry.stopbands) for plan in plans
        ),
        "unavailable_channel_bandwidth_hz": unavailable_channel_bandwidth_hz,
        "estimation_window_count": model.window_count,
        "detection_test_count_per_channel": model.test_count_per_channel,
        "total_detection_test_count": (
            model.test_count_per_channel * model.channel_count
        ),
    }
    detected_line_rows = tuple(
        {
            "recording": vhdr.stem,
            "channel": channel.channel_name,
            "frequency_hz": line.position_hz,
            "raw_p_value": line.raw_p_value,
            "corrected_p_value": line.corrected_p_value,
            "window_indices": ",".join(
                str(value) for value in line.window_indices
            ),
            "harmonic": "" if line.harmonic is None else line.harmonic,
        }
        for channel in model.channels
        for line in channel.lines
    )
    return RecordingDiagnosis(
        model_row,
        detected_line_rows,
        tuple(
            notch.artifact_manifest_rows(
                vhdr.stem,
                model,
                plans,
                bands,
                settings,
            )
        ),
    )


def run(args: argparse.Namespace) -> None:
    """Fit every requested recording and write its statistical line catalogue."""
    from decomb import effective
    from decomb.config import load_config

    config = load_config(getattr(args, "config", None))
    bids_root = config.path("bids_root", override=getattr(args, "bids_root", None))
    output_dir = config.path("diagnosis_dir", override=getattr(args, "output_dir", None))
    settings = notch.HarmonicNotchSettings.from_config(config)
    bands = notch.analysed_bands_from_config(config)
    runs = recordings.discover_runs(
        bids_root,
        subjects=getattr(args, "subjects", None),
        task="*",
    )

    results = []
    for index, vhdr in enumerate(runs, start=1):
        diagnosis = _diagnose_recording(vhdr, settings, bands)
        results.append(diagnosis)
        model_row = diagnosis.model_row
        affected = int(model_row["n_affected_eeg_channels"])
        tested = int(model_row["n_tested_eeg_channels"])
        line_count = int(model_row["n_detected_channel_lines"])
        unavailable = float(model_row["unavailable_channel_bandwidth_hz"])
        print(
            f"[{index}/{len(runs)}] {vhdr.stem[:44]:44s} "
            f"{affected}/{tested} affected channel(s), "
            f"{line_count} detected channel-line(s), "
            f"{unavailable:.3f} channel-Hz unavailable"
        )

    model_rows = [diagnosis.model_row for diagnosis in results]
    detected_line_rows = [
        row for diagnosis in results for row in diagnosis.detected_line_rows
    ]
    stopband_rows = [
        row for diagnosis in results for row in diagnosis.stopband_rows
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    recordings.write_tsv_atomic(pd.DataFrame(model_rows), output_dir / "model.tsv")
    recordings.write_tsv_atomic(
        pd.DataFrame(detected_line_rows),
        output_dir / "detected_lines.tsv",
    )
    recordings.write_tsv_atomic(pd.DataFrame(stopband_rows), output_dir / "stopbands.tsv")
    effective_path = effective.write(
        config,
        settings,
        output_dir / "effective_config_diagnose.txt",
        stage="diagnose",
    )
    print(f"Diagnosed {len(runs)} recording(s) without modifying data")
    print(f"  wrote {output_dir / 'model.tsv'}")
    print(f"  wrote {output_dir / 'detected_lines.tsv'}")
    print(f"  wrote {output_dir / 'stopbands.tsv'}")
    print(f"  wrote {effective_path}")
