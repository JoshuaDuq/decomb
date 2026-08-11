#!/usr/bin/env python3
"""Build the README notch-geometry comparison from one real BIDS recording."""

from __future__ import annotations

import argparse
from pathlib import Path

from decomb import comparison, notch, recordings
from decomb.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--subject", required=True, help="BIDS subject label, including sub-")
    parser.add_argument(
        "--recording-index",
        type=int,
        default=1,
        help="one-based recording index within the selected subject",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("notch_comparison_real.png"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.recording_index < 1:
        raise ValueError("--recording-index must be positive.")

    config = load_config(args.config)
    settings = notch.HarmonicNotchSettings.from_config(config)
    source_root = config.path("bids_root")
    runs = recordings.discover_runs(
        source_root,
        subjects=[args.subject],
        task="*",
    )
    if args.recording_index > len(runs):
        raise IndexError(
            f"{args.subject} has {len(runs)} recording(s), not {args.recording_index}."
        )

    source_vhdr = runs[args.recording_index - 1]
    raw = recordings.read_bids_raw(source_vhdr)
    measured = comparison.measure_notch_comparison(raw, settings)
    duration_minutes = measured.duration_s / 60.0
    description = f"{source_vhdr.stem} ({duration_minutes:.1f} min)"
    comparison.figure_notch_comparison(
        measured,
        args.output,
        recording_description=description,
    )

    frequency_range_hz = settings.frequency_range_hz
    decomb_width_hz = comparison.unavailable_width_hz(
        measured.decomb_plan,
        frequency_range_hz,
    )
    traditional_width_hz = comparison.unavailable_width_hz(
        measured.traditional_plan,
        frequency_range_hz,
    )
    print(f"recording: {source_vhdr}")
    print(f"decomb unavailable: {decomb_width_hz:.3f} Hz")
    print(f"MNE defaults unavailable: {traditional_width_hz:.3f} Hz")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
