"""Check every recording decomb will read, before a stage spends an hour finding out.

diagnose and benchmark read the whole cohort into one measurement, so a single
unreadable or odd-rate file aborts the stage rather than skipping the recording.
This reports every such file in one pass so they can be dealt with up front.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mne

from decomb import remove
from decomb.config import load_config

mne.set_log_level("ERROR")


def main() -> int:
    bids_root = Path(sys.argv[1])
    config = load_config(None)
    settings = remove.RemovalSettings.from_config(config)

    runs = remove.discover_runs(bids_root, subjects=None, task=settings.task)
    print(f"{len(runs)} recording(s) under {bids_root}\n")

    ok: list[tuple[Path, float, float, int]] = []
    bad: list[tuple[Path, str]] = []

    for index, vhdr in enumerate(runs, start=1):
        try:
            remove.parse_channel_scaling(vhdr)
            raw = remove.read_bids_raw(vhdr)
            rate = float(raw.info["sfreq"])
            duration = raw.n_times / rate
            ok.append((vhdr, rate, duration, len(raw.ch_names)))
            print(f"[{index:3d}/{len(runs)}] {vhdr.name[:60]:60s} {rate:7.1f} Hz "
                  f"{duration:8.1f} s  {len(raw.ch_names):3d} ch")
        except Exception as error:  # noqa: BLE001 - reporting every failure is the point
            bad.append((vhdr, f"{type(error).__name__}: {error}"))
            print(f"[{index:3d}/{len(runs)}] {vhdr.name[:60]:60s} FAILED {error}")

    print()
    if bad:
        print(f"{len(bad)} unreadable recording(s):")
        for vhdr, reason in bad:
            print(f"  {vhdr}\n    {reason}")
        print()

    if not ok:
        return 1

    rates = sorted({rate for _, rate, _, _ in ok})
    channels = sorted({count for _, _, _, count in ok})
    shortest = min(duration for _, _, duration, _ in ok)
    longest = max(duration for _, _, duration, _ in ok)

    print(f"sampling rate(s): {rates}")
    print(f"channel count(s): {channels}")
    print(f"duration: {shortest:.1f} s shortest, {longest:.1f} s longest")

    window = settings.estimation_window_s
    burst = float(config.get("benchmark.probe.burst_centre_s"))
    print(f"\nestimation window {window} s; benchmark burst centre {burst} s")

    problems = []
    if len(rates) > 1:
        problems.append("recordings do not share a sampling rate; diagnose will refuse")
    if len(channels) > 1:
        problems.append(f"channel counts differ across recordings: {channels}")
    if shortest < window:
        problems.append(f"shortest recording {shortest:.1f} s is under one {window} s window")
    if burst >= shortest:
        problems.append(
            f"benchmark burst centre {burst} s is outside the shortest recording {shortest:.1f} s"
        )

    if problems:
        print("\nproblems:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("\nno blocking problems found")

    return 1 if bad or problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
