"""Which adaptive window of a recording fails the comb fit, and by how much.

Uses `remove.run_spectra`, so the per-window spectra are the ones the plan is actually
fitted from, then replays the estimator's own robust fit on each. Distinguishes a window
too quiet to show its comb from one whose peaks genuinely disagree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mne
import numpy as np

from decomb import estimators, remove
from decomb.config import load_config

mne.set_log_level("ERROR")


def window_fit(freqs, prominence, settings, f0, low, high, need):
    """Candidates admitted in one window, and how many survive the robust fit."""
    rows = []
    for k in range(low, high + 1):
        mask = np.abs(freqs - k * f0) <= settings.search_hz
        if not mask.any():
            continue
        pick = np.flatnonzero(mask)[np.argmax(prominence[mask])]
        if prominence[pick] >= settings.min_prominence_db:
            rows.append((float(k), float(freqs[pick]), float(prominence[pick])))

    if not rows:
        return 0, 0, float("nan")

    ks = np.array([row[0] for row in rows])
    pos = np.array([row[1] for row in rows])
    weights = np.array([row[2] for row in rows])
    try:
        kept, _, _, fundamental = estimators._fit_consistent_harmonics(
            ks, pos, weights,
            min_harmonics=need,
            max_harmonic_residual_hz=settings.max_harmonic_residual_hz,
        )
        return len(rows), len(kept), fundamental
    except ValueError as error:
        return len(rows), int(str(error).split()[1]), float("nan")


def main() -> int:
    # None so DECOMB_CONFIG can point this at a trial config without editing the file.
    config = load_config(None)
    settings = remove.RemovalSettings.from_config(config)
    f0 = settings.nominal_fundamental_hz
    low, high = settings.harmonic_range
    need = settings.min_harmonics_for_fit

    total_failures = 0
    for argument in sys.argv[1:]:
        vhdr = Path(argument)
        raw = remove.read_bids_raw(vhdr)
        sfreq = float(raw.info["sfreq"])
        _, per_window, bounds = remove.run_spectra(raw, settings)

        failures = []
        lines = []
        for index, ((freqs, _, prominence), (start, stop)) in enumerate(zip(per_window, bounds)):
            admitted, in_grid, fundamental = window_fit(
                freqs, prominence, settings, f0, low, high, need
            )
            failed = in_grid < need
            if failed:
                failures.append((index, admitted, in_grid))
            lines.append(
                f"{index:>4} {start / sfreq:>8.1f} {stop / sfreq:>8.1f} "
                f"{admitted:>9} {in_grid:>8} {fundamental:>10.6f}"
                f"{'  <-- FAILS' if failed else ''}"
            )

        # The margin, not just the verdict: a cohort that only just clears the floor is a
        # different situation from one that clears it by twenty harmonics, and the choice
        # of window length should be made on that distribution rather than on pass/fail.
        grids = [int(line.split()[4]) for line in lines]
        worst = min(grids)
        worst_window = int(grids.index(worst))
        status = f"{len(failures)} failing window(s)" if failures else "all windows clear"
        print(f"{vhdr.name}: {len(bounds)} windows, need {need}, "
              f"worst {worst} (win {worst_window})  --  {status}")
        if failures:
            print(f"{'win':>4} {'start_s':>8} {'stop_s':>8} {'admitted':>9} "
                  f"{'in_grid':>8} {'f0_hz':>10}")
            for line in lines:
                print(line)
        print()
        total_failures += len(failures)

    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
