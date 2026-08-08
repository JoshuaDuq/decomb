"""Why does one recording's comb fit fall below min_harmonics_for_fit?

Runs the same scaffold `remove` fits per recording, over a chosen harmonic range, and
reports each candidate peak's residual against the fitted grid so a too-sparse comb can
be told apart from a comb whose peaks scatter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mne
import numpy as np

from decomb import remove
from decomb.config import load_config

mne.set_log_level("ERROR")


def main() -> int:
    vhdr = Path(sys.argv[1])
    config = load_config("decomb.yaml")
    settings = remove.RemovalSettings.from_config(config)

    raw = remove.read_bids_raw(vhdr)
    freqs, spectrum_db, prominence = remove.run_spectrum(raw, settings)

    f0 = settings.nominal_fundamental_hz
    low, high = settings.harmonic_range
    print(f"{vhdr.name}")
    print(f"nominal f0 {f0} Hz, fit range {low}-{high}, "
          f"max_harmonic_residual {settings.max_harmonic_residual_hz} Hz")
    print(f"min_prominence_db {settings.min_prominence_db}, search_hz {settings.search_hz}\n")

    # Peak nearest each nominal harmonic, with the prominence the fit would weight it by.
    rows = []
    for k in range(low, high + 1):
        target = k * f0
        window = np.abs(freqs - target) <= settings.search_hz
        if not window.any():
            continue
        index = np.flatnonzero(window)[np.argmax(prominence[window])]
        rows.append((k, float(freqs[index]), float(prominence[index])))

    admitted = [row for row in rows if row[2] >= settings.min_prominence_db]
    print(f"{len(admitted)} of {len(rows)} harmonics clear min_prominence_db")

    if not admitted:
        return 1

    ks = np.array([row[0] for row in admitted], dtype=float)
    pos = np.array([row[1] for row in admitted])
    weights = np.array([row[2] for row in admitted])

    # Weighted least squares through the origin -- the seed the robust fit starts from.
    fitted = float(np.sum(weights * ks * pos) / np.sum(weights * ks**2))
    residual = pos - ks * fitted
    within = np.abs(residual) <= settings.max_harmonic_residual_hz
    print(f"seed fundamental {fitted:.6f} Hz")
    print(f"{int(within.sum())} of {len(admitted)} within "
          f"{settings.max_harmonic_residual_hz} Hz of that grid "
          f"(need {settings.min_harmonics_for_fit})\n")

    print(f"{'k':>4} {'peak_hz':>10} {'expected':>10} {'resid_mHz':>10} {'prom_dB':>8}  ")
    for (k, peak, prom), res, ok in zip(rows and admitted, residual, within):
        print(f"{k:>4} {peak:>10.4f} {k * fitted:>10.4f} {res * 1000:>10.1f} {prom:>8.2f}  "
              f"{'' if ok else 'OUT'}")

    print()
    print("residual spread of the in-grid peaks: "
          f"RMS {np.sqrt(np.mean(residual[within] ** 2)) * 1000:.1f} mHz, "
          f"max |resid| {np.abs(residual[within]).max() * 1000:.1f} mHz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
