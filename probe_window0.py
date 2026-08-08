"""What distinguishes the opening window from the rest of a recording.

Compares, per adaptive window, the comb's strength against the broadband floor. A window
whose comb is weak has no comb to fit; a window whose floor is raised has one that is
masked. The two call for different answers, so the point is to tell them apart.
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
    sfreq = float(raw.info["sfreq"])
    _, per_window, bounds = remove.run_spectra(raw, settings)

    f0 = settings.nominal_fundamental_hz
    low, high = settings.harmonic_range
    band = settings.cost_band_hz

    print(f"{vhdr.name}\n")
    print(f"{'win':>4} {'start_s':>8} {'med_prom':>9} {'p90_prom':>9} "
          f"{'floor_dB':>9} {'rms_uV':>9}")

    data = raw.get_data(picks=mne.pick_types(raw.info, eeg=True, exclude=()))

    for index, ((freqs, spectrum_db, prominence), (start, stop)) in enumerate(
        zip(per_window, bounds)
    ):
        proms = []
        for k in range(low, high + 1):
            mask = np.abs(freqs - k * f0) <= settings.search_hz
            if mask.any():
                proms.append(float(prominence[mask].max()))
        proms = np.array(proms)

        inside = (freqs >= band[0]) & (freqs <= band[1])
        floor = float(np.median(spectrum_db[inside]))
        rms = float(np.sqrt(np.mean(data[:, start:stop] ** 2)) * 1e6)

        print(f"{index:>4} {start / sfreq:>8.1f} {np.median(proms):>9.2f} "
              f"{np.percentile(proms, 90):>9.2f} {floor:>9.1f} {rms:>9.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
