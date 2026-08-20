"""Is the removal confined to the bandwidth the manifest declares?

Measured on the difference signal, source minus derivative, which is exactly what was
removed. Comparing two spectra instead would confound the removal with the spectral
estimator: at 0.1 Hz bins a Welch estimate smears a narrow removal across neighbours, and
a control on a known stationary line shows that spread vanishing entirely once the
segment is long enough. Segments here match what `verify` uses.
"""
import os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd

CONFIG = "/Users/joduq24/Desktop/decomb/decomb.yaml"
# same rule as notch.measure_removal_confinement: twice the subtraction fit window,
# which puts eight bins across a declared interval
SEGMENT_MULTIPLE = 2.0


def _merge(iv):
    out = []
    for lo, hi in sorted(iv):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _one(payload):
    name, fir_intervals, subtracted_hz = payload
    import mne
    from decomb import recordings
    from decomb.config import load_config
    config = load_config(CONFIG)
    src, deriv = config.path("bids_root"), config.path("output_root")
    vhdr = src / name.split("_")[0] / "eeg" / f"{name}.vhdr"
    raw = recordings.read_bids_raw(vhdr)
    cln = recordings.read_bids_raw(recordings.derivative_vhdr_path(vhdr, src, deriv))
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    sfreq = float(raw.info["sfreq"])
    removed = raw.get_data(picks=picks) - cln.get_data(picks=picks)
    from decomb import notch, subtraction
    settings = notch.HarmonicNotchSettings.from_config(config)
    n_fft = int(SEGMENT_MULTIPLE * subtraction.fit_window_s(settings) * sfreq)
    power, freqs = mne.time_frequency.psd_array_welch(
        removed, sfreq, fmin=1.0, fmax=min(100.0, np.nextafter(sfreq / 2, 0.0)),
        n_fft=n_fft, n_per_seg=n_fft, n_overlap=n_fft // 2, average="mean",
        window="hamming", remove_dc=True, verbose="ERROR")
    power = power.mean(axis=0)
    total = power.sum()
    row = {"recording": name, "bin_hz": sfreq / n_fft,
           "n_subtracted": len(subtracted_hz)}
    # confinement against the width declared today, and against a halved width
    for half, key in ((2.0 / subtraction.fit_window_s(settings), "inside_pm0p10_pct"),
                      (1.0 / subtraction.fit_window_s(settings), "inside_pm0p05_pct")):
        intervals = _merge(
            list(fir_intervals) + [(f - half, f + half) for f in subtracted_hz]
        )
        inside = np.zeros(freqs.size, bool)
        for lo, hi in intervals:
            inside |= (freqs >= lo) & (freqs <= hi)
        row[key] = 100.0 * power[inside].sum() / total
    return row


def main():
    manifest = pd.read_csv(sys.argv[1], sep="\t", keep_default_na=False)
    jobs = []
    for rec, block in manifest.groupby("recording"):
        fir = [(float(a), float(b)) for a, b, kind in
               zip(block.unavailable_low_hz, block.unavailable_high_hz, block.kind)
               if str(a) != "" and str(b) != "" and str(kind) != "subtracted"]
        subtracted = [
            float(value)
            for row in block.loc[block.kind == "subtracted", "subtracted_frequencies_hz"]
            for value in str(row).split(";")
            if value
        ]
        jobs.append((rec, _merge(fir), sorted(set(subtracted))))
    rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(_one, j): j[0] for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception as e:
                print(f"[{i}] {futs[f]} FAILED {type(e).__name__}: {e}", flush=True)
                continue
            pd.DataFrame(rows).to_csv(sys.argv[2], sep="\t", index=False)
            if i % 20 == 0:
                print(f"[{i}/{len(jobs)}]", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
