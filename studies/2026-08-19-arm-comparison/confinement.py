"""Is the removal confined to the bandwidth the manifest declares?

Measured on the difference signal, source minus derivative, which is exactly what was
removed. Comparing two spectra instead would confound the removal with the spectral
estimator: at 0.1 Hz bins a Welch estimate smears a narrow removal across neighbours, and
a control on a known stationary line shows that spread vanishing entirely by 30 s
segments. Bins here are 30 s, fine enough to resolve the removal.
"""
import os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd

CONFIG = "/Users/joduq24/Desktop/decomb/decomb.yaml"
SEGMENT_S = 30.0


def _merge(iv):
    out = []
    for lo, hi in sorted(iv):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _one(payload):
    name, intervals = payload
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
    n_fft = int(SEGMENT_S * sfreq)
    power, freqs = mne.time_frequency.psd_array_welch(
        removed, sfreq, fmin=1.0, fmax=min(100.0, np.nextafter(sfreq / 2, 0.0)),
        n_fft=n_fft, n_per_seg=n_fft, n_overlap=n_fft // 2, average="mean",
        window="hamming", remove_dc=True, verbose="ERROR")
    power = power.mean(axis=0)
    total = power.sum()
    row = {"recording": name, "bin_hz": sfreq / n_fft}
    for margin, key in ((0.0, "inside_declared_pct"), (0.05, "inside_plus_50mhz_pct")):
        inside = np.zeros(freqs.size, bool)
        for lo, hi in intervals:
            inside |= (freqs >= lo - margin) & (freqs <= hi + margin)
        row[key] = 100.0 * power[inside].sum() / total
    return row


def main():
    manifest = pd.read_csv(sys.argv[1], sep="\t", keep_default_na=False)
    jobs = []
    for rec, block in manifest.groupby("recording"):
        iv = [(float(a), float(b)) for a, b in
              zip(block.unavailable_low_hz, block.unavailable_high_hz)
              if str(a) != "" and str(b) != ""]
        jobs.append((rec, _merge(iv)))
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
