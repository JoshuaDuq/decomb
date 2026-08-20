"""Per-bin impact outside the declaration: removed power relative to source power."""
import os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd

CONFIG = "/Users/joduq24/Desktop/decomb/decomb.yaml"


def _merge(iv):
    out = []
    for lo, hi in sorted(tuple(x) for x in iv):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _one(payload):
    name, fir, subtracted = payload
    import mne
    from decomb import notch, recordings, subtraction
    from decomb.config import load_config
    config = load_config(CONFIG)
    st = notch.HarmonicNotchSettings.from_config(config)
    src, deriv = config.path("bids_root"), config.path("output_root")
    vhdr = src / name.split("_")[0] / "eeg" / f"{name}.vhdr"
    raw = recordings.read_bids_raw(vhdr)
    cln = recordings.read_bids_raw(recordings.derivative_vhdr_path(vhdr, src, deriv))
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    sfreq = float(raw.info["sfreq"])
    source = raw.get_data(picks=picks)
    removed = source - cln.get_data(picks=picks)
    n = int(2.0 * subtraction.fit_window_s(st) * sfreq)
    def spec(x):
        p, fr = mne.time_frequency.psd_array_welch(
            x, sfreq, fmin=1.0, fmax=min(100.0, np.nextafter(sfreq/2, 0.0)), n_fft=n,
            n_per_seg=n, n_overlap=n//2, average="mean", window="hamming",
            remove_dc=True, verbose="ERROR")
        return p.mean(axis=0), fr
    rp, freqs = spec(removed)
    sp, _ = spec(source)
    row = {"recording": name}
    for half, tag in ((2.0/subtraction.fit_window_s(st), "pm0p10"),
                      (1.0/subtraction.fit_window_s(st), "pm0p05")):
        iv = _merge(list(fir) + [(f-half, f+half) for f in subtracted])
        inside = np.zeros(freqs.size, bool)
        for lo, hi in iv:
            inside |= (freqs >= lo) & (freqs <= hi)
        frac = rp[~inside] / np.maximum(sp[~inside], 1e-30)
        db = 10*np.log10(np.maximum(1.0 - np.minimum(frac, 0.999999), 1e-12))
        row[f"{tag}_worst_bin_db"] = float(np.abs(db).max())
        row[f"{tag}_p999_bin_db"] = float(np.percentile(np.abs(db), 99.9))
        row[f"{tag}_bins_over_0p5db"] = int((np.abs(db) > 0.5).sum())
        row[f"{tag}_bins_outside"] = int((~inside).sum())
    return row


def main():
    m = pd.read_csv(sys.argv[1], sep="\t", keep_default_na=False)
    jobs = []
    for rec, blk in m.groupby("recording"):
        fir = [(float(a), float(b)) for a, b, k in
               zip(blk.unavailable_low_hz, blk.unavailable_high_hz, blk.kind)
               if str(a) != "" and str(b) != "" and str(k) != "subtracted"]
        subs = [float(v) for r in blk.loc[blk.kind == "subtracted", "subtracted_frequencies_hz"]
                for v in str(r).split(";") if v]
        jobs.append((rec, _merge(fir), sorted(set(subs))))
    rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(_one, j): j[0] for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception as e:
                print(f"[{i}] {futs[f]} FAILED {e}", flush=True); continue
            pd.DataFrame(rows).to_csv(sys.argv[2], sep="\t", index=False)
            if i % 25 == 0: print(f"[{i}/{len(jobs)}]", flush=True)
    if not rows:
        raise SystemExit("every recording failed; no output written")
    print(f"done: {len(rows)} recordings", flush=True)


if __name__ == "__main__":
    main()
