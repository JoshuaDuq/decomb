"""Full stats on the written derivative: comb, residuals, availability, fidelity."""
import os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = "/Users/joduq24/Desktop/decomb/decomb.yaml"
COMB_HZ = 1.2


def _one(name):
    import mne
    from scipy.signal import medfilt
    from decomb import notch, recordings
    from decomb.config import load_config

    config = load_config(CONFIG)
    settings = notch.HarmonicNotchSettings.from_config(config)
    source_root, out_root = config.path("bids_root"), config.path("output_root")
    vhdr = source_root / name.split("_")[0] / "eeg" / f"{name}.vhdr"
    raw = recordings.read_bids_raw(vhdr)
    cleaned_raw = recordings.read_bids_raw(
        recordings.derivative_vhdr_path(vhdr, source_root, out_root))
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    sfreq = float(raw.info["sfreq"])
    before, after = raw.get_data(picks=picks), cleaned_raw.get_data(picks=picks)
    n_fft = recordings.estimation_window_samples(sfreq, settings.estimation_window_s)
    nyq = np.nextafter(sfreq / 2.0, 0.0)

    def spec(d):
        p, fr = mne.time_frequency.psd_array_welch(
            d, sfreq, fmin=1.0, fmax=min(100.0, nyq), n_fft=n_fft, n_per_seg=n_fft,
            n_overlap=n_fft // 2, average="mean", window="hamming", remove_dc=True,
            verbose="ERROR")
        return 10 * np.log10(p.mean(axis=0) * 1e12), fr

    bdb, bfr = spec(before)
    adb, afr = spec(after)

    def comb_db(db, fr):
        res = db - medfilt(db, 41)
        vals = []
        for k in range(int(np.ceil(20.0 / COMB_HZ)), int(95.0 / COMB_HZ) + 1):
            off = np.abs(fr - k * COMB_HZ)
            pk, gp = off <= 0.11, (off >= 0.35 * COMB_HZ) & (off <= 0.5 * COMB_HZ)
            if pk.any() and gp.sum() >= 2:
                vals.append(res[pk].max() - np.median(res[gp]))
        return float(np.median(vals)) if vals else np.nan

    def prom(db, fr, c):
        off = np.abs(fr - c)
        pk, nr = off <= 0.11, (off > 1.0) & (off <= 4.0)
        return float(db[pk].max() - np.median(db[nr])) if pk.any() and nr.sum() >= 3 else np.nan

    teeth = np.arange(1, int(min(95.0, nyq) / COMB_HZ) + 1) * COMB_HZ
    teeth = teeth[(teeth >= 20.0) & (teeth < nyq)]
    tooth_prom_before = np.array([prom(bdb, bfr, t) for t in teeth])
    tooth_prom_after = np.array([prom(adb, afr, t) for t in teeth])

    # per-recording null on the derivative: prominence far from any tooth
    ok = (afr >= 20.0) & (afr <= 95.0)
    for t in teeth:
        ok &= ~(np.abs(afr - t) < 0.5)
    nulls = np.array([prom(adb, afr, f) for f in afr[ok]])
    nulls = nulls[np.isfinite(nulls)]
    p99 = float(np.percentile(nulls, 99)) if nulls.size else np.nan

    ch_b = before - before.mean(axis=1, keepdims=True)
    ch_a = after - after.mean(axis=1, keepdims=True)
    return {
        "recording": name, "participant": name.split("_")[0],
        "comb_db_before": round(comb_db(bdb, bfr), 2),
        "comb_db_after": round(comb_db(adb, afr), 2),
        "teeth_tested": int(np.isfinite(tooth_prom_before).sum()),
        "teeth_proud_2db_before": int(np.nansum(tooth_prom_before > 2.0)),
        "teeth_proud_2db_after": int(np.nansum(tooth_prom_after > 2.0)),
        "teeth_above_null_p99_after": int(np.nansum(tooth_prom_after > p99)),
        "null_p99_db": round(p99, 2) if np.isfinite(p99) else np.nan,
        "null_bins": int(nulls.size),
        "correlation": round(float(np.corrcoef(ch_b.ravel(), ch_a.ravel())[0, 1]), 5),
        "change_rms": round(float(np.sqrt(((after - before) ** 2).mean())
                                  / np.sqrt((before ** 2).mean())), 5),
    }


def main():
    names = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(_one, n): n for n in names}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception as e:
                print(f"[{i}/{len(names)}] {futs[f]} FAILED {type(e).__name__}: {e}", flush=True)
                continue
            pd.DataFrame(rows).to_csv(sys.argv[2], sep="\t", index=False)
            r = rows[-1]
            print(f"[{i}/{len(names)}] {r['recording'][:34]} comb {r['comb_db_before']:+.2f}"
                  f"->{r['comb_db_after']:+.2f}  teeth>2dB {r['teeth_proud_2db_before']:2d}"
                  f"->{r['teeth_proud_2db_after']:2d}", flush=True)
    print(f"done -> {sys.argv[2]}", flush=True)


if __name__ == "__main__":
    main()
