"""Pre-hoc comb evidence per recording.

Measures, from the ORIGINAL spectrum only (no filtering, no recovery), quantities that
are available before the decision to declare comb_fundamental_hz is taken. Goal: find a
predictor that separates the 5 recordings where declaring excavates the comb (-6 dB)
from the 85 where it does not.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os, sys, time
import numpy as np, pandas as pd

CONFIG = Path("/Users/joduq24/Desktop/decomb/decomb.yaml")
COMB_HZ, TOOTH_DB = 1.200, 1.0
STARTED = time.perf_counter()
def log(m): print(f"[{time.perf_counter()-STARTED:7.1f}s] {m}", flush=True)


def _one(name):
    for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    import mne
    from scipy.signal import medfilt
    from decomb import recordings
    from decomb.config import load_config
    config = load_config(CONFIG)
    raw = recordings.read_bids_raw(
        config.path("bids_root") / name.split("_")[0] / "eeg" / f"{name}.vhdr")
    if recordings.acquisition_segments(raw) != ((0, raw.n_times),):
        return None
    sfreq = float(raw.info["sfreq"])
    n_fft = recordings.estimation_window_samples(sfreq, 10.0)
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    p, fr = mne.time_frequency.psd_array_welch(
        raw.get_data(picks=picks), sfreq, fmin=1.0, fmax=min(100.0, sfreq/2*0.999),
        n_fft=n_fft, n_per_seg=n_fft, n_overlap=n_fft//2, average="mean",
        window="hamming", remove_dc=True, verbose="ERROR")
    db = 10*np.log10(p.mean(axis=0)*1e12)

    def prom(c):
        pk = np.abs(fr-c) <= 0.11; nr = (np.abs(fr-c) > 1.0) & (np.abs(fr-c) <= 4.0)
        return float(db[pk].max()-np.median(db[nr])) if pk.any() and nr.sum() >= 3 else np.nan

    nyq = sfreq/2.0
    teeth = np.arange(1, int(min(95.0, nyq)/COMB_HZ)+1)*COMB_HZ
    teeth = teeth[(teeth >= 20.0) & (teeth < nyq)]
    tp = np.array([prom(t) for t in teeth])
    tpf = tp[np.isfinite(tp)]

    # comb_db measured on the ORIGINAL spectrum: is there a comb here at all?
    res = db - medfilt(db, 41); vals = []
    for k in range(int(np.ceil(20.0/COMB_HZ)), int(95.0/COMB_HZ)+1):
        off = np.abs(fr - k*COMB_HZ); pk = off <= 0.11
        gp = (off >= 0.35*COMB_HZ) & (off <= 0.5*COMB_HZ)
        if pk.any() and gp.sum() >= 2:
            vals.append(res[pk].max()-np.median(res[gp]))
    return {
        "recording": name, "participant": name.split("_")[0],
        "orig_comb_db": round(float(np.median(vals)), 3) if vals else np.nan,
        "n_teeth": int(teeth.size),
        "n_strong_teeth": int(np.nansum(tp > TOOTH_DB)),
        "frac_strong_teeth": round(float(np.nansum(tp > TOOTH_DB)/max(teeth.size,1)), 4),
        "median_tooth_prom_db": round(float(np.median(tpf)), 3) if tpf.size else np.nan,
        "p90_tooth_prom_db": round(float(np.percentile(tpf, 90)), 3) if tpf.size else np.nan,
    }


def main():
    names = [l.strip() for l in open(sys.argv[1])]
    log(f"{len(names)} recordings")
    rows = []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("W", "8"))) as ex:
        futs = {ex.submit(_one, n): n for n in names}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as e:
                log(f"[{i}/{len(names)}] {futs[f]}: FAILED {type(e).__name__}: {e}"); continue
            if r is None:
                log(f"[{i}/{len(names)}] {futs[f]}: skipped (segmented)"); continue
            rows.append(r)
            log(f"[{i}/{len(names)}] {r['recording']}: orig_comb {r['orig_comb_db']:+.2f} "
                f"strong_teeth {r['n_strong_teeth']}/{r['n_teeth']}")
    pd.DataFrame(rows).sort_values("recording").to_csv(sys.argv[2], sep="\t", index=False)
    log(f"wrote {sys.argv[2]} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
