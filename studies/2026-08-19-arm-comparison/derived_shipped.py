"""Measure the shipped subtraction pipeline, arm-comparable to derived_probe.py.

Drives the shipped code path -- subtraction.subtract_authorized, damage_intervals,
band_availability_from_intervals, clean_until_no_supported_lines -- rather than the
prototype's inline steps, so what is measured is what `decomb apply` computes. Adds a
per-recording null calibration, because a fixed 2 dB floor sits under the noise of its
own statistic in some recordings.
"""
import os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = "/Users/joduq24/Desktop/decomb/decomb.yaml"
T0 = time.perf_counter()
def log(m): print(f"[{time.perf_counter()-T0:7.1f}s] {m}", flush=True)
BANDS = (("delta", 1, 3.9), ("theta", 4, 7.9), ("alpha", 8, 12.9), ("beta", 13, 30), ("gamma", 30.1, 80))


def _merge(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _overlap(intervals, lo, hi):
    return sum(max(0.0, min(b, hi) - max(a, lo)) for a, b in intervals)


def _measure(name):
    import mne
    from decomb import notch, recordings, subtraction
    from decomb.config import load_config

    config = load_config(CONFIG)
    settings = notch.HarmonicNotchSettings.from_config(config)
    raw = recordings.read_bids_raw(config.path("bids_root") / name.split("_")[0] / "eeg" / f"{name}.vhdr")
    if recordings.acquisition_segments(raw) != ((0, raw.n_times),):
        return None

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    sfreq = float(raw.info["sfreq"])
    original = raw.get_data(picks=picks)

    # --- the shipped path ---
    evidence = notch.fit_harmonic_round(raw, settings, round_index=1)
    recovered, record = subtraction.subtract_authorized(raw, evidence, settings, n_jobs=1)
    result = notch.clean_until_no_supported_lines(recovered, settings, n_jobs=1)
    cleaned = result.cleaned.get_data(picks=picks)

    targets = record.frequencies_hz
    n_sh = 0 if evidence.scanner_harmonics is None else len(evidence.scanner_harmonics.supporting_harmonics)

    n_fft = recordings.estimation_window_samples(sfreq, 10.0)
    def spec(d):
        p, fr = mne.time_frequency.psd_array_welch(
            d, sfreq, fmin=1.0, fmax=min(100.0, sfreq / 2 * 0.999), n_fft=n_fft,
            n_per_seg=n_fft, n_overlap=n_fft // 2, average="mean", window="hamming",
            remove_dc=True, verbose="ERROR")
        return 10 * np.log10(p.mean(axis=0) * 1e12), fr

    cdb, cfr = spec(cleaned)
    def prom(c):
        pk = np.abs(cfr - c) <= 0.11
        nr = (np.abs(cfr - c) > 1.0) & (np.abs(cfr - c) <= 4.0)
        return float(cdb[pk].max() - np.median(cdb[nr])) if pk.any() and nr.sum() >= 3 else np.nan

    from scipy.signal import medfilt
    def comb_db(db, fr, f0=1.2):
        r = db - medfilt(db, 41)
        vals = []
        for k in range(int(np.ceil(20.0 / f0)), int(95.0 / f0) + 1):
            off = np.abs(fr - k * f0)
            pk = off <= 0.11
            gp = (off >= 0.35 * f0) & (off <= 0.5 * f0)
            if pk.any() and gp.sum() >= 2:
                vals.append(r[pk].max() - np.median(r[gp]))
        return float(np.median(vals)) if vals else np.nan

    nyq = sfreq / 2.0
    teeth = np.arange(1, int(min(95.0, nyq) / 1.2) + 1) * 1.2
    teeth = teeth[(teeth >= 20.0) & (teeth < nyq)]
    target_array = np.array(targets) if targets else np.array([])
    check = np.unique(np.concatenate([target_array, teeth]))
    proms = np.array([prom(f) for f in check])

    # per-recording null: prominence far from any target or tooth
    rng = np.random.default_rng(7)
    nulls = []
    for f in rng.uniform(20.0, 95.0, 4000):
        if target_array.size and np.min(np.abs(target_array - f)) < 0.5:
            continue
        if np.min(np.abs(teeth - f)) < 0.5:
            continue
        pr = prom(f)
        if np.isfinite(pr):
            nulls.append(pr)
        if len(nulls) >= 400:
            break
    nulls = np.array(nulls)
    null_p99 = float(np.percentile(nulls, 99)) if nulls.size else np.nan

    damage = list(subtraction.damage_intervals(targets, settings))
    fir = [(sb.low_hz, sb.high_hz) for r in result.rounds for sb in r.filter_plan.stopbands]
    merged_all = _merge(damage + fir)
    merged_fir = _merge(fir)

    row = {
        "recording": name, "participant": name.split("_")[0], "arm": "derived_shipped",
        "n_targets": len(targets), "n_scanner_harmonics": n_sh,
        "fir_rounds": len(result.rounds), "n_fir_stopbands": len(fir),
        "peaks_above_2dB": int(np.nansum(proms > 2.0)),
        "peaks_above_null_p99": int(np.nansum(proms > null_p99)) if np.isfinite(null_p99) else -1,
        "null_p99_db": round(null_p99, 2) if np.isfinite(null_p99) else np.nan,
        "null_frac_over_2db": round(float((nulls > 2.0).mean()), 4) if nulls.size else np.nan,
        "comb_db": round(comb_db(cdb, cfr), 2),
        "damage_half_hz": 2.0 * settings.frequency_bin_width_hz,
        "unavail_hz_total": round(sum(b - a for a, b in merged_all), 2),
        "unavail_hz_fir_only": round(sum(b - a for a, b in merged_fir), 2),
    }
    ch_o = original - original.mean(axis=1, keepdims=True)
    ch_c = cleaned - cleaned.mean(axis=1, keepdims=True)
    row["correlation"] = round(float(np.corrcoef(ch_o.ravel(), ch_c.ravel())[0, 1]), 5)
    row["change_rms"] = round(float(np.sqrt(((cleaned - original) ** 2).mean()) / np.sqrt((original ** 2).mean())), 5)
    shares = notch.band_availability_from_intervals(merged_all, BANDS)
    shares_fir = notch.band_availability_from_intervals(merged_fir, BANDS)
    for bn, lo, hi in BANDS:
        row[f"{bn}_kept"] = round(shares[f"{bn}_retained_share"], 4)
        row[f"{bn}_kept_fir_only"] = round(shares_fir[f"{bn}_retained_share"], 4)
    return row


def main():
    names = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    out = Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_measure, n): n for n in names}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as e:
                log(f"[{i}/{len(names)}] {futs[f]} FAILED {type(e).__name__}: {e}")
                continue
            if r is None:
                log(f"[{i}/{len(names)}] {futs[f]} skipped (segmented acquisition)")
                continue
            rows.append(r)
            pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
            log(f"[{i}/{len(names)}] {r['recording'][:34]} targets {r['n_targets']:3d} "
                f"peaks {r['peaks_above_2dB']:3d} (null-cal {r['peaks_above_null_p99']:3d}) "
                f"comb {r['comb_db']:+.2f} gamma_kept {r['gamma_kept']:.3f} "
                f"(fir-only {r['gamma_kept_fir_only']:.3f})")
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
