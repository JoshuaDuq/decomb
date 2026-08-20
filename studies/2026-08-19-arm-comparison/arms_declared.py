"""Four arms on ONE availability basis: every arm declares its own subtraction damage.

The `combined` arm recorded in arm_combined.tsv computes availability from its FIR plans
only, while it also subtracts at a 20 s window. Comparing that against the shipped arm's
fully-declared number is not like for like. Here every arm declares
  FIR:         plan.unavailable_edges()   (stopband + transition, as the manifest does)
  subtraction: +/- 2 bins = +/- 2 / fit_window_s around each removed frequency
so the arms are finally comparable.

Arms:
  notching        current shipping behaviour, no subtraction
  tuned           reproduction of `combined`: subtract ordinary+strong teeth at 20 s,
                  notch what still stands proud, then converge
  derived         the shipped pipeline: subtract authorized set at 10 s, then converge
  derived_w20fit  as derived, but the subtraction fit uses 20 s while detection stays
                  at 10 s -- halves the damage zone without touching detection
"""
import os, sys, time
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = "/Users/joduq24/Desktop/decomb/decomb.yaml"
COMB_HZ, FLOOR_DB, TOOTH_SUBTRACT_DB = 1.200, 2.0, 1.0
CLUSTER_GAP_HZ, MARGIN_HZ, TUNED_WINDOW_S = 0.30, 0.125, 20.0
ARMS = ("notching", "tuned", "derived", "derived_w20fit")
T0 = time.perf_counter()
def log(m): print(f"[{time.perf_counter()-T0:7.1f}s] {m}", flush=True)


def _merge(iv):
    out = []
    for lo, hi in sorted(iv):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _clusters(frequencies, gap=CLUSTER_GAP_HZ):
    groups = []
    for f in sorted(frequencies):
        if groups and f - groups[-1][-1] <= gap:
            groups[-1].append(f)
        else:
            groups.append([f])
    return groups


def _damage(frequencies, fit_window_s):
    """The +/- 2-bin interval a fit at this window destroys, merged."""
    half = 2.0 / float(fit_window_s)
    return _merge([(float(f) - half, float(f) + half) for f in frequencies])


def _measure(name):
    import mne
    from dataclasses import replace as _replace
    from scipy.signal import medfilt
    from decomb import notch, recordings, recovery, recovery_benchmark, subtraction
    from decomb.config import load_config

    config = load_config(CONFIG)
    base = notch.HarmonicNotchSettings.from_config(config)
    bands = notch.analysed_bands_from_config(config)
    raw = recordings.read_bids_raw(
        config.path("bids_root") / name.split("_")[0] / "eeg" / f"{name}.vhdr")
    if recordings.acquisition_segments(raw) != ((0, raw.n_times),):
        return []

    sfreq = float(raw.info["sfreq"])
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    original = raw.get_data(picks=picks)
    n_fft = recordings.estimation_window_samples(sfreq, 10.0)
    nyquist = np.nextafter(sfreq / 2.0, 0.0)
    teeth = np.arange(1, int(min(95.0, nyquist) / COMB_HZ) + 1) * COMB_HZ
    teeth = teeth[(teeth >= 20.0) & (teeth < nyquist)]

    def spec(d):
        p, fr = mne.time_frequency.psd_array_welch(
            d, sfreq, fmin=1.0, fmax=min(100.0, nyquist), n_fft=n_fft, n_per_seg=n_fft,
            n_overlap=n_fft // 2, average="mean", window="hamming", remove_dc=True,
            verbose="ERROR")
        return 10 * np.log10(p.mean(axis=0) * 1e12), fr

    def prom(db, fr, c):
        pk = np.abs(fr - c) <= 0.11
        nr = (np.abs(fr - c) > 1.0) & (np.abs(fr - c) <= 4.0)
        return float(db[pk].max() - np.median(db[nr])) if pk.any() and nr.sum() >= 3 else np.nan

    def comb_db(db, fr, f0=COMB_HZ):
        res = db - medfilt(db, 41)
        vals = []
        for k in range(int(np.ceil(20.0 / f0)), int(95.0 / f0) + 1):
            off = np.abs(fr - k * f0)
            pk = off <= 0.11
            gp = (off >= 0.35 * f0) & (off <= 0.5 * f0)
            if pk.any() and gp.sum() >= 2:
                vals.append(res[pk].max() - np.median(res[gp]))
        return float(np.median(vals)) if vals else np.nan

    odb, ofr = spec(original)
    ev = notch.fit_harmonic_round(raw, base, round_index=1)

    # the tuned arm's own target set, from its own manifest reading
    rows = []
    if ev.model.channels:
        rows.extend(notch.line_manifest_rows(name, ev.model, ev.plans, bands, base, round_index=1))
    if ev.scanner_harmonics is not None:
        rows.extend(notch.scanner_harmonic_manifest_rows(
            name, ev.scanner_harmonics, ev.scanner_plan, bands, base, round_index=1))
    for r in rows:
        r["removal_round"] = 1
    ordinary = np.asarray(
        recovery_benchmark.targets_from_manifest(pd.DataFrame(rows), name).ordinary_frequencies_hz
    ) if rows else np.array([])
    tooth_prom = np.array([prom(odb, ofr, f) for f in teeth])
    strong_teeth = teeth[np.nan_to_num(tooth_prom, nan=-99) > TOOTH_SUBTRACT_DB]

    out = []
    for arm in ARMS:
        settings = _replace(base, comb_fundamental_hz=COMB_HZ) if arm == "tuned" else base
        damage, extra_plans, subtracted = [], [], ()

        if arm == "notching":
            prepared = raw
        elif arm == "tuned":
            if ordinary.size == 0:
                continue
            merged_targets = list(ordinary)
            for t in strong_teeth:
                if np.min(np.abs(ordinary - t)) > 0.05:
                    merged_targets.append(float(t))
            subtracted = tuple(sorted(set(merged_targets)))
            prepared = recovery_benchmark.recover_with_multitaper(
                raw, recovery_benchmark.RecoveryTargets(subtracted, ()),
                window_s=TUNED_WINDOW_S, n_jobs=1)
            damage = _damage(subtracted, TUNED_WINDOW_S)
            rdb, rfr = spec(prepared.get_data(picks=picks))
            candidates = np.unique(np.concatenate([np.array(subtracted), teeth]))
            spans = []
            for group in _clusters(candidates):
                best = np.nanmax([prom(rdb, rfr, f) for f in group])
                if np.isfinite(best) and best > FLOOR_DB:
                    spans.append((min(group) - MARGIN_HZ, max(group) + MARGIN_HZ))
            if spans:
                plan = notch.HarmonicNotchPlan(
                    tuple(notch.HarmonicStopband((), a, b, "isolated") for a, b in _merge(spans)),
                    settings.transition_bandwidth_hz)
                extra_plans.append(plan)
                prepared = notch.apply_harmonic_notches(prepared, plan, n_jobs=1)
        else:
            fit_window_s = TUNED_WINDOW_S if arm == "derived_w20fit" else base.estimation_window_s
            subtracted = subtraction.authorized_frequencies(ev, base)
            if subtracted:
                res = recovery.subtract_multitaper_sinusoids(
                    original, sfreq, subtracted, window_s=fit_window_s, n_jobs=1)
                prepared = raw.copy()
                prepared._data[picks] = res.cleaned_data
            else:
                prepared = raw.copy()
            damage = _damage(subtracted, fit_window_s)

        result = notch.clean_until_no_supported_lines(prepared, settings, n_jobs=1)
        cleaned = result.cleaned.get_data(picks=picks)
        fir = [e for p in extra_plans for e in p.unavailable_edges()]
        fir += [e for r in result.rounds for e in r.filter_plan.unavailable_edges()]

        cdb, cfr = spec(cleaned)
        check = np.unique(np.concatenate(
            [np.array(subtracted) if len(subtracted) else np.array([]), teeth]))
        proms = np.array([prom(cdb, cfr, f) for f in check])
        rng = np.random.default_rng(7)
        nulls = []
        tarr = np.array(subtracted) if len(subtracted) else np.array([])
        for f in rng.uniform(20.0, 95.0, 4000):
            if tarr.size and np.min(np.abs(tarr - f)) < 0.5:
                continue
            if np.min(np.abs(teeth - f)) < 0.5:
                continue
            pr = prom(cdb, cfr, f)
            if np.isfinite(pr):
                nulls.append(pr)
            if len(nulls) >= 400:
                break
        nulls = np.array(nulls)
        p99 = float(np.percentile(nulls, 99)) if nulls.size else np.nan

        all_iv, fir_iv = _merge(list(damage) + fir), _merge(fir)
        shares = notch.band_availability_from_intervals(all_iv, bands)
        shares_fir = notch.band_availability_from_intervals(fir_iv, bands)
        ch_o = original - original.mean(axis=1, keepdims=True)
        ch_c = cleaned - cleaned.mean(axis=1, keepdims=True)
        row = {
            "recording": name, "participant": name.split("_")[0], "arm": arm,
            "n_subtracted": len(subtracted), "fir_rounds": len(result.rounds),
            "n_fir_blocks": len(fir_iv),
            "peaks_above_2dB": int(np.nansum(proms > 2.0)),
            "peaks_above_null_p99": int(np.nansum(proms > p99)) if np.isfinite(p99) else -1,
            "null_p99_db": round(p99, 2) if np.isfinite(p99) else np.nan,
            "comb_db": round(comb_db(cdb, cfr), 2),
            "unavail_hz_total": round(sum(b - a for a, b in all_iv), 2),
            "unavail_hz_fir_only": round(sum(b - a for a, b in fir_iv), 2),
            "correlation": round(float(np.corrcoef(ch_o.ravel(), ch_c.ravel())[0, 1]), 5),
            "change_rms": round(float(np.sqrt(((cleaned - original) ** 2).mean())
                                      / np.sqrt((original ** 2).mean())), 5),
        }
        for bn, lo, hi in bands:
            row[f"{bn}_kept"] = round(shares[f"{bn}_retained_share"], 4)
            row[f"{bn}_kept_fir_only"] = round(shares_fir[f"{bn}_retained_share"], 4)
        out.append(row)
    return out


def main():
    names = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    out = Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_measure, n): n for n in names}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                got = f.result()
            except Exception as e:
                log(f"[{i}/{len(names)}] {futs[f]} FAILED {type(e).__name__}: {e}")
                continue
            if not got:
                log(f"[{i}/{len(names)}] {futs[f]} skipped")
                continue
            rows += got
            pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
            by = {r["arm"]: r for r in got}
            log(f"[{i}/{len(names)}] {got[0]['recording'][:34]} " + "  ".join(
                f"{a[:8]} g{by[a]['gamma_kept']:.3f}" for a in ARMS if a in by))
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
