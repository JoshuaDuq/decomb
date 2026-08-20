"""Does the shipped three-stage pipeline reproduce the measured `tuned` arm?"""
import os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

CONFIG = "/Users/joduq24/Desktop/decomb/decomb.yaml"
COMB_HZ = 1.2


def _merge(iv):
    out = []
    for lo, hi in sorted(iv):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _measure(name):
    import mne
    from dataclasses import replace as _replace
    from scipy.signal import medfilt
    from decomb import notch, recordings, residual, subtraction
    from decomb.config import load_config

    config = load_config(CONFIG)
    settings = _replace(
        notch.HarmonicNotchSettings.from_config(config), comb_fundamental_hz=COMB_HZ
    )
    bands = notch.analysed_bands_from_config(config)
    raw = recordings.read_bids_raw(
        config.path("bids_root") / name.split("_")[0] / "eeg" / f"{name}.vhdr")
    if recordings.acquisition_segments(raw) != ((0, raw.n_times),):
        return None
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    sfreq = float(raw.info["sfreq"])
    original = raw.get_data(picks=picks)

    # --- the shipped path, exactly as clean_harmonic_run runs it ---
    evidence = notch.fit_harmonic_round(raw, settings, round_index=1)
    recovered, record = subtraction.subtract_authorized(raw, evidence, settings, n_jobs=1)
    threshold = residual.fit_threshold_stage(recovered, record.frequencies_hz, settings)
    plan = threshold.plan(settings)
    if plan is not None:
        recovered = notch.apply_harmonic_notches(recovered, plan, n_jobs=1)
    result = notch.clean_until_no_supported_lines(recovered, settings, n_jobs=1)
    cleaned = result.cleaned.get_data(picks=picks)

    n_fft = recordings.estimation_window_samples(sfreq, 10.0)
    def spec(d):
        p, fr = mne.time_frequency.psd_array_welch(
            d, sfreq, fmin=1.0, fmax=min(100.0, np.nextafter(sfreq/2.0, 0.0)),
            n_fft=n_fft, n_per_seg=n_fft, n_overlap=n_fft // 2, average="mean",
            window="hamming", remove_dc=True, verbose="ERROR")
        return 10 * np.log10(p.mean(axis=0) * 1e12), fr

    cdb, cfr = spec(cleaned)
    res = cdb - medfilt(cdb, 41)
    vals = []
    for k in range(int(np.ceil(20.0 / COMB_HZ)), int(95.0 / COMB_HZ) + 1):
        off = np.abs(cfr - k * COMB_HZ)
        pk = off <= 0.11
        gp = (off >= 0.35 * COMB_HZ) & (off <= 0.5 * COMB_HZ)
        if pk.any() and gp.sum() >= 2:
            vals.append(res[pk].max() - np.median(res[gp]))

    damage = list(subtraction.damage_intervals(record.frequencies_hz, record.window_s))
    fir = list(plan.unavailable_edges()) if plan is not None else []
    fir += [e for r in result.rounds for e in r.filter_plan.unavailable_edges()]
    shares = notch.band_availability_from_intervals(_merge(damage + fir), bands)
    ch_o = original - original.mean(axis=1, keepdims=True)
    ch_c = cleaned - cleaned.mean(axis=1, keepdims=True)
    return {
        "recording": name, "arm": "shipped_tuned",
        "n_subtracted": len(record.frequencies_hz),
        "n_threshold_stopbands": len(threshold.stopbands),
        "fir_rounds": len(result.rounds),
        "comb_db": round(float(np.median(vals)), 2) if vals else np.nan,
        "gamma_kept": round(shares["gamma_retained_share"], 4),
        "beta_kept": round(shares["beta_retained_share"], 4),
        "correlation": round(float(np.corrcoef(ch_o.ravel(), ch_c.ravel())[0, 1]), 5),
        "change_rms": round(float(np.sqrt(((cleaned - original) ** 2).mean())
                                  / np.sqrt((original ** 2).mean())), 5),
    }


def main():
    names = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(_measure, n) for n in names]):
            r = f.result()
            if r:
                rows.append(r)
                print(f"  {r['recording'][:34]} subtracted {r['n_subtracted']:3d} "
                      f"thresh {r['n_threshold_stopbands']:2d} comb {r['comb_db']:+.2f} "
                      f"gamma {r['gamma_kept']:.3f}", flush=True)
    got = pd.DataFrame(rows).set_index("recording").sort_index()
    tuned = pd.read_csv(sys.argv[2], sep="\t").query("arm == 'tuned'").set_index("recording")
    tuned = tuned.loc[got.index]
    print("\n=== shipped vs measured tuned arm ===")
    print(f"{'recording':36s} {'n_sub':>12s} {'comb_db':>14s} {'gamma':>14s}")
    for r in got.index:
        print(f"{r[:36]:36s} {got.n_subtracted[r]:5d}/{tuned.n_subtracted[r]:<6d} "
              f"{got.comb_db[r]:+6.2f}/{tuned.comb_db[r]:+6.2f}  "
              f"{got.gamma_kept[r]:.4f}/{tuned.gamma_kept[r]:.4f}")
    ok = (
        (got.n_subtracted == tuned.n_subtracted).all()
        and np.allclose(got.comb_db, tuned.comb_db, atol=0.02)
        and np.allclose(got.gamma_kept, tuned.gamma_kept, atol=0.002)
    )
    print("\nREPRODUCES TUNED ARM" if ok else "\nDOES NOT REPRODUCE -- investigate")


if __name__ == "__main__":
    main()
