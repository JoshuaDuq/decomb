"""Two fixes over the previous best configuration.

FIX 1 -- cluster before thresholding. Adjacent detected bins are one physical line, so
the residual floor is applied to the cluster's strongest bin and the notch spans the
whole cluster plus a full transition band on each side. Previously the floor ran per
bin, so one bin of a line could be notched while its neighbour sat in the filter's
transition and survived 17 dB louder.

FIX 2 -- let comb teeth face the floor. 56% of surviving peaks were 1.2 Hz teeth, which
the configuration never considered. They are added as CANDIDATES for the residual test
rather than as subtraction targets: subtracting all ~60 of them was measured to cost
fidelity (0.9957 -> 0.9840) for little gain, whereas notching only the few that stand
proud costs a handful of narrow stopbands.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os, sys, time, traceback
import numpy as np, pandas as pd

BENCH = Path(__file__).parent
CONFIG = Path("/Users/joduq24/Desktop/decomb/decomb.yaml")
WINDOW_S, FLOOR_DB, COMB_HZ = 20.0, 2.0, 1.200
TOOTH_SUBTRACT_DB = 1.0   # only teeth standing this proud are worth fitting
CLUSTER_GAP_HZ = 0.30      # bins closer than this belong to one line
MARGIN_HZ = 0.125          # half-width added outside a cluster
STARTED = time.perf_counter()


def log(m): print(f"[{time.perf_counter()-STARTED:8.1f}s] {m}", flush=True)


def _clusters(frequencies, gap=CLUSTER_GAP_HZ):
    groups = []
    for f in sorted(frequencies):
        if groups and f - groups[-1][-1] <= gap:
            groups[-1].append(f)
        else:
            groups.append([f])
    return groups


def _measure(name, arms):
    for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    import mne
    from scipy.signal import medfilt
    from decomb import notch, recordings, recovery_benchmark, recovery_evaluation
    from decomb.config import load_config
    from dataclasses import replace as _replace
    config = load_config(CONFIG)
    base_settings = notch.HarmonicNotchSettings.from_config(config)
    bands = notch.analysed_bands_from_config(config)
    raw = recordings.read_bids_raw(
        config.path("bids_root") / name.split("_")[0] / "eeg" / f"{name}.vhdr")
    if recordings.acquisition_segments(raw) != ((0, raw.n_times),):
        return []
    settings = base_settings
    ev = notch.fit_harmonic_round(raw, settings, round_index=1)
    rows = []
    if ev.model.channels:
        rows.extend(notch.line_manifest_rows(name, ev.model, ev.plans, bands, settings, round_index=1))
    if ev.scanner_harmonics is not None:
        rows.extend(notch.scanner_harmonic_manifest_rows(
            name, ev.scanner_harmonics, ev.scanner_plan, bands, settings, round_index=1))
    for r in rows: r["removal_round"] = 1
    if not rows: return []
    targets = recovery_benchmark.targets_from_manifest(pd.DataFrame(rows), name)
    ordinary = np.asarray(targets.ordinary_frequencies_hz)
    if ordinary.size == 0: return []

    sfreq = float(raw.info["sfreq"]); n_fft = recordings.estimation_window_samples(sfreq, 10.0)
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    original = raw.get_data(picks=picks)
    nyquist = np.nextafter(sfreq/2.0, 0.0)
    teeth = np.arange(1, int(min(95.0, nyquist)/COMB_HZ)+1)*COMB_HZ
    teeth = teeth[(teeth >= 20.0) & (teeth < nyquist)]

    def spec(d):
        p, fr = mne.time_frequency.psd_array_welch(
            d, sfreq, fmin=1.0, fmax=min(100.0, nyquist), n_fft=n_fft, n_per_seg=n_fft,
            n_overlap=n_fft//2, average="mean", window="hamming", remove_dc=True,
            verbose="ERROR")
        return 10*np.log10(p.mean(axis=0)*1e12), fr

    def prom(db, fr, c):
        pk = np.abs(fr-c) <= 0.11; nr = (np.abs(fr-c) > 1.0) & (np.abs(fr-c) <= 4.0)
        return float(db[pk].max()-np.median(db[nr])) if pk.any() and nr.sum() >= 3 else np.nan

    def level(db, fr, c):
        pk = np.abs(fr-c) <= 0.11
        return float(db[pk].max()) if pk.any() else np.nan

    def comb_db(db, fr, f0=COMB_HZ, lo=20.0, hi=95.0):
        res = db - medfilt(db, 41); vals = []
        for k in range(int(np.ceil(lo/f0)), int(hi/f0)+1):
            off = np.abs(fr - k*f0); pk = off <= 0.11
            gp = (off >= 0.35*f0) & (off <= 0.5*f0)
            if pk.any() and gp.sum() >= 2:
                vals.append(res[pk].max()-np.median(res[gp]))
        return float(np.median(vals)) if vals else np.nan

    odb, ofr = spec(original)
    # teeth that actually stand proud before any correction
    tooth_prom = np.array([prom(odb, ofr, f) for f in teeth])
    strong_teeth = teeth[np.nan_to_num(tooth_prom, nan=-99) > TOOTH_SUBTRACT_DB]
    out = []
    for arm in arms:
        extra, chosen_count = (), 0
        # the declared-fundamental arm re-detects on the 1.2 Hz grid
        settings = (_replace(base_settings, comb_fundamental_hz=COMB_HZ)
                    if arm in ("notching_declared", "lines_declared", "combined")
                    else base_settings)
        if arm.startswith("notching"):
            prepared = raw
        else:
            subtract = ordinary
            if arm in ("comb_subtracted", "combined"):
                merged_targets = list(ordinary)
                for t in strong_teeth:
                    if np.min(np.abs(ordinary - t)) > 0.05:
                        merged_targets.append(float(t))
                subtract = np.array(sorted(set(merged_targets)))
            prepared = recovery_benchmark.recover_with_multitaper(
                raw, recovery_benchmark.RecoveryTargets(tuple(subtract), ()),
                window_s=WINDOW_S, n_jobs=1)
            rdb, rfr = spec(prepared.get_data(picks=picks))
            candidates = (np.unique(np.concatenate([ordinary, teeth]))
                          if arm == "fixed_with_comb"
                          else np.unique(np.concatenate([subtract, teeth]))
                          if arm in ("comb_subtracted", "combined") else ordinary)
            spans = []
            for group in _clusters(candidates):
                best = np.nanmax([prom(rdb, rfr, f) for f in group])
                if np.isfinite(best) and best > FLOOR_DB:
                    spans.append((min(group)-MARGIN_HZ, max(group)+MARGIN_HZ))
            merged = []
            for lo, hi in sorted(spans):
                if merged and lo <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
                else:
                    merged.append((lo, hi))
            chosen_count = len(merged)
            if merged:
                extra = (notch.HarmonicNotchPlan(
                    tuple(notch.HarmonicStopband((), a, b, "isolated") for a, b in merged),
                    settings.transition_bandwidth_hz),)
                prepared = notch.apply_harmonic_notches(prepared, extra[0], n_jobs=1)
        result = notch.clean_until_no_supported_lines(prepared, settings, n_jobs=1)
        cleaned = result.cleaned.get_data(picks=picks)
        plans = (*extra, *(r.filter_plan for r in result.rounds))
        avail = (notch._band_availability_fields(notch.merge_recording_plans(plans), bands)
                 if plans else {f"{n}_retained_share": 1.0 for n, _, _ in bands})
        metrics = recovery_evaluation.measure_preservation(
            original, cleaned, sfreq, bands, window_s=settings.estimation_window_s)
        cdb, cfr = spec(cleaned)
        check = np.unique(np.concatenate([ordinary, teeth]))
        proms = np.array([prom(cdb, cfr, f) for f in check])
        supp = np.array([level(cdb, cfr, f) - level(odb, ofr, f) for f in check])
        out.append({"recording": name, "participant": name.split("_")[0], "arm": arm,
                    "stopbands": chosen_count, "of_lines": int(ordinary.size),
                    "peaks_above_2dB": int(np.nansum(proms > 2.0)),
                    "worst_prom_db": round(float(np.nanmax(proms)), 2),
                    "worst_level_db": round(float(np.nanmax(
                        [level(cdb, cfr, f) for f in check])), 2),
                    "median_suppression_db": round(float(np.nanmedian(supp)), 2),
                    "comb_db": round(comb_db(cdb, cfr), 2),
                    "correlation": round(metrics.signal_correlation, 5),
                    "change_rms": round(metrics.normalized_change_rms, 5),
                    "fir_rounds": len(result.rounds),
                    **{f"{b.name}_kept": round(avail[f"{b.name}_retained_share"], 4)
                       for b in metrics.bands}})
    return out


def _worker(payload):
    name, arms = payload
    try:
        return name, _measure(name, arms), None
    except Exception:
        return name, [], traceback.format_exc(limit=3)


def main():
    from decomb import recordings
    from decomb.config import load_config
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    runs = recordings.discover_runs(load_config(CONFIG).path("bids_root"), None, task="*")
    names = [p.stem for p in runs]
    if limit:
        names = ["sub-0005_task-thermalactive_run-6_eeg",
                 "sub-0001_task-thermalactive_run-6_eeg",
                 "sub-0006_task-thermalactive_run-5_eeg",
                 "sub-0011_task-thermalactive_run-4_eeg",
                 "sub-0013_task-thermalactive_run-6_eeg",
                 "sub-0010_task-thermalactive_run-1_eeg"][:limit]
    arms = (tuple(sys.argv[2].split(","))
            if len(sys.argv) > 2
            else ("notching", "notching_declared", "fixed_lines_only",
                  "comb_subtracted", "lines_declared"))
    output = BENCH / (f"arm_{'_'.join(arms)}_probe.tsv" if limit
                      else f"arm_{'_'.join(arms)}.tsv")
    log(f"{len(names)} recordings x {len(arms)} arms -> {output.name}")
    rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_worker, (n, arms)) for n in names]
        for done, fut in enumerate(as_completed(futures), start=1):
            name, measured, error = fut.result()
            if error:
                log(f"[{done}/{len(names)}] {name} FAILED: {error.splitlines()[-1]}"); continue
            rows.extend(measured)
            recordings.write_tsv_atomic(pd.DataFrame(rows), output)
            by = {r["arm"]: r for r in measured}
            log(f"[{done}/{len(names)}] {name}: " + "  ".join(
                f"{a}: gamma {by[a]['gamma_kept']:.3f} comb {by[a]['comb_db']:+.2f} "
                f"peaks {by[a]['peaks_above_2dB']}" for a in arms if a in by))
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
