"""PSD before vs after the combined method, using the user's exact MNE Welch settings."""
import importlib.util, sys, os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, mne
from dataclasses import replace as _replace
from pathlib import Path

STUDY = Path("/Users/joduq24/Desktop/decomb/studies/2026-08-19-arm-comparison/fixed_config.py")
spec_ = importlib.util.spec_from_file_location("study", STUDY)
study = importlib.util.module_from_spec(spec_); sys.modules["study"] = study
spec_.loader.exec_module(study)

from decomb import notch, recordings, recovery_benchmark
from decomb.config import load_config

N_JOBS = 8


def combined_arm(name):
    """Reproduce the `combined` arm and return (original_raw, cleaned_raw)."""
    config = load_config(study.CONFIG)
    base = notch.HarmonicNotchSettings.from_config(config)
    bands = notch.analysed_bands_from_config(config)
    raw = recordings.read_bids_raw(
        config.path("bids_root") / name.split("_")[0] / "eeg" / f"{name}.vhdr")
    ev = notch.fit_harmonic_round(raw, base, round_index=1)
    rows = []
    if ev.model.channels:
        rows.extend(notch.line_manifest_rows(name, ev.model, ev.plans, bands, base, round_index=1))
    if ev.scanner_harmonics is not None:
        rows.extend(notch.scanner_harmonic_manifest_rows(
            name, ev.scanner_harmonics, ev.scanner_plan, bands, base, round_index=1))
    for r in rows: r["removal_round"] = 1
    targets = recovery_benchmark.targets_from_manifest(pd.DataFrame(rows), name)
    ordinary = np.asarray(targets.ordinary_frequencies_hz)

    sfreq = float(raw.info["sfreq"])
    n_fft = recordings.estimation_window_samples(sfreq, 10.0)
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    nyq = np.nextafter(sfreq / 2.0, 0.0)
    teeth = np.arange(1, int(min(95.0, nyq) / study.COMB_HZ) + 1) * study.COMB_HZ
    teeth = teeth[(teeth >= 20.0) & (teeth < nyq)]

    def spec(d):
        p, fr = mne.time_frequency.psd_array_welch(
            d, sfreq, fmin=1.0, fmax=min(100.0, nyq), n_fft=n_fft, n_per_seg=n_fft,
            n_overlap=n_fft // 2, average="mean", window="hamming", remove_dc=True,
            verbose="ERROR")
        return 10 * np.log10(p.mean(axis=0) * 1e12), fr

    def prom(db, fr, c):
        pk = np.abs(fr - c) <= 0.11; nr = (np.abs(fr - c) > 1.0) & (np.abs(fr - c) <= 4.0)
        return float(db[pk].max() - np.median(db[nr])) if pk.any() and nr.sum() >= 3 else np.nan

    odb, ofr = spec(raw.get_data(picks=picks))
    tp = np.array([prom(odb, ofr, f) for f in teeth])
    strong = teeth[np.nan_to_num(tp, nan=-99) > study.TOOTH_SUBTRACT_DB]

    settings = _replace(base, comb_fundamental_hz=study.COMB_HZ)
    merged_targets = list(ordinary)
    for t in strong:
        if np.min(np.abs(ordinary - t)) > 0.05:
            merged_targets.append(float(t))
    subtract = np.array(sorted(set(merged_targets)))
    print(f"  subtracting {subtract.size} targets ({ordinary.size} lines + "
          f"{subtract.size - ordinary.size} teeth)", flush=True)
    prepared = recovery_benchmark.recover_with_multitaper(
        raw, recovery_benchmark.RecoveryTargets(tuple(subtract), ()),
        window_s=study.WINDOW_S, n_jobs=N_JOBS)

    rdb, rfr = spec(prepared.get_data(picks=picks))
    spans = []
    for group in study._clusters(np.unique(np.concatenate([subtract, teeth]))):
        best = np.nanmax([prom(rdb, rfr, f) for f in group])
        if np.isfinite(best) and best > study.FLOOR_DB:
            spans.append((min(group) - study.MARGIN_HZ, max(group) + study.MARGIN_HZ))
    merged = []
    for lo, hi in sorted(spans):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    print(f"  notching {len(merged)} residual clusters above {study.FLOOR_DB} dB", flush=True)
    if merged:
        plan = notch.HarmonicNotchPlan(
            tuple(notch.HarmonicStopband((), a, b, "isolated") for a, b in merged),
            settings.transition_bandwidth_hz)
        prepared = notch.apply_harmonic_notches(prepared, plan, n_jobs=N_JOBS)
    result = notch.clean_until_no_supported_lines(prepared, settings, n_jobs=N_JOBS)
    print(f"  {len(result.rounds)} FIR round(s)", flush=True)
    return raw, result.cleaned


def user_psd(raw):
    """The user's preprocessing and Welch settings, verbatim."""
    raw = raw.copy().load_data(verbose="ERROR")
    raw.filter(1, 100, verbose="ERROR")
    annot, _bads = mne.preprocessing.annotate_amplitude(raw, peak=30e-6, verbose="ERROR")
    raw.set_annotations(annot + raw.annotations)
    raw.set_montage("standard_1020", on_missing="ignore", verbose="ERROR")
    return raw.compute_psd(method="welch", fmin=1, fmax=100,
                           n_fft=10000, n_per_seg=10000, n_overlap=5000,
                           window="hamming", picks="eeg",
                           reject_by_annotation=False, verbose="ERROR")


def main():
    name = sys.argv[1]
    out = Path(sys.argv[2])
    print(f"running combined arm on {name}", flush=True)
    raw, cleaned = combined_arm(name)
    for label, r in (("before", raw), ("after", cleaned)):
        sp = user_psd(r)
        for lo, hi, tag in ((30, 50, "30-50Hz"), (1, 100, "full")):
            fig = sp.plot(spatial_colors=True, dB=True, amplitude=False, show=False)
            for ax in fig.axes:
                if ax.get_ylabel():
                    ax.set_xlim(lo, hi)
            fig.suptitle(f"{name}  —  {label} (combined method)", fontsize=10)
            p = out / f"{name.split('_task')[0]}_{name.split('run-')[1][0]}_{label}_{tag}.png"
            fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
            print(f"  wrote {p.name}", flush=True)


if __name__ == "__main__":
    main()
