"""Injection retention measured AT 1.2 Hz comb teeth.

The published injection tables anchored on 1.111 Hz TR harmonics and covered one
recording. This aims the same harness at consecutive 1.2 Hz teeth, on several
recordings, by giving each recording custom bands centred on its own tooth pairs.
"""
from __future__ import annotations
import subprocess, sys, os
from pathlib import Path
import numpy as np, pandas as pd
from dataclasses import replace as _replace

S = Path(__file__).parent
PATCHED = S / "patched_src"
CONFIG = "/Users/joduq24/Desktop/decomb/decomb.yaml"
RECS = ["sub-0010_task-thermalactive_run-1_eeg",
        "sub-0003_task-thermalactive_run-1_eeg",
        "sub-0013_task-thermalactive_run-3_eeg"]
PAIRS_PER_RECORDING = 2

CFG_TEMPLATE = """paths:
  bids_root: "/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg"
  output_root: "/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg_decombed_auto"
removal:
  scanner_repetition_time_s: 0.9
  scanner_trigger_event_name: "Volume/V  1"
  comb_fundamental_hz: 1.2
execution:
  n_jobs: 8
frequency_bands:
  delta: [1.0, 1.05]
  theta: [4.0, 4.05]
  alpha: [8.0, 8.05]
  beta: [13.0, 13.05]
  gamma: [30.1, 30.15]
"""


def tooth_pairs(name, config, base, bands_std):
    from decomb import notch, recordings, recovery_benchmark
    raw = recordings.read_bids_raw(
        config.path("bids_root") / name.split("_")[0] / "eeg" / f"{name}.vhdr")
    st = _replace(base, comb_fundamental_hz=1.2)
    ev = notch.fit_harmonic_round(raw, st, round_index=1)
    rows = []
    if ev.model.channels:
        rows.extend(notch.line_manifest_rows(name, ev.model, ev.plans, bands_std, st, round_index=1))
    if ev.scanner_harmonics is not None:
        rows.extend(notch.scanner_harmonic_manifest_rows(
            name, ev.scanner_harmonics, ev.scanner_plan, bands_std, st, round_index=1))
    for r in rows: r["removal_round"] = 1
    mf = pd.DataFrame(rows)
    t = recovery_benchmark.targets_from_manifest(mf, name)
    f = np.sort(np.asarray(t.all_frequencies_hz))
    on = np.abs(f - np.round(f / 1.2) * 1.2) < 0.06
    pairs = [(f[i], f[i + 1]) for i in range(len(f) - 1)
             if on[i] and on[i + 1] and 1.1 < f[i + 1] - f[i] < 1.3 and f[i] >= 20]
    return mf, pairs


def main():
    from decomb import notch
    from decomb.config import load_config
    out = S / "comb_injection"; out.mkdir(exist_ok=True)
    config = load_config(CONFIG)
    base = notch.HarmonicNotchSettings.from_config(config)
    bands_std = notch.analysed_bands_from_config(config)

    for name in RECS:
        mf, pairs = tooth_pairs(name, config, base, bands_std)
        pairs = pairs[:PAIRS_PER_RECORDING]
        if not pairs:
            print(f"{name}: no on-grid tooth pair, skipping", flush=True); continue
        cfg = CFG_TEMPLATE
        for a, b in pairs:
            mid = (a + b) / 2.0
            cfg += f"  comb{int(round(a)):d}: [{mid-1.0:.4f}, {mid+1.0:.4f}]\n"
        short = f"{name.split('_task')[0]}_run{name.split('run-')[1][0]}"
        cfg_path = out / f"{short}.yaml"; cfg_path.write_text(cfg)
        mf_path = out / f"{short}_manifest.tsv"
        mf.to_csv(mf_path, sep="\t", index=False)
        res = out / f"{short}_comb_injection.tsv"
        if res.exists():
            print(f"{name}: {res.name} exists, skipping", flush=True); continue
        print(f"=== {name}: teeth {[(round(a,3), round(b,3)) for a,b in pairs]} ===", flush=True)
        env = dict(os.environ, PYTHONPATH=str(PATCHED))
        cmd = [sys.executable, "-m", "decomb.neural_recovery_validation",
               "--config", str(cfg_path), "--manifest", str(mf_path),
               "--output", str(res), "--recordings", name,
               "--residual-protocol", "frozen", "--residual-stage", "lines",
               "--n-jobs", "8"]
        p = subprocess.run(cmd, env=env)
        print(f"  exit={p.returncode}", flush=True)


if __name__ == "__main__":
    main()
