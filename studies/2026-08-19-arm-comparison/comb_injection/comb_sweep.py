"""Retention vs offset from a 1.2 Hz comb tooth, swept at fixed settings.

The CLI harness only probes three fixed offsets (0, 2*bin_width, midpoint). This calls
the same paired-trial function directly with arbitrary offsets, so nothing but the
injection frequency changes: identical notch settings, identical background cleaning
(computed once per recording and reused), identical RNG per (recording, kind) so the
injected realisation is the same at every offset.

Only `stationary` and `intermittent` are swept: their injection targets do not depend on
frequency_bin_width_hz at all, and they are the worst-affected kinds.
"""
from __future__ import annotations
import importlib.util, sys, os, time
from pathlib import Path
import numpy as np, pandas as pd
from dataclasses import replace as _replace

S = Path(__file__).parent
CONFIG = "/Users/joduq24/Desktop/decomb/decomb.yaml"
OFFSETS = tuple(float(x) for x in os.environ.get("OFFSETS",
    "0,0.025,0.05,0.075,0.10,0.15,0.20,0.30,0.45,0.60").split(","))
WINDOW_S = float(os.environ["WINDOW_S"]) if os.environ.get("WINDOW_S") else None
KINDS = ("stationary", "intermittent")
RECS = ["sub-0010_task-thermalactive_run-1_eeg",
        "sub-0003_task-thermalactive_run-1_eeg",
        "sub-0013_task-thermalactive_run-3_eeg"]
T0 = time.perf_counter()
def log(m): print(f"[{time.perf_counter()-T0:7.1f}s] {m}", flush=True)


def main():
    spec = importlib.util.spec_from_file_location("ci", S / "comb_injection.py")
    ci = importlib.util.module_from_spec(spec); sys.modules["ci"] = ci
    spec.loader.exec_module(ci)
    from decomb import notch, recordings, recovery_benchmark
    from decomb import neural_recovery_validation as nrv
    from decomb.config import load_config

    config = load_config(CONFIG)
    base = notch.HarmonicNotchSettings.from_config(config)
    bands_std = notch.analysed_bands_from_config(config)
    if WINDOW_S is not None:
        base = _replace(base, estimation_window_s=WINDOW_S)
    bw = base.frequency_bin_width_hz
    log(f"estimation_window_s={base.estimation_window_s} bin_width={bw}")
    out = Path(sys.argv[1])
    rows = []

    for name in RECS:
        mf, pairs = ci.tooth_pairs(name, config, base, bands_std)
        if not pairs:
            log(f"{name}: no tooth pair, skipping"); continue
        a, b = pairs[0]
        mid = (a + b) / 2.0
        band = (f"comb{int(round(a))}", mid - 1.0, mid + 1.0)
        path = config.path("bids_root") / name.split("_")[0] / "eeg" / f"{name}.vhdr"
        raw = recordings.read_bids_raw(path)
        targets = recovery_benchmark.targets_from_manifest(mf, name)
        log(f"{name}: tooth {a:.3f} Hz (harmonic {a/1.2:.0f}), cleaning background once")
        bg, paired_trial = nrv.prepare_background(
            raw, targets, base, residual_protocol="frozen",
            spatial_rank=None, n_jobs=8, residual_stage="lines")
        log(f"  background terminal null = {bg.terminal_residual_detector_null}")
        participant = recordings.subject_of(path)
        for kind in KINDS:
            rng_key = (name, band[0], kind, "sweep")
            for off in OFFSETS:
                placement = nrv.FrequencyPlacement(
                    band[0], band[1], band[2],
                    "exact" if off == 0.0 else "near",
                    float(a), float(b), float(a + off))
                target = nrv.injection_target(
                    placement, kind, frequency_bin_width_hz=bw,
                    component_to_background_db=nrv.COMPONENT_TO_BACKGROUND_DB)
                t = time.perf_counter()
                measured = paired_trial(
                    raw, bg, targets, placement, target, nrv._trial_rng(rng_key),
                    recording=name, participant=participant,
                    channel_name=nrv.INJECTION_CHANNEL_NAME,
                    notch_settings=base,
                    recovery_window_s=base.estimation_window_s,
                    n_jobs=8, residual_stage="lines")
                for r in measured:
                    r["offset_hz"] = off
                    r["tooth_hz"] = float(a)
                rows.extend(measured)
                pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
                fin = measured[-1]
                log(f"  {kind:12s} +{off:.3f} Hz: remaining={float(fin['remaining_fraction']):.3f} "
                    f"error={float(fin['component_error_fraction']):.3f} ({time.perf_counter()-t:.0f}s)")
    log(f"done -> {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
