"""Pair the shipped subtraction arm against current notching and the tuned arm."""
import sys

import numpy as np
import pandas as pd

EXCAVATION = [
    "sub-0003_task-thermalactive_run-1_eeg",
    "sub-0006_task-thermalactive_run-1_eeg",
    "sub-0006_task-thermalactive_run-2_eeg",
    "sub-0013_task-thermalactive_run-1_eeg",
    "sub-0014_task-thermalactive_run-6_eeg",
]
OUTLIER = "sub-0008"


def load():
    shipped = pd.read_csv("derived_shipped.tsv", sep="\t")
    combined = pd.read_csv("arm_combined.tsv", sep="\t")
    notching = pd.read_csv("final_config.tsv", sep="\t").query("arm == 'notching'")
    return shipped, combined, notching


def summarise(frame, label, mask=None):
    d = frame if mask is None else frame[mask]
    return {
        "arm": label,
        "n": len(d),
        "peaks_2dB_mean": round(d["peaks_above_2dB"].mean(), 1),
        "comb_db_median": round(d["comb_db"].median(), 2),
        "abs_comb_db_mean": round(d["comb_db"].abs().mean(), 2),
        "abs_comb_db_max": round(d["comb_db"].abs().max(), 2),
        "correlation_mean": round(d["correlation"].mean(), 4),
        "change_rms_mean": round(d["change_rms"].mean(), 4),
        "gamma_kept_mean": round(d["gamma_kept"].mean(), 3),
        "beta_kept_mean": round(d["beta_kept"].mean(), 3),
        "alpha_kept_mean": round(d["alpha_kept"].mean(), 3),
    }


def main():
    shipped, combined, notching = load()
    out = sys.stdout

    common = sorted(set(shipped.recording) & set(combined.recording) & set(notching.recording))
    print(f"paired recordings: {len(common)}", file=out)

    def sub(frame):
        return frame[frame.recording.isin(common)].set_index("recording").loc[common].reset_index()

    s, c, n = sub(shipped), sub(combined), sub(notching)
    no8 = ~s.recording.str.startswith(OUTLIER)
    no8_exc = no8 & ~s.recording.isin(EXCAVATION)

    print("\n== whole cohort (all paired recordings) ==", file=out)
    rows = [summarise(n, "notching (shipping)"), summarise(c, "combined (tuned)"), summarise(s, "derived (shipped)")]
    print(pd.DataFrame(rows).to_string(index=False), file=out)

    print(f"\n== excluding {OUTLIER} (bad ECG) ==", file=out)
    rows = [
        summarise(n, "notching (shipping)", no8),
        summarise(c, "combined (tuned)", no8),
        summarise(s, "derived (shipped)", no8),
    ]
    print(pd.DataFrame(rows).to_string(index=False), file=out)

    print(f"\n== excluding {OUTLIER} and the 5 stale excavation recordings ==", file=out)
    rows = [
        summarise(n, "notching (shipping)", no8_exc),
        summarise(c, "combined (tuned)", no8_exc),
        summarise(s, "derived (shipped)", no8_exc),
    ]
    print(pd.DataFrame(rows).to_string(index=False), file=out)

    print("\n== null-calibrated residual peaks, derived arm only ==", file=out)
    print(f"raw 2 dB count           mean {s.peaks_above_2dB.mean():.1f}  median {s.peaks_above_2dB.median():.0f}", file=out)
    print(f"per-recording p99 count  mean {s.peaks_above_null_p99.mean():.1f}  median {s.peaks_above_null_p99.median():.0f}", file=out)
    print(f"null p99 (dB)            mean {s.null_p99_db.mean():.2f}  min {s.null_p99_db.min():.2f}  max {s.null_p99_db.max():.2f}", file=out)
    print(f"recordings whose null p99 exceeds 2 dB: {(s.null_p99_db > 2.0).sum()} of {len(s)}", file=out)
    print(f"noise bins over 2 dB, mean share: {s.null_frac_over_2db.mean():.3f}", file=out)

    print("\n== gamma availability, derived arm ==", file=out)
    print(f"declared (subtraction damage + FIR): {s.gamma_kept.mean():.3f}", file=out)
    print(f"counting FIR stopbands only:         {s.gamma_kept_fir_only.mean():.3f}", file=out)
    print(f"current notching:                    {n.gamma_kept.mean():.3f}", file=out)

    print(f"\n== {OUTLIER}, reported separately ==", file=out)
    cols = ["recording", "n_targets", "peaks_above_2dB", "peaks_above_null_p99", "comb_db", "correlation", "change_rms", "gamma_kept"]
    print(s[~no8][cols].to_string(index=False), file=out)

    print("\n== paired deltas, derived minus notching (excluding sub-0008) ==", file=out)
    for col in ("comb_db", "correlation", "change_rms", "gamma_kept"):
        d = s.loc[no8, col].values - n.loc[no8, col].values
        wins = int((d > 0).sum()) if col != "change_rms" else int((d < 0).sum())
        print(f"{col:14s} mean {d.mean():+.4f}  median {np.median(d):+.4f}  better in {wins}/{no8.sum()}", file=out)

    print("\n== paired deltas, derived minus combined (excluding sub-0008 and excavation) ==", file=out)
    for col in ("comb_db", "correlation", "change_rms", "gamma_kept"):
        d = s.loc[no8_exc, col].values - c.loc[no8_exc, col].values
        wins = int((d > 0).sum()) if col != "change_rms" else int((d < 0).sum())
        print(f"{col:14s} mean {d.mean():+.4f}  median {np.median(d):+.4f}  better in {wins}/{no8_exc.sum()}", file=out)


if __name__ == "__main__":
    main()
