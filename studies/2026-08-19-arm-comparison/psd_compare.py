"""Faithful user-spec PSD figures plus a matched-scale before/after comparison."""
import importlib.util, sys, os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, mne
from pathlib import Path

spec_ = importlib.util.spec_from_file_location(
    "psd_after", Path(__file__).parent / "psd_after.py")
pa = importlib.util.module_from_spec(spec_); sys.modules["psd_after"] = pa
spec_.loader.exec_module(pa)


def main():
    name, out = sys.argv[1], Path(sys.argv[2])
    short = f"{name.split('_task')[0]}_run{name.split('run-')[1][0]}"
    print(f"=== {name} ===", flush=True)
    raw, cleaned = pa.combined_arm(name)

    psds = {}
    for label, r in (("before", raw), ("after", cleaned)):
        sp = pa.user_psd(r)
        psds[label] = (sp.get_data(), sp.freqs)
        for lo, hi, tag in ((30, 50, "30-50Hz"), (1, 100, "full")):
            fig = sp.plot(spatial_colors=True, dB=True, amplitude=False, show=False)
            for ax in fig.axes:
                if ax.get_ylabel():
                    ax.set_xlim(lo, hi)
            fig.suptitle(f"{name}  —  {label} (combined method)", fontsize=10)
            fig.savefig(out / f"{short}_{label}_{tag}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)
    np.savez_compressed(out / f"{short}_psd.npz",
                        before=psds["before"][0], after=psds["after"][0],
                        freqs=psds["before"][1])

    # matched-scale comparison so the two panels are visually comparable
    for lo, hi, tag in ((30, 50, "30-50Hz"), (1, 100, "full")):
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, sharey=True)
        for ax, label in zip(axes, ("before", "after")):
            data, freqs = psds[label]
            m = (freqs >= lo) & (freqs <= hi)
            ax.plot(freqs[m], 10 * np.log10(data[:, m].T * 1e12), lw=0.5, alpha=0.6)
            ax.set_ylabel("Power (dB/Hz re 1 µV²)")
            ax.set_title(f"{label}", loc="left", fontsize=11)
            ax.grid(alpha=0.3, ls=":")
        axes[-1].set_xlabel("Frequency (Hz)")
        axes[0].set_xlim(lo, hi)
        d = np.concatenate([10 * np.log10(psds[k][0][:, (psds[k][1] >= lo) &
                                                        (psds[k][1] <= hi)] * 1e12).ravel()
                            for k in ("before", "after")])
        axes[0].set_ylim(np.percentile(d, 0.05) - 3, np.percentile(d, 99.99) + 3)
        fig.suptitle(f"{name} — combined method, matched scale", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"{short}_compare_{tag}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    print(f"  wrote figures for {short}", flush=True)


if __name__ == "__main__":
    main()
