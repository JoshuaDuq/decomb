"""Full-spectrum MNE PSD of the derivative with every declared-excluded zone marked."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne

from decomb import notch, recordings
from decomb.config import load_config

CONFIG = "/Users/joduq24/Desktop/decomb/decomb.yaml"
ZONE = "#E8A0A8"


def user_psd(path):
    """The user's preprocessing and Welch settings, verbatim."""
    raw = mne.io.read_raw_brainvision(str(path), misc=["ECG"], verbose="ERROR")
    raw.load_data(verbose="ERROR")
    raw.filter(1, 100, verbose="ERROR")
    annot, _ = mne.preprocessing.annotate_amplitude(raw, peak=30e-6, verbose="ERROR")
    raw.set_annotations(annot + raw.annotations)
    raw.set_montage("standard_1020", on_missing="ignore", verbose="ERROR")
    return raw.compute_psd(method="welch", fmin=1, fmax=100, n_fft=10000,
                           n_per_seg=10000, n_overlap=5000, window="hamming",
                           picks="eeg", reject_by_annotation=False, verbose="ERROR")


def main():
    name, out = sys.argv[1], sys.argv[2]
    config = load_config(CONFIG)
    src, deriv = config.path("bids_root"), config.path("output_root")
    manifest = notch._read_manifest(config.path("removal_dir") / notch.MANIFEST_NAME)
    block = manifest.loc[manifest["recording"] == name]
    zones = notch.merged_intervals(
        (float(a), float(b))
        for a, b in zip(block["unavailable_low_hz"], block["unavailable_high_hz"])
        if str(a) != "" and str(b) != ""
    )
    vhdr = src / name.split("_")[0] / "eeg" / f"{name}.vhdr"
    spectra = {
        "source": user_psd(vhdr),
        "derivative": user_psd(recordings.derivative_vhdr_path(vhdr, src, deriv)),
    }
    # one decibel scale for both, taken from the source, so the pair is comparable
    reference = 10 * np.log10(spectra["source"].get_data() * 1e12)
    pad = 0.04 * (reference.max() - reference.min())
    ylim = (reference.min() - pad, reference.max() + pad)
    excluded = sum(hi - lo for lo, hi in zones)

    for label, spectrum in spectra.items():
        fig = spectrum.plot(spatial_colors=True, dB=True, amplitude=False, show=False)
        fig.set_size_inches(11.0, 4.6)
        for ax in fig.axes:
            if not ax.get_ylabel():
                continue
            for lo, hi in zones:
                ax.axvspan(lo, hi, color=ZONE, alpha=0.55, lw=0, zorder=0)
            ax.set_xlim(1, 100)
            ax.set_ylim(*ylim)
            handle = plt.Rectangle((0, 0), 1, 1, color=ZONE, alpha=0.55)
            ax.legend(
                [handle],
                [f"declared unavailable: {len(zones)} zones, {excluded:.1f} Hz of 99 Hz"],
                loc="lower left", frameon=True, framealpha=0.9, edgecolor="none",
                fontsize=9, handlelength=1.6, borderaxespad=0.4)
        fig.suptitle(f"{name} — {label}, shared scale", fontsize=10)
        path = f"{out}_{label}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")
    print(f"  {len(zones)} zones, {excluded:.2f} Hz of 99 Hz excluded")


if __name__ == "__main__":
    main()
