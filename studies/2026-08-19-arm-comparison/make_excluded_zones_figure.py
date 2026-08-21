"""Cohort spectrum with the frequencies removal declares unavailable."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

MM = 1 / 25.4
INK, SPREAD, ZONE = "#1B1B1B", "#C9D4DE", "#B04A57"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#4A4A4A", "figure.dpi": 300, "savefig.dpi": 300,
})

d = np.load(sys.argv[1])
freqs, frac = d["freqs"], d["excluded_fraction"]
n, hours = int(d["n_recordings"]), float(d["hours"])
db = {k: 10 * np.log10(d[k] * 1e12) for k in ("before", "after")}
ylim = (db["before"].min() - 2, db["before"].max() + 2)

for label, title in (("before", "Source"), ("after", "Derivative")):
    fig, (ax, strip) = plt.subplots(
        2, 1, figsize=(160 * MM, 88 * MM), sharex=True,
        gridspec_kw={"height_ratios": [5.2, 1.0], "hspace": 0.10})
    values = db[label]
    ax.fill_between(freqs, values.min(axis=0), values.max(axis=0),
                    color=SPREAD, lw=0, zorder=1)
    ax.plot(freqs, values.mean(axis=0), color=INK, lw=1.1, zorder=3)
    ax.set_ylim(*ylim)
    ax.set_ylabel("Power spectral density\n(dB/Hz re 1 µV²)")
    ax.text(0.985, 0.94, title, transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold")
    ax.text(0.985, 0.845, f"{n} recordings · {hours:.1f} h · 63 sensors",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#666666")
    handles = [
        plt.Line2D([], [], color=INK, lw=1.1, label="sensor mean"),
        plt.Rectangle((0, 0), 1, 1, color=SPREAD, label="sensor range"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, handlelength=1.5,
              labelspacing=0.25, borderaxespad=0.3)

    strip.fill_between(freqs, 0, 100 * frac, color=ZONE, lw=0)
    strip.set_ylim(0, 100)
    strip.set_yticks([0, 100])
    strip.set_yticklabels(["0", "100"])
    strip.set_ylabel("declared\nunavailable (%)", fontsize=8, labelpad=2)
    strip.set_xlabel("Frequency (Hz)")
    strip.set_xlim(1, 100)
    strip.xaxis.set_major_locator(MultipleLocator(10))
    strip.tick_params(labelsize=8)

    path = f"{sys.argv[2]}_{label}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {path}")
