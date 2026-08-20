"""The README figure: before and after, and the bandwidth it costs."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MM = 1 / 25.4
INK, SOURCE, DERIV = "#1B1B1B", "#9AA3AB", "#2F6690"
REMOVED, LOST = "#DCE6EE", "#F3DFE1"
COMB = 1.2
LO, HI = 37.0, 52.0
TRANS = 0.061111 / 2.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#4A4A4A", "figure.dpi": 300, "savefig.dpi": 300,
})

d = np.load(sys.argv[1])
freqs = d["freqs"]
m = (freqs >= LO) & (freqs <= HI)
f, raw, final = freqs[m], d["raw"][m], d["final"][m]

# everything the manifest declares unavailable: the +/-0.1 Hz a subtracted line costs,
# and the stopbands plus transitions of every filter that ran
filtered = [tuple(x) for x in d["damage"]]
filtered += [(a - TRANS, b + TRANS) for a, b in d["threshold_spans"]]
filtered += [tuple(x) for x in d["fir_spans"]]
merged = []
for lo, hi in sorted(filtered):
    if merged and lo <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], hi)
    else:
        merged.append([lo, hi])

fig, ax = plt.subplots(figsize=(165 * MM, 76 * MM))
ymin, ymax = -12.5, 6.0

for lo, hi in merged:
    if hi > LO and lo < HI:
        ax.axvspan(max(lo, LO), min(hi, HI), ymin=0, ymax=1, color=LOST, lw=0, zorder=1)
ax.fill_between(f, final, raw, where=raw >= final, color=REMOVED, lw=0,
                interpolate=True, zorder=2)
ax.plot(f, raw, color=SOURCE, lw=1.2, zorder=3, solid_capstyle="round")
ax.plot(f, final, color=DERIV, lw=1.3, zorder=4, solid_capstyle="round")

ax.set_xlim(LO, HI); ax.set_ylim(ymin, ymax)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power spectral density  (dB/Hz re 1 µV²)")
ax.set_xticks(np.arange(38, 53, 2))
ax.set_yticks([-10, -5, 0, 5])

handles = [
    plt.Line2D([], [], color=SOURCE, lw=1.1, label="source"),
    plt.Line2D([], [], color=DERIV, lw=1.3, label="derivative"),
    plt.Rectangle((0, 0), 1, 1, color=REMOVED, label="power removed"),
    plt.Rectangle((0, 0), 1, 1, color=LOST, label="frequencies declared unavailable"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4,
          frameon=False, handlelength=1.6, columnspacing=1.6, borderaxespad=0)
ax.text(0.985, 0.955, "stopbands continue below the axis", transform=ax.transAxes,
        ha="right", va="top", fontsize=8, color="#8A8A8A")

fig.tight_layout(pad=0.5)
fig.savefig(sys.argv[2], bbox_inches="tight", pad_inches=0.03)
print(f"wrote {sys.argv[2]}")
