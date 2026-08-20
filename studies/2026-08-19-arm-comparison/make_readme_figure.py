"""README methods figure: mechanism, then cohort evidence, then cost."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

MM = 1 / 25.4
RAW, SUB, NOTCH, CONV = "#222222", "#4477AA", "#EE6677", "#228833"
GREY, TOOTH = "#888888", "#CCCCCC"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4, "lines.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 400, "savefig.dpi": 400,
})

d = np.load(sys.argv[1])
freqs = d["freqs"]
stats = pd.read_csv(sys.argv[2], sep="\t")
manifest = pd.read_csv(sys.argv[3], sep="\t", keep_default_na=False)

fig = plt.figure(figsize=(183 * MM, 108 * MM))
gs = fig.add_gridspec(2, 2, left=0.062, right=0.988, top=0.945, bottom=0.085,
                      hspace=0.52, wspace=0.24, width_ratios=[1.0, 1.12])


def letter(ax, ch, dx=-0.052, dy=1.10):
    ax.text(dx, dy, ch, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="top", ha="left")


# --- a: mechanism, high zoom -------------------------------------------------
LO, HI = 41.5, 44.5
ax = fig.add_subplot(gs[0, 0])
m = (freqs >= LO) & (freqs <= HI)
for lo, hi in d["threshold_spans"]:
    if hi > LO and lo < HI:
        ax.axvspan(max(lo, LO), min(hi, HI), color="#FBE4E7", lw=0, zorder=0)
ax.plot(freqs[m], d["raw"][m], color="#BBBBBB", lw=1.8, label="source", zorder=2,
        solid_capstyle="round")
ax.plot(freqs[m], d["subtracted"][m], color=SUB, lw=1.0, label="after subtraction", zorder=3)
ax.plot(freqs[m], d["final"][m], color=NOTCH, lw=1.0, ls=(0, (2.6, 2.0)),
        label="after notching", zorder=4)
ax.set_xlim(LO, HI); ax.set_ylim(-15.5, 4.5)
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (dB/Hz re 1 µV²)")
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.legend(loc="upper right", frameon=False, handlelength=1.5, borderaxespad=0.1,
          labelspacing=0.22)
ax.annotate("subtraction empties\none bin (±0.1 Hz)", xy=(42.0, -6.6),
            xytext=(41.60, -13.4), fontsize=6, color=SUB, ha="left", linespacing=1.25,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=SUB, shrinkA=1, shrinkB=2))
ax.annotate("notching removes\na contiguous band", xy=(43.45, -12.6),
            xytext=(43.60, -6.6), fontsize=6, color=NOTCH, ha="left", linespacing=1.25,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=NOTCH, shrinkA=1, shrinkB=2))
letter(ax, "a")

# --- b: where and how deep each stage removes --------------------------------
inner = gs[0, 1].subgridspec(3, 1, hspace=0.18)
band = (freqs >= 20) & (freqs <= 95)
pairs = [("raw", "subtracted", SUB, "subtract"),
         ("subtracted", "notched", NOTCH, "notch residue"),
         ("notched", "final", CONV, "converge")]
for i, (a, b, colour, label) in enumerate(pairs):
    axb = fig.add_subplot(inner[i])
    drop = d[a][band] - d[b][band]
    hit = drop > 0.5
    axb.vlines(freqs[band][hit], 0, drop[hit], color=colour, lw=0.55, alpha=0.95)
    axb.set_xlim(20, 95); axb.set_ylim(0, 44)
    axb.set_yticks([0, 40])
    axb.text(0.014, 0.98, f"{label}  ·  {hit.sum()} bins, median {np.median(drop[hit]):.1f} dB",
             transform=axb.transAxes, ha="left", va="top", fontsize=6.2, color=colour)
    if i < 2:
        axb.set_xticklabels([])
        axb.spines["bottom"].set_visible(False)
        axb.tick_params(bottom=False)
    else:
        axb.set_xlabel("Frequency (Hz)")
    if i == 1:
        axb.set_ylabel("Power removed (dB)")
    if i == 0:
        letter(axb, "b", dx=-0.088, dy=1.34)

# --- c: cohort comb, paired ---------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
before, after = stats.comb_db_before.values, stats.comb_db_after.values
rng = np.random.default_rng(0)
x0 = 0 + rng.uniform(-0.045, 0.045, before.size)
x1 = 1 + rng.uniform(-0.045, 0.045, after.size)
for a, b, xa, xb in zip(before, after, x0, x1):
    ax.plot([xa, xb], [a, b], color=GREY, lw=0.35, alpha=0.45, zorder=1)
ax.scatter(x0, before, s=4.5, color=RAW, zorder=3, linewidths=0)
ax.scatter(x1, after, s=4.5, color=SUB, zorder=3, linewidths=0)
ax.axhline(0, color="#BBBBBB", lw=0.6, ls=(0, (3, 3)), zorder=0)
for x, v, colour in ((0, before, RAW), (1, after, SUB)):
    ax.plot([x - 0.17, x + 0.17], [np.median(v)] * 2, color=colour, lw=1.8, zorder=5)
    ax.text(x - 0.24 if x == 0 else x + 0.24, np.median(v), f"{np.median(v):+.2f} dB",
            ha="right" if x == 0 else "left", va="center", fontsize=6.6,
            color=colour, fontweight="bold", zorder=6)
ax.set_xlim(-0.68, 1.68); ax.set_xticks([0, 1])
ax.set_xticklabels(["source", "derivative"])
ax.set_ylabel("Comb prominence (dB)")
ax.text(0.5, 0.965, f"n = {before.size} recordings", transform=ax.transAxes,
        ha="center", va="top", fontsize=6.5, color="#555555")
letter(ax, "c")

# --- d: availability cost -----------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
bands = ["delta", "theta", "alpha", "beta", "gamma"]
per = manifest.groupby("recording")[[f"{b}_retained_share" for b in bands]].first().astype(float)
data = [100 * per[f"{b}_retained_share"].values for b in bands]
bp = ax.boxplot(data, positions=range(len(bands)), widths=0.5, showfliers=False,
                medianprops=dict(color=RAW, lw=1.3), whiskerprops=dict(lw=0.6, color=GREY),
                capprops=dict(lw=0.6, color=GREY),
                boxprops=dict(lw=0.6, color=GREY), patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("#EAF0F6"); patch.set_edgecolor(GREY)
for i, v in enumerate(data):
    ax.scatter(i + rng.uniform(-0.13, 0.13, v.size), v, s=2.6, color=SUB,
               alpha=0.55, zorder=3, linewidths=0)
    ax.text(i, 103.2, f"{v.mean():.1f}", ha="center", fontsize=6.3, color=RAW)
ax.set_xticks(range(len(bands)))
ax.set_xticklabels([b.capitalize() for b in bands])
ax.set_ylim(50, 106); ax.set_ylabel("Bandwidth retained (%)")
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.text(0.5, 0.055, f"n = {len(per)} recordings", transform=ax.transAxes,
        ha="center", fontsize=6.5, color="#555555")
letter(ax, "d")

fig.savefig(sys.argv[4], bbox_inches="tight", pad_inches=0.012)
print(f"wrote {sys.argv[4]}")
