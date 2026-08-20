"""Three README figures, one idea each."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MM = 1 / 25.4
INK, ACCENT, MUTED, FAINT = "#1B1B1B", "#3B6FA0", "#9A9A9A", "#E8EDF2"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "#555555",
    "figure.dpi": 300, "savefig.dpi": 300,
})

stages = np.load(sys.argv[1])
stats = pd.read_csv(sys.argv[2], sep="\t")
manifest = pd.read_csv(sys.argv[3], sep="\t", keep_default_na=False)
out = sys.argv[4]

# ============================ 1. what the pipeline removes ====================
LO, HI = 38.0, 50.0
freqs = stages["freqs"]
m = (freqs >= LO) & (freqs <= HI)
fig, ax = plt.subplots(figsize=(130 * MM, 62 * MM))
# the shaded area is exactly the power the pipeline removed
ax.fill_between(freqs[m], stages["final"][m], stages["raw"][m],
                where=stages["raw"][m] >= stages["final"][m],
                color=FAINT, lw=0, zorder=1, interpolate=True)
ax.plot(freqs[m], stages["raw"][m], color=MUTED, lw=1.0, zorder=2)
ax.plot(freqs[m], stages["final"][m], color=ACCENT, lw=1.2, zorder=3)
ax.set_xlim(LO, HI); ax.set_ylim(-13, 5.5)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power spectral density\n(dB/Hz re 1 µV²)")
ax.set_xticks(np.arange(38, 51, 2))
ax.set_yticks([-10, -5, 0, 5])
handles = [
    plt.Line2D([], [], color=MUTED, lw=1.0, label="source"),
    plt.Line2D([], [], color=ACCENT, lw=1.2, label="derivative"),
    plt.Rectangle((0, 0), 1, 1, color=FAINT, label="power removed"),
]
ax.legend(handles=handles, loc="upper left", frameon=False, handlelength=1.5,
          labelspacing=0.3, borderaxespad=0.1, ncol=3, columnspacing=1.2)
ax.text(0.015, 0.055, "notches continue below the axis", transform=ax.transAxes,
        ha="left", fontsize=7.5, color=MUTED)
fig.tight_layout(pad=0.4)
fig.savefig(f"{out}/comb_removal_spectrum.png", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ============================ 2. across the cohort ============================
before, after = stats.comb_db_before.values, stats.comb_db_after.values
fig, ax = plt.subplots(figsize=(88 * MM, 82 * MM))
lim = (-1.6, 5.6)
ax.plot(lim, lim, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
ax.axhline(0, color="#CFCFCF", lw=0.7, zorder=1)
ax.scatter(before, after, s=17, facecolor=ACCENT, edgecolor="white",
           linewidth=0.4, alpha=0.9, zorder=3)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel("Comb prominence, source (dB)")
ax.set_ylabel("Comb prominence, derivative (dB)")
ax.set_xticks([0, 2, 4]); ax.set_yticks([0, 2, 4])
ax.text(0.97, 0.955, "no change", transform=ax.transAxes, ha="right", va="top",
        color=MUTED, fontsize=8, rotation=39, rotation_mode="anchor")
ax.text(5.4, 0.42, "fully removed", ha="right", va="bottom", color=ACCENT, fontsize=8)
ax.text(0.03, 0.955, f"n = {before.size} recordings", transform=ax.transAxes,
        ha="left", va="top", fontsize=8, color=INK)
fig.tight_layout(pad=0.4)
fig.savefig(f"{out}/comb_removal_cohort.png", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ============================ 3. what it costs ================================
bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
per = manifest.groupby("recording")[[f"{b.lower()}_retained_share" for b in bands]].first()
per = per.astype(float) * 100
fig, ax = plt.subplots(figsize=(112 * MM, 62 * MM))
rng = np.random.default_rng(0)
for i, b in enumerate(bands):
    v = per[f"{b.lower()}_retained_share"].values
    ax.scatter(v, i + rng.uniform(-0.16, 0.16, v.size), s=8, color=ACCENT,
               alpha=0.28, linewidths=0, zorder=2)
    ax.plot([v.mean()], [i], marker="|", ms=15, mew=1.7, color=INK, zorder=4)
    ax.text(101.5, i, f"{v.mean():.1f}%", va="center", ha="left", fontsize=8.5,
            fontweight="medium", color=INK)
ax.set_yticks(range(len(bands))); ax.set_yticklabels(bands)
ax.invert_yaxis()
ax.set_xlim(48, 108); ax.set_xticks([50, 60, 70, 80, 90, 100])
ax.set_xlabel("Bandwidth retained after removal (%)")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.text(0.015, 0.965, f"n = {len(per)} recordings", transform=ax.transAxes,
        va="top", fontsize=8, color=MUTED)
fig.tight_layout(pad=0.4)
fig.savefig(f"{out}/availability_cost.png", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("wrote comb_removal_spectrum.png, comb_removal_cohort.png, availability_cost.png")
