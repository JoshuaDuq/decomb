#!/usr/bin/env python3
"""Measure what each removal stage costs, and on which artifact.

    python docs/make_notch_figure.py

README section 2 says a wide FIR notch removes its full stop-band, including frequencies
where no artifact was measured, and section 6.3 says `apply` and `notch` are counterparts
rather than competitors. Both are prose. This measures them.

Nothing here is a claim about which tool is better. Each column is the artifact structure
one of them exists for, and each wins its own column: a comb is sparse, so removing it line
by line costs a fraction of what covering the same lines with notches costs; a cluster is
not, so the notch clears it and target-by-target fitting does not.

The second row is the point. What a transform did to the artifact is visible in a spectrum;
what it cost a signal carrying no artifact is not, and that is the quantity the prose claims.
So a broadband recording with nothing in it goes through both transforms unchanged, which is
the same construction `benchmark` measures band cost with.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_figure import (  # noqa: E402
    AFTER_COLOR,
    AXIS_INK,
    BAND_HZ,
    BEFORE_COLOR,
    COMB_HARMONICS,
    COMB_PROMINENCE_DB,
    FAINT_INK,
    FUNDAMENTAL_HZ,
    GRID,
    MAINS_HZ,
    MUTED_INK,
    N_CHANNELS,
    RHYTHM_HZ,
    RHYTHM_PROMINENCE_DB,
    SFREQ_HZ,
    TITLE_INK,
    _background,
    _rhythm,
    _scale_for,
    local_background_db,
    write_config,
)

DURATION_S = 300.0
N_RECORDINGS = 1
"""One recording. Nothing here is a cohort statistic -- both arms are transforms measured on
the same data, and a second recording would only average two draws of the same noise."""

CLUSTER_HZ = 20.0
CLUSTER_SPAN_HZ = 1.0
CLUSTER_PEAKS = 12
CLUSTER_WANDER_HZ = 0.08
"""A dense cluster: twelve peaks over one hertz, each wandering across the recording.

The wander is what makes it a cluster rather than twelve lines, and it is measured rather
than asserted. Detection on this finds 18 peaks over 3 dB where 12 were planted, of which 8
are narrow enough to be called lines; the stationary version yields 12, all 12 targetable,
which the removal simply takes. Placed below the comb's harmonic range, clear of the rhythm
and of the mains band, so each column shows one structure and not a mixture.
"""
CLUSTER_PROMINENCE_DB = 18.0

COST_BAND_HZ = (28.0, 95.0)
"""The span `removal.cost_band_hz` defaults to, so the number this figure prints is the one
`benchmark` reports."""

NOTCH_FILTER_LENGTH = "30s"
"""MNE cannot build this filter at its own default.

``filter_length='auto'`` raises on a 1.2 Hz comb -- "the requested filter length 1651 is too
short for the requested 0.11 Hz transition band" -- because the notch is narrower than an
automatically sized FIR can resolve. Every other notch parameter is left at MNE's default,
including the ``freq/200`` widths and the 1 Hz transitions, which is where its cost lives.
"""

COMB_VIEW_HZ = (60.5, 68.0)
CLUSTER_VIEW_HZ = (18.6, 21.4)

NOTCH_COLOR = "#EDA100"


def _cluster(times, scale):
    """Peaks packed inside one hertz, each drifting on its own slow schedule."""
    centres = CLUSTER_HZ + np.linspace(-CLUSTER_SPAN_HZ / 2, CLUSTER_SPAN_HZ / 2, CLUSTER_PEAKS)
    out = np.zeros((N_CHANNELS, times.size))
    for index, centre in enumerate(centres):
        drift = CLUSTER_WANDER_HZ * np.sin(2 * np.pi * times / DURATION_S + index)
        phase = 2 * np.pi * np.cumsum(centre + drift) / SFREQ_HZ
        out += scale * np.sin(phase + index)
    return out


def write_dataset(root: Path) -> None:
    """One BIDS recording carrying a sparse comb, a dense cluster and a broad rhythm."""
    import mne
    from mne_bids import BIDSPath, write_raw_bids

    mne.set_log_level("ERROR")
    times = np.arange(int(SFREQ_HZ * DURATION_S)) / SFREQ_HZ
    mid_harmonic = (COMB_HARMONICS.start + COMB_HARMONICS.stop) // 2

    def tone(rng, sample_times, scale):
        return scale * np.sin(2 * np.pi * FUNDAMENTAL_HZ * mid_harmonic * sample_times)

    comb_v = _scale_for(COMB_PROMINENCE_DB, FUNDAMENTAL_HZ * mid_harmonic, times, tone)
    rhythm_v = _scale_for(RHYTHM_PROMINENCE_DB, RHYTHM_HZ, times, _rhythm)
    cluster_v = _scale_for(
        CLUSTER_PROMINENCE_DB, CLUSTER_HZ, times, lambda rng, t, scale: _cluster(t, scale)
    )
    print(f"  comb amplitude    {comb_v:.3e} V -> {COMB_PROMINENCE_DB:g} dB")
    print(
        f"  cluster amplitude {cluster_v:.3e} V -> "
        f"{CLUSTER_PROMINENCE_DB:g} dB at {CLUSTER_HZ:g} Hz"
    )
    print(f"  rhythm scale      {rhythm_v:.3e}")

    rng = np.random.default_rng(1000)
    data = _background(rng, times)
    for harmonic in COMB_HARMONICS:
        data += comb_v * np.sin(2 * np.pi * FUNDAMENTAL_HZ * harmonic * times + harmonic)
    data += _cluster(times, cluster_v)
    data += _rhythm(np.random.default_rng(5000), times, rhythm_v)

    info = mne.create_info([f"EEG{i:02d}" for i in range(N_CHANNELS)], SFREQ_HZ, "eeg")
    write_raw_bids(
        mne.io.RawArray(data, info, verbose="ERROR"),
        BIDSPath(subject="0000", task="rest", datatype="eeg", root=root, extension=".vhdr"),
        format="BrainVision",
        allow_preload=True,
        verbose="ERROR",
    )


def notch_frequencies() -> np.ndarray:
    """Every comb harmonic a practitioner would hand to `notch_filter`, mains excluded."""
    return np.asarray(
        [
            FUNDAMENTAL_HZ * harmonic
            for harmonic in COMB_HARMONICS
            if abs(FUNDAMENTAL_HZ * harmonic - MAINS_HZ) > 1.0
        ],
        dtype=float,
    )


def apply_notch(data: np.ndarray, frequencies: np.ndarray, widths=None) -> np.ndarray:
    """MNE's own notch, at MNE's own defaults but for a filter long enough to build."""
    from mne.filter import notch_filter

    return notch_filter(
        np.asarray(data, dtype=float).copy(),
        SFREQ_HZ,
        frequencies,
        notch_widths=widths,
        filter_length=NOTCH_FILTER_LENGTH,
        verbose="ERROR",
    )


def spectrum(data):
    """The Welch estimator `decomb psd` uses, on one array."""
    import mne

    from decomb import psd as psd_stage

    raw = mne.io.RawArray(
        np.asarray(data, dtype=float),
        mne.create_info([f"EEG{i:02d}" for i in range(np.shape(data)[0])], SFREQ_HZ, "eeg"),
        verbose="ERROR",
    )
    return psd_stage.channel_median_psd(raw, psd_stage.PsdSettings(band_hz=BAND_HZ))


def measure(root: Path, work: Path):
    """Both transforms, applied to the recording and to a probe that carries no artifact."""
    from decomb import remove

    settings = remove.RemovalSettings.from_config(
        {
            "removal": {
                "harmonic_range": [COMB_HARMONICS.start, COMB_HARMONICS.stop - 1],
                "removal_harmonic_range": [COMB_HARMONICS.start, COMB_HARMONICS.stop - 1],
                "high_hz": 99.0,
            }
        }
    )
    runs = list(remove.discover_runs(root, subjects=None, task="rest"))
    print("\n--- fitting the removal plan " + "-" * 45)
    plans = remove.build_run_plans(runs, settings)
    plan = plans[runs[0].stem]

    raw = remove.read_bids_raw(runs[0])
    original = raw.get_data()
    decombed = remove.clean_continuous_raw(raw.copy(), plan, settings).get_data()
    notched_comb = apply_notch(original, notch_frequencies())
    notched_cluster = apply_notch(
        original,
        np.asarray([CLUSTER_HZ]),
        widths=np.asarray([CLUSTER_SPAN_HZ + 2.0 * CLUSTER_WANDER_HZ]),
    )

    # A recording with nothing in it, through the identical transforms. This is the only
    # way to say what a transform costs rather than what it removed.
    import mne

    times = np.arange(original.shape[1]) / SFREQ_HZ
    probe = np.random.default_rng(77).normal(size=(N_CHANNELS, times.size)) * 5e-7
    probe_raw = mne.io.RawArray(
        probe,
        mne.create_info([f"EEG{i:02d}" for i in range(N_CHANNELS)], SFREQ_HZ, "eeg"),
        verbose="ERROR",
    )
    probe_decombed = remove.clean_continuous_raw(probe_raw.copy(), plan, settings).get_data()
    probe_notched_comb = apply_notch(probe, notch_frequencies())
    probe_notched_cluster = apply_notch(
        probe,
        np.asarray([CLUSTER_HZ]),
        widths=np.asarray([CLUSTER_SPAN_HZ + 2.0 * CLUSTER_WANDER_HZ]),
    )

    freqs, before = spectrum(original)
    return {
        "freqs": freqs,
        "before": before,
        "decomb": spectrum(decombed)[1],
        "notch_comb": spectrum(notched_comb)[1],
        "notch_cluster": spectrum(notched_cluster)[1],
        "probe": spectrum(probe)[1],
        "probe_decomb": spectrum(probe_decombed)[1],
        "probe_notch_comb": spectrum(probe_notched_comb)[1],
        "probe_notch_cluster": spectrum(probe_notched_cluster)[1],
    }


def attenuation_db(measured, key):
    """How much of a signal that carried no artifact each transform took, per frequency."""
    from decomb.spectral import to_db

    return to_db(measured["probe"]) - to_db(measured[key])


def band_share(measured, key, threshold_db=1.0):
    freqs = measured["freqs"]
    inside = (freqs >= COST_BAND_HZ[0]) & (freqs <= COST_BAND_HZ[1])
    return float(np.mean(attenuation_db(measured, key)[inside] > threshold_db))


def cost_hz(measured, key, threshold_db=1.0):
    """Spectrum a transform took, in hertz, wherever it took it.

    A share of `cost_band_hz` is the number `benchmark` reports, and it is the wrong one for
    a transform aimed outside that band: the cluster sits at 20 Hz, so notching it costs
    0.000 of 28-95 Hz while plainly costing something. Hertz is comparable everywhere.
    """
    freqs = measured["freqs"]
    spacing = float(freqs[1] - freqs[0])
    return float(np.sum(attenuation_db(measured, key) > threshold_db) * spacing)


def peak_over_background(measured, key, centre_hz, reach_hz):
    from decomb.spectral import to_db

    freqs = measured["freqs"]
    background = local_background_db(freqs, to_db(measured["before"]))
    inside = np.abs(freqs - centre_hz) <= reach_hz
    return float(np.max((to_db(measured[key]) - background)[inside]))


def draw(measured, path: Path) -> None:
    from decomb.spectral import to_db

    freqs = measured["freqs"]
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 7.6), layout="constrained")

    columns = (
        (
            "a sparse comb — `apply`'s case",
            COMB_VIEW_HZ,
            "decomb",
            "notch_comb",
            "probe_decomb",
            "probe_notch_comb",
            f"{len(notch_frequencies())} notches, one per harmonic",
        ),
        (
            "a dense cluster — `notch`'s case",
            CLUSTER_VIEW_HZ,
            "decomb",
            "notch_cluster",
            "probe_decomb",
            "probe_notch_cluster",
            f"one notch, {CLUSTER_SPAN_HZ + 2 * CLUSTER_WANDER_HZ:g} Hz wide",
        ),
    )

    for column, (title, view, decomb_key, notch_key, cost_decomb, cost_notch, note) in enumerate(
        columns
    ):
        inside = (freqs >= view[0]) & (freqs <= view[1])

        top = axes[0][column]
        top.plot(freqs[inside], to_db(measured["before"])[inside], color=BEFORE_COLOR, lw=2.6)
        top.plot(freqs[inside], to_db(measured[decomb_key])[inside], color=AFTER_COLOR, lw=1.1)
        top.plot(
            freqs[inside], to_db(measured[notch_key])[inside], color=NOTCH_COLOR, lw=1.1, ls="--"
        )
        top.set_title(title, loc="left", fontsize=10, color=TITLE_INK, pad=6)
        centre, reach = (FUNDAMENTAL_HZ * 52, 0.1) if column == 0 else (CLUSTER_HZ, CLUSTER_SPAN_HZ)
        top.set_title(
            f"artifact {peak_over_background(measured, 'before', centre, reach):.1f} → "
            f"{peak_over_background(measured, decomb_key, centre, reach):.1f} dB fitted, "
            f"{peak_over_background(measured, notch_key, centre, reach):.1f} notched",
            loc="right",
            fontsize=8.5,
            color=FAINT_INK,
            pad=7,
        )
        top.set_ylabel("median PSD (dB)" if column == 0 else "", fontsize=9, color=TITLE_INK)

        bottom = axes[1][column]
        bottom.fill_between(
            freqs[inside],
            0.0,
            attenuation_db(measured, cost_notch)[inside],
            color=NOTCH_COLOR,
            alpha=0.35,
            lw=0,
        )
        bottom.plot(
            freqs[inside], attenuation_db(measured, cost_notch)[inside], color=NOTCH_COLOR, lw=1.1
        )
        bottom.plot(
            freqs[inside], attenuation_db(measured, cost_decomb)[inside], color=AFTER_COLOR, lw=1.1
        )
        bottom.axhline(1.0, color=FAINT_INK, lw=0.7, ls=":")
        bottom.set_title(
            f"what it cost a signal carrying no artifact   ({note})",
            loc="left",
            fontsize=9,
            color=MUTED_INK,
            pad=6,
        )
        bottom.set_ylabel("attenuation (dB)" if column == 0 else "", fontsize=9, color=TITLE_INK)
        bottom.set_xlabel("frequency (Hz)", fontsize=9, color=TITLE_INK)

        for axis in (top, bottom):
            axis.set_xlim(*view)
            axis.grid(axis="y", color=GRID, lw=0.6)
            axis.set_axisbelow(True)
            axis.tick_params(labelsize=8, colors=MUTED_INK, length=3, width=0.8)
            for side, spine in axis.spines.items():
                spine.set_visible(side in ("left", "bottom"))
                spine.set_color(AXIS_INK)

    axes[0][0].legend(
        handles=[
            plt.Line2D([], [], color=BEFORE_COLOR, lw=2.6, label="before"),
            plt.Line2D([], [], color=AFTER_COLOR, lw=1.4, label="after decomb"),
            plt.Line2D([], [], color=NOTCH_COLOR, lw=1.4, ls="--", label="after MNE notch"),
        ],
        loc="lower left",
        frameon=False,
        fontsize=8.5,
        labelcolor=TITLE_INK,
    )
    decomb_share = band_share(measured, "probe_decomb")
    notch_share = band_share(measured, "probe_notch_comb")
    figure.suptitle(
        f"Clearing the comb costs {decomb_share:.0%} of "
        f"{COST_BAND_HZ[0]:g}–{COST_BAND_HZ[1]:g} Hz by fitting its lines, and "
        f"{notch_share:.0%} by notching them.\nThe cluster is not resolvable into lines, so "
        "the fit leaves it standing and the notch is what clears it.",
        fontsize=11,
        color=TITLE_INK,
        linespacing=1.5,
    )
    figure.savefig(path, dpi=200)
    plt.close(figure)


def report(measured) -> None:
    from decomb.spectral import to_db

    freqs = measured["freqs"]
    print("\n=== measured on the figure's own data " + "=" * 40)
    for label, key in (
        ("decomb", "probe_decomb"),
        ("notch, one per harmonic", "probe_notch_comb"),
        ("notch, one over the cluster", "probe_notch_cluster"),
    ):
        print(
            f"  cost of {label:<28} {cost_hz(measured, key):6.2f} Hz attenuated >1 dB "
            f"({cost_hz(measured, key, 3.0):.2f} Hz >3 dB); "
            f"{band_share(measured, key):.3f} of {COST_BAND_HZ[0]:g}-{COST_BAND_HZ[1]:g} Hz"
        )

    background = local_background_db(freqs, to_db(measured["before"]))
    for label, key in (
        ("before", "before"),
        ("after decomb", "decomb"),
        ("after notch", "notch_cluster"),
    ):
        spectrum_db = to_db(measured[key])
        inside = np.abs(freqs - CLUSTER_HZ) <= CLUSTER_SPAN_HZ
        print(
            f"  cluster peak {label:<14} "
            f"{float(np.max(spectrum_db[inside] - background[inside])):5.1f} dB over background"
        )
    comb_hz = [FUNDAMENTAL_HZ * k for k in COMB_HARMONICS if abs(FUNDAMENTAL_HZ * k - MAINS_HZ) > 1]
    for label, key in (
        ("before", "before"),
        ("after decomb", "decomb"),
        ("after notch", "notch_comb"),
    ):
        spectrum_db = to_db(measured[key])
        peaks = [
            float(np.max((spectrum_db - background)[np.abs(freqs - hz) <= 0.1])) for hz in comb_hz
        ]
        print(f"  comb median  {label:<14} {np.median(peaks):5.1f} dB over background")
    rhythm = np.abs(freqs - RHYTHM_HZ) <= 1.0
    for label, key in (
        ("before", "before"),
        ("after decomb", "decomb"),
        ("after notch", "notch_comb"),
    ):
        spectrum_db = to_db(measured[key])
        print(
            f"  rhythm peak  {label:<14} "
            f"{float(np.max(spectrum_db[rhythm] - background[rhythm])):5.1f} dB over background"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "notch_comparison.png",
        help="where the figure goes (default: docs/notch_comparison.png)",
    )
    parser.add_argument("--keep", type=Path, default=None, help="keep the generated dataset here")
    args = parser.parse_args(argv)

    work = args.keep or Path(tempfile.mkdtemp(prefix="decomb-notch-figure-"))
    work.mkdir(parents=True, exist_ok=True)
    root = work / "bids"
    if root.exists():
        shutil.rmtree(root)

    print(f"building {N_RECORDINGS} x {DURATION_S:g} s recording in {root}")
    write_dataset(root)
    write_config(work / "decomb.yaml", root, work)

    measured = measure(root, work)
    report(measured)
    draw(measured, args.output)
    print(f"\nwrote {args.output}")
    if args.keep is None:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
