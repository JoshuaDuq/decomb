#!/usr/bin/env python3
"""Build the README figure by running the real workflow on a synthetic dataset.

    python docs/make_figure.py

Nothing here is drawn by hand. The script writes three recordings whose contents it
knows exactly, runs ``diagnose``, ``benchmark`` and ``apply`` through their ordinary
entry points, measures the result with the same Welch estimator ``decomb psd`` uses, and
plots that. Every number in the caption is printed as it is measured, so the figure and
the claims made about it can be checked and regenerated after a change.

The dataset carries three things the figure needs to show:

* a comb at ``FUNDAMENTAL_HZ`` over ``COMB_HARMONICS``, which the removal should take;
* a broad rhythm sitting on one of those harmonics, which it must not take;
* a comb harmonic inside the mains band, which belongs to a different stage.

The last two are the point, and the rhythm is the one that matters. The question a reader
brings to a line remover is whether it will eat their gamma band, and the answer is not a
prominence threshold -- it is that a rhythm is whole hertz wide while a line is a tenth of
one. Planting a rhythm *on top of* a harmonic shows the distinction doing its work: the
line inside the rhythm goes, and the rhythm does not.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SFREQ_HZ = 250.0
DURATION_S = 300.0
N_CHANNELS = 4
N_RECORDINGS = 3

FUNDAMENTAL_HZ = 1.2
COMB_HARMONICS = range(24, 80)

COMB_PROMINENCE_DB = 12.0
"""How far the comb stands above the pink background, at the middle of its span.

Stated as a prominence rather than an amplitude, and the amplitude is solved for, because
this is the number that decides whether the figure resembles the case the tool is for. Far
above the background every criterion becomes trivially satisfiable; far below it there is
nothing to remove.
"""

RHYTHM_HZ = 42.0
"""Centre of the planted rhythm. Exactly ``35 * FUNDAMENTAL_HZ``, so a comb harmonic sits
inside it and the two have to be told apart on width alone."""
RHYTHM_SD_HZ = 1.2
"""Spectral width of the rhythm, in Hz. Nearly two orders above ``max_line_width_hz``,
which is the whole basis on which the removal distinguishes the two."""
RHYTHM_PROMINENCE_DB = 9.0

MAINS_HZ = 60.0
"""Comb harmonic 50 exactly, which is why it falls inside the default mains band."""

BACKGROUND_V = 3e-8
BAND_HZ = (1.0, 100.0)

PANELS = (
    ("delta", 1.0, 3.9),
    ("theta", 4.0, 7.9),
    ("alpha", 8.0, 12.9),
    ("beta", 13.0, 30.0),
    ("gamma", 30.1, 80.0),
    ("", 80.0, 99.0),
)
"""One panel per analysed band, plus the unnamed span the removal reaches past them.

Drawing 1-100 Hz on a single axis puts several frequency bins in every pixel, so a comb
line 1.2 Hz from its neighbour is sub-pixel: such a figure can show that the lines went
and cannot show whether they went surgically. A panel per band is about one bin per pixel,
which is what it takes to see the shape of an individual notch -- and it puts the result
in the units the analyses are actually reported in.
"""

BEFORE_COLOR = "#EF8F6C"
AFTER_COLOR = "#2A78D6"
"""Two hues, checked for separation under protanopia and deuteranopia rather than chosen.

Weight carries the same distinction a second time: the before trace is thick and sits
underneath, the after trace is thin and sits on top, so the pair reads as one line emerging
from another even where colour does not survive -- printing, projection, a colourblind
reader.
"""

TITLE_INK = "#111827"
MUTED_INK = "#6B7280"
FAINT_INK = "#9CA3AF"
AXIS_INK = "#D1D5DB"
GRID = "#E9EBEF"

CALLOUT_INK = "#334155"
CALLOUT_FILL = "#F8FAFC"
CALLOUT_EDGE = "#94A3B8"
"""Slate, not red. These two peaks are the removal behaving correctly, and a warning colour
would say the opposite of what the annotation says."""

LABEL_STRIP = 0.55
"""Headroom added above the data, as a fraction of the panel's own span, on panels that
carry a callout. It buys a clear strip across the top of the panel that no trace enters."""

LABEL_Y = 0.81
"""Where the callouts sit, in axes fractions. Inside the strip ``LABEL_STRIP`` opens: with
that headroom the data reaches 0.66, and a two-line box centred here spans 0.70 to 0.92."""


def _background(rng, times):
    """Pink background: white noise shaped by 1/sqrt(f) in the frequency domain."""
    spectrum = np.fft.rfft(rng.normal(size=(N_CHANNELS, times.size)), axis=-1)
    freqs = np.fft.rfftfreq(times.size, 1.0 / SFREQ_HZ)
    freqs[0] = freqs[1]
    return np.fft.irfft(spectrum / np.sqrt(freqs), n=times.size, axis=-1) * BACKGROUND_V


def _rhythm(rng, times, scale):
    """Band-limited noise with a Gaussian spectral envelope: a rhythm, not a line.

    A rhythm is a resonance, so it has to be generated as one -- noise shaped in the
    frequency domain -- rather than as a sinusoid. That is exactly what makes it whole
    hertz wide, and width is the only thing separating it from the comb line inside it.
    """
    noise = rng.normal(size=(N_CHANNELS, times.size))
    freqs = np.fft.rfftfreq(times.size, 1.0 / SFREQ_HZ)
    envelope = np.exp(-0.5 * ((freqs - RHYTHM_HZ) / RHYTHM_SD_HZ) ** 2)
    return np.fft.irfft(np.fft.rfft(noise, axis=-1) * envelope, n=times.size, axis=-1) * scale


def _scale_for(prominence_db: float, frequency_hz: float, times, make_signal) -> float:
    """Solve for the scale at which ``make_signal`` stands ``prominence_db`` above background.

    Measured rather than derived from window constants: one background-only spectrum gives
    the local level, and a signal's spectral peak scales as the square of its amplitude, so
    a single reference measurement fixes the scale. Stating the figure's signals in decibels
    over background is what makes them comparable to a real recording's.
    """
    import mne

    from decomb import psd as psd_stage

    reference_v = 1e-8
    rng = np.random.default_rng(9_999)
    background = _background(rng, times)
    with_tone = background + make_signal(np.random.default_rng(4_242), times, reference_v)
    settings = psd_stage.PsdSettings(band_hz=BAND_HZ)

    def spectrum(data):
        raw = mne.io.RawArray(
            data,
            mne.create_info([f"EEG{i:02d}" for i in range(N_CHANNELS)], SFREQ_HZ, "eeg"),
            verbose="ERROR",
        )
        return psd_stage.channel_median_psd(raw, settings)

    freqs, quiet = spectrum(background)
    _, loud = spectrum(with_tone)
    bin_index = int(np.argmin(np.abs(freqs - frequency_hz)))
    nearby = (np.abs(freqs - frequency_hz) > 5.0) & (np.abs(freqs - frequency_hz) < 12.0)
    level = float(np.median(quiet[nearby]))
    excess_at_reference = float(loud[bin_index] - quiet[bin_index])
    if excess_at_reference <= 0:
        raise RuntimeError("The reference tone did not register above the background.")
    wanted = level * (10.0 ** (prominence_db / 10.0) - 1.0)
    return reference_v * float(np.sqrt(wanted / excess_at_reference))


def write_dataset(root: Path) -> tuple[float, float]:
    """Three BrainVision recordings in BIDS layout, with no events of any kind."""
    import mne
    from mne_bids import BIDSPath, write_raw_bids

    mne.set_log_level("ERROR")
    times = np.arange(int(SFREQ_HZ * DURATION_S)) / SFREQ_HZ

    mid_harmonic = (COMB_HARMONICS.start + COMB_HARMONICS.stop) // 2

    def tone(rng, sample_times, scale):
        return scale * np.sin(2 * np.pi * FUNDAMENTAL_HZ * mid_harmonic * sample_times)

    comb_v = _scale_for(COMB_PROMINENCE_DB, FUNDAMENTAL_HZ * mid_harmonic, times, tone)
    rhythm_v = _scale_for(RHYTHM_PROMINENCE_DB, RHYTHM_HZ, times, _rhythm)
    print(
        f"  comb amplitude  {comb_v:.3e} V -> {COMB_PROMINENCE_DB:g} dB at "
        f"{FUNDAMENTAL_HZ * mid_harmonic:g} Hz"
    )
    print(
        f"  rhythm scale    {rhythm_v:.3e}   -> {RHYTHM_PROMINENCE_DB:g} dB at "
        f"{RHYTHM_HZ:g} Hz, sd {RHYTHM_SD_HZ:g} Hz"
    )

    for index in range(N_RECORDINGS):
        rng = np.random.default_rng(1000 + index)
        data = _background(rng, times)

        for harmonic in COMB_HARMONICS:
            # Each harmonic gets its own phase, so they do not sum into a spike train.
            data += comb_v * np.sin(2 * np.pi * FUNDAMENTAL_HZ * harmonic * times + harmonic)
        data += _rhythm(np.random.default_rng(5000 + index), times, rhythm_v)

        info = mne.create_info([f"EEG{i:02d}" for i in range(N_CHANNELS)], SFREQ_HZ, "eeg")
        raw = mne.io.RawArray(data, info, verbose="ERROR")
        write_raw_bids(
            raw,
            BIDSPath(
                subject=f"{index:04d}",
                task="rest",
                datatype="eeg",
                root=root,
                extension=".vhdr",
            ),
            format="BrainVision",
            allow_preload=True,
            verbose="ERROR",
        )


def write_config(path: Path, root: Path, work: Path) -> None:
    path.write_text(
        textwrap.dedent(f"""\
            paths:
              bids_root: "{root}"
              output_root: "{work}/cleaned"
              diagnosis_dir: "{work}/diagnosis"
              removal_dir: "{work}/removal"
            dataset:
              task: "rest"
            removal:
              harmonic_range: [{COMB_HARMONICS.start}, {COMB_HARMONICS.stop - 1}]
              removal_harmonic_range: [{COMB_HARMONICS.start}, {COMB_HARMONICS.stop - 1}]
              high_hz: 99.0
        """),
        encoding="utf-8",
    )


def run_stage(stage: str, config: Path) -> None:
    import subprocess

    print(f"\n--- decomb {stage} " + "-" * (58 - len(stage)))
    result = subprocess.run(
        [sys.executable, "-m", "decomb.cli", stage, "--config", str(config)],
        capture_output=True,
        text=True,
    )
    print("\n".join((result.stdout or result.stderr).strip().splitlines()[-8:]))
    if result.returncode != 0:
        raise SystemExit(f"`decomb {stage}` failed:\n{result.stderr[-3000:]}")


def measure(source_root: Path, cleaned_root: Path):
    """Channel-median Welch spectra of both arms, on one grid, via the psd stage's own code."""
    from decomb import psd, remove

    settings = psd.PsdSettings(band_hz=BAND_HZ)
    before, after = [], []
    freqs = None
    for vhdr in remove.discover_runs(source_root, subjects=None, task="rest"):
        run_freqs, source = psd.channel_median_psd(remove.read_bids_raw(vhdr), settings)
        _, cleaned = psd.channel_median_psd(
            remove.read_bids_raw(cleaned_root / vhdr.relative_to(source_root)), settings
        )
        freqs = run_freqs if freqs is None else freqs
        before.append(source)
        after.append(cleaned)
    return freqs, np.median(before, axis=0), np.median(after, axis=0)


def local_background_db(freqs, spectrum_db, half_width_hz=2.0):
    """Running median with a window wide enough to hold several comb spacings."""
    half = max(int(round(half_width_hz / float(freqs[1] - freqs[0]))), 1)
    padded = np.pad(spectrum_db, half, mode="edge")
    window = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    return np.median(window, axis=1)


def prominence_at(freqs, spectrum_db, background_db, frequency_hz, reach_hz=0.3):
    inside = np.flatnonzero(np.abs(freqs - frequency_hz) <= reach_hz)
    peak = inside[int(np.argmax(spectrum_db[inside]))]
    return float(freqs[peak]), float(spectrum_db[peak] - background_db[peak])


def draw(freqs, before, after, path: Path) -> dict[str, tuple[float, float]]:
    from decomb.spectral import to_db

    before_db, after_db = to_db(before), to_db(after)
    background_db = local_background_db(freqs, after_db)

    survivors = {
        "rhythm": prominence_at(freqs, after_db, background_db, RHYTHM_HZ, reach_hz=1.0),
        "mains": prominence_at(freqs, after_db, background_db, MAINS_HZ),
    }

    figure, axes = plt.subplots(3, 2, figsize=(13.5, 8.8), layout="constrained")
    flat = axes.ravel()

    for axis, (name, low_hz, high_hz) in zip(flat, PANELS):
        inside = (freqs >= low_hz) & (freqs <= high_hz)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color=GRID, lw=0.6)
        for side, spine in axis.spines.items():
            spine.set_visible(side in ("left", "bottom"))
            spine.set_color(AXIS_INK)
        axis.plot(
            freqs[inside], before_db[inside], color=BEFORE_COLOR, lw=2.6, solid_capstyle="round"
        )
        axis.plot(freqs[inside], after_db[inside], color=AFTER_COLOR, lw=0.9)
        axis.set_xlim(low_hz, high_hz)
        axis.tick_params(labelsize=8, colors=MUTED_INK, length=3, width=0.8)

        # Where a survivor is labelled the panel gets a clear strip above the data, wide
        # enough that the box sits entirely inside it: nothing is drawn over the trace.
        labelled = [key for key, (hz, _) in survivors.items() if low_hz <= hz <= high_hz]
        low_db = float(np.nanmin(after_db[inside]))
        high_db = float(np.nanmax(before_db[inside]))
        span = high_db - low_db
        headroom = LABEL_STRIP if labelled else 0.12
        axis.set_ylim(low_db - 0.05 * span, high_db + headroom * span)

        # Band name and change count ride above the frame rather than inside it, so no
        # panel has text competing with its own trace for the same pixels.
        axis.set_title(
            f"{name}   {low_hz:g}–{high_hz:g} Hz" if name else f"{low_hz:g}–{high_hz:g} Hz",
            loc="left",
            fontsize=9.5,
            color=TITLE_INK,
            pad=6,
        )
        touched = np.abs(before_db[inside] - after_db[inside]) > 1.0
        axis.set_title(
            "unchanged" if not touched.any() else f"{int(touched.sum())} bins moved >1 dB",
            loc="right",
            fontsize=8,
            color=FAINT_INK,
            pad=7,
        )

        # The annotations are the point of the figure: a tool that removes everything it
        # can find is easy to write and impossible to trust. Both survivors sit in gamma.
        # The text is placed in axes coordinates and the arrow tip in data coordinates, so
        # a label can never escape its panel and push the layout around; the two boxes are
        # given halves of the strip, which is what keeps them off each other.
        for key, text, x_frac in (
            (
                "rhythm",
                f"a rhythm {2.355 * RHYTHM_SD_HZ:.1f} Hz wide on harmonic "
                f"{round(RHYTHM_HZ / FUNDAMENTAL_HZ)}\n"
                "the line inside it went, the rhythm did not",
                0.24,
            ),
            (
                "mains",
                f"{survivors['mains'][0]:.2f} Hz, +{survivors['mains'][1]:.1f} dB, "
                "inside the mains band\nwhich `notch` takes whole",
                0.70,
            ),
        ):
            if key not in labelled:
                continue
            peak_hz, _ = survivors[key]
            peak_db = float(after_db[int(np.argmin(np.abs(freqs - peak_hz)))])
            axis.annotate(
                text,
                xy=(peak_hz, peak_db),
                xycoords="data",
                xytext=(x_frac, LABEL_Y),
                textcoords="axes fraction",
                fontsize=7.5,
                color=CALLOUT_INK,
                ha="center",
                va="center",
                linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.45", fc=CALLOUT_FILL, ec=CALLOUT_EDGE, lw=0.7),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=CALLOUT_EDGE,
                    lw=0.9,
                    shrinkB=5,
                    patchA=None,
                ),
            )

    # Inside the first panel rather than beside the title: an outside legend competes with
    # the suptitle for the same strip, and the delta panel's lower left is empty in every
    # spectrum this figure can produce -- the trace falls left to right.
    flat[0].legend(
        handles=[
            plt.Line2D([], [], color=BEFORE_COLOR, lw=2.6, label="before removal"),
            plt.Line2D([], [], color=AFTER_COLOR, lw=1.4, label="after removal"),
        ],
        loc="lower left",
        frameon=False,
        fontsize=8.5,
        labelcolor=TITLE_INK,
        handlelength=2.2,
    )
    figure.supxlabel("frequency (Hz)", fontsize=10, color=TITLE_INK)
    figure.supylabel("median PSD (dB re 1 V²/Hz)", fontsize=10, color=TITLE_INK)
    figure.suptitle(
        f"A {FUNDAMENTAL_HZ} Hz comb removed from three synthetic recordings, "
        "one panel per analysed band. Two peaks are left standing on purpose.",
        fontsize=11.5,
        color=TITLE_INK,
    )

    change = after_db - before_db
    figure.savefig(path, dpi=200)
    plt.close(figure)

    # What the caption may claim, measured rather than asserted.
    comb_hz = np.array([FUNDAMENTAL_HZ * k for k in COMB_HARMONICS])
    on_a_line = np.zeros(freqs.size, dtype=bool)
    for centre in (*comb_hz, RHYTHM_HZ):
        on_a_line |= np.abs(freqs - centre) <= 0.3
    on_a_line |= np.abs(freqs - RHYTHM_HZ) <= 4.0 * RHYTHM_SD_HZ
    drawn = (freqs >= PANELS[0][1]) & (freqs <= PANELS[-1][2])
    off_line = drawn & ~on_a_line
    print("\n=== measured on the figure's own data " + "=" * 40)
    for key, (peak_hz, prominence) in survivors.items():
        print(f"  survivor ({key:8s}) {peak_hz:6.2f} Hz  +{prominence:.1f} dB over background")
    removed = [
        prominence_at(freqs, to_db(before), local_background_db(freqs, to_db(before)), f)[1]
        for f in comb_hz
        if abs(f - MAINS_HZ) > 0.5
    ]
    left = [
        prominence_at(freqs, after_db, background_db, f)[1]
        for f in comb_hz
        if abs(f - MAINS_HZ) > 0.5
    ]
    print(f"  comb harmonics targeted: {len(removed)}")
    print(
        f"  their prominence before: {np.median(removed):.1f} dB median, {max(removed):.1f} worst"
    )
    print(f"  their prominence after : {np.median(left):.1f} dB median, {max(left):.1f} worst")
    print(f"  off-line change        : max |{np.max(np.abs(change[off_line])):.3f}| dB")
    return survivors


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "psd_before_after.png",
        help="where the figure goes (default: docs/psd_before_after.png)",
    )
    parser.add_argument(
        "--keep",
        type=Path,
        default=None,
        help="keep the generated dataset and stage outputs here instead of a temp directory",
    )
    args = parser.parse_args(argv)

    work = args.keep or Path(tempfile.mkdtemp(prefix="decomb-figure-"))
    work.mkdir(parents=True, exist_ok=True)
    root = work / "bids"
    if root.exists():
        shutil.rmtree(root)
    config = work / "decomb.yaml"

    print(f"building {N_RECORDINGS} x {DURATION_S:g} s recordings in {root}")
    write_dataset(root)
    write_config(config, root, work)

    for stage in ("diagnose", "benchmark", "apply"):
        run_stage(stage, config)

    freqs, before, after = measure(root, work / "cleaned")
    draw(freqs, before, after, args.output)
    print(f"\nwrote {args.output}")
    if args.keep is None:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
