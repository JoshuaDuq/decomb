#!/usr/bin/env python3
"""The cost comparison again, on a real recording instead of a generated one.

    python docs/make_notch_figure_real.py --config trials/sub0011.yaml

`docs/make_notch_figure.py` measures both stages on synthetic data, where the truth is
known exactly and anyone with the repository can regenerate it. This runs the same two
transforms over a real EEG-fMRI recording, where the truth is not known and the data is not
public. It is evidence that the synthetic result is not an artifact of the simulation, and
it is not a substitute for it.

Only one of the two structures survives the move. The comb does: a real 1.2 Hz comb over
harmonics 22-77 is exactly the sparse case. The dense cluster does not -- surveyed on
sub-0011, the densest 1.5 Hz span of the channel-median spectrum holds three peaks, all
19 mHz wide and individually resolvable, and the densest span on any single channel is seven
peaks at 10 Hz on POz, which is alpha. A cluster that would justify `notch` is simply not
present in this recording, and the figure says so rather than promoting three resolvable
lines into one.

So the second column shows the whole analysed band instead: the same measurement, over
every frequency the removal reaches rather than the six harmonics the first column magnifies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_figure import (  # noqa: E402
    AFTER_COLOR,
    AXIS_INK,
    BEFORE_COLOR,
    FAINT_INK,
    GRID,
    MUTED_INK,
    TITLE_INK,
)
from make_notch_figure import (  # noqa: E402
    NOTCH_COLOR,
    NOTCH_FILTER_LENGTH,
    attenuation_db,
    cost_hz,
)

PROBE_V = 5e-7
"""Amplitude of the broadband probe. Any level does: attenuation is a ratio."""


def collect_targets(plan, tolerance_hz: float) -> np.ndarray:
    """Every distinct line the plan actually authorized, across its windows.

    The notch arm is given exactly these, so both transforms aim at the same frequencies and
    the comparison is of how each one takes them rather than of what each one decided to
    take.
    """
    values = sorted(float(target) for window in plan.windows for target in window.targets_hz)
    distinct: list[float] = []
    for value in values:
        if not distinct or value - distinct[-1] > tolerance_hz:
            distinct.append(value)
    return np.asarray(distinct, dtype=float)


def merge_for_notch(targets: np.ndarray, separation_hz: float):
    """Group targets a notch filter cannot keep apart, and give each group one stop-band.

    On real data this is not a refinement, it is the only way the arm runs at all. Handed
    sub-0011's 57 authorized targets, `notch_filter` raises "Stop bands are not sufficiently
    separated": three of the sources sit within 0.19 Hz of each other, far inside the 1 Hz
    transition MNE puts on either side of every notch. A practitioner meeting that error has
    to widen and merge, so that is what this does -- and the widening is itself part of what
    the notch costs, which is why the figure reports the result rather than hiding the merge.
    """
    groups: list[list[float]] = []
    for target in sorted(float(value) for value in targets):
        if groups and target - groups[-1][-1] <= separation_hz:
            groups[-1].append(target)
        else:
            groups.append([target])
    centres = np.asarray([0.5 * (group[0] + group[-1]) for group in groups], dtype=float)
    widths = np.asarray(
        [max(group[-1] - group[0], centre / 200.0) for group, centre in zip(groups, centres)],
        dtype=float,
    )
    return centres, widths


def stage_as_bids(vhdr: Path, root: Path, subject: str, task: str) -> Path:
    """Copy a bare BrainVision recording into a BIDS layout and return its path there.

    Not every recording worth measuring is already in a BIDS tree -- the baseline
    acquisitions sit in the source directory as plain BrainVision triples. Staging them is
    better than reading them directly: every stage in this pipeline takes channel types from
    BIDS metadata rather than from the BrainVision header, and a figure that quietly used a
    different rule for picking EEG channels would not be measuring the same thing.
    """
    import mne
    from mne_bids import BIDSPath, write_raw_bids

    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    path = BIDSPath(subject=subject, task=task, datatype="eeg", root=root, extension=".vhdr")
    write_raw_bids(
        raw, path, format="BrainVision", allow_preload=True, overwrite=True, verbose="ERROR"
    )
    return Path(path.fpath)


def buildable_stop_bands(targets: np.ndarray, sampling_frequency_hz: float):
    """The least merging that lets `notch_filter` build at all, found by asking it.

    A synthetic comb is evenly spaced and MNE builds 55 notches from it without complaint.
    A real recording is not: sub-0011's 57 targets include 16 gaps under 1.19 Hz and one of
    0.439 Hz, and at that spacing the stop bands overlap at every transition bandwidth MNE
    will accept -- below 0.05 Hz it asks for a filter longer than the one it was given.

    So the merge threshold is not chosen, it is discovered: widen until the filter exists,
    and report where that landed. Trying it on a few seconds of zeros costs nothing and
    keeps the search away from the recording.
    """
    from mne.filter import notch_filter

    probe = np.zeros((1, int(round(60.0 * sampling_frequency_hz))))
    separation = 0.0
    while separation <= 5.0:
        centres, widths = merge_for_notch(targets, separation)
        try:
            notch_filter(
                probe.copy(),
                sampling_frequency_hz,
                centres,
                notch_widths=widths,
                filter_length=NOTCH_FILTER_LENGTH,
                verbose="ERROR",
            )
        except ValueError:
            separation += 0.05
            continue
        return centres, widths, separation
    raise SystemExit("No merge of these targets produces a filter MNE will build.")


def measure(vhdr: Path, settings, cache: Path | None = None):
    """Fit the plan, then put the recording and a bare probe through both transforms.

    The fit and the two cleanings are the expensive part -- an hour on a 63-channel, 493 s
    recording -- so their spectra are cached. The notch arm is seconds and is never cached,
    because it is the arm most likely to need adjusting.
    """
    import mne
    from mne.filter import notch_filter

    from decomb import psd as psd_stage
    from decomb import remove

    raw = remove.read_bids_raw(vhdr)
    sampling_frequency_hz = float(raw.info["sfreq"])
    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    info = mne.pick_info(raw.info, picks, copy=True)
    original = raw.get_data(picks=picks)
    probe = np.random.default_rng(77).normal(size=original.shape) * PROBE_V

    def spectrum(values):
        one = mne.io.RawArray(np.asarray(values, dtype=float), info, verbose="ERROR")
        return psd_stage.channel_median_psd(one, psd_stage.PsdSettings(band_hz=(1.0, 100.0)))

    if cache is not None and cache.exists():
        held = np.load(cache)
        freqs, targets = held["freqs"], held["targets"]
        fitted = {key: held[key] for key in ("before", "decomb", "probe", "probe_decomb")}
        print(f"  reusing the fitted arm from {cache}", flush=True)
    else:
        print(f"\n--- fitting the removal plan for {vhdr.stem} " + "-" * 20, flush=True)
        plan = remove.build_run_plans([vhdr], settings)[vhdr.stem]
        targets = collect_targets(plan, settings.line_claim_hz)
        print(
            f"  {targets.size} distinct authorized targets, "
            f"{targets.min():.2f}-{targets.max():.2f} Hz",
            flush=True,
        )
        print("  cleaning the recording, then a bare probe", flush=True)
        decombed = remove.clean_continuous_raw(raw.copy(), plan, settings).get_data(picks=picks)
        probe_raw = mne.io.RawArray(probe, info, verbose="ERROR")
        probe_decombed = remove.clean_continuous_raw(probe_raw.copy(), plan, settings).get_data()
        freqs, before = spectrum(original)
        fitted = {
            "before": before,
            "decomb": spectrum(decombed)[1],
            "probe": spectrum(probe)[1],
            "probe_decomb": spectrum(probe_decombed)[1],
        }
        if cache is not None:
            np.savez(cache, freqs=freqs, targets=targets, **fitted)
            print(f"  cached the fitted arm to {cache}", flush=True)

    centres, widths, separation = buildable_stop_bands(targets, sampling_frequency_hz)
    merged = (
        f", merged from targets closer than {separation:.2f} Hz"
        if centres.size < targets.size
        else ""
    )
    print(f"  notch arm: {centres.size} stop-bands for {targets.size} targets{merged}", flush=True)
    notched = notch_filter(
        original.copy(),
        sampling_frequency_hz,
        centres,
        notch_widths=widths,
        filter_length=NOTCH_FILTER_LENGTH,
        verbose="ERROR",
    )
    probe_notched = notch_filter(
        probe.copy(),
        sampling_frequency_hz,
        centres,
        notch_widths=widths,
        filter_length=NOTCH_FILTER_LENGTH,
        verbose="ERROR",
    )

    return {
        "freqs": freqs,
        "targets": targets,
        "stop_bands": centres,
        **fitted,
        "notch_comb": spectrum(notched)[1],
        "probe_notch_comb": spectrum(probe_notched)[1],
    }


def draw(measured, path: Path, label: str, band_hz) -> None:
    from decomb.spectral import to_db

    freqs = measured["freqs"]
    targets = measured["targets"]
    middle = float(np.median(targets))
    zoom = (middle - 3.6, middle + 3.9)

    # Detail above, aggregate below, each the full width of the page. Side by side, the
    # whole-band panel holds 57 lines in the space of a column and reads as noise.
    figure, axes = plt.subplots(
        2, 1, figsize=(13.0, 8.0), layout="constrained", height_ratios=(1.1, 1.0)
    )

    top, bottom = axes
    inside = (freqs >= zoom[0]) & (freqs <= zoom[1])
    top.plot(freqs[inside], to_db(measured["before"])[inside], color=BEFORE_COLOR, lw=2.4)
    top.plot(freqs[inside], to_db(measured["decomb"])[inside], color=AFTER_COLOR, lw=1.1)
    top.plot(
        freqs[inside], to_db(measured["notch_comb"])[inside], color=NOTCH_COLOR, lw=1.1, ls="--"
    )
    top.set_title(
        f"six harmonics, magnified   ({zoom[0]:.0f}–{zoom[1]:.0f} Hz)",
        loc="left",
        fontsize=10,
        color=TITLE_INK,
        pad=6,
    )
    top.set_title(
        f"fit {cost_hz(measured, 'probe_decomb', view=zoom):.1f} · notch "
        f"{cost_hz(measured, 'probe_notch_comb', view=zoom):.1f} Hz of the "
        f"{zoom[1] - zoom[0]:.1f} Hz shown",
        loc="right",
        fontsize=8.5,
        color=FAINT_INK,
        pad=7,
    )
    top.set_ylabel("median PSD (dB)", fontsize=9, color=TITLE_INK)
    top.set_xlabel("frequency (Hz)", fontsize=9, color=TITLE_INK)
    top.set_xlim(*zoom)

    band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    bottom.fill_between(
        freqs[band],
        0.0,
        attenuation_db(measured, "probe_notch_comb")[band],
        color=NOTCH_COLOR,
        alpha=0.30,
        lw=0,
    )
    bottom.plot(
        freqs[band], attenuation_db(measured, "probe_notch_comb")[band], color=NOTCH_COLOR, lw=0.8
    )
    bottom.plot(
        freqs[band], attenuation_db(measured, "probe_decomb")[band], color=AFTER_COLOR, lw=0.8
    )
    bottom.axhline(1.0, color=FAINT_INK, lw=0.7, ls=":")
    bottom.set_title(
        "cost to a signal carrying no artifact, over every frequency the removal reaches",
        loc="left",
        fontsize=9.5,
        color=MUTED_INK,
        pad=6,
    )
    bottom.set_title(
        f"fit {cost_hz(measured, 'probe_decomb', view=band_hz):.1f} · notch "
        f"{cost_hz(measured, 'probe_notch_comb', view=band_hz):.1f} Hz of "
        f"{band_hz[1] - band_hz[0]:.0f} Hz",
        loc="right",
        fontsize=8.5,
        color=FAINT_INK,
        pad=7,
    )
    bottom.set_ylabel("attenuation (dB)", fontsize=9, color=TITLE_INK)
    bottom.set_xlabel("frequency (Hz)", fontsize=9, color=TITLE_INK)
    bottom.set_xlim(*band_hz)

    for axis in (top, bottom):
        axis.grid(axis="y", color=GRID, lw=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=8, colors=MUTED_INK, length=3, width=0.8)
        for side, spine in axis.spines.items():
            spine.set_visible(side in ("left", "bottom"))
            spine.set_color(AXIS_INK)

    figure.legend(
        handles=[
            plt.Line2D([], [], color=BEFORE_COLOR, lw=2.6, label="before"),
            plt.Line2D([], [], color=AFTER_COLOR, lw=1.4, label="after decomb"),
            plt.Line2D([], [], color=NOTCH_COLOR, lw=1.4, ls="--", label="after MNE notch"),
        ],
        loc="outside lower center",
        ncols=3,
        frameon=False,
        fontsize=9,
        labelcolor=TITLE_INK,
        handlelength=2.6,
    )
    fit = cost_hz(measured, "probe_decomb", view=band_hz)
    notch = cost_hz(measured, "probe_notch_comb", view=band_hz)
    figure.suptitle(
        f"{label}: the same {targets.size} lines taken two ways. Fitting them costs "
        f"{fit:.1f} Hz of {band_hz[1] - band_hz[0]:.0f} Hz,\nnotching them costs "
        f"{notch:.1f} Hz. This recording carries no cluster, so only the sparse case appears.",
        fontsize=11,
        color=TITLE_INK,
        linespacing=1.5,
    )
    figure.savefig(path, dpi=200)
    plt.close(figure)


def report(measured, band_hz) -> None:
    from decomb.spectral import to_db

    freqs = measured["freqs"]
    targets = measured["targets"]
    print("\n=== measured on the recording " + "=" * 46)
    for label, key in (("decomb", "probe_decomb"), ("MNE notch", "probe_notch_comb")):
        print(
            f"  cost of {label:<12} {cost_hz(measured, key, view=band_hz):6.2f} Hz of "
            f"{band_hz[1] - band_hz[0]:.0f} Hz attenuated >1 dB "
            f"({cost_hz(measured, key, 3.0, view=band_hz):.2f} Hz >3 dB)"
        )
    half = int(round(2.0 / float(freqs[1] - freqs[0])))
    padded = np.pad(to_db(measured["before"]), half, mode="edge")
    background = np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1), axis=-1)
    for label, key in (
        ("before", "before"),
        ("after decomb", "decomb"),
        ("after notch", "notch_comb"),
    ):
        peaks = [
            float(np.max((to_db(measured[key]) - background)[np.abs(freqs - hz) <= 0.1]))
            for hz in targets
        ]
        print(
            f"  target median {label:<14} {np.median(peaks):6.1f} dB over background "
            f"(worst {max(peaks):.1f})"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="a decomb config file")
    parser.add_argument(
        "--subject",
        default=None,
        help="restrict discovery to this subject; the subject label to stage --vhdr under",
    )
    parser.add_argument(
        "--vhdr",
        type=Path,
        default=None,
        help="a bare BrainVision recording to stage into BIDS and measure instead",
    )
    parser.add_argument("--stage-task", default="baseline", help="task label to stage --vhdr under")
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=None,
        help="where --vhdr is staged (default: a directory beside the recording)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="reuse (or write) the fitted arm here; the fit is the hour-long part",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "notch_comparison_real.png",
    )
    args = parser.parse_args(argv)

    from decomb import remove
    from decomb.config import load_config

    config = load_config(args.config)
    settings = remove.RemovalSettings.from_config(config.data)
    root = Path(config.data["paths"]["bids_root"])
    if args.vhdr is not None:
        if not args.subject:
            raise SystemExit("--vhdr needs --subject: a staged recording has to be labelled")
        stage_root = args.stage_root or args.vhdr.parent / "_staged_bids"
        print(f"staging {args.vhdr.name} into {stage_root}", flush=True)
        runs = [stage_as_bids(args.vhdr, stage_root, args.subject, args.stage_task)]
    else:
        runs = list(
            remove.discover_runs(
                root,
                subjects=[args.subject] if args.subject else None,
                task=settings.task,
            )
        )
    if not runs:
        raise SystemExit(f"no runs found under {root}")
    measured = measure(runs[0], settings, cache=args.cache)
    band_hz = (settings.low_hz, settings.high_hz)
    report(measured, band_hz)
    draw(measured, args.output, runs[0].stem, band_hz)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
