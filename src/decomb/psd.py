"""Before-and-after power spectra of the removal, computed with MNE.

    decomb psd

Answers the question a reader asks first: what did this actually take out? Three figures
from Welch spectra computed by ``Raw.compute_psd`` on the source and harmonic-notch
derivative: an overview, readable frequency tiles, and one panel per recording.

The tiled figure exists because the overview cannot answer the second question. Drawing
the whole band on one axis puts several frequency bins in every pixel, so a comb line is
sub-pixel from its neighbour: the overview shows that the lines went, and cannot show
whether they went surgically or took their surroundings with them.

It runs after ``apply`` and compares the source against the single delivered derivative.

Unlike the stages that transform or certify, this one accepts ``--subjects``: a figure of
part of a cohort is a smaller figure, not a false claim about the whole one.

The comparison is only worth reading if both sides were measured identically, so the
spectra are computed with one set of parameters, on the same channels, over the same
samples, and the stage refuses when the recordings do not correspond.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from decomb import recordings  # noqa: E402
from decomb.spectral import to_db  # noqa: E402


@dataclass(frozen=True)
class PsdSettings:
    """PSD geometry derived from the correction's stationarity window."""

    window_s: float
    band_hz: tuple[float, float]
    """Welch segment length, in seconds. Sets the resolution to ``1 / window_s``.

    Match it to the removal's estimation window so the figure resolves what the fit
    resolved. Resolving the comb spacing needs far less; resolving a close line pair needs
    all of it.
    """
    def __post_init__(self) -> None:
        if not np.isfinite(self.window_s) or self.window_s <= 0.0:
            raise ValueError("window_s must be finite and positive.")
        low_hz, high_hz = self.band_hz
        if not 0.0 <= low_hz < high_hz <= 100.0:
            raise ValueError("band_hz must increase inside [0, 100] Hz.")

    @property
    def overlap(self) -> float:
        return 0.5

    @property
    def panel_span_hz(self) -> float:
        return 10.0

    @classmethod
    def from_config(cls, config) -> PsdSettings:
        from decomb.notch import HarmonicNotchSettings

        correction = HarmonicNotchSettings.from_config(config)
        return cls(
            window_s=correction.estimation_window_s,
            band_hz=correction.frequency_range_hz,
        )


def channel_median_psd(raw, settings: PsdSettings) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD of the EEG channels, and their median, via ``Raw.compute_psd``.

    EEG only: ECG and EOG carry different units and amplitudes, and averaging them in would
    put a millivolt trace on the same axis as a microvolt one.

    The median rather than the mean across channels, because one bad channel should not set
    the level of a cohort figure.
    """
    sampling_frequency_hz = float(raw.info["sfreq"])
    n_fft = int(round(settings.window_s * sampling_frequency_hz))
    if n_fft > raw.n_times:
        raise ValueError(
            f"The recording holds {raw.n_times} samples, fewer than the {n_fft} of one "
            f"{settings.window_s:g} s Welch segment. Lower "
            "`removal.estimation_window_s`."
        )
    spectrum = raw.compute_psd(
        method="welch",
        picks="eeg",
        fmin=settings.band_hz[0],
        fmax=min(settings.band_hz[1], np.nextafter(sampling_frequency_hz / 2.0, 0.0)),
        n_fft=n_fft,
        n_overlap=int(round(n_fft * settings.overlap)),
        n_per_seg=n_fft,
        verbose="ERROR",
    )
    psd, freqs = spectrum.get_data(return_freqs=True)
    return freqs, np.median(psd, axis=0)


def compare_recording(
    source_vhdr: Path,
    derivative_vhdrs: Sequence[tuple[str, Path]],
    settings: PsdSettings,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    """One recording's spectrum before and after each derivative, on one frequency grid.

    Every arm is read through the same BIDS reader and measured with the same parameters,
    and the geometry is checked rather than assumed: a derivative that has drifted in
    channel set, length or sampling rate is not a comparison, it is two different
    recordings on one axis.
    """
    source = recordings.read_bids_raw(source_vhdr)
    duration_s = source.n_times / float(source.info["sfreq"])
    freqs, source_psd = channel_median_psd(source, settings)
    spectra = {"source": source_psd}
    for label, path in derivative_vhdrs:
        derivative = recordings.read_bids_raw(path)
        if derivative.ch_names != source.ch_names:
            raise ValueError(f"{label} {path.name}: channel set differs from the source.")
        if derivative.n_times != source.n_times:
            raise ValueError(f"{label} {path.name}: length differs from the source.")
        if not np.isclose(derivative.info["sfreq"], source.info["sfreq"]):
            raise ValueError(f"{label} {path.name}: sampling rate differs from the source.")
        derivative_freqs, psd = channel_median_psd(derivative, settings)
        if not np.array_equal(derivative_freqs, freqs):
            raise ValueError(f"{label} {path.name}: spectra landed on a different grid.")
        spectra[label] = psd
        del derivative
    del source
    return freqs, spectra, duration_s


def analysis_bands_from_config(config) -> tuple[tuple[str, float, float], ...]:
    """The study's own bands where it defines them, so the labels match its analyses."""
    defined = config.get("frequency_bands") or {}
    if not isinstance(defined, dict):
        raise ValueError("frequency_bands must be a mapping of name to [low, high].")
    bands = []
    for name, edges in defined.items():
        low_hz, high_hz = (float(value) for value in edges)
        if high_hz <= low_hz:
            raise ValueError(f"frequency_bands.{name} must have increasing edges.")
        bands.append((str(name), low_hz, high_hz))
    return tuple(bands)


def panel_edges(band_hz: tuple[float, float], span_hz: float) -> tuple[tuple[float, float], ...]:
    """Consecutive equal spans tiling the plotted band, the last one truncated."""
    low_hz, high_hz = band_hz
    starts = np.arange(low_hz, high_hz, span_hz)
    return tuple((float(start), float(min(start + span_hz, high_hz))) for start in starts)


def bands_covering(low_hz: float, high_hz: float, bands) -> str:
    """The named bands a span overlaps, for the panel label."""
    return ", ".join(
        name
        for name, band_low, band_high in bands
        if min(high_hz, band_high) > max(low_hz, band_low)
    )


def figure_band_panels(
    freqs,
    arms: dict[str, np.ndarray],
    path: Path,
    *,
    settings: PsdSettings,
    cohort_description: str,
    bands=(),
) -> None:
    """The same spectra tiled into readable spans, one panel per range.

    Each panel autoscales on its own contents, so a deep notch in one span does not
    flatten the spectrum in another.
    """
    edges = panel_edges((float(freqs[0]), float(freqs[-1])), settings.panel_span_hz)
    figure, axes = plt.subplots(len(edges), 1, figsize=(13, 1.9 * len(edges)), squeeze=False)
    medians = {label: to_db(np.median(stack, axis=0)) for label, stack in arms.items()}
    for index, (low_hz, high_hz) in enumerate(edges):
        axis = axes[index][0]
        inside = (freqs >= low_hz) & (freqs <= high_hz)
        for label in _draw_order(medians):
            colour, name, width = _style(label)
            axis.plot(freqs[inside], medians[label][inside], color=colour, lw=width, label=name)
        axis.set_xlim(low_hz, high_hz)
        covered = bands_covering(low_hz, high_hz, bands)
        axis.set_ylabel(f"{low_hz:g}-{high_hz:g} Hz", fontsize=8)
        if covered:
            axis.text(
                0.995,
                0.93,
                covered,
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="#6B7280",
            )
        if index == 0:
            axis.legend(loc="upper left", fontsize=8)
        axis.tick_params(labelsize=8)
    axes[0][0].set_title(
        f"Power spectra before and after removal: {cohort_description}. "
        f"{settings.panel_span_hz:g} Hz per panel; Welch via MNE."
    )
    axes[-1][0].set_xlabel("frequency (Hz)")
    figure.supylabel("median PSD (dB re 1 V²/Hz)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


ARM_STYLE = {
    # The source is drawn wide and pale beneath the derivatives, so where nothing changed
    # it survives as a halo instead of vanishing under the trace on top. Without that,
    # "this line was removed" and "this feature was already here" render identically --
    # which matters wherever the source spectrum already carried structure this pass never
    # aimed at, such as the periodic residue of an upstream gradient correction.
    "source": ("#F3A28E", "before correction", 2.2),
    "harmonic-notched": ("#111827", "after automatic line notches", 0.7),
}
DEFAULT_STYLE = ("#6B7280", "", 0.7)


def _style(label: str) -> tuple[str, str, float]:
    colour, name, width = ARM_STYLE.get(label, DEFAULT_STYLE)
    return colour, name or label, width


def _draw_order(arms) -> tuple[str, ...]:
    """Source first so everything else lands on top of it."""
    return ("source", *(label for label in arms if label != "source"))


def figure_cohort(
    freqs,
    arms: dict[str, np.ndarray],
    path: Path,
    *,
    cohort_description: str,
) -> None:
    """Cohort median spectrum, and what the removal took, on one frequency axis."""
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, height_ratios=[2, 1])
    for label in _draw_order(arms):
        colour, name, width = _style(label)
        axes[0].plot(
            freqs, to_db(np.median(arms[label], axis=0)), color=colour, lw=width, label=name
        )
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("median PSD (dB re 1 V²/Hz)")
    axes[0].set_title(
        f"Power spectra before and after removal: {cohort_description}. Welch via MNE."
    )

    # What changed, which is the question the top panel only implies. Negative is removed.
    source = to_db(np.median(arms["source"], axis=0))
    for label in _draw_order(arms)[1:]:
        colour, name, _ = _style(label)
        stack = arms[label]
        change = to_db(np.median(stack, axis=0)) - source
        axes[1].plot(freqs, change, color=colour, lw=0.6, label=name)
        # "Added power here" and "removed power here" are different claims and should not
        # render identically. Over-subtraction against a cluster is what produces the first.
        gained = change > 0.0
        axes[1].plot(
            freqs[gained],
            change[gained],
            linestyle="none",
            marker="|",
            markersize=3,
            color="#C1442E",
        )
    axes[1].axhline(0.0, color="#6B7280", lw=0.6, ls="--")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("change (dB)")
    axes[1].set_xlim(freqs[0], freqs[-1])
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def figure_per_recording(
    freqs,
    per_recording: dict[str, dict[str, np.ndarray]],
    path: Path,
    *,
    cohort_description: str,
) -> None:
    """One panel per recording, so a single bad one cannot hide inside a cohort median."""
    names = sorted(per_recording)
    columns = min(3, len(names))
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.5 * columns, 2.8 * rows),
        squeeze=False,
        sharex=True,
    )
    for index, name in enumerate(names):
        axis = axes[index // columns][index % columns]
        for label in _draw_order(per_recording[name]):
            colour, _, width = _style(label)
            axis.plot(freqs, to_db(per_recording[name][label]), color=colour, lw=width * 0.7)
        axis.set_title(name, fontsize=8)
        axis.set_xlim(freqs[0], freqs[-1])
    for index in range(len(names), rows * columns):
        axes[index // columns][index % columns].axis("off")
    figure.supxlabel("frequency (Hz)")
    figure.supylabel("median PSD (dB re 1 V²/Hz)")
    figure.suptitle(f"Before and after automatic line notching: {cohort_description}")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.995))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def dataset_description(
    recording_count: int,
    participant_count: int,
    total_duration_s: float,
) -> str:
    """Compact grammatically correct description for generated figure titles."""
    recording_label = "recording" if recording_count == 1 else "recordings"
    participant_label = "participant" if participant_count == 1 else "participants"
    return (
        f"{recording_count} {recording_label} from {participant_count} "
        f"{participant_label} ({total_duration_s / 3600.0:.1f} h EEG)"
    )


def run(args: argparse.Namespace) -> None:
    """Compare the source against every derivative that exists."""
    import time

    from decomb.config import load_config

    config = load_config(getattr(args, "config", None))
    settings = PsdSettings.from_config(config)
    source_root = config.path("bids_root", override=getattr(args, "bids_root", None))
    report_dir = config.path("removal_dir", override=getattr(args, "report_dir", None))

    candidates = [
        (
            "harmonic-notched",
            config.path("output_root", override=getattr(args, "output_root", None)),
        )
    ]
    available = [(label, root) for label, root in candidates if root.is_dir()]
    if not available:
        raise FileNotFoundError(
            "Nothing to compare against: no cleaned dataset exists yet. Run `decomb apply` first."
        )

    subjects = getattr(args, "subjects", None)
    runs = recordings.discover_runs(source_root, subjects=subjects, task="*")
    print(f"Measuring {len(runs)} recordings from {source_root}")
    for label, root in available:
        print(f"  against {label}: {root}")
    print(f"  Welch, {settings.window_s:g} s segments, {settings.overlap:.0%} overlap, EEG only")

    per_recording: dict[str, dict[str, np.ndarray]] = {}
    total_duration_s = 0.0
    freqs = None
    for index, vhdr in enumerate(runs, start=1):
        started = time.time()
        derivatives = [(label, root / vhdr.relative_to(source_root)) for label, root in available]
        missing = [str(path) for _, path in derivatives if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{vhdr.stem}: derivative missing at {missing[0]}")
        run_freqs, spectra, duration_s = compare_recording(vhdr, derivatives, settings)
        if freqs is None:
            freqs = run_freqs
        elif not np.array_equal(run_freqs, freqs):
            raise ValueError(f"{vhdr.stem}: spectra landed on a different grid from the first.")
        per_recording[vhdr.stem] = spectra
        total_duration_s += duration_s
        print(f"[{index}/{len(runs)}] {vhdr.stem[:44]:44s} ({time.time() - started:.0f}s)")

    arms = {
        label: np.stack([per_recording[name][label] for name in sorted(per_recording)])
        for label in ("source", *(label for label, _ in available))
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    participant_count = len({recordings.subject_of(vhdr) for vhdr in runs})
    cohort_description = dataset_description(
        len(runs),
        participant_count,
        total_duration_s,
    )
    figure_cohort(
        freqs,
        arms,
        report_dir / "psd_before_after.png",
        cohort_description=cohort_description,
    )
    figure_band_panels(
        freqs,
        arms,
        report_dir / "psd_before_after_panels.png",
        settings=settings,
        cohort_description=cohort_description,
        bands=analysis_bands_from_config(config),
    )
    figure_per_recording(
        freqs,
        per_recording,
        report_dir / "psd_before_after_per_recording.png",
        cohort_description=cohort_description,
    )
    np.savez_compressed(
        report_dir / "psd_before_after.npz",
        freqs=freqs,
        recordings=np.array(sorted(per_recording)),
        **arms,
    )

    source_db = to_db(np.median(arms["source"], axis=0))
    print("\nmedian change over the cohort:")
    for label, _ in available:
        change = to_db(np.median(arms[label], axis=0)) - source_db
        worst = int(np.argmin(change))
        print(
            f"  {label:14s} deepest {change[worst]:7.2f} dB at {freqs[worst]:7.3f} Hz; "
            f"bins changed by more than 1 dB: {np.mean(np.abs(change) > 1.0):.1%}"
        )
    print(f"\n  wrote {report_dir / 'psd_before_after.png'}")
    print(f"  wrote {report_dir / 'psd_before_after_panels.png'}")
    print(f"  wrote {report_dir / 'psd_before_after_per_recording.png'}")
    print(f"  wrote {report_dir / 'psd_before_after.npz'}")
