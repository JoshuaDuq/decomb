"""Before-and-after power spectra of the removal, drawn by MNE.

    decomb psd

Two figures from one recording: the source spectrum and the line-notch derivative's,
each an MNE per-channel Welch spectrum with sensor-position colours and the sensor
inset. Nothing else is drawn, so what a reader compares is the same plot twice.

The pair is only worth reading if both sides were measured identically, so the spectra
are computed with one set of parameters, on the same channels, over the same samples,
and on one shared decibel scale. The stage refuses when the recordings do not
correspond, and it runs after ``apply``.

Unlike the stages that transform or certify, this one accepts ``--subjects``: a figure of
part of a cohort is a smaller figure, not a false claim about the whole one.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from decomb import recordings  # noqa: E402

BEFORE_NAME = "psd_before.png"
AFTER_NAME = "psd_after.png"


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
        values = np.asarray(self.band_hz, dtype=float)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError("band_hz must contain two finite values.")
        low_hz, high_hz = values
        if not 0.0 <= low_hz < high_hz:
            raise ValueError("band_hz must contain increasing non-negative values.")

    @property
    def overlap(self) -> float:
        return 0.5

    @classmethod
    def from_config(cls, config) -> PsdSettings:
        from decomb.notch import HarmonicNotchSettings

        correction = HarmonicNotchSettings.from_config(config)
        return cls(
            window_s=correction.estimation_window_s,
            band_hz=correction.frequency_range_hz,
        )


def channel_spectrum(raw, settings: PsdSettings):
    """Per-channel Welch spectrum of the EEG channels, computed by MNE.

    Every channel is kept rather than reduced to one summary trace, because the figure's
    subject is what the removal did to each channel.

    EEG only: ECG and EOG carry different units and amplitudes, and drawing them on the
    same axis would put a millivolt trace beside a microvolt one.
    """
    sampling_frequency_hz = float(raw.info["sfreq"])
    n_fft = recordings.estimation_window_samples(
        sampling_frequency_hz,
        settings.window_s,
    )
    if n_fft > raw.n_times:
        raise ValueError(
            f"The recording holds {raw.n_times} samples, fewer than the {n_fft} of one "
            f"{settings.window_s:g} s Welch segment. Lower "
            "`removal.estimation_window_s`."
        )
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    if len(picks) == 0:
        raise ValueError("PSD estimation requires at least one EEG channel.")
    bounds = recordings.valid_window_bounds(
        raw,
        window_s=settings.window_s,
        overlap=settings.overlap,
    )
    data = raw.get_data(picks=picks)
    windows = np.stack(
        [data[:, start:stop] for start, stop in bounds],
        axis=1,
    )
    power, frequencies_hz = mne.time_frequency.psd_array_welch(
        windows,
        sampling_frequency_hz,
        fmin=settings.band_hz[0],
        fmax=min(settings.band_hz[1], np.nextafter(sampling_frequency_hz / 2.0, 0.0)),
        n_fft=n_fft,
        n_per_seg=n_fft,
        n_overlap=0,
        average="mean",
        window="hamming",
        remove_dc=True,
        verbose="ERROR",
    )
    channel_info = mne.pick_info(raw.info, picks, copy=True)
    return mne.time_frequency.SpectrumArray(
        power.mean(axis=1),
        channel_info,
        frequencies_hz,
        verbose="ERROR",
    )


def require_correspondence(source, derivative, label: str) -> None:
    """Refuse a pair that is two different recordings rather than one comparison."""
    if derivative.ch_names != source.ch_names:
        raise ValueError(f"{label}: channel set differs from the source.")
    if derivative.n_times != source.n_times:
        raise ValueError(f"{label}: length differs from the source.")
    if not np.isclose(derivative.info["sfreq"], source.info["sfreq"]):
        raise ValueError(f"{label}: sampling rate differs from the source.")


def align_bad_channels(source, derivative) -> tuple[str, ...]:
    """Give both sides one set of bad channels, so the pair draws them the same way.

    MNE draws bad channels in grey dashes and open sensors in the inset. A derivative that
    lost the source's marking would render those channels as ordinary coloured traces, and
    the pair would then differ by which channels are bad as well as by the correction. The
    union is used so a channel either side distrusts is never quietly drawn as good.
    """
    union = tuple(
        sorted(set(source.info["bads"]) | set(derivative.info["bads"]))
    )
    source.info["bads"] = list(union)
    derivative.info["bads"] = list(union)
    return union


def figure_spectrum(spectrum, path: Path, *, title: str, ylim=None, dpi: int = 200):
    """One MNE spectrum figure, per channel, in sensor-position colours.

    ``spatial_colors`` is what makes a channel identifiable: its line takes the colour of
    its position in the inset, so a channel that moved can be found on the head rather
    than merely counted. Bad channels stay in MNE's grey dashes.
    """
    figure = spectrum.plot(
        spatial_colors=True,
        dB=True,
        amplitude=False,
        show=False,
    )
    if ylim is not None:
        for axis in figure.axes:
            if axis.get_ylabel():
                axis.set_ylim(ylim)
    figure.suptitle(title, fontsize=12, fontweight="bold")
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def shared_decibel_limits(*spectra) -> tuple[float, float]:
    """One decibel scale for every figure in the pair.

    Two spectra on two autoscaled axes cannot be compared by eye: the removal would move
    the axis as much as it moves the data, and a deep notch would redraw the baseline it
    is supposed to be measured against.
    """
    lows, highs = [], []
    for spectrum in spectra:
        # MNE plots decibels relative to 1 µV²/Hz, so the same scaling is applied here
        # rather than reading limits back off a drawn axis. Bad channels are included
        # because the figure draws them: get_data drops them by default, which would set
        # a scale that clips the very traces sitting furthest from the rest.
        decibels = 10.0 * np.log10(
            np.maximum(spectrum.get_data(exclude=()) * 1e12, 1e-30)
        )
        finite = decibels[np.isfinite(decibels)]
        if finite.size == 0:
            raise ValueError("A spectrum held no finite power.")
        lows.append(float(np.min(finite)))
        highs.append(float(np.max(finite)))
    low, high = min(lows), max(highs)
    margin = 0.04 * (high - low) or 1.0
    return low - margin, high + margin


def run(args: argparse.Namespace) -> None:
    """Draw one recording's spectrum before and after the correction."""
    from decomb.config import load_config

    config = load_config(getattr(args, "config", None))
    settings = PsdSettings.from_config(config)
    source_root = config.path("bids_root", override=getattr(args, "bids_root", None))
    derivative_root = config.path(
        "output_root", override=getattr(args, "output_root", None)
    )
    report_dir = config.path("removal_dir", override=getattr(args, "report_dir", None))
    if not derivative_root.is_dir():
        raise FileNotFoundError(
            "Nothing to compare against: no cleaned dataset exists yet. "
            "Run `decomb apply` first."
        )

    runs = recordings.discover_runs(
        source_root,
        subjects=getattr(args, "subjects", None),
        task="*",
    )
    if not runs:
        raise FileNotFoundError(f"No recordings were discovered under {source_root}.")
    source_vhdr = runs[0]
    derivative_vhdr = recordings.derivative_vhdr_path(
        source_vhdr,
        source_root,
        derivative_root,
    )
    if not derivative_vhdr.is_file():
        raise FileNotFoundError(f"{source_vhdr.stem}: derivative missing at {derivative_vhdr}")

    # One recording, named rather than implied: the pair is a demonstration of the
    # correction, and a figure that silently stood for a different recording each run
    # would not be one.
    if len(runs) > 1:
        print(f"{len(runs)} recordings discovered; drawing the first. --subjects selects another.")
    print(f"Drawing {source_vhdr.stem}")
    print(f"  Welch, {settings.window_s:g} s segments, {settings.overlap:.0%} overlap, EEG only")

    source = recordings.read_bids_raw(source_vhdr)
    derivative = recordings.read_bids_raw(derivative_vhdr)
    require_correspondence(source, derivative, derivative_vhdr.name)
    bads = align_bad_channels(source, derivative)
    if bads:
        print(f"  bad channels drawn in grey on both: {', '.join(bads)}")
    before = channel_spectrum(source, settings)
    after = channel_spectrum(derivative, settings)
    limits = shared_decibel_limits(before, after)

    report_dir.mkdir(parents=True, exist_ok=True)
    figure_spectrum(
        before,
        report_dir / BEFORE_NAME,
        title=f"Before correction — {source_vhdr.stem}",
        ylim=limits,
    )
    figure_spectrum(
        after,
        report_dir / AFTER_NAME,
        title=f"After correction — {source_vhdr.stem}",
        ylim=limits,
    )
    print(f"  shared scale {limits[0]:.1f} to {limits[1]:.1f} dB/Hz re 1 µV²")
    print(f"  wrote {report_dir / BEFORE_NAME}")
    print(f"  wrote {report_dir / AFTER_NAME}")
