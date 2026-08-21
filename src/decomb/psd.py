"""Cohort-averaged before-and-after power spectra, drawn by MNE.

    decomb psd

Two figures summarize the selected recordings: source and line-notch derivative spectra,
each averaged equally by recording in linear power before MNE draws the per-channel
spectrum with sensor-position colours and the sensor inset.

The pair is only worth reading if both sides were measured identically, so the spectra
are computed with one set of parameters, on the same channels, over the same samples,
and on one shared decibel scale. The stage refuses when the recordings do not
correspond, and it runs after ``apply``.

Unlike the stages that transform or certify, this one accepts ``--subjects``: a figure of
part of a cohort reports its own recording count and analysed duration.
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
DECLARED_BEFORE_NAME = "psd_before_declared.png"
DECLARED_AFTER_NAME = "psd_after_declared.png"


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


@dataclass(frozen=True)
class CohortSpectrumPair:
    """Equal-recording source and derivative spectra with cohort extent."""

    before: object
    after: object
    recording_count: int
    analysed_hours: float


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


def average_channel_spectra(spectra):
    """Average recording-level power after excluding each run's bad channels."""
    import mne

    items = tuple(spectra)
    if not items:
        raise ValueError("A cohort spectrum requires recordings.")
    reference = items[0]
    for spectrum in items[1:]:
        if spectrum.ch_names != reference.ch_names:
            raise ValueError("Cohort spectra require the same EEG channels in one order.")
        if not np.array_equal(spectrum.freqs, reference.freqs):
            raise ValueError("Cohort spectra require one shared frequency grid.")

    powers = np.stack(
        [spectrum.get_data(exclude=()) for spectrum in items],
        axis=0,
    )
    included = np.array(
        [
            [name not in spectrum.info["bads"] for name in reference.ch_names]
            for spectrum in items
        ],
        dtype=bool,
    )
    recording_counts = included.sum(axis=0)
    if np.any(recording_counts == 0):
        missing = [
            name
            for name, count in zip(
                reference.ch_names,
                recording_counts,
                strict=True,
            )
            if count == 0
        ]
        raise ValueError(
            f"Every cohort channel must be good in at least one recording: {missing}"
        )
    mean_power = np.sum(
        powers * included[:, :, np.newaxis],
        axis=0,
    ) / recording_counts[:, np.newaxis]
    channel_info = reference.info.copy()
    channel_info["bads"] = []
    return mne.time_frequency.SpectrumArray(
        mean_power,
        channel_info,
        reference.freqs,
        verbose="ERROR",
    )


def analysed_duration_hours(raws, settings: PsdSettings) -> float:
    """Total acquisition time eligible for complete Welch windows, in hours."""
    total_seconds = 0.0
    for raw in raws:
        sampling_frequency_hz = float(raw.info["sfreq"])
        window_samples = recordings.estimation_window_samples(
            sampling_frequency_hz,
            settings.window_s,
        )
        total_seconds += sum(
            stop - start
            for start, stop in recordings.acquisition_segments(raw)
            if stop - start >= window_samples
        ) / sampling_frequency_hz
    if not np.isfinite(total_seconds) or total_seconds <= 0.0:
        raise ValueError("Analysed duration must be finite and positive.")
    return float(total_seconds / 3_600.0)


def cohort_spectrum_pair(recording_pairs, settings: PsdSettings) -> CohortSpectrumPair:
    """Compute equally weighted channel spectra for corresponding recordings."""
    before_spectra = []
    after_spectra = []
    total_hours = 0.0
    recording_count = 0
    for label, source, derivative in recording_pairs:
        require_correspondence(source, derivative, label)
        align_bad_channels(source, derivative)
        before_spectra.append(channel_spectrum(source, settings))
        after_spectra.append(channel_spectrum(derivative, settings))
        total_hours += analysed_duration_hours((source,), settings)
        recording_count += 1
    if recording_count == 0:
        raise ValueError("A cohort spectrum requires recording pairs.")
    return CohortSpectrumPair(
        before=average_channel_spectra(before_spectra),
        after=average_channel_spectra(after_spectra),
        recording_count=recording_count,
        analysed_hours=total_hours,
    )


def mean_band_availability_percent(
    manifest,
    *,
    band_names: tuple[str, ...],
) -> dict[str, float]:
    """Mean the cumulative retained share once per recording."""
    if manifest.empty:
        raise ValueError("Band availability requires manifest rows.")
    required = {
        "recording",
        *(f"{band_name}_retained_share" for band_name in band_names),
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Band availability columns are missing: {sorted(missing)}")

    recording_rows = []
    for recording, rows in manifest.groupby("recording", sort=False):
        retained = rows[
            [f"{band_name}_retained_share" for band_name in band_names]
        ].to_numpy(dtype=float)
        if not np.all(retained == retained[0]):
            raise ValueError(
                f"{recording}: manifest rows disagree on cumulative band availability."
            )
        recording_rows.append(retained[0])

    availability = {}
    values_by_recording = np.asarray(recording_rows)
    for band_index, band_name in enumerate(band_names):
        values = values_by_recording[:, band_index]
        if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{band_name}: retained shares must lie in [0, 1].")
        availability[band_name] = float(100.0 * values.mean())
    return availability


def _cohort_band_availability_percent(
    manifest_path: Path,
    *,
    recording_names: tuple[str, ...],
    band_names: tuple[str, ...],
) -> dict[str, float]:
    """Read exactly the selected recordings before summarizing terminal plans."""
    import pandas as pd

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Line-notch manifest missing at {manifest_path}")
    manifest = pd.read_csv(manifest_path, sep="\t", float_precision="round_trip")
    selected = manifest.loc[manifest["recording"].isin(recording_names)]
    actual_recordings = set(selected["recording"].astype(str))
    expected_recordings = set(recording_names)
    if actual_recordings != expected_recordings:
        missing = sorted(expected_recordings - actual_recordings)
        raise ValueError(f"Band availability is missing recordings: {missing}")
    return mean_band_availability_percent(selected, band_names=band_names)


def require_correspondence(source, derivative, label: str) -> None:
    """Refuse a pair that is two different recordings rather than one comparison."""
    if derivative.ch_names != source.ch_names:
        raise ValueError(f"{label}: channel set differs from the source.")
    if derivative.n_times != source.n_times:
        raise ValueError(f"{label}: length differs from the source.")
    if not np.isclose(derivative.info["sfreq"], source.info["sfreq"]):
        raise ValueError(f"{label}: sampling rate differs from the source.")


def align_bad_channels(source, derivative) -> tuple[str, ...]:
    """Give both sides one bad-channel set before cohort exclusion.

    The union ensures that a channel distrusted on either side contributes to neither
    source nor derivative cohort means for that recording.
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
    than merely counted.
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


def declared_unavailable_fraction(manifest, frequencies_hz, recording_names):
    """Share of recordings whose manifest declares each frequency unavailable.

    Counted per recording, not per row: a recording that names one frequency in fifty
    rows has still given it up once.
    """
    from decomb import notch

    required = {"recording", "unavailable_low_hz", "unavailable_high_hz"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"Declared bandwidth needs manifest columns: {sorted(missing)}"
        )
    frequencies = np.asarray(frequencies_hz, dtype=float)
    wanted = set(recording_names)
    count = np.zeros(frequencies.size)
    for recording, rows in manifest.groupby("recording", sort=False):
        if recording not in wanted:
            continue
        intervals = notch.merged_intervals(
            (float(low_hz), float(high_hz))
            for low_hz, high_hz in zip(
                rows["unavailable_low_hz"], rows["unavailable_high_hz"], strict=True
            )
            if str(low_hz).strip() != "" and str(high_hz).strip() != ""
        )
        declared = np.zeros(frequencies.size, dtype=bool)
        for low_hz, high_hz in intervals:
            declared |= (frequencies >= low_hz) & (frequencies <= high_hz)
        count += declared
    return count / max(len(wanted), 1)


def figure_spectrum_with_declared(
    spectrum,
    fraction,
    path: Path,
    *,
    title: str,
    ylim=None,
    dpi: int = 300,
):
    """Cohort spectrum above the share of recordings that gave up each frequency.

    The sensor mean and range replace one line per channel: with 63 channels overplotted
    the comb is lost in its own traces. The declaration goes in its own panel rather than
    as shading, so neither layer competes with the other.
    """
    frequencies = spectrum.freqs
    decibels = 10.0 * np.log10(spectrum.get_data() * 1e12)
    figure, (upper, lower) = plt.subplots(
        2,
        1,
        figsize=(6.3, 3.5),
        sharex=True,
        gridspec_kw={"height_ratios": [5.2, 1.0], "hspace": 0.10},
    )
    upper.fill_between(
        frequencies,
        decibels.min(axis=0),
        decibels.max(axis=0),
        color="#C9D4DE",
        lw=0,
    )
    upper.plot(frequencies, decibels.mean(axis=0), color="#1B1B1B", lw=1.1)
    if ylim is not None:
        upper.set_ylim(ylim)
    upper.set_ylabel("Power spectral density\n(dB/Hz re 1 µV²)", fontsize=9)
    upper.set_title(title, fontsize=10, fontweight="bold", loc="right")
    upper.legend(
        handles=[
            plt.Line2D([], [], color="#1B1B1B", lw=1.1, label="sensor mean"),
            plt.Rectangle((0, 0), 1, 1, color="#C9D4DE", label="sensor range"),
        ],
        loc="lower left",
        frameon=False,
        fontsize=8,
        handlelength=1.5,
        labelspacing=0.25,
    )
    lower.fill_between(frequencies, 0.0, 100.0 * np.asarray(fraction),
                       color="#B04A57", lw=0)
    lower.set_ylim(0, 100)
    lower.set_yticks([0, 100])
    lower.set_ylabel("declared\nunavailable (%)", fontsize=8, labelpad=2)
    lower.set_xlabel("Frequency (Hz)", fontsize=9)
    lower.set_xlim(frequencies[0], frequencies[-1])
    for axis in (upper, lower):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8)
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
        # rather than reading limits back off a drawn axis.
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


def _read_recording_pairs(runs, source_root: Path, derivative_root: Path):
    """Yield source/derivative pairs after requiring every derivative file."""
    for index, source_vhdr in enumerate(runs, start=1):
        derivative_vhdr = recordings.derivative_vhdr_path(
            source_vhdr,
            source_root,
            derivative_root,
        )
        if not derivative_vhdr.is_file():
            raise FileNotFoundError(
                f"{source_vhdr.stem}: derivative missing at {derivative_vhdr}"
            )
        print(f"[{index}/{len(runs)}] measuring {source_vhdr.stem}")
        yield (
            source_vhdr.stem,
            recordings.read_bids_raw(source_vhdr),
            recordings.read_bids_raw(derivative_vhdr),
        )


def run(args: argparse.Namespace) -> None:
    """Draw equal-recording cohort spectra before and after the correction."""
    import pandas as pd

    from decomb import notch
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
    print(
        f"Measuring {len(runs)} recordings with {settings.window_s:g} s Welch "
        f"segments, {settings.overlap:.0%} overlap, EEG only"
    )
    cohort = cohort_spectrum_pair(
        _read_recording_pairs(runs, source_root, derivative_root),
        settings,
    )
    limits = shared_decibel_limits(cohort.before, cohort.after)
    extent = f"{cohort.recording_count} recordings · {cohort.analysed_hours:.2f} h"

    report_dir.mkdir(parents=True, exist_ok=True)
    figure_spectrum(
        cohort.before,
        report_dir / BEFORE_NAME,
        title=f"Before correction — {extent}",
        ylim=limits,
    )
    figure_spectrum(
        cohort.after,
        report_dir / AFTER_NAME,
        title=f"After correction — {extent}",
        ylim=limits,
    )
    recording_names = tuple(run.stem for run in runs)
    manifest_path = derivative_root / notch.MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Line-notch manifest missing at {manifest_path}")
    manifest = pd.read_csv(
        manifest_path, sep="\t", float_precision="round_trip", keep_default_na=False
    )
    fraction = declared_unavailable_fraction(
        manifest, cohort.before.freqs, recording_names
    )
    figure_spectrum_with_declared(
        cohort.before,
        fraction,
        report_dir / DECLARED_BEFORE_NAME,
        title=f"Source — {extent}",
        ylim=limits,
    )
    figure_spectrum_with_declared(
        cohort.after,
        fraction,
        report_dir / DECLARED_AFTER_NAME,
        title=f"Derivative — {extent}",
        ylim=limits,
    )
    print("  BIDS-bad channels excluded within each recording before averaging")
    band_names = tuple(name for name, _, _ in notch.analysed_bands_from_config(config))
    availability = _cohort_band_availability_percent(
        manifest_path,
        recording_names=recording_names,
        band_names=band_names,
    )
    for band_name, retained_percent in availability.items():
        print(f"  {band_name} availability {retained_percent:.3f}%")
    print(f"  analysed duration {cohort.analysed_hours:.4f} h")
    print(f"  shared scale {limits[0]:.1f} to {limits[1]:.1f} dB/Hz re 1 µV²")
    print(f"  wrote {report_dir / BEFORE_NAME}")
    print(f"  wrote {report_dir / AFTER_NAME}")
    print(f"  wrote {report_dir / DECLARED_BEFORE_NAME}")
    print(f"  wrote {report_dir / DECLARED_AFTER_NAME}")
    print(
        f"  {100 * (fraction == 0).mean():.1f}% of frequencies declared by no recording, "
        f"{100 * (fraction == 1).mean():.1f}% by all {len(recording_names)}"
    )
