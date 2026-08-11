"""Fair real-data comparison with conventional MNE notch defaults.

Both arms remove the frequencies selected by decomb.  Only filter geometry differs:
decomb uses the measured trajectory envelopes, while the reference arm uses MNE's
documented default width (frequency / 200) and 1 Hz transition bandwidth.  Reference
bands are merged when their transitions overlap because MNE refuses overlapping FIR
stopbands.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from decomb import harmonics, notch
from decomb.spectral import to_db

MNE_DEFAULT_NOTCH_WIDTH_DIVISOR = 200.0
MNE_DEFAULT_TRANSITION_BANDWIDTH_HZ = 1.0


@dataclass(frozen=True)
class NotchComparison:
    """Real spectrum and filter geometry for two notch implementations."""

    frequencies_hz: np.ndarray
    source_psd: np.ndarray
    decomb_psd: np.ndarray
    traditional_psd: np.ndarray
    decomb_plan: notch.HarmonicNotchPlan
    traditional_plan: notch.HarmonicNotchPlan
    duration_s: float

    def __post_init__(self) -> None:
        spectra = (self.source_psd, self.decomb_psd, self.traditional_psd)
        if self.frequencies_hz.ndim != 1 or self.frequencies_hz.size < 2:
            raise ValueError("Comparison frequencies must be a one-dimensional grid.")
        if any(spectrum.shape != self.frequencies_hz.shape for spectrum in spectra):
            raise ValueError("Every comparison spectrum must use the frequency grid.")
        if not np.all(np.isfinite(self.frequencies_hz)) or any(
            not np.all(np.isfinite(spectrum)) for spectrum in spectra
        ):
            raise ValueError("Comparison spectra must contain only finite values.")
        if any(np.any(spectrum <= 0.0) for spectrum in spectra):
            raise ValueError("Comparison power must be positive.")
        if not np.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("Comparison duration must be finite and positive.")


def traditional_notch_plan(
    model: harmonics.AdaptiveCombModel,
    settings,
) -> notch.HarmonicNotchPlan:
    """Use decomb's detected centres with conventional MNE FIR geometry."""
    stopbands = [
        _traditional_stopband(
            measured_band.centre_hz,
            measured_band.harmonics,
            measured_band.kind,
        )
        for measured_band in notch.observed_line_intervals(model, settings)
    ]
    merged = notch._merge_stopbands(
        stopbands,
        minimum_gap_hz=MNE_DEFAULT_TRANSITION_BANDWIDTH_HZ,
    )
    return notch.HarmonicNotchPlan(
        merged,
        transition_bandwidth_hz=MNE_DEFAULT_TRANSITION_BANDWIDTH_HZ,
    )


def _traditional_stopband(
    position_hz: float,
    harmonic_numbers: tuple[int, ...],
    kind: str,
) -> notch.HarmonicStopband:
    width_hz = position_hz / MNE_DEFAULT_NOTCH_WIDTH_DIVISOR
    return notch.HarmonicStopband(
        harmonic_numbers,
        position_hz - width_hz / 2.0,
        position_hz + width_hz / 2.0,
        kind=kind,
    )


def unavailable_width_hz(
    plan: notch.HarmonicNotchPlan,
    frequency_range_hz: Sequence[float],
) -> float:
    """Total planned stopband and transition width inside a frequency range."""
    low_hz, high_hz = (float(value) for value in frequency_range_hz)
    if high_hz <= low_hz:
        raise ValueError("The comparison frequency range must be increasing.")
    return sum(
        max(0.0, min(high_hz, upper_hz) - max(low_hz, lower_hz))
        for lower_hz, upper_hz in plan.unavailable_edges()
    )


def measure_notch_comparison(raw, settings) -> NotchComparison:
    """Apply both notch geometries to one in-memory recording and measure them equally."""
    from decomb import psd

    model = notch.fit_harmonic_model(raw, settings)
    decomb_plan = notch.plan_harmonic_stopbands(model, settings)
    traditional_plan = traditional_notch_plan(model, settings)
    decomb_raw = notch.apply_harmonic_notches(raw, decomb_plan)
    traditional_raw = notch.apply_harmonic_notches(raw, traditional_plan)
    psd_settings = psd.PsdSettings(
        window_s=settings.estimation_window_s,
        band_hz=settings.frequency_range_hz,
    )

    frequencies_hz, source_psd = psd.channel_median_psd(raw, psd_settings)
    decomb_frequencies_hz, decomb_psd = psd.channel_median_psd(
        decomb_raw,
        psd_settings,
    )
    traditional_frequencies_hz, traditional_psd = psd.channel_median_psd(
        traditional_raw,
        psd_settings,
    )
    if not np.array_equal(frequencies_hz, decomb_frequencies_hz) or not np.array_equal(
        frequencies_hz,
        traditional_frequencies_hz,
    ):
        raise ValueError("Notch comparison arms landed on different frequency grids.")
    return NotchComparison(
        frequencies_hz=frequencies_hz,
        source_psd=source_psd,
        decomb_psd=decomb_psd,
        traditional_psd=traditional_psd,
        decomb_plan=decomb_plan,
        traditional_plan=traditional_plan,
        duration_s=raw.n_times / float(raw.info["sfreq"]),
    )


def figure_notch_comparison(
    comparison: NotchComparison,
    path: Path,
    *,
    recording_description: str,
) -> None:
    """Plot real-data outcomes and the bandwidth each notch geometry invalidates."""
    import matplotlib.pyplot as plt

    frequencies_hz = comparison.frequencies_hz
    frequency_range_hz = (float(frequencies_hz[0]), float(frequencies_hz[-1]))
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13.0, 7.2),
        sharex=True,
        height_ratios=(2.4, 1.0),
        layout="constrained",
    )

    spectrum_axis, geometry_axis = axes
    spectrum_axis.plot(
        frequencies_hz,
        to_db(comparison.source_psd),
        color="#F3A28E",
        linewidth=2.2,
        label="before correction",
    )
    spectrum_axis.plot(
        frequencies_hz,
        to_db(comparison.decomb_psd),
        color="#111827",
        linewidth=0.8,
        label="decomb measured-width notches",
    )
    spectrum_axis.plot(
        frequencies_hz,
        to_db(comparison.traditional_psd),
        color="#2563A6",
        linewidth=0.8,
        linestyle="--",
        label="same centres, MNE default notch geometry",
    )
    spectrum_axis.set_ylabel("median PSD (dB re 1 V²/Hz)")
    spectrum_axis.set_title(
        "Measured narrow notches versus conventional notch geometry\n"
        f"{recording_description}; real EEG; identical detected centres"
    )
    spectrum_axis.legend(loc="upper right", fontsize=8)

    geometries = (
        (comparison.decomb_plan, 1.0, "#111827"),
        (comparison.traditional_plan, 0.0, "#2563A6"),
    )
    for plan, y_position, colour in geometries:
        for low_hz, high_hz in plan.unavailable_edges():
            clipped_low_hz = max(frequency_range_hz[0], low_hz)
            clipped_high_hz = min(frequency_range_hz[1], high_hz)
            if clipped_high_hz > clipped_low_hz:
                geometry_axis.hlines(
                    y_position,
                    clipped_low_hz,
                    clipped_high_hz,
                    color=colour,
                    linewidth=7.0,
                )
        width_hz = unavailable_width_hz(plan, frequency_range_hz)
        geometry_axis.text(
            frequency_range_hz[1],
            y_position + 0.16,
            f"{width_hz:.1f} Hz unavailable",
            ha="right",
            va="bottom",
            fontsize=8,
            color=colour,
        )
    geometry_axis.set_yticks([0.0, 1.0], ["MNE defaults", "decomb"])
    geometry_axis.set_ylim(-0.45, 1.5)
    geometry_axis.set_xlabel("frequency (Hz)")
    geometry_axis.set_ylabel("stopbands + transitions")
    geometry_axis.set_xlim(frequency_range_hz)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)
