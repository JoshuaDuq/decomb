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


_SURFACE_COLOUR = "#FCFCFB"
_GRID_COLOUR = "#E3E2DD"
_MUTED_COLOUR = "#52514E"
_SOURCE_COLOUR = "#EB6834"
_DECOMB_COLOUR = "#0B0B0B"
_TRADITIONAL_COLOUR = "#2A78D6"
_FONT_STACK = ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")

# One window this wide spans several detected lines while still rendering a single
# stopband wider than a pixel, which a 0-100 Hz axis cannot do at README width.
_DETAIL_WINDOW_HZ = 6.0

# A window the reference arm has covered above this share shows one solid block rather
# than individual stopbands, so it cannot carry the per-notch geometry.
_DETAIL_SATURATION_SHARE = 0.95


def _covered_width_hz(
    edges: Sequence[tuple[float, float]],
    low_hz: float,
    high_hz: float,
) -> float:
    """Total width of non-overlapping intervals inside a range."""
    return sum(
        max(0.0, min(high_hz, upper_hz) - max(low_hz, lower_hz)) for lower_hz, upper_hz in edges
    )


def _available_mask(
    plan: notch.HarmonicNotchPlan,
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    """Grid points that survive a plan: outside every stopband and every transition."""
    mask = np.ones(frequencies_hz.size, dtype=bool)
    for low_hz, high_hz in plan.unavailable_edges():
        mask &= ~((frequencies_hz >= low_hz) & (frequencies_hz <= high_hz))
    return mask


def _detail_window_hz(
    comparison: NotchComparison,
    frequency_range_hz: tuple[float, float],
) -> tuple[float, float]:
    """Pick the window where the two geometries differ most and both stay legible."""
    low_hz, high_hz = frequency_range_hz
    if high_hz - low_hz <= _DETAIL_WINDOW_HZ:
        return (low_hz, high_hz)

    decomb_edges = comparison.decomb_plan.unavailable_edges()
    traditional_edges = comparison.traditional_plan.unavailable_edges()
    best_start_hz: float | None = None
    best_score_hz = 0.0
    for start_hz in np.arange(low_hz, high_hz - _DETAIL_WINDOW_HZ, 0.25):
        stop_hz = start_hz + _DETAIL_WINDOW_HZ
        traditional_hz = _covered_width_hz(traditional_edges, start_hz, stop_hz)
        if traditional_hz > _DETAIL_SATURATION_SHARE * _DETAIL_WINDOW_HZ:
            continue
        score_hz = traditional_hz - _covered_width_hz(decomb_edges, start_hz, stop_hz)
        if score_hz > best_score_hz:
            best_score_hz = score_hz
            best_start_hz = float(start_hz)
    if best_start_hz is None:
        return (high_hz - _DETAIL_WINDOW_HZ, high_hz)
    return (best_start_hz, best_start_hz + _DETAIL_WINDOW_HZ)


def _style_axis(axis) -> None:
    """Recessive chrome: hairline rules, no top or right spine, muted ticks."""
    axis.set_facecolor(_SURFACE_COLOUR)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(_GRID_COLOUR)
        axis.spines[side].set_linewidth(0.8)
    axis.tick_params(colors=_MUTED_COLOUR, labelsize=9, length=3, width=0.8)


def _draw_retained_row(
    axis,
    frequencies_hz: np.ndarray,
    spectrum_db: np.ndarray,
    mask: np.ndarray,
    colour: str,
) -> None:
    """Draw a spectrum only where it survives, so the gaps carry the message.

    Drawing the notches themselves is what turns a 0-100 Hz axis into a barcode, and
    the dive carries nothing: what a reader needs is which frequencies are left.
    """
    axis.plot(
        frequencies_hz,
        np.where(mask, spectrum_db, np.nan),
        color=colour,
        linewidth=0.9,
        solid_capstyle="round",
    )
    # A surviving sliver can be one grid point wide, and a one-point run has no length
    # to stroke, so those would silently vanish from the row that most depends on them.
    isolated = mask & ~np.r_[False, mask[:-1]] & ~np.r_[mask[1:], False]
    if isolated.any():
        axis.plot(
            frequencies_hz[isolated],
            spectrum_db[isolated],
            linestyle="none",
            marker=".",
            markersize=1.8,
            markeredgewidth=0.0,
            color=colour,
        )


def _window_geometry(
    plan: notch.HarmonicNotchPlan,
    detail_range_hz: tuple[float, float],
) -> tuple[list[tuple[float, float]], str]:
    """Intervals inside the window, with a plain description of their size and spacing."""
    low_hz, high_hz = detail_range_hz
    spans = [
        (max(low_hz, lower_hz), min(high_hz, upper_hz))
        for lower_hz, upper_hz in plan.unavailable_edges()
        if min(high_hz, upper_hz) > max(low_hz, lower_hz)
    ]
    if not spans:
        return spans, "nothing removed here"

    widths_hz = [upper_hz - lower_hz for lower_hz, upper_hz in spans]
    gaps_hz = [later[0] - earlier[1] for earlier, later in zip(spans, spans[1:])]
    description = f"{len(spans)} stopband{'s' if len(spans) > 1 else ''} of "
    description += f"{np.median(widths_hz):.2f} Hz"
    if gaps_hz:
        description += f", leaving {np.median(gaps_hz):.2f} Hz gaps"
    return spans, description


def _draw_geometry_panel(
    axis,
    comparison: NotchComparison,
    detail_range_hz: tuple[float, float],
) -> None:
    """Why the rows above differ: both filters over one window, drawn to scale.

    Deliberately geometry and not spectra.  Every arm reproduces the same spectrum
    wherever it survives, so overlaying the three in one window renders them as a single
    curve that changes colour, which reads as one recoloured line rather than as three.
    """
    rows = (
        (comparison.decomb_plan, "decomb", 0.3, _DECOMB_COLOUR),
        (comparison.traditional_plan, "MNE defaults", -0.3, _TRADITIONAL_COLOUR),
    )
    bar_height = 0.26
    for plan, _name, y_position, colour in rows:
        spans, description = _window_geometry(plan, detail_range_hz)
        axis.broken_barh(
            [(low_hz, high_hz - low_hz) for low_hz, high_hz in spans],
            (y_position - bar_height / 2.0, bar_height),
            facecolor=colour,
            edgecolor=_SURFACE_COLOUR,
            linewidth=0.4,
        )
        axis.annotate(
            description,
            xy=(detail_range_hz[1], y_position + bar_height / 2.0),
            xytext=(0.0, 5.0),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=10,
            color=colour,
        )

    axis.set_yticks([y_position for _, _, y_position, _ in rows], [name for _, name, _, _ in rows])
    for tick_label, (_, _, _, colour) in zip(axis.get_yticklabels(), rows):
        tick_label.set_color(colour)
        tick_label.set_fontsize(10)
    axis.set_ylim(-0.56, 0.62)
    axis.set_xlim(detail_range_hz)
    axis.tick_params(axis="y", length=0.0)
    axis.spines["left"].set_visible(False)
    axis.set_xlabel("frequency (Hz)", fontsize=9, color=_MUTED_COLOUR)


def figure_notch_comparison(
    comparison: NotchComparison,
    path: Path,
    *,
    recording_description: str,
) -> None:
    """Plot the spectrum each notch geometry leaves behind, and one window up close."""
    import matplotlib.pyplot as plt

    frequencies_hz = comparison.frequencies_hz
    frequency_range_hz = (float(frequencies_hz[0]), float(frequencies_hz[-1]))
    span_hz = frequency_range_hz[1] - frequency_range_hz[0]
    detail_range_hz = _detail_window_hz(comparison, frequency_range_hz)
    source_db = to_db(comparison.source_psd)

    # The retained samples sit on the uncorrected spectrum to within hundredths of a dB,
    # so overplotting the three arms would hide two of them.  Small multiples of one
    # curve turn the comparison into what it actually is: how much of it is left.
    rows = (
        (source_db, np.ones(frequencies_hz.size, bool), _SOURCE_COLOUR, "before correction", None),
        (
            to_db(comparison.decomb_psd),
            _available_mask(comparison.decomb_plan, frequencies_hz),
            _DECOMB_COLOUR,
            "after decomb measured-width notches",
            comparison.decomb_plan,
        ),
        (
            to_db(comparison.traditional_psd),
            _available_mask(comparison.traditional_plan, frequencies_hz),
            _TRADITIONAL_COLOUR,
            "after MNE default notch geometry",
            comparison.traditional_plan,
        ),
    )

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": _FONT_STACK}):
        figure = plt.figure(figsize=(11.5, 7.4), layout="constrained")
        figure.set_facecolor(_SURFACE_COLOUR)
        grid = figure.add_gridspec(4, 1, height_ratios=(1.0, 1.0, 1.0, 0.8), hspace=0.06)

        retained_axes = []
        for index, (spectrum_db, mask, colour, name, plan) in enumerate(rows):
            shared = {"sharex": retained_axes[0], "sharey": retained_axes[0]} if index else {}
            axis = figure.add_subplot(grid[index], **shared)
            retained_axes.append(axis)
            _style_axis(axis)
            axis.grid(axis="y", color=_GRID_COLOUR, linewidth=0.6)
            axis.set_axisbelow(True)
            _draw_retained_row(axis, frequencies_hz, spectrum_db, mask, colour)

            available_hz = (
                span_hz
                if plan is None
                else span_hz - unavailable_width_hz(plan, frequency_range_hz)
            )
            # Both row labels sit on the title baseline above the axes, so the curve
            # keeps the whole row instead of giving up headroom to text.
            axis.set_title(name, fontsize=10, color=colour, loc="left", pad=6.0)
            axis.annotate(
                f"{available_hz:.1f} Hz of {span_hz:.0f} Hz left to analyse",
                xy=(1.0, 1.0),
                xycoords="axes fraction",
                xytext=(0.0, 6.0),
                textcoords="offset points",
                ha="right",
                va="baseline",
                fontsize=10,
                color=colour,
            )
            axis.yaxis.set_major_locator(plt.MaxNLocator(3))
            if index < len(rows) - 1:
                axis.tick_params(labelbottom=False)

        retained_axes[0].set_xlim(frequency_range_hz)
        retained_axes[0].set_ylim(source_db.min() - 4.0, source_db.max() + 4.0)
        retained_axes[1].set_ylabel("median PSD (dB re 1 V²/Hz)", fontsize=9, color=_MUTED_COLOUR)
        retained_axes[-1].set_xlabel("frequency (Hz)", fontsize=9, color=_MUTED_COLOUR)

        # The panel below is a different frequency scale, so mark the window it covers
        # on every row.  Without it the reader has no cue that the axes are not shared.
        for axis in retained_axes:
            axis.axvspan(*detail_range_hz, color=_MUTED_COLOUR, alpha=0.08, linewidth=0.0, zorder=0)
        retained_axes[-1].annotate(
            "the window below",
            xy=(detail_range_hz[1], 0.06),
            xycoords=("data", "axes fraction"),
            xytext=(7.0, 0.0),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9,
            color=_MUTED_COLOUR,
        )

        geometry_axis = figure.add_subplot(grid[3])
        _style_axis(geometry_axis)
        _draw_geometry_panel(geometry_axis, comparison, detail_range_hz)
        geometry_axis.set_title(
            "why: the filters themselves, over one 6 Hz window at "
            f"{detail_range_hz[0]:.1f}–{detail_range_hz[1]:.1f} Hz, drawn to scale",
            fontsize=9.5,
            color=_MUTED_COLOUR,
            loc="left",
            pad=10.0,
        )

        figure.suptitle(
            "Both arms remove the same detected lines. Only the filter geometry differs.",
            fontsize=13.5,
            color=_DECOMB_COLOUR,
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        # The recording is recorded in PNG text chunks rather than printed under the
        # figure: the provenance travels with the file without spending a caption on it.
        figure.savefig(
            path,
            dpi=200,
            facecolor=_SURFACE_COLOUR,
            metadata={
                "Title": "decomb measured-width notches versus MNE default notch geometry",
                "Description": (
                    f"{recording_description} · real EEG · identical detected centres · "
                    "each spectrum is drawn only where that arm leaves the band usable"
                ),
            },
        )
        plt.close(figure)
