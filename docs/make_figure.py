#!/usr/bin/env python3
"""Generate the README figures from deterministic simulated continuous EEG."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from decomb import notch, recordings  # noqa: E402
from decomb.config import load_config  # noqa: E402

SOURCE_COLOUR = "#D96C4A"
CLEANED_COLOUR = "#176B87"
TEST_COLOUR = "#2F855A"
STOPBAND_COLOUR = "#C2413B"
TRANSITION_COLOUR = "#E9A23B"
INK = "#17202A"
MUTED_INK = "#5D6873"
GRID_COLOUR = "#D8DEE4"


@dataclass(frozen=True)
class SimulationSettings:
    """Known signals used to demonstrate suppression and off-grid preservation."""

    sampling_frequency_hz: float
    duration_s: float
    n_channels: int
    random_seed: int
    noise_standard_deviation_v: float
    noise_spectral_exponent: float
    comb_amplitude_v: float
    comb_decay_exponent: float
    test_oscillation_hz: float
    test_oscillation_amplitude_v: float
    representative_harmonic: int
    analysis_margin_s: float
    overview_band_hz: tuple[float, float]
    detail_band_hz: tuple[float, float]

    def __post_init__(self) -> None:
        positive = (
            "sampling_frequency_hz",
            "duration_s",
            "n_channels",
            "noise_standard_deviation_v",
            "comb_amplitude_v",
            "test_oscillation_hz",
            "test_oscillation_amplitude_v",
            "representative_harmonic",
            "analysis_margin_s",
        )
        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"simulation.{name} must be finite and positive.")
        if self.noise_spectral_exponent < 0.0:
            raise ValueError("simulation.noise_spectral_exponent must be non-negative.")
        if self.comb_decay_exponent < 0.0:
            raise ValueError("simulation.comb_decay_exponent must be non-negative.")
        for name in ("overview_band_hz", "detail_band_hz"):
            low_hz, high_hz = getattr(self, name)
            if not 0.0 <= low_hz < high_hz < self.sampling_frequency_hz / 2.0:
                raise ValueError(f"simulation.{name} must increase below the Nyquist frequency.")
        if 2.0 * self.analysis_margin_s >= self.duration_s:
            raise ValueError("simulation.analysis_margin_s removes the entire recording.")

    @classmethod
    def from_config(cls, config) -> SimulationSettings:
        block = dict(config.get("simulation") or {})
        known = {entry.name for entry in fields(cls)}
        unknown = set(block) - known
        if unknown:
            raise ValueError(f"Unknown `simulation` setting(s): {sorted(unknown)}.")
        missing = known - set(block)
        if missing:
            raise ValueError(f"Missing `simulation` setting(s): {sorted(missing)}.")
        integers = {"n_channels", "random_seed", "representative_harmonic"}
        ranges = {"overview_band_hz", "detail_band_hz"}
        values = {}
        for name, value in block.items():
            if name in integers:
                values[name] = int(value)
            elif name in ranges:
                values[name] = tuple(float(edge) for edge in value)
            else:
                values[name] = float(value)
        return cls(**values)


@dataclass(frozen=True)
class SimulationResult:
    """Known input, delivered output, and the exact filter plan between them."""

    original: object
    cleaned: object
    plan: notch.HarmonicNotchPlan
    correction: notch.HarmonicNotchSettings
    simulation: SimulationSettings


def build_simulation(config_path: Path) -> SimulationResult:
    """Create deterministic EEG and pass it through the current production transform."""
    config = load_config(config_path)
    correction = notch.HarmonicNotchSettings.from_config(config)
    simulation = SimulationSettings.from_config(config)
    original = _simulate_raw(correction, simulation)
    model = notch.fit_harmonic_model(original, correction)
    plan = notch.plan_harmonic_stopbands(model, correction)
    cleaned = notch.apply_harmonic_notches(
        original,
        plan,
        filter_jobs=correction.filter_jobs,
    )
    _validate_demonstration(plan, correction, simulation)
    return SimulationResult(original, cleaned, plan, correction, simulation)


def _simulate_raw(correction, simulation: SimulationSettings):
    import mne

    sample_count = int(round(simulation.duration_s * simulation.sampling_frequency_hz))
    times_s = np.arange(sample_count) / simulation.sampling_frequency_hz
    generator = np.random.default_rng(simulation.random_seed)
    data = _coloured_noise(generator, simulation, sample_count)

    first_harmonic, last_harmonic = correction.removal_harmonic_range
    channel_scales = generator.uniform(0.85, 1.15, simulation.n_channels)
    for harmonic in range(first_harmonic, last_harmonic + 1):
        frequency_hz = harmonic * correction.nominal_fundamental_hz
        amplitude_v = simulation.comb_amplitude_v * (harmonic / first_harmonic) ** (
            -simulation.comb_decay_exponent
        )
        phases = generator.uniform(0.0, 2.0 * np.pi, simulation.n_channels)
        data += (
            amplitude_v
            * channel_scales[:, np.newaxis]
            * np.sin(2.0 * np.pi * frequency_hz * times_s[np.newaxis, :] + phases[:, np.newaxis])
        )

    test_phases = generator.uniform(0.0, 2.0 * np.pi, simulation.n_channels)
    data += simulation.test_oscillation_amplitude_v * np.sin(
        2.0 * np.pi * simulation.test_oscillation_hz * times_s[np.newaxis, :]
        + test_phases[:, np.newaxis]
    )
    info = mne.create_info(
        [f"EEG {index + 1:03d}" for index in range(simulation.n_channels)],
        simulation.sampling_frequency_hz,
        ch_types="eeg",
    )
    return mne.io.RawArray(data, info, verbose="ERROR")


def _coloured_noise(
    generator: np.random.Generator,
    simulation: SimulationSettings,
    sample_count: int,
) -> np.ndarray:
    white = generator.normal(size=(simulation.n_channels, sample_count))
    frequencies_hz = np.fft.rfftfreq(
        sample_count,
        1.0 / simulation.sampling_frequency_hz,
    )
    reference_hz = max(1.0 / simulation.duration_s, np.finfo(float).eps)
    shaping = np.maximum(frequencies_hz, reference_hz) ** (
        -simulation.noise_spectral_exponent / 2.0
    )
    noise = np.fft.irfft(
        np.fft.rfft(white, axis=-1) * shaping,
        n=sample_count,
        axis=-1,
    )
    noise -= noise.mean(axis=-1, keepdims=True)
    noise /= noise.std(axis=-1, keepdims=True)
    return noise * simulation.noise_standard_deviation_v


def _validate_demonstration(plan, correction, simulation) -> None:
    harmonic = simulation.representative_harmonic
    if (
        not correction.removal_harmonic_range[0]
        <= harmonic
        <= (correction.removal_harmonic_range[1])
    ):
        raise ValueError("simulation.representative_harmonic must lie in removal_harmonic_range.")
    representative = next(
        (stopband for stopband in plan.stopbands if harmonic in stopband.harmonics),
        None,
    )
    if representative is None:
        raise ValueError("The representative harmonic was not supported by the simulation.")
    unavailable = dict(zip(plan.stopbands, plan.unavailable_edges()))[representative]
    if unavailable[0] <= simulation.test_oscillation_hz <= unavailable[1]:
        raise ValueError("The test oscillation lies inside the declared unavailable interval.")


def tone_amplitude_ratio(result: SimulationResult, frequency_hz: float) -> float:
    """Median channel amplitude after correction divided by its known input amplitude."""
    simulation = result.simulation
    margin_samples = int(round(simulation.analysis_margin_s * simulation.sampling_frequency_hz))
    sample_slice = slice(margin_samples, -margin_samples)
    before = _tone_amplitudes(
        result.original.get_data()[:, sample_slice],
        frequency_hz,
        simulation.sampling_frequency_hz,
    )
    after = _tone_amplitudes(
        result.cleaned.get_data()[:, sample_slice],
        frequency_hz,
        simulation.sampling_frequency_hz,
    )
    return float(np.median(after / before))


def _tone_amplitudes(
    data: np.ndarray,
    frequency_hz: float,
    sampling_frequency_hz: float,
) -> np.ndarray:
    times_s = np.arange(data.shape[-1]) / sampling_frequency_hz
    basis = np.column_stack(
        (
            np.sin(2.0 * np.pi * frequency_hz * times_s),
            np.cos(2.0 * np.pi * frequency_hz * times_s),
        )
    )
    coefficients, *_ = np.linalg.lstsq(basis, data.T, rcond=None)
    return np.linalg.norm(coefficients, axis=0)


def _spectra(result: SimulationResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import mne

    picks = mne.pick_types(result.original.info, eeg=True, exclude=())
    frequencies_hz, before_power = recordings.psd(
        result.original,
        picks,
        result.correction,
    )
    after_frequencies_hz, after_power = recordings.psd(
        result.cleaned,
        picks,
        result.correction,
    )
    if not np.array_equal(frequencies_hz, after_frequencies_hz):
        raise ValueError("The simulated before/after spectra use different grids.")
    tiny = np.finfo(float).tiny
    before_db = 10.0 * np.log10(np.maximum(np.median(before_power, axis=0), tiny))
    after_db = 10.0 * np.log10(np.maximum(np.median(after_power, axis=0), tiny))
    return frequencies_hz, before_db, after_db


def render_figures(
    result: SimulationResult,
    overview_path: Path,
    detail_path: Path,
) -> None:
    """Render the overview and audit-detail figures used in the README."""
    frequencies_hz, before_db, after_db = _spectra(result)
    _render_overview(
        result,
        frequencies_hz,
        before_db,
        after_db,
        overview_path,
    )
    _render_detail(
        result,
        frequencies_hz,
        before_db,
        after_db,
        detail_path,
    )


def _render_overview(
    result,
    frequencies_hz,
    before_db,
    after_db,
    output_path: Path,
) -> None:
    low_hz, high_hz = result.simulation.overview_band_hz
    inside = (frequencies_hz >= low_hz) & (frequencies_hz <= high_hz)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12.5, 7.0),
        sharex=True,
        height_ratios=(2.2, 1.0),
    )
    axes[0].plot(
        frequencies_hz[inside],
        before_db[inside],
        color=SOURCE_COLOUR,
        linewidth=1.15,
        label="simulated input",
    )
    axes[0].plot(
        frequencies_hz[inside],
        after_db[inside],
        color=CLEANED_COLOUR,
        linewidth=0.9,
        label="after automatic harmonic notches",
    )
    axes[0].set_ylabel("median PSD (dB re 1 V²/Hz)")
    axes[0].set_title("Evidence-bounded removal of a simulated harmonic comb", loc="left")
    axes[0].legend(frameon=False, ncols=2, loc="lower left")

    change_db = after_db - before_db
    axes[1].plot(
        frequencies_hz[inside],
        change_db[inside],
        color=CLEANED_COLOUR,
        linewidth=0.9,
    )
    axes[1].axhline(0.0, color=MUTED_INK, linewidth=0.7)
    axes[1].set_ylabel("change (dB)")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylim(min(-45.0, float(np.nanmin(change_db[inside]))), 3.0)

    harmonic_count = sum(len(stopband.harmonics) for stopband in result.plan.stopbands)
    test_change_db = 20.0 * np.log10(
        tone_amplitude_ratio(result, result.simulation.test_oscillation_hz)
    )
    axes[0].text(
        0.99,
        0.97,
        f"{harmonic_count} supported harmonics\noff-grid test tone: {test_change_db:+.2f} dB",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        color=INK,
        bbox={"facecolor": "white", "edgecolor": GRID_COLOUR, "pad": 6},
    )
    _style_axes(axes)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _render_detail(
    result,
    frequencies_hz,
    before_db,
    after_db,
    output_path: Path,
) -> None:
    simulation = result.simulation
    harmonic = simulation.representative_harmonic
    stopband_index = next(
        index
        for index, stopband in enumerate(result.plan.stopbands)
        if harmonic in stopband.harmonics
    )
    stopband = result.plan.stopbands[stopband_index]
    unavailable = result.plan.unavailable_edges()[stopband_index]
    low_hz, high_hz = simulation.detail_band_hz
    inside = (frequencies_hz >= low_hz) & (frequencies_hz <= high_hz)

    figure, axis = plt.subplots(figsize=(12.5, 4.8))
    axis.axvspan(
        unavailable[0],
        unavailable[1],
        color=TRANSITION_COLOUR,
        alpha=0.22,
        label="declared unavailable (stopband + transitions)",
    )
    axis.axvspan(
        stopband.low_hz,
        stopband.high_hz,
        color=STOPBAND_COLOUR,
        alpha=0.28,
        label="requested stopband",
    )
    axis.plot(
        frequencies_hz[inside],
        before_db[inside],
        color=SOURCE_COLOUR,
        linewidth=1.4,
        label="simulated input",
    )
    axis.plot(
        frequencies_hz[inside],
        after_db[inside],
        color=CLEANED_COLOUR,
        linewidth=1.1,
        label="after correction",
    )
    axis.axvline(
        simulation.test_oscillation_hz,
        color=TEST_COLOUR,
        linestyle="--",
        linewidth=1.1,
        label="off-grid test oscillation",
    )

    harmonic_ratio = tone_amplitude_ratio(
        result,
        harmonic * result.correction.nominal_fundamental_hz,
    )
    test_ratio = tone_amplitude_ratio(result, simulation.test_oscillation_hz)
    axis.text(
        0.99,
        0.97,
        f"harmonic {harmonic}: {20.0 * np.log10(harmonic_ratio):+.1f} dB\n"
        f"off-grid oscillation: {20.0 * np.log10(test_ratio):+.2f} dB",
        transform=axis.transAxes,
        ha="right",
        va="top",
        color=INK,
        bbox={"facecolor": "white", "edgecolor": GRID_COLOUR, "pad": 6},
    )
    axis.set_xlim(low_hz, high_hz)
    axis.set_xlabel("frequency (Hz)")
    axis.set_ylabel("median PSD (dB re 1 V²/Hz)")
    axis.set_title(
        "A removed harmonic is unavailable; a nearby off-grid oscillation remains",
        loc="left",
    )
    axis.legend(frameon=False, ncols=2, loc="lower right", fontsize=8.5)
    _style_axes((axis,))
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _style_axes(axes) -> None:
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(GRID_COLOUR)
        axis.grid(axis="y", color=GRID_COLOUR, linewidth=0.55, alpha=0.65)
        axis.tick_params(colors=MUTED_INK)
        axis.xaxis.label.set_color(INK)
        axis.yaxis.label.set_color(INK)
        axis.title.set_color(INK)


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("simulated_readme.yaml"),
    )
    parser.add_argument(
        "--overview",
        type=Path,
        default=repository / "docs" / "psd_before_after.png",
    )
    parser.add_argument(
        "--detail",
        type=Path,
        default=repository / "docs" / "harmonic_detail.png",
    )
    arguments = parser.parse_args()

    result = build_simulation(arguments.config)
    render_figures(result, arguments.overview, arguments.detail)
    harmonic_hz = (
        result.simulation.representative_harmonic * result.correction.nominal_fundamental_hz
    )
    test_oscillation_change_db = 20.0 * np.log10(
        tone_amplitude_ratio(result, result.simulation.test_oscillation_hz)
    )
    print(f"wrote {arguments.overview}")
    print(f"wrote {arguments.detail}")
    print(
        f"representative harmonic amplitude: "
        f"{20.0 * np.log10(tone_amplitude_ratio(result, harmonic_hz)):+.2f} dB"
    )
    print(f"off-grid test amplitude: {test_oscillation_change_db:+.2f} dB")


if __name__ == "__main__":
    main()
