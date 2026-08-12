"""Automatic recording-specific FIR notches for supported sinusoidal lines.

The transform makes no claim to recover neural activity at a removed frequency. Its
manifest therefore records every stopband and transition as unavailable for inference.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from decomb import __version__, lines, recordings, spectral

FIR_DESIGN = "firwin"
FIR_PAD = "reflect_limited"
FIR_PHASE = "zero"
FIR_WINDOW = "hamming"
MANIFEST_NAME = "line_notch_manifest.tsv"
VERIFICATION_NAME = "line_notch_verification.tsv"


@dataclass(frozen=True)
class HarmonicNotchSettings:
    """The stationarity horizon and study frequency range supplied by the user."""

    estimation_window_s: float
    familywise_error_rate: float
    frequency_range_hz: tuple[float, float]

    def __post_init__(self) -> None:
        if not np.isfinite(self.estimation_window_s) or self.estimation_window_s <= 0.0:
            raise ValueError("removal.estimation_window_s must be finite and positive.")
        if not np.isfinite(self.familywise_error_rate) or not (
            0.0 < self.familywise_error_rate < 1.0
        ):
            raise ValueError(
                "removal.familywise_error_rate must lie strictly between zero and one."
            )
        values = np.asarray(self.frequency_range_hz, dtype=float)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError(
                "removal.frequency_range_hz must contain two finite values."
            )
        low_hz, high_hz = (float(value) for value in values)
        if not 0.0 <= low_hz < high_hz:
            raise ValueError(
                "removal.frequency_range_hz must contain increasing non-negative values."
            )

    @property
    def estimation_overlap(self) -> float:
        """Hann's constant-overlap-add hop, derived from the window itself."""
        return 0.5

    @property
    def spectral_resolution_hz(self) -> float:
        return spectral.hann_resolution_hz(self.estimation_window_s)

    @property
    def transition_bandwidth_hz(self) -> float:
        """Total notch transition width across both edges."""
        return 3.3 / self.estimation_window_s

    @property
    def per_edge_transition_bandwidth_hz(self) -> float:
        """Transition width MNE uses to derive the automatic FIR length."""
        return self.transition_bandwidth_hz / 2.0

    @property
    def frequency_bin_width_hz(self) -> float:
        return 1.0 / self.estimation_window_s

    @classmethod
    def from_config(cls, config) -> HarmonicNotchSettings:
        block = dict(config.get("removal") or {})
        known = {entry.name for entry in fields(cls)}
        unknown = set(block) - known
        if unknown:
            raise ValueError(
                f"Unknown `removal` setting(s): {sorted(unknown)}. "
                f"Known settings are {sorted(known)}."
            )
        missing = known - set(block)
        if missing:
            raise ValueError(f"Missing `removal` setting(s): {sorted(missing)}.")
        try:
            frequency_range = tuple(float(value) for value in block["frequency_range_hz"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                "removal.frequency_range_hz must contain two numeric values."
            ) from error
        return cls(
            estimation_window_s=float(block["estimation_window_s"]),
            familywise_error_rate=float(block["familywise_error_rate"]),
            frequency_range_hz=frequency_range,
        )


@dataclass(frozen=True)
class HarmonicStopband:
    """One measured comb or isolated-line interval unavailable after filtering."""

    harmonics: tuple[int, ...]
    low_hz: float
    high_hz: float
    kind: str = "comb"

    def __post_init__(self) -> None:
        if self.kind not in {"comb", "isolated", "mixed"}:
            raise ValueError("Stopband kind must be comb, isolated, or mixed.")
        if tuple(sorted(set(self.harmonics))) != self.harmonics:
            raise ValueError("Stopband harmonics must be sorted unique integers.")
        if self.kind == "comb" and not self.harmonics:
            raise ValueError("A comb stopband requires at least one harmonic.")
        if self.kind == "isolated" and self.harmonics:
            raise ValueError("An isolated stopband cannot claim a comb harmonic.")
        if any(harmonic < 1 for harmonic in self.harmonics):
            raise ValueError("Stopband harmonics must be positive.")
        if not np.all(np.isfinite((self.low_hz, self.high_hz))):
            raise ValueError("Stopband edges must be finite.")
        if self.low_hz <= 0.0 or self.high_hz <= self.low_hz:
            raise ValueError("Stopband edges must be positive and increasing.")

    @property
    def centre_hz(self) -> float:
        return (self.low_hz + self.high_hz) / 2.0

    @property
    def width_hz(self) -> float:
        return self.high_hz - self.low_hz


@dataclass(frozen=True)
class HarmonicNotchPlan:
    """The complete fixed FIR geometry for one recording."""

    stopbands: tuple[HarmonicStopband, ...]
    transition_bandwidth_hz: float

    def __post_init__(self) -> None:
        if not self.stopbands:
            raise ValueError("A line-notch plan requires at least one supported frequency.")
        if not np.isfinite(self.transition_bandwidth_hz) or self.transition_bandwidth_hz <= 0.0:
            raise ValueError("The transition bandwidth must be finite and positive.")
        if any(
            later.low_hz < earlier.high_hz
            for earlier, later in zip(
                self.stopbands[:-1],
                self.stopbands[1:],
                strict=True,
            )
        ):
            raise ValueError("Harmonic stopbands must be sorted and non-overlapping.")

    def unavailable_edges(self) -> tuple[tuple[float, float], ...]:
        """Intervals unsuitable for inference, including the FIR transitions."""
        half_transition_hz = self.transition_bandwidth_hz / 2.0
        return tuple(
            (
                stopband.low_hz - half_transition_hz,
                stopband.high_hz + half_transition_hz,
            )
            for stopband in self.stopbands
        )


@dataclass(frozen=True)
class ChannelNotchPlan:
    """One EEG channel and the complete FIR geometry supported for it."""

    channel_name: str
    geometry: HarmonicNotchPlan

    def __post_init__(self) -> None:
        if not self.channel_name.strip():
            raise ValueError("A channel notch plan requires a channel name.")


@dataclass(frozen=True)
class HarmonicFilterDesign:
    """Exact length and measured response of MNE's FIR design."""

    length_samples: int
    length_s: float
    minimum_stopband_attenuation_db: float
    maximum_passband_deviation_db: float

    def manifest_fields(self) -> dict[str, float | int]:
        """Machine-readable filter properties repeated for each recording row."""
        return {
            "fir_filter_length_samples": self.length_samples,
            "fir_filter_length_s": self.length_s,
            "fir_minimum_stopband_attenuation_db": (
                self.minimum_stopband_attenuation_db
            ),
            "fir_maximum_passband_deviation_db": self.maximum_passband_deviation_db,
        }


def _mne_passband_edges(
    plan: HarmonicNotchPlan,
) -> tuple[np.ndarray, np.ndarray]:
    """MNE band-stop passband edges around the declared unavailable intervals."""
    half_transition_hz = plan.transition_bandwidth_hz / 2.0
    low_pass_edges_hz = np.array(
        [stopband.low_hz - half_transition_hz for stopband in plan.stopbands],
        dtype=float,
    )
    high_pass_edges_hz = np.array(
        [stopband.high_hz + half_transition_hz for stopband in plan.stopbands],
        dtype=float,
    )
    return low_pass_edges_hz, high_pass_edges_hz


def characterize_harmonic_filter(
    sampling_frequency_hz: float,
    plan: HarmonicNotchPlan,
) -> HarmonicFilterDesign:
    """Design MNE's exact FIR coefficients and summarize their response."""
    import mne
    from scipy.fft import next_fast_len
    from scipy.signal import freqz

    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0.0:
        raise ValueError("The sampling frequency must be finite and positive.")
    nyquist_hz = sampling_frequency_hz / 2.0
    unavailable_edges = plan.unavailable_edges()
    if unavailable_edges[0][0] <= 0.0 or unavailable_edges[-1][1] >= nyquist_hz:
        raise ValueError("The notch transitions must lie strictly inside (0, Nyquist).")

    low_pass_edges_hz, high_pass_edges_hz = _mne_passband_edges(plan)
    per_edge_transition_hz = plan.transition_bandwidth_hz / 2.0
    coefficients = mne.filter.create_filter(
        data=None,
        sfreq=sampling_frequency_hz,
        l_freq=high_pass_edges_hz,
        h_freq=low_pass_edges_hz,
        filter_length="auto",
        l_trans_bandwidth=per_edge_transition_hz,
        h_trans_bandwidth=per_edge_transition_hz,
        method="fir",
        phase=FIR_PHASE,
        fir_window=FIR_WINDOW,
        fir_design=FIR_DESIGN,
        verbose="ERROR",
    )

    response_points = next_fast_len(max(16_384, 8 * len(coefficients)))
    frequencies_hz, response = freqz(
        coefficients,
        worN=response_points,
        fs=sampling_frequency_hz,
    )
    boundary_frequencies_hz = np.array(
        [
            0.0,
            nyquist_hz,
            *(edge for stopband in plan.stopbands for edge in (stopband.low_hz, stopband.high_hz)),
            *(edge for unavailable in unavailable_edges for edge in unavailable),
        ],
        dtype=float,
    )
    _, boundary_response = freqz(
        coefficients,
        worN=boundary_frequencies_hz,
        fs=sampling_frequency_hz,
    )
    frequencies_hz = np.concatenate((frequencies_hz, boundary_frequencies_hz))
    response = np.concatenate((response, boundary_response))
    gain_db = 20.0 * np.log10(np.maximum(np.abs(response), np.finfo(float).tiny))

    inside_stopband = np.zeros(frequencies_hz.shape, dtype=bool)
    unavailable = np.zeros(frequencies_hz.shape, dtype=bool)
    for stopband, unavailable_interval in zip(
        plan.stopbands,
        unavailable_edges,
        strict=True,
    ):
        inside_stopband |= (frequencies_hz >= stopband.low_hz) & (
            frequencies_hz <= stopband.high_hz
        )
        unavailable |= (frequencies_hz >= unavailable_interval[0]) & (
            frequencies_hz <= unavailable_interval[1]
        )

    return HarmonicFilterDesign(
        length_samples=len(coefficients),
        length_s=len(coefficients) / sampling_frequency_hz,
        minimum_stopband_attenuation_db=float(-np.max(gain_db[inside_stopband])),
        maximum_passband_deviation_db=float(np.max(np.abs(gain_db[~unavailable]))),
    )


def observed_line_intervals(model, settings) -> list[HarmonicStopband]:
    """Resolution-limited intervals around statistically supported frequencies."""
    location_uncertainty_hz = settings.frequency_bin_width_hz / 2.0
    grouped: dict[int | None, list[float]] = {}
    isolated_index = -1
    for line in model.lines:
        key = line.harmonic
        if key is None:
            key = isolated_index
            isolated_index -= 1
        grouped.setdefault(key, []).append(line.position_hz)

    intervals = []
    for label, positions_hz in grouped.items():
        low_hz = min(positions_hz) - location_uncertainty_hz
        high_hz = max(positions_hz) + location_uncertainty_hz
        centre_hz = (low_hz + high_hz) / 2.0
        if high_hz - low_hz < settings.spectral_resolution_hz:
            half_width_hz = settings.spectral_resolution_hz / 2.0
            low_hz = centre_hz - half_width_hz
            high_hz = centre_hz + half_width_hz
        harmonic = label if label >= 1 else None
        intervals.append(
            HarmonicStopband(
                () if harmonic is None else (harmonic,),
                low_hz,
                high_hz,
                kind="isolated" if harmonic is None else "comb",
            )
        )
    return intervals


def _merge_stopbands(
    stopbands: list[HarmonicStopband],
    *,
    minimum_gap_hz: float,
) -> tuple[HarmonicStopband, ...]:
    """Merge intervals without enough passband for their filter transitions."""
    merged: list[HarmonicStopband] = []
    for stopband in sorted(stopbands, key=lambda band: band.low_hz):
        if not merged or stopband.low_hz > merged[-1].high_hz + minimum_gap_hz:
            merged.append(stopband)
            continue
        previous = merged[-1]
        merged[-1] = HarmonicStopband(
            harmonics=tuple(sorted((*previous.harmonics, *stopband.harmonics))),
            low_hz=previous.low_hz,
            high_hz=max(previous.high_hz, stopband.high_hz),
            kind=(previous.kind if previous.kind == stopband.kind else "mixed"),
        )
    return tuple(merged)


def plan_harmonic_stopbands(model, settings) -> HarmonicNotchPlan:
    """Build the narrowest plan justified by measured harmonic positions."""
    transition_bandwidth_hz = settings.transition_bandwidth_hz
    stopbands = _merge_stopbands(
        observed_line_intervals(model, settings),
        minimum_gap_hz=transition_bandwidth_hz,
    )
    return HarmonicNotchPlan(stopbands, transition_bandwidth_hz)


def plan_channel_notches(
    model: lines.ArtifactModel,
    settings: HarmonicNotchSettings,
) -> tuple[ChannelNotchPlan, ...]:
    """Build the independently supported FIR geometry for each affected channel."""
    return tuple(
        ChannelNotchPlan(
            channel.channel_name,
            plan_harmonic_stopbands(channel, settings),
        )
        for channel in model.channels
    )


def eeg_channel_names(raw) -> tuple[str, ...]:
    """EEG channel names in the order decomb tests and filters them."""
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    if len(picks) == 0:
        raise ValueError("Line detection requires at least one EEG channel.")
    return tuple(raw.ch_names[index] for index in picks)


def _thomson_f_p_values(raw, settings) -> tuple[np.ndarray, np.ndarray]:
    """Raw Thomson F-test p-values for every EEG channel, correction-independent."""
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    if len(picks) == 0:
        raise ValueError("Line detection requires at least one EEG channel.")
    data = raw.get_data(picks=picks)
    if not np.all(np.isfinite(data)):
        raise ValueError("EEG data must contain only finite values.")
    bounds = recordings.valid_window_bounds(
        raw,
        window_s=settings.estimation_window_s,
        overlap=settings.estimation_overlap,
    )
    windows = np.stack(
        [data[:, start:stop] for start, stop in bounds],
        axis=0,
    )
    sampling_frequency_hz = float(raw.info["sfreq"])
    maximum_hz = min(
        settings.frequency_range_hz[1],
        float(np.nextafter(sampling_frequency_hz / 2.0, 0.0)),
    )
    return lines.thomson_f_p_values(
        windows,
        sampling_frequency_hz,
        frequency_range_hz=(settings.frequency_range_hz[0], maximum_hz),
    )


def detect_channel_lines(
    raw,
    settings,
    *,
    correction: str = "holm",
) -> lines.LineDetectionResult:
    """Thomson F-test detections for every EEG channel, with a chosen correction.

    ``correction`` defaults to Holm, decomb's real pipeline. Validation studies pass
    ``"bonferroni"`` or ``"none"`` to compare alternative per-channel multiplicity
    procedures on the identical tests and windows.
    """
    frequencies_hz, p_values = _thomson_f_p_values(raw, settings)
    return lines.detect_lines_from_p_values(
        frequencies_hz,
        p_values,
        familywise_error_rate=settings.familywise_error_rate,
        correction=correction,
    )


def detect_channel_lines_every_correction(
    raw,
    settings,
) -> dict[str, lines.LineDetectionResult]:
    """Detections for every correction procedure from one shared Thomson F-test pass.

    A cohort-scale ablation comparing Holm, Bonferroni, and uncorrected detection would
    otherwise repeat the dominant cost -- the Thomson F-test itself -- once per
    correction. This computes it once and reuses it for all of :data:`lines.CORRECTIONS`.
    """
    frequencies_hz, p_values = _thomson_f_p_values(raw, settings)
    return {
        correction: lines.detect_lines_from_p_values(
            frequencies_hz,
            p_values,
            familywise_error_rate=settings.familywise_error_rate,
            correction=correction,
        )
        for correction in lines.CORRECTIONS
    }


def fit_harmonic_model(raw, settings, *, correction: str = "holm"):
    """Fit channel-level corrected Thomson lines and descriptive harmonics."""
    result = detect_channel_lines(raw, settings, correction=correction)
    return lines.build_artifact_model(
        result,
        channel_names=eeg_channel_names(raw),
        frequency_bin_width_hz=settings.frequency_bin_width_hz,
        spectral_resolution_hz=settings.spectral_resolution_hz,
        familywise_error_rate=settings.familywise_error_rate,
    )


def apply_harmonic_notches(raw, plan: HarmonicNotchPlan):
    """Return a copy with the evidence-bounded line intervals removed from EEG."""
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    if len(picks) == 0:
        raise ValueError("Harmonic notching requires at least one EEG channel.")
    filtered = raw.copy()
    _apply_harmonic_notches(filtered, plan, picks)
    return filtered


def apply_channel_notches(raw, plans: Sequence[ChannelNotchPlan]):
    """Apply each statistically supported geometry only to its EEG channel."""
    import mne

    channel_plans = tuple(plans)
    if not channel_plans:
        return raw.copy()
    channel_names = [plan.channel_name for plan in channel_plans]
    if len(channel_names) != len(set(channel_names)):
        raise ValueError("Each EEG channel may have only one notch plan.")

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=())
    eeg_names = {raw.ch_names[index] for index in eeg_picks}
    invalid = set(channel_names) - eeg_names
    if invalid:
        raise ValueError(f"Notch plans reference non-EEG channels: {sorted(invalid)}.")

    geometries: dict[HarmonicNotchPlan, list[str]] = {}
    for channel_plan in channel_plans:
        geometries.setdefault(channel_plan.geometry, []).append(
            channel_plan.channel_name
        )

    filtered = raw.copy()
    for geometry, names in geometries.items():
        _apply_harmonic_notches(filtered, geometry, names)
    return filtered


def _apply_harmonic_notches(raw, plan, picks):
    """Apply one validated FIR geometry to selected channels in place."""
    nyquist_hz = float(raw.info["sfreq"]) / 2.0
    unavailable_edges = plan.unavailable_edges()
    if unavailable_edges[0][0] <= 0.0:
        raise ValueError("The first line-notch transition reaches 0 Hz.")
    if unavailable_edges[-1][1] >= nyquist_hz:
        raise ValueError(
            f"The last line-notch transition reaches the {nyquist_hz:g} Hz Nyquist limit."
        )

    raw.notch_filter(
        freqs=np.array(
            [stopband.centre_hz for stopband in plan.stopbands],
            dtype=float,
        ),
        picks=picks,
        notch_widths=np.array(
            [stopband.width_hz for stopband in plan.stopbands],
            dtype=float,
        ),
        trans_bandwidth=plan.transition_bandwidth_hz,
        method="fir",
        filter_length="auto",
        phase=FIR_PHASE,
        fir_window=FIR_WINDOW,
        fir_design=FIR_DESIGN,
        pad=FIR_PAD,
        skip_by_annotation=recordings.ACQUISITION_BOUNDARY_ANNOTATIONS,
        n_jobs=-1,
        verbose="ERROR",
    )


def _interval_overlap_hz(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    """Length shared by two closed frequency intervals."""
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def harmonic_exclusion_rows(
    recording: str,
    plan: HarmonicNotchPlan,
    analysed_bands: tuple[tuple[str, float, float], ...],
) -> list[dict[str, float | str]]:
    """Describe exact stopbands and the total analysis bandwidth they invalidate."""
    unavailable_edges = plan.unavailable_edges()
    unavailable_shares = {
        name: sum(
            _interval_overlap_hz(interval, (low_hz, high_hz)) for interval in unavailable_edges
        )
        / (high_hz - low_hz)
        for name, low_hz, high_hz in analysed_bands
    }

    rows: list[dict[str, float | str]] = []
    for stopband, unavailable in zip(
        plan.stopbands,
        unavailable_edges,
        strict=True,
    ):
        row: dict[str, float | str] = {
            "recording": recording,
            "kind": stopband.kind,
            "harmonics": ";".join(str(harmonic) for harmonic in stopband.harmonics),
            "stopband_low_hz": stopband.low_hz,
            "stopband_high_hz": stopband.high_hz,
            "unavailable_low_hz": unavailable[0],
            "unavailable_high_hz": unavailable[1],
            "transition_bandwidth_hz": plan.transition_bandwidth_hz,
        }
        for name, share in unavailable_shares.items():
            row[f"{name}_unavailable_share"] = share
            row[f"{name}_retained_share"] = 1.0 - share
        rows.append(row)
    return rows


def artifact_manifest_rows(
    recording: str,
    model: lines.ArtifactModel,
    plans: Sequence[ChannelNotchPlan],
    analysed_bands: tuple[tuple[str, float, float], ...],
    settings: HarmonicNotchSettings,
) -> list[dict[str, float | int | str]]:
    """Attach channel-local statistical evidence to every channel stopband."""
    channel_models = {channel.channel_name: channel for channel in model.channels}
    channel_plans = tuple(plans)
    if {plan.channel_name for plan in channel_plans} != set(channel_models):
        raise ValueError("Channel plans must cover exactly the affected EEG channels.")
    if not channel_plans:
        return [_null_artifact_manifest_row(recording, model, settings)]

    manifest_rows = []
    for channel_plan in channel_plans:
        channel = channel_models[channel_plan.channel_name]
        rows = harmonic_exclusion_rows(
            recording,
            channel_plan.geometry,
            analysed_bands,
        )
        assigned_positions = []
        fundamental_hz: float | str = (
            "" if channel.fundamental_hz is None else channel.fundamental_hz
        )
        comb_p_value: float | str = (
            ""
            if channel.comb_corrected_p_value is None
            else channel.comb_corrected_p_value
        )
        for row, stopband in zip(
            rows,
            channel_plan.geometry.stopbands,
            strict=True,
        ):
            supported = tuple(
                line
                for line in channel.lines
                if stopband.low_hz <= line.position_hz <= stopband.high_hz
            )
            if not supported:
                raise ValueError(
                    "Every channel stopband must contain a statistically supported line."
                )
            assigned_positions.extend(line.position_hz for line in supported)
            row.update(
                {
                    "outcome": "artifact_detected",
                    "channel": channel.channel_name,
                    "detected_line_frequencies_hz": ";".join(
                        f"{line.position_hz:.17g}" for line in supported
                    ),
                    "detected_line_raw_p_values": ";".join(
                        f"{line.raw_p_value:.17g}" for line in supported
                    ),
                    "detected_line_corrected_p_values": ";".join(
                        f"{line.corrected_p_value:.17g}" for line in supported
                    ),
                    "detected_line_window_indices": ";".join(
                        ",".join(str(index) for index in line.window_indices)
                        for line in supported
                    ),
                    "detected_line_harmonics": ";".join(
                        "" if line.harmonic is None else str(line.harmonic)
                        for line in supported
                    ),
                    "fundamental_hz": fundamental_hz,
                    "comb_corrected_p_value": comb_p_value,
                    "multiple_testing_method": "holm",
                    "familywise_error_rate": settings.familywise_error_rate,
                    "estimation_window_count": model.window_count,
                    "tested_eeg_channel_count": model.channel_count,
                    "detection_test_count_per_channel": (
                        model.test_count_per_channel
                    ),
                    "total_detection_test_count": (
                        model.test_count_per_channel * model.channel_count
                    ),
                }
            )
        expected_positions = [line.position_hz for line in channel.lines]
        if sorted(assigned_positions) != expected_positions:
            raise ValueError(
                "Every statistically supported channel line must belong to one stopband."
            )
        manifest_rows.extend(rows)
    return manifest_rows


def _null_artifact_manifest_row(
    recording: str,
    model: lines.ArtifactModel,
    settings: HarmonicNotchSettings,
) -> dict[str, float | int | str]:
    """Represent a valid null result without inventing filter geometry."""
    return {
        "recording": recording,
        "outcome": "no_artifact_detected",
        "channel": "",
        "kind": "",
        "harmonics": "",
        "stopband_low_hz": "",
        "stopband_high_hz": "",
        "unavailable_low_hz": "",
        "unavailable_high_hz": "",
        "transition_bandwidth_hz": "",
        "detected_line_frequencies_hz": "",
        "detected_line_raw_p_values": "",
        "detected_line_corrected_p_values": "",
        "detected_line_window_indices": "",
        "detected_line_harmonics": "",
        "fundamental_hz": "",
        "comb_corrected_p_value": "",
        "multiple_testing_method": "holm",
        "familywise_error_rate": settings.familywise_error_rate,
        "estimation_window_count": model.window_count,
        "tested_eeg_channel_count": model.channel_count,
        "detection_test_count_per_channel": model.test_count_per_channel,
        "total_detection_test_count": (
            model.test_count_per_channel * model.channel_count
        ),
    }


def harmonic_plan_from_rows(
    rows: Sequence[Mapping[str, object]],
) -> HarmonicNotchPlan:
    """Reconstruct one recording's immutable filter geometry from its manifest."""
    if not rows:
        raise ValueError("A recording's harmonic notch manifest has no stopbands.")
    transition_bandwidths_hz = {float(row["transition_bandwidth_hz"]) for row in rows}
    if len(transition_bandwidths_hz) != 1:
        raise ValueError("A recording must have one transition bandwidth.")
    stopbands = tuple(
        sorted(
            (
                HarmonicStopband(
                    harmonics=(
                        ()
                        if pd.isna(row["harmonics"]) or str(row["harmonics"]) == ""
                        else tuple(
                            int(float(value))
                            for value in str(row["harmonics"]).split(";")
                        )
                    ),
                    low_hz=float(row["stopband_low_hz"]),
                    high_hz=float(row["stopband_high_hz"]),
                    kind=str(row["kind"]),
                )
                for row in rows
            ),
            key=lambda stopband: stopband.low_hz,
        )
    )
    return HarmonicNotchPlan(stopbands, transition_bandwidths_hz.pop())


def channel_plans_from_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[ChannelNotchPlan, ...]:
    """Reconstruct every channel-specific FIR geometry from a manifest block."""
    if not rows:
        raise ValueError("A recording's line-notch manifest has no stopbands.")
    outcomes = {str(row["outcome"]) for row in rows}
    if outcomes == {"no_artifact_detected"}:
        if len(rows) != 1:
            raise ValueError("A null recording must have exactly one manifest row.")
        return ()
    if outcomes != {"artifact_detected"}:
        raise ValueError("A recording cannot mix artifact and null manifest outcomes.")
    channels = []
    for row in rows:
        channel_name = str(row["channel"])
        if channel_name not in channels:
            channels.append(channel_name)
    return tuple(
        ChannelNotchPlan(
            channel_name,
            harmonic_plan_from_rows(
                [row for row in rows if str(row["channel"]) == channel_name]
            ),
        )
        for channel_name in channels
    )


def analysed_bands_from_config(config) -> tuple[tuple[str, float, float], ...]:
    """Return the canonical analysis bands whose unavailable shares are reported."""
    defined = config.get("frequency_bands") or {}
    if not isinstance(defined, dict):
        raise ValueError("frequency_bands must be a mapping of name to [low, high].")
    bands = []
    for name, edges in defined.items():
        if not str(name).strip():
            raise ValueError("frequency_bands names must not be empty.")
        low_hz, high_hz = (float(value) for value in edges)
        if high_hz <= low_hz:
            raise ValueError(f"frequency_bands.{name} must have increasing edges.")
        bands.append((str(name), low_hz, high_hz))
    return tuple(bands)


def band_power(
    frequencies_hz: np.ndarray,
    psd: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    """Total power across the channel-mean spectrum between two edges."""
    frequency_array = np.asarray(frequencies_hz, dtype=float)
    inside = (frequency_array >= low_hz) & (frequency_array <= high_hz)
    if not np.any(inside):
        raise ValueError(f"No frequency bin lies in {low_hz:g}-{high_hz:g} Hz.")
    return float(np.mean(np.asarray(psd, dtype=float)[..., inside].sum(axis=-1)))


def _change_db(before: float, after: float) -> float:
    """Return power change in decibels, with negative values denoting attenuation."""
    if before <= 0.0:
        raise ValueError("Reference power must be positive.")
    return 10.0 * np.log10(max(after, np.finfo(float).tiny) / before)


def _measure_stopband_changes(
    raw_before,
    raw_after,
    plan: HarmonicNotchPlan,
    settings,
) -> tuple[float, ...]:
    """Measure power change inside each declared stopband."""
    import mne

    picks = mne.pick_types(raw_before.info, eeg=True, exclude=())
    frequencies_hz, before_psd = recordings.psd(raw_before, picks, settings)
    after_frequencies_hz, after_psd = recordings.psd(raw_after, picks, settings)
    if not np.array_equal(frequencies_hz, after_frequencies_hz):
        raise ValueError("Before and after spectra use different frequency grids.")
    return tuple(
        _change_db(
            band_power(frequencies_hz, before_psd, stopband.low_hz, stopband.high_hz),
            band_power(frequencies_hz, after_psd, stopband.low_hz, stopband.high_hz),
        )
        for stopband in plan.stopbands
    )


def _measure_channel_stopband_changes(
    raw_before,
    raw_after,
    plans: Sequence[ChannelNotchPlan],
    settings,
) -> tuple[float, ...]:
    """Measure each stopband on the channel to which it was applied."""
    channel_plans = tuple(plans)
    if not channel_plans:
        return ()
    picks = [raw_before.ch_names.index(plan.channel_name) for plan in channel_plans]
    frequencies_hz, before_psd = recordings.psd(raw_before, picks, settings)
    after_frequencies_hz, after_psd = recordings.psd(raw_after, picks, settings)
    if not np.array_equal(frequencies_hz, after_frequencies_hz):
        raise ValueError("Before and after spectra use different frequency grids.")
    return tuple(
        _change_db(
            band_power(
                frequencies_hz,
                before_psd[channel_index],
                stopband.low_hz,
                stopband.high_hz,
            ),
            band_power(
                frequencies_hz,
                after_psd[channel_index],
                stopband.low_hz,
                stopband.high_hz,
            ),
        )
        for channel_index, channel_plan in enumerate(channel_plans)
        for stopband in channel_plan.geometry.stopbands
    )


def clean_harmonic_run(
    vhdr: Path,
    output_root: Path,
    source_root: Path,
    settings,
    analysed_bands: tuple[tuple[str, float, float], ...],
) -> list[dict[str, float | str]]:
    """Fit, notch, write, and audit one continuous BrainVision recording."""
    raw = recordings.read_bids_raw(vhdr)
    model = fit_harmonic_model(raw, settings)
    plans = plan_channel_notches(model, settings)
    sampling_frequency_hz = float(raw.info["sfreq"])
    filter_designs = {
        plan.channel_name: characterize_harmonic_filter(
            sampling_frequency_hz,
            plan.geometry,
        )
        for plan in plans
    }
    filtered = apply_channel_notches(raw, plans)

    destination_vhdr = recordings.derivative_vhdr_path(
        vhdr,
        source_root,
        output_root,
    )
    recordings.write_brainvision_sidecars(vhdr, destination_vhdr)
    recordings.write_eeg_binary(
        destination_vhdr,
        destination_vhdr.with_suffix(".eeg"),
        filtered.get_data(),
        filtered.ch_names,
    )

    written = recordings.read_bids_raw(destination_vhdr)
    expected = filtered.get_data()
    representable = recordings.quantized_eeg_data(
        destination_vhdr,
        expected,
        filtered.ch_names,
    )
    if not np.array_equal(written.get_data(), representable):
        deviation_v = float(np.max(np.abs(written.get_data() - representable)))
        raise RuntimeError(
            f"{vhdr.name}: written data differs from its exact BrainVision "
            f"quantization by as much as {deviation_v:.3e} V."
        )
    deviation_v = 0.0

    rows = artifact_manifest_rows(
        vhdr.stem,
        model,
        plans,
        analysed_bands,
        settings,
    )
    changes_db = _measure_channel_stopband_changes(raw, filtered, plans, settings)
    artifact_rows = [row for row in rows if row["outcome"] == "artifact_detected"]
    for row, change_db in zip(artifact_rows, changes_db, strict=True):
        row.update(filter_designs[str(row["channel"])].manifest_fields())
        row["in_stopband_change_db"] = change_db
    for row in rows:
        if row["outcome"] == "no_artifact_detected":
            row.update(
                {
                    "fir_filter_length_samples": "",
                    "fir_filter_length_s": "",
                    "fir_minimum_stopband_attenuation_db": "",
                    "fir_maximum_passband_deviation_db": "",
                    "in_stopband_change_db": "",
                }
            )
        row["roundtrip_deviation_v"] = deviation_v
    return rows


def relative_source_dataset_url(source_root: Path, derivative_root: Path) -> str:
    """Filesystem-relative BIDS source URL from the published derivative root."""
    relative_path = os.path.relpath(source_root.resolve(), derivative_root.resolve())
    return Path(relative_path).as_posix()


def write_harmonic_derivative_description(
    output_root: Path,
    source_dataset_url: str,
    settings,
) -> Path:
    """Declare the automatic harmonic-notch output and its inference boundary."""
    import json

    import mne
    import scipy

    path = output_root / "dataset_description.json"
    if not path.is_file():
        raise FileNotFoundError(f"Source dataset description was not mirrored to {path}.")
    described = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(described, dict):
        raise ValueError("BIDS dataset_description.json must contain a JSON object.")

    described["DatasetType"] = "derivative"
    if "BIDSVersion" not in described:
        raise ValueError("BIDS dataset_description.json must declare BIDSVersion.")
    described["Name"] = "decomb line-notched EEG"
    existing = described.get("GeneratedBy", [])
    if not isinstance(existing, list) or not all(isinstance(entry, dict) for entry in existing):
        raise ValueError("BIDS GeneratedBy must be a list of objects.")
    generated = [entry for entry in existing if entry.get("Name") != "decomb"]
    generated.append(
        {
            "Name": "decomb",
            "Version": __version__,
            "Description": (
                "Thomson multitaper F tests identified sinusoidal components with Holm "
                "family-wise correction across all continuous estimation windows and "
                "tested frequencies within each EEG channel. Each significant frequency "
                "was removed only from its supported channel with a zero-phase MNE FIR "
                "notch. Harmonic classification never added an unobserved target. "
                "Stopbands and transitions are unavailable for inference and are listed "
                f"per channel in {MANIFEST_NAME}."
            ),
            "Parameters": {
                "multiple_testing_method": "holm",
                "familywise_error_unit": "eeg_channel",
                "filter_scope": "statistically_supported_channels",
                "library_versions": {
                    "mne": mne.__version__,
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                },
                "method": "fir",
                "filter_length": "auto",
                "phase": FIR_PHASE,
                "fir_window": FIR_WINDOW,
                "fir_design": FIR_DESIGN,
                "pad": FIR_PAD,
                "skip_by_annotation": list(
                    recordings.ACQUISITION_BOUNDARY_ANNOTATIONS
                ),
                **{
                    name: list(value) if isinstance(value, tuple) else value
                    for name, value in asdict(settings).items()
                },
                "spectral_resolution_hz": settings.spectral_resolution_hz,
                "transition_bandwidth_hz": settings.transition_bandwidth_hz,
                "per_edge_transition_bandwidth_hz": (
                    settings.per_edge_transition_bandwidth_hz
                ),
            },
        }
    )
    described["GeneratedBy"] = generated
    if not source_dataset_url:
        raise ValueError("The source dataset URL must not be empty.")
    described["SourceDatasets"] = [{"URL": source_dataset_url}]
    path.write_text(json.dumps(described, indent=2) + "\n", encoding="utf-8")
    return path


def settings_for_verification(
    derivative_root: Path,
    current_settings: HarmonicNotchSettings,
) -> HarmonicNotchSettings:
    """Return immutable apply-time settings, refusing a current-config mismatch."""
    import json

    import mne
    import scipy

    path = derivative_root / "dataset_description.json"
    if not path.is_file():
        raise FileNotFoundError(f"No derivative description at {path}.")
    described = json.loads(path.read_text(encoding="utf-8"))
    generated_by = described.get("GeneratedBy")
    if not isinstance(generated_by, list):
        raise ValueError("The derivative description must contain a GeneratedBy list.")
    decomb_entries = [
        entry
        for entry in generated_by
        if isinstance(entry, Mapping) and entry.get("Name") == "decomb"
    ]
    if len(decomb_entries) != 1:
        raise ValueError("The derivative description must contain one decomb GeneratedBy entry.")
    parameters = decomb_entries[0].get("Parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("The decomb GeneratedBy entry must contain Parameters.")
    try:
        applied_settings = HarmonicNotchSettings(
            estimation_window_s=float(parameters["estimation_window_s"]),
            familywise_error_rate=float(parameters["familywise_error_rate"]),
            frequency_range_hz=tuple(
                float(value) for value in parameters["frequency_range_hz"]
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "The decomb derivative does not contain valid apply-time settings."
        ) from error

    expected_provenance = {
        "multiple_testing_method": "holm",
        "familywise_error_unit": "eeg_channel",
        "filter_scope": "statistically_supported_channels",
        "library_versions": {
            "mne": mne.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "method": "fir",
        "filter_length": "auto",
        "phase": FIR_PHASE,
        "fir_window": FIR_WINDOW,
        "fir_design": FIR_DESIGN,
        "pad": FIR_PAD,
        "skip_by_annotation": list(recordings.ACQUISITION_BOUNDARY_ANNOTATIONS),
        "spectral_resolution_hz": applied_settings.spectral_resolution_hz,
        "transition_bandwidth_hz": applied_settings.transition_bandwidth_hz,
        "per_edge_transition_bandwidth_hz": (
            applied_settings.per_edge_transition_bandwidth_hz
        ),
    }
    provenance_mismatches = [
        name
        for name, expected in expected_provenance.items()
        if parameters.get(name) != expected
    ]
    if provenance_mismatches:
        raise ValueError(
            "The derivative's apply-time filter provenance does not reproduce "
            f"the implemented filter: {provenance_mismatches}."
        )

    mismatches = [
        field.name
        for field in fields(HarmonicNotchSettings)
        if getattr(current_settings, field.name) != getattr(applied_settings, field.name)
    ]
    if mismatches:
        raise ValueError(
            f"Current removal setting(s) {mismatches} do not match those recorded "
            "during apply. Verify with the original configuration."
        )
    return applied_settings


def run(args: argparse.Namespace) -> None:
    """Apply automatic evidence-bounded harmonic notches to a complete BIDS dataset."""
    import time

    import mne

    from decomb import effective
    from decomb.config import load_config

    mne.set_log_level("ERROR")
    config = load_config(getattr(args, "config", None))
    source_root = config.path("bids_root", override=getattr(args, "bids_root", None))
    output_root = config.path("output_root", override=getattr(args, "output_root", None))
    report_dir = config.path("removal_dir", override=getattr(args, "report_dir", None))
    settings = HarmonicNotchSettings.from_config(config)
    analysed_bands = analysed_bands_from_config(config)
    runs = recordings.discover_runs(source_root, subjects=None, task="*")

    if output_root.exists():
        raise FileExistsError(
            f"Refusing to mix a new derivative with existing output: {output_root}"
        )
    staging = output_root.with_name(f".{output_root.name}.staging")
    if staging.exists():
        raise FileExistsError(
            f"Incomplete staging output exists at {staging}; inspect it before retrying."
        )
    staging.mkdir(parents=True)
    print(f"Applying automatic line notches to {len(runs)} recordings")
    print(f"  copied {recordings.mirror_sidecars(source_root, staging)} sidecars")

    rows: list[dict[str, float | str]] = []
    for index, vhdr in enumerate(runs, start=1):
        started = time.time()
        measured = clean_harmonic_run(
            vhdr,
            staging,
            source_root,
            settings,
            analysed_bands,
        )
        rows.extend(measured)
        artifact_rows = [
            row for row in measured if row["outcome"] == "artifact_detected"
        ]
        if not artifact_rows:
            print(
                f"[{index}/{len(runs)}] {vhdr.stem[:44]:44s} "
                f"no supported line; copied unchanged "
                f"({time.time() - started:.0f}s)"
            )
            continue
        stopband_width_hz = sum(
            float(row["stopband_high_hz"]) - float(row["stopband_low_hz"])
            for row in artifact_rows
        )
        median_change_db = float(
            np.median(
                [float(row["in_stopband_change_db"]) for row in artifact_rows]
            )
        )
        print(
            f"[{index}/{len(runs)}] {vhdr.stem[:44]:44s} "
            f"{len(artifact_rows)} stopbands, {stopband_width_hz:.3f} Hz, "
            f"median {median_change_db:+.1f} dB ({time.time() - started:.0f}s)"
        )

    frame = pd.DataFrame(rows)
    recordings.write_tsv_atomic(frame, staging / MANIFEST_NAME)
    source_dataset_url = relative_source_dataset_url(source_root, output_root)
    described = write_harmonic_derivative_description(
        staging,
        source_dataset_url,
        settings,
    )
    os.replace(staging, output_root)

    report_dir.mkdir(parents=True, exist_ok=True)
    recordings.write_tsv_atomic(frame, report_dir / MANIFEST_NAME)
    effective_path = effective.write(
        config,
        settings,
        report_dir / "effective_config_apply.txt",
        stage="apply",
    )
    print(f"  declared {output_root / described.name} a derivative of {source_root}")
    print(f"  wrote {report_dir / MANIFEST_NAME}")
    print(f"  wrote {effective_path}")


def _validate_matching_recordings(original, cleaned) -> None:
    """Fail when a purported derivative does not match its source geometry."""
    if original.ch_names != cleaned.ch_names:
        raise ValueError("Source and cleaned recordings have different channel names.")
    if original.get_channel_types() != cleaned.get_channel_types():
        raise ValueError("Source and cleaned recordings have different channel types.")
    if original.n_times != cleaned.n_times:
        raise ValueError("Source and cleaned recordings have different sample counts.")
    if float(original.info["sfreq"]) != float(cleaned.info["sfreq"]):
        raise ValueError("Source and cleaned recordings have different sampling frequencies.")


def _validate_filter_design(
    manifest_rows: Sequence[Mapping[str, object]],
    design: HarmonicFilterDesign,
) -> None:
    """Require every manifest row to reproduce MNE's exact FIR design."""
    expected = design.manifest_fields()
    for row in manifest_rows:
        for name, value in expected.items():
            if name not in row:
                raise ValueError(f"Manifest filter design is missing {name!r}.")
            recorded = row[name]
            if isinstance(value, int):
                matches = int(recorded) == value
            else:
                matches = float(recorded) == value
            if not matches:
                raise ValueError(
                    f"Manifest filter design {name!r}={recorded!r} does not "
                    f"reproduce {value!r}."
                )


def _validate_manifest_evidence(
    manifest_rows: Sequence[Mapping[str, object]],
    settings: HarmonicNotchSettings,
) -> None:
    """Require every target to carry valid channel-level Holm evidence."""
    outcomes = {str(row["outcome"]) for row in manifest_rows}
    if not outcomes <= {"artifact_detected", "no_artifact_detected"}:
        raise ValueError("Manifest contains an unknown statistical outcome.")
    if len(outcomes) != 1:
        raise ValueError("A recording cannot mix artifact and null outcomes.")
    methods = {str(row["multiple_testing_method"]) for row in manifest_rows}
    if methods != {"holm"}:
        raise ValueError("Manifest targets must use Holm multiple-testing correction.")
    error_rates = {float(row["familywise_error_rate"]) for row in manifest_rows}
    if error_rates != {settings.familywise_error_rate}:
        raise ValueError("Manifest family-wise error rate differs from apply settings.")

    count_fields = (
        "estimation_window_count",
        "tested_eeg_channel_count",
        "detection_test_count_per_channel",
        "total_detection_test_count",
    )
    counts = {}
    for name in count_fields:
        values = {float(row[name]) for row in manifest_rows}
        if len(values) != 1 or next(iter(values)) <= 0.0:
            raise ValueError(f"Manifest {name} must be one positive count.")
        if not next(iter(values)).is_integer():
            raise ValueError(f"Manifest {name} must be an integer.")
        counts[name] = int(next(iter(values)))
    expected_total = (
        counts["tested_eeg_channel_count"]
        * counts["detection_test_count_per_channel"]
    )
    if counts["total_detection_test_count"] != expected_total:
        raise ValueError("Manifest total test count does not match its channel families.")

    if outcomes == {"no_artifact_detected"}:
        if len(manifest_rows) != 1:
            raise ValueError("A null recording must have exactly one manifest row.")
        row = manifest_rows[0]
        null_fields = (
            "channel",
            "kind",
            "harmonics",
            "stopband_low_hz",
            "stopband_high_hz",
            "detected_line_frequencies_hz",
            "detected_line_raw_p_values",
            "detected_line_corrected_p_values",
            "detected_line_window_indices",
            "detected_line_harmonics",
            "fundamental_hz",
            "comb_corrected_p_value",
        )
        if any(not _missing(row[name]) for name in null_fields):
            raise ValueError("A null result cannot contain line or filter evidence.")
        return

    channel_names = [str(row["channel"]) for row in manifest_rows]
    if any(not name.strip() for name in channel_names):
        raise ValueError("Manifest channel names must not be empty.")
    for channel_name in dict.fromkeys(channel_names):
        channel_rows = [
            row for row in manifest_rows if str(row["channel"]) == channel_name
        ]
        _validate_channel_manifest_evidence(channel_rows, settings, counts)


def _validate_refitted_evidence(
    manifest_rows: Sequence[Mapping[str, object]],
    refitted_rows: Sequence[Mapping[str, object]],
) -> None:
    """Require the manifest's statistical authorization to equal a source refit."""
    if _authorization_records(manifest_rows) != _authorization_records(refitted_rows):
        raise ValueError(
            "Manifest does not match refitted statistical evidence from the source."
        )


def _authorization_records(
    rows: Sequence[Mapping[str, object]],
) -> tuple[tuple[object, ...], ...]:
    """Return a stable semantic representation of manifest authorization."""
    records = []
    for row in rows:
        frequencies_hz = _semicolon_floats(
            row["detected_line_frequencies_hz"]
        )
        records.append(
            (
                _text(row["outcome"]),
                _text(row["channel"]),
                _text(row["kind"]),
                _semicolon_ints(row["harmonics"]),
                _optional_float(row["stopband_low_hz"]),
                _optional_float(row["stopband_high_hz"]),
                _optional_float(row["transition_bandwidth_hz"]),
                frequencies_hz,
                _semicolon_floats(row["detected_line_raw_p_values"]),
                _semicolon_floats(row["detected_line_corrected_p_values"]),
                _window_index_groups(row["detected_line_window_indices"]),
                _line_harmonics(
                    row["detected_line_harmonics"],
                    len(frequencies_hz),
                ),
                _optional_float(row["fundamental_hz"]),
                _optional_float(row["comb_corrected_p_value"]),
                _text(row["multiple_testing_method"]),
                float(row["familywise_error_rate"]),
                int(row["estimation_window_count"]),
                int(row["tested_eeg_channel_count"]),
                int(row["detection_test_count_per_channel"]),
                int(row["total_detection_test_count"]),
            )
        )
    return tuple(sorted(records, key=lambda record: (record[1], record[4] or -1.0)))


def _validate_channel_manifest_evidence(
    rows: Sequence[Mapping[str, object]],
    settings: HarmonicNotchSettings,
    counts: Mapping[str, int],
) -> None:
    """Validate one affected channel's model and line evidence."""
    fundamentals = {
        None if _missing(row["fundamental_hz"]) else float(row["fundamental_hz"])
        for row in rows
    }
    comb_p_values = {
        None
        if _missing(row["comb_corrected_p_value"])
        else float(row["comb_corrected_p_value"])
        for row in rows
    }
    if len(fundamentals) != 1 or len(comb_p_values) != 1:
        raise ValueError("Manifest comb evidence must be channel-level and consistent.")
    fundamental_hz = fundamentals.pop()
    comb_p_value = comb_p_values.pop()
    if (fundamental_hz is None) != (comb_p_value is None):
        raise ValueError("Manifest comb fundamental and corrected p-value must occur together.")
    if fundamental_hz is not None:
        if not np.isfinite(fundamental_hz) or fundamental_hz <= 0.0:
            raise ValueError("Manifest comb fundamental must be finite and positive.")
        if not 0.0 <= comb_p_value < settings.familywise_error_rate:
            raise ValueError("Manifest comb is not statistically supported.")

    positions = []
    for row in rows:
        frequencies_hz = _semicolon_floats(row["detected_line_frequencies_hz"])
        raw_p_values = _semicolon_floats(row["detected_line_raw_p_values"])
        corrected_p_values = _semicolon_floats(
            row["detected_line_corrected_p_values"]
        )
        window_groups = str(row["detected_line_window_indices"]).split(";")
        line_count = len(frequencies_hz)
        if not line_count or not (
            len(raw_p_values)
            == len(corrected_p_values)
            == len(window_groups)
            == line_count
        ):
            raise ValueError(
                "Every detected line requires raw, corrected, and window evidence."
            )
        low_hz = float(row["stopband_low_hz"])
        high_hz = float(row["stopband_high_hz"])
        for frequency_hz, raw_p_value, corrected_p_value, window_group in zip(
            frequencies_hz,
            raw_p_values,
            corrected_p_values,
            window_groups,
            strict=True,
        ):
            if not low_hz <= frequency_hz <= high_hz:
                raise ValueError("Every detected line must lie inside its stopband.")
            if not 0.0 <= raw_p_value <= corrected_p_value:
                raise ValueError("Manifest raw and Holm-adjusted p-values are invalid.")
            if corrected_p_value >= settings.familywise_error_rate:
                raise ValueError("Every filtered line must be statistically supported.")
            window_indices = tuple(int(value) for value in window_group.split(","))
            if (
                not window_indices
                or window_indices != tuple(sorted(set(window_indices)))
                or window_indices[0] < 0
                or window_indices[-1] >= counts["estimation_window_count"]
            ):
                raise ValueError("Manifest supporting window indices are invalid.")
            positions.append(frequency_hz)
    if len(positions) != len(set(positions)):
        raise ValueError("A channel line may belong to only one stopband.")


def _missing(value: object) -> bool:
    return bool(pd.isna(value) or str(value) == "")


def _text(value: object) -> str:
    return "" if _missing(value) else str(value)


def _optional_float(value: object) -> float | None:
    return None if _missing(value) else float(value)


def _semicolon_ints(value: object) -> tuple[int, ...]:
    if _missing(value):
        return ()
    return tuple(int(float(entry)) for entry in str(value).split(";"))


def _window_index_groups(value: object) -> tuple[tuple[int, ...], ...]:
    if _missing(value):
        return ()
    return tuple(
        tuple(int(index) for index in group.split(","))
        for group in str(value).split(";")
    )


def _line_harmonics(
    value: object,
    line_count: int,
) -> tuple[int | None, ...]:
    if _missing(value):
        return tuple(None for _ in range(line_count))
    labels = tuple(
        None if entry == "" else int(float(entry))
        for entry in str(value).split(";")
    )
    if len(labels) != line_count:
        raise ValueError("Manifest harmonic labels must match detected lines.")
    return labels


def _semicolon_floats(value: object) -> tuple[float, ...]:
    if _missing(value):
        return ()
    values = tuple(float(entry) for entry in str(value).split(";"))
    if not np.all(np.isfinite(values)):
        raise ValueError("Manifest line evidence must contain finite values.")
    return values


def _validate_exact_derivative(
    original,
    cleaned,
    cleaned_vhdr: Path,
    plans: Sequence[ChannelNotchPlan],
) -> float:
    """Reapply the declared FIR and require exact written BrainVision samples."""
    _validate_matching_recordings(original, cleaned)
    filtered = apply_channel_notches(original, plans)
    expected = recordings.quantized_eeg_data(
        cleaned_vhdr,
        filtered.get_data(),
        filtered.ch_names,
    )
    actual = cleaned.get_data()
    if np.array_equal(actual, expected):
        return 0.0
    deviation_v = float(np.max(np.abs(actual - expected)))
    differing_samples = int(np.count_nonzero(actual != expected))
    raise RuntimeError(
        f"{cleaned_vhdr.name}: written data does not equal the declared FIR "
        f"derivative after BrainVision quantization ({differing_samples} samples "
        f"differ; maximum deviation {deviation_v:.3e} V)."
    )


def verify_harmonic_run(
    source_vhdr: Path,
    cleaned_vhdr: Path,
    manifest_rows: Sequence[Mapping[str, object]],
    settings,
) -> list[dict[str, float | str]]:
    """Reproduce a written recording from its declared filter geometry."""
    original = recordings.read_bids_raw(source_vhdr)
    cleaned = recordings.read_bids_raw(cleaned_vhdr)
    _validate_matching_recordings(original, cleaned)
    _validate_manifest_evidence(manifest_rows, settings)
    refitted_model = fit_harmonic_model(original, settings)
    refitted_plans = plan_channel_notches(refitted_model, settings)
    refitted_rows = artifact_manifest_rows(
        source_vhdr.stem,
        refitted_model,
        refitted_plans,
        (),
        settings,
    )
    _validate_refitted_evidence(manifest_rows, refitted_rows)
    plans = channel_plans_from_rows(manifest_rows)
    sampling_frequency_hz = float(original.info["sfreq"])
    for channel_plan in plans:
        channel_rows = [
            row
            for row in manifest_rows
            if str(row["channel"]) == channel_plan.channel_name
        ]
        design = characterize_harmonic_filter(
            sampling_frequency_hz,
            channel_plan.geometry,
        )
        _validate_filter_design(channel_rows, design)
    maximum_sample_deviation_v = _validate_exact_derivative(
        original,
        cleaned,
        cleaned_vhdr,
        plans,
    )
    changes_db = _measure_channel_stopband_changes(
        original,
        cleaned,
        plans,
        settings,
    )
    rows = []
    if not plans:
        return [
            {
                "recording": source_vhdr.stem,
                "outcome": "no_artifact_detected",
                "channel": "",
                "kind": "",
                "harmonics": "",
                "stopband_low_hz": "",
                "stopband_high_hz": "",
                "unavailable_low_hz": "",
                "unavailable_high_hz": "",
                "verified_stopband_change_db": "",
                "maximum_sample_deviation_v": maximum_sample_deviation_v,
            }
        ]
    change_index = 0
    for channel_plan in plans:
        for stopband, unavailable in zip(
            channel_plan.geometry.stopbands,
            channel_plan.geometry.unavailable_edges(),
            strict=True,
        ):
            rows.append(
                {
                    "recording": source_vhdr.stem,
                    "outcome": "artifact_detected",
                    "channel": channel_plan.channel_name,
                    "kind": stopband.kind,
                    "harmonics": ";".join(str(value) for value in stopband.harmonics),
                    "stopband_low_hz": stopband.low_hz,
                    "stopband_high_hz": stopband.high_hz,
                    "unavailable_low_hz": unavailable[0],
                    "unavailable_high_hz": unavailable[1],
                    "verified_stopband_change_db": changes_db[change_index],
                    "maximum_sample_deviation_v": maximum_sample_deviation_v,
                }
            )
            change_index += 1
    return rows


def run_verify(args: argparse.Namespace) -> None:
    """Audit the written line-notch derivative without refitting its targets."""
    from decomb import effective
    from decomb.config import load_config

    config = load_config(getattr(args, "config", None))
    source_root = config.path("bids_root", override=getattr(args, "bids_root", None))
    cleaned_root = config.path("output_root", override=getattr(args, "output_root", None))
    report_dir = config.path("removal_dir", override=getattr(args, "report_dir", None))
    current_settings = HarmonicNotchSettings.from_config(config)
    settings = settings_for_verification(cleaned_root, current_settings)
    runs = recordings.discover_runs(source_root, subjects=None, task="*")
    manifest_path = cleaned_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No line-notch manifest at {manifest_path}. Run `decomb apply` first."
        )
    manifest = pd.read_csv(manifest_path, sep="\t", float_precision="round_trip")
    required = {
        "recording",
        "outcome",
        "channel",
        "kind",
        "harmonics",
        "fundamental_hz",
        "comb_corrected_p_value",
        "detected_line_frequencies_hz",
        "detected_line_raw_p_values",
        "detected_line_corrected_p_values",
        "detected_line_window_indices",
        "detected_line_harmonics",
        "multiple_testing_method",
        "familywise_error_rate",
        "estimation_window_count",
        "tested_eeg_channel_count",
        "detection_test_count_per_channel",
        "total_detection_test_count",
        "stopband_low_hz",
        "stopband_high_hz",
        "transition_bandwidth_hz",
        "fir_filter_length_samples",
        "fir_filter_length_s",
        "fir_minimum_stopband_attenuation_db",
        "fir_maximum_passband_deviation_db",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Line-notch manifest is missing columns: {sorted(missing)}")
    recording_names = {vhdr.stem for vhdr in runs}
    if set(manifest["recording"]) != recording_names:
        raise ValueError("Line-notch manifest does not cover exactly the source recordings.")

    rows: list[dict[str, float | str]] = []
    for vhdr in runs:
        block = manifest.loc[manifest["recording"] == vhdr.stem]
        rows.extend(
            verify_harmonic_run(
                vhdr,
                recordings.derivative_vhdr_path(vhdr, source_root, cleaned_root),
                block.to_dict("records"),
                settings,
            )
        )
    frame = pd.DataFrame(rows)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / VERIFICATION_NAME
    recordings.write_tsv_atomic(frame, output_path)
    effective_path = effective.write(
        config,
        settings,
        report_dir / "effective_config_verify.txt",
        stage="verify",
    )
    changes_db = pd.to_numeric(
        frame["verified_stopband_change_db"],
        errors="coerce",
    ).dropna()
    summary = (
        "no filter was authorized"
        if changes_db.empty
        else f"median stopband change {changes_db.median():+.1f} dB"
    )
    print(f"Verified {len(runs)} recordings: {summary}")
    print(f"  wrote {output_path}")
    print(f"  wrote {effective_path}")
