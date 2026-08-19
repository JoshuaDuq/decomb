"""Automatic recording-specific FIR notches for supported lines and scanner combs.

The transform makes no claim to recover neural activity at a removed frequency. Its
manifest therefore records every stopband and transition as unavailable for inference.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

from decomb import __version__, lines, recordings

FIR_DESIGN = "firwin"
FIR_TRANSITION_REFERENCE_WINDOW_S = 54.0
FIR_PAD = "reflect_limited"
FIR_PHASE = "zero"
FIR_WINDOW = "hamming"
MANIFEST_NAME = "line_notch_manifest.tsv"
MULTIPLE_TESTING_METHOD = (
    "source_bonferroni_two_shape_union_then_holm_"
    "residual_holm_and_scanner_bonferroni"
)
SCANNER_HARMONIC_ESTIMATION_WINDOW_S = 4.0
VERIFICATION_NAME = "line_notch_verification.tsv"
MANIFEST_REQUIRED_COLUMNS = frozenset(
    {
        "recording",
        "removal_round",
        "outcome",
        "channel",
        "kind",
        "harmonics",
        "fundamental_hz",
        "scanner_family_corrected_p_value",
        "scanner_supporting_harmonics",
        "scanner_repetition_time_s",
        "scanner_trigger_event_name",
        "detected_line_frequencies_hz",
        "detected_line_input_p_values",
        "detected_line_corrected_p_values",
        "detected_line_window_indices",
        "multiple_testing_method",
        "multiple_testing_scope",
        "familywise_error_rate",
        "round_familywise_error_rate",
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
)


@dataclass(frozen=True)
class HarmonicNotchSettings:
    """Statistical settings and the scanner timing supplied by the user."""

    estimation_window_s: float
    familywise_error_rate: float
    frequency_range_hz: tuple[float, float]
    scanner_repetition_time_s: float = 0.9
    scanner_trigger_event_name: str = "Volume/V  1"
    comb_fundamental_hz: float | None = None

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
        if (
            not np.isfinite(self.scanner_repetition_time_s)
            or self.scanner_repetition_time_s <= 0.0
        ):
            raise ValueError(
                "removal.scanner_repetition_time_s must be finite and positive."
            )
        if not isinstance(self.scanner_trigger_event_name, str) or not (
            self.scanner_trigger_event_name.strip()
        ):
            raise ValueError(
                "removal.scanner_trigger_event_name must be a non-empty string."
            )
        if self.comb_fundamental_hz is not None and (
            not np.isfinite(self.comb_fundamental_hz)
            or self.comb_fundamental_hz <= 0.0
        ):
            raise ValueError(
                "removal.comb_fundamental_hz must be finite and positive when set."
            )

    @property
    def comb_fundamental(self) -> float:
        """The comb's fundamental, declared outright or derived from the TR.

        A periodic device does not have to run at the volume rate. When the artifact's
        source has its own known rate -- a cold head at 72 cycles per minute is
        1.2 Hz, against a 1.1111 Hz volume rate for a 0.9 s TR -- the trigger-derived
        grid lands between the teeth and tests frequencies the artifact never occupies.

        Declaring the rate keeps the property the trigger anchoring exists to protect:
        the grid is still fixed before any spectrum is inspected, so it cannot be
        fished out of the data. It is a stated fact about the hardware, exactly as the
        TR is, and the trigger check still validates the recording's timing.
        """
        if self.comb_fundamental_hz is not None:
            return float(self.comb_fundamental_hz)
        return 1.0 / self.scanner_repetition_time_s

    @property
    def estimation_overlap(self) -> float:
        """Fixed overlap between successive estimation windows."""
        return 0.5

    @property
    def transition_bandwidth_hz(self) -> float:
        """Total notch transition width across both edges."""
        return 3.3 / self.filter_resolution_window_s

    @property
    def filter_resolution_window_s(self) -> float:
        """Fixed selectivity reference for the MNE Hamming FIR geometry."""
        return FIR_TRANSITION_REFERENCE_WINDOW_S

    @property
    def per_edge_transition_bandwidth_hz(self) -> float:
        """Transition width MNE uses to derive the automatic FIR length."""
        return self.transition_bandwidth_hz / 2.0

    @property
    def frequency_bin_width_hz(self) -> float:
        return 1.0 / self.estimation_window_s

    @property
    def scanner_harmonics_stopband_width_hz(self) -> float:
        """Width fixed by the 4 s scanner-comb localization horizon."""
        return 1.0 / SCANNER_HARMONIC_ESTIMATION_WINDOW_S

    @property
    def supported_scanner_harmonic_stopband_width_hz(self) -> float:
        """Width covering the local background used to establish a visible tooth."""
        return (
            self.scanner_harmonics_stopband_width_hz
            + 2.0 * lines.PERSISTENT_PEAK_SMOOTHING_HZ
        )

    @property
    def ordinary_line_stopband_width_hz(self) -> float:
        """Minimum width that removes visible structure around an authorized line."""
        return max(
            self.frequency_bin_width_hz,
            self.scanner_harmonics_stopband_width_hz,
        )

    @property
    def alpha_spending_rule(self) -> str:
        return "alpha / (round * (round + 1))"

    def error_rate_for_round(self, round_index: int) -> float:
        """Alpha-spending rate whose infinite sequence sums to the configured FWER."""
        if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index < 1:
            raise ValueError("round_index must be a positive integer.")
        return self.familywise_error_rate / (round_index * (round_index + 1))

    def for_round(self, round_index: int) -> HarmonicNotchSettings:
        """Settings carrying one round's share of the recording-wide error budget."""
        return HarmonicNotchSettings(
            estimation_window_s=self.estimation_window_s,
            familywise_error_rate=self.error_rate_for_round(round_index),
            frequency_range_hz=self.frequency_range_hz,
            scanner_repetition_time_s=self.scanner_repetition_time_s,
            scanner_trigger_event_name=self.scanner_trigger_event_name,
        )

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
            scanner_repetition_time_s=float(
                block["scanner_repetition_time_s"]
            ),
            scanner_trigger_event_name=block["scanner_trigger_event_name"],
            comb_fundamental_hz=(
                None
                if block["comb_fundamental_hz"] is None
                else float(block["comb_fundamental_hz"])
            ),
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
class ScannerHarmonicEvidence:
    """Recording-level evidence for trigger-prespecified scanner harmonics."""

    fundamental_hz: float
    corrected_p_value: float
    supporting_harmonics: tuple[int, ...]
    window_count: int = 1
    channel_count: int = 1
    frequency_count: int = 1

    def __post_init__(self) -> None:
        if not np.isfinite(self.fundamental_hz) or self.fundamental_hz <= 0.0:
            raise ValueError("A scanner comb requires a positive finite fundamental.")
        if not 0.0 <= self.corrected_p_value < 1.0:
            raise ValueError("A scanner-comb p-value must lie in [0, 1).")
        if (
            self.supporting_harmonics
            != tuple(sorted(set(self.supporting_harmonics)))
            or not self.supporting_harmonics
            or self.supporting_harmonics[0] < 1
        ):
            raise ValueError(
                "Scanner-harmonic evidence requires sorted unique positive harmonics."
            )
        if min(self.window_count, self.channel_count, self.frequency_count) < 1:
            raise ValueError("Scanner-harmonic test dimensions must be positive.")


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


@dataclass(frozen=True)
class HarmonicRemovalRound:
    """One Holm-supported residual model and its recording-wide FIR geometry."""

    model: lines.LineModel
    plans: tuple[ChannelNotchPlan, ...]
    filter_plan: HarmonicNotchPlan
    in_stopband_changes_db: tuple[float, ...]
    scanner_harmonics: ScannerHarmonicEvidence | None = None
    scanner_plan: HarmonicNotchPlan | None = None

    def __post_init__(self) -> None:
        affected_channels = {channel.channel_name for channel in self.model.channels}
        planned_channels = {plan.channel_name for plan in self.plans}
        if planned_channels != affected_channels:
            raise ValueError(
                "A removal round must plan every affected channel exactly once."
            )
        if not affected_channels and self.scanner_harmonics is None:
            raise ValueError("A removal round requires line or scanner-comb evidence.")
        if (self.scanner_harmonics is None) != (self.scanner_plan is None):
            raise ValueError(
                "Scanner-comb evidence and its complete notch plan must occur together."
            )
        geometries = tuple(plan.geometry for plan in self.plans)
        if self.scanner_plan is not None:
            geometries = (*geometries, self.scanner_plan)
        if self.filter_plan != merge_recording_plans(geometries):
            raise ValueError(
                "A removal round's recording plan must equal the union of its evidence."
            )
        stopband_count = sum(len(plan.geometry.stopbands) for plan in self.plans)
        if self.scanner_plan is not None:
            stopband_count += len(self.scanner_plan.stopbands)
        if len(self.in_stopband_changes_db) != stopband_count:
            raise ValueError("A removal round requires one change per stopband.")
        if not np.all(np.isfinite(self.in_stopband_changes_db)):
            raise ValueError("Removal-round stopband changes must be finite.")


@dataclass(frozen=True)
class HarmonicCleaningResult:
    """Converged recording, its supported removal rounds, and terminal null fit."""

    cleaned: object
    rounds: tuple[HarmonicRemovalRound, ...]
    residual_model: lines.LineModel
    residual_scanner_harmonics: ScannerHarmonicEvidence | None = None

    def __post_init__(self) -> None:
        if self.residual_model.channels or self.residual_scanner_harmonics is not None:
            raise ValueError("A converged cleaning result requires a null residual model.")


@dataclass(frozen=True)
class HarmonicRoundEvidence:
    """Joint line and trigger-anchored evidence fitted before one removal round."""

    model: lines.LineModel
    plans: tuple[ChannelNotchPlan, ...]
    scanner_harmonics: ScannerHarmonicEvidence | None
    scanner_plan: HarmonicNotchPlan | None

    def __post_init__(self) -> None:
        affected_channels = {channel.channel_name for channel in self.model.channels}
        if {plan.channel_name for plan in self.plans} != affected_channels:
            raise ValueError("Round evidence must plan every affected channel exactly once.")
        if (self.scanner_harmonics is None) != (self.scanner_plan is None):
            raise ValueError(
                "Scanner-comb evidence and its complete notch plan must occur together."
            )

    @property
    def filter_plan(self) -> HarmonicNotchPlan | None:
        """Union of every statistically authorized geometry, or none for a joint null."""
        geometries = tuple(plan.geometry for plan in self.plans)
        if self.scanner_plan is not None:
            geometries = (*geometries, self.scanner_plan)
        return None if not geometries else merge_recording_plans(geometries)


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
    minimum_width_hz = settings.ordinary_line_stopband_width_hz
    location_uncertainty_hz = minimum_width_hz / 2.0
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
        if high_hz - low_hz < minimum_width_hz:
            half_width_hz = minimum_width_hz / 2.0
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
            harmonics=tuple(
                sorted({*previous.harmonics, *stopband.harmonics})
            ),
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


def plan_scanner_harmonic_notches(
    evidence: ScannerHarmonicEvidence,
    settings: HarmonicNotchSettings,
    *,
    maximum_hz: float,
) -> HarmonicNotchPlan:
    """Plan a notch at each supported scanner harmonic and nowhere else."""
    upper_hz = min(float(maximum_hz), settings.frequency_range_hz[1])
    if not np.isfinite(upper_hz) or upper_hz <= settings.frequency_range_hz[0]:
        raise ValueError("The scanner-comb upper frequency must exceed the study minimum.")
    first_harmonic = max(
        1,
        int(np.ceil(settings.frequency_range_hz[0] / evidence.fundamental_hz)),
    )
    last_harmonic = int(np.floor(upper_hz / evidence.fundamental_hz))
    if last_harmonic < first_harmonic:
        raise ValueError("No scanner harmonic lies inside the recording's study range.")

    planned_harmonics = evidence.supporting_harmonics
    if any(
        harmonic < first_harmonic or harmonic > last_harmonic
        for harmonic in planned_harmonics
    ):
        raise ValueError("A supported scanner harmonic lies outside the study range.")

    stopbands = []
    for harmonic in planned_harmonics:
        width_hz = settings.supported_scanner_harmonic_stopband_width_hz
        centre_hz = harmonic * evidence.fundamental_hz
        stopbands.append(
            HarmonicStopband(
                (harmonic,),
                centre_hz - width_hz / 2.0,
                centre_hz + width_hz / 2.0,
            )
        )
    return HarmonicNotchPlan(
        _merge_stopbands(
            stopbands,
            minimum_gap_hz=settings.transition_bandwidth_hz,
        ),
        settings.transition_bandwidth_hz,
    )


def _plan_scanner_harmonics_for_recording(
    raw,
    evidence: ScannerHarmonicEvidence,
    settings: HarmonicNotchSettings,
) -> HarmonicNotchPlan:
    """Keep the complete comb and its transitions strictly below Nyquist."""
    half_unavailable_width_hz = (
        settings.scanner_harmonics_stopband_width_hz
        + settings.transition_bandwidth_hz
    ) / 2.0
    maximum_hz = min(
        settings.frequency_range_hz[1],
        np.nextafter(
            float(raw.info["sfreq"]) / 2.0 - half_unavailable_width_hz,
            0.0,
        ),
    )
    return plan_scanner_harmonic_notches(
        evidence,
        settings,
        maximum_hz=maximum_hz,
    )


def plan_channel_notches(
    model: lines.LineModel,
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


def plan_recording_notches(
    model: lines.LineModel,
    settings: HarmonicNotchSettings,
) -> HarmonicNotchPlan:
    """Union all supported channel intervals into one spatially invariant plan."""
    if not model.channels:
        raise ValueError("A recording notch plan requires statistical evidence.")
    channel_plans = plan_channel_notches(model, settings)
    return recording_plan_from_channel_plans(channel_plans)


def recording_plan_from_channel_plans(
    plans: Sequence[ChannelNotchPlan],
) -> HarmonicNotchPlan:
    """Merge channel-local evidence into the FIR applied to every EEG channel."""
    channel_plans = tuple(plans)
    if not channel_plans:
        raise ValueError("A recording notch plan requires channel evidence.")
    return merge_recording_plans(
        tuple(plan.geometry for plan in channel_plans)
    )


def merge_recording_plans(
    plans: Sequence[HarmonicNotchPlan],
) -> HarmonicNotchPlan:
    """Merge supported FIR geometries into one recording-wide plan."""
    filter_plans = tuple(plans)
    if not filter_plans:
        raise ValueError("At least one notch plan is required.")
    transition_bandwidths_hz = {
        plan.transition_bandwidth_hz for plan in filter_plans
    }
    if len(transition_bandwidths_hz) != 1:
        raise ValueError("Channel evidence must use one transition bandwidth.")
    transition_bandwidth_hz = transition_bandwidths_hz.pop()
    channel_stopbands = [
        stopband
        for plan in filter_plans
        for stopband in plan.stopbands
    ]
    return HarmonicNotchPlan(
        _merge_stopbands(
            channel_stopbands,
            minimum_gap_hz=transition_bandwidth_hz,
        ),
        transition_bandwidth_hz,
    )


def eeg_channel_names(raw) -> tuple[str, ...]:
    """Non-bad EEG channel names in the order decomb tests them."""
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(picks) == 0:
        raise ValueError("Line detection requires at least one non-bad EEG channel.")
    return tuple(raw.ch_names[index] for index in picks)


def scanner_fundamental_hz(raw, settings: HarmonicNotchSettings) -> float:
    """Return the configured scanner frequency after validating its exact markers."""
    descriptions = np.asarray(raw.annotations.description, dtype=str)
    trigger_times_s = np.asarray(raw.annotations.onset, dtype=float)[
        descriptions == settings.scanner_trigger_event_name
    ]
    if trigger_times_s.size < 2:
        raise ValueError(
            f"Configured scanner trigger event {settings.scanner_trigger_event_name!r} "
            "is not present at least twice in the recording annotations."
        )

    sampling_frequency_hz = float(raw.info["sfreq"])
    tolerance_s = 0.5 / sampling_frequency_hz
    trigger_intervals_s = np.diff(trigger_times_s)
    if not np.allclose(
        trigger_intervals_s,
        settings.scanner_repetition_time_s,
        rtol=0.0,
        atol=tolerance_s,
    ):
        raise ValueError(
            f"Scanner trigger intervals do not equal the configured "
            f"{settings.scanner_repetition_time_s:g} s TR within half a sample."
        )
    return settings.comb_fundamental


def detect_scanner_harmonics(
    frequencies_hz: np.ndarray,
    p_values: np.ndarray,
    *,
    fundamental_hz: float,
    familywise_error_rate: float,
) -> ScannerHarmonicEvidence | None:
    """Bonferroni-test every harmonic fixed in advance by scanner timing."""
    frequencies = np.asarray(frequencies_hz, dtype=float)
    probabilities = np.asarray(p_values, dtype=float)
    if (
        frequencies.ndim != 1
        or frequencies.size < 2
        or not np.all(np.isfinite(frequencies))
        or not np.all(np.diff(frequencies) > 0.0)
    ):
        raise ValueError(
            "Scanner-harmonic frequencies require at least two finite increasing bins."
        )
    if probabilities.ndim != 3 or probabilities.shape[-1] != frequencies.size:
        raise ValueError(
            "Scanner-comb p-values must have window, channel, and frequency axes."
        )
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("Scanner-comb p-values must be finite probabilities.")
    if not np.isfinite(fundamental_hz) or fundamental_hz <= 0.0:
        raise ValueError("The scanner fundamental must be finite and positive.")
    if not np.isfinite(familywise_error_rate) or not (
        0.0 < familywise_error_rate < 1.0
    ):
        raise ValueError("familywise_error_rate must lie strictly between zero and one.")

    first_harmonic = max(1, int(np.ceil(frequencies[0] / fundamental_hz)))
    last_harmonic = int(np.floor(frequencies[-1] / fundamental_hz))
    harmonics = np.arange(first_harmonic, last_harmonic + 1)
    if harmonics.size < 1:
        return None

    bin_width_hz = float(np.median(np.diff(frequencies)))
    harmonic_p_values = []
    for harmonic in harmonics:
        target_hz = harmonic * fundamental_hz
        distances_hz = np.abs(frequencies - target_hz)
        nearest = distances_hz <= np.min(distances_hz) + np.finfo(float).eps
        group = probabilities[..., nearest]
        harmonic_p_values.append(
            min(1.0, float(np.min(group)) * group.size)
            if np.min(distances_hz) <= bin_width_hz / 2.0 + np.finfo(float).eps
            else 1.0
        )

    harmonic_probabilities = np.asarray(harmonic_p_values)
    corrected_harmonic_probabilities = np.minimum(
        1.0,
        harmonics.size * harmonic_probabilities,
    )
    supporting = tuple(
        int(harmonic)
        for harmonic, probability in zip(
            harmonics,
            corrected_harmonic_probabilities,
            strict=True,
        )
        if probability < familywise_error_rate
    )
    if not supporting:
        return None
    ordered_probabilities = np.sort(corrected_harmonic_probabilities)
    authorization_index = 1 if len(supporting) >= 2 else 0
    corrected_p_value = float(ordered_probabilities[authorization_index])
    return ScannerHarmonicEvidence(
        fundamental_hz=float(fundamental_hz),
        corrected_p_value=corrected_p_value,
        supporting_harmonics=supporting,
        window_count=probabilities.shape[0],
        channel_count=probabilities.shape[1],
        frequency_count=harmonics.size,
    )


def _eeg_estimation_windows(raw, settings) -> tuple[np.ndarray, float, float]:
    """Return validated as-recorded EEG windows and their spectral limits."""
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(picks) < 2:
        raise ValueError(
            "Line detection requires at least two non-bad EEG channels."
        )
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
    return windows, sampling_frequency_hz, maximum_hz


def _thomson_f_p_values(raw, settings) -> tuple[np.ndarray, np.ndarray]:
    """Raw Thomson F-test p-values for every as-recorded non-bad EEG channel."""
    windows, sampling_frequency_hz, maximum_hz = _eeg_estimation_windows(
        raw,
        settings,
    )
    return lines.thomson_f_p_values(
        windows,
        sampling_frequency_hz,
        frequency_range_hz=(settings.frequency_range_hz[0], maximum_hz),
    )


def _line_test_p_values(raw, settings) -> tuple[np.ndarray, np.ndarray]:
    """Combined coherent-sinusoid and persistent narrowband p-values."""
    windows, sampling_frequency_hz, maximum_hz = _eeg_estimation_windows(
        raw,
        settings,
    )
    return lines.line_test_p_values(
        windows,
        sampling_frequency_hz,
        frequency_range_hz=(settings.frequency_range_hz[0], maximum_hz),
    )


def detect_channel_lines(
    raw,
    settings,
) -> lines.LineDetectionResult:
    """Complementary line-shape tests with recording-family Holm correction."""
    frequencies_hz, p_values = _line_test_p_values(raw, settings)
    return lines.detect_lines_from_p_values(
        frequencies_hz,
        p_values,
        familywise_error_rate=settings.familywise_error_rate,
    )


def detect_residual_channel_lines(raw, settings) -> lines.LineDetectionResult:
    """Thomson-only refit after filtering has shaped the local power spectrum."""
    frequencies_hz, p_values = _thomson_f_p_values(raw, settings)
    return lines.detect_lines_from_p_values(
        frequencies_hz,
        p_values,
        familywise_error_rate=settings.familywise_error_rate,
    )


def _detect_lines_for_round(
    raw,
    settings,
    round_index: int,
) -> lines.LineDetectionResult:
    """Use the source-only peak family once, then coherent residual refits."""
    if round_index == 1:
        return detect_channel_lines(raw, settings)
    return detect_residual_channel_lines(raw, settings)


def fit_harmonic_model(raw, settings, *, round_index: int = 1):
    """Fit recording-family Holm-corrected spectral lines."""
    round_settings = settings.for_round(round_index)
    result = _detect_lines_for_round(raw, round_settings, round_index)
    return _line_model_from_detection(raw, result)


def fit_harmonic_round(
    raw,
    settings: HarmonicNotchSettings,
    *,
    round_index: int = 1,
) -> HarmonicRoundEvidence:
    """Fit the exact joint evidence used by one production removal round."""
    round_settings = settings.for_round(round_index)
    line_settings = replace(
        round_settings,
        familywise_error_rate=round_settings.familywise_error_rate / 2.0,
    )
    line_detection = _detect_lines_for_round(
        raw,
        line_settings,
        round_index,
    )
    model = _line_model_from_detection(raw, line_detection)
    plans = plan_channel_notches(model, round_settings)

    scanner_settings = replace(
        round_settings,
        estimation_window_s=SCANNER_HARMONIC_ESTIMATION_WINDOW_S,
        familywise_error_rate=round_settings.familywise_error_rate / 2.0,
    )
    scanner_frequencies_hz, scanner_p_values = _thomson_f_p_values(
        raw,
        scanner_settings,
    )
    scanner_harmonics = detect_scanner_harmonics(
        scanner_frequencies_hz,
        scanner_p_values,
        fundamental_hz=scanner_fundamental_hz(raw, settings),
        familywise_error_rate=scanner_settings.familywise_error_rate,
    )
    scanner_plan = (
        None
        if scanner_harmonics is None
        else _plan_scanner_harmonics_for_recording(raw, scanner_harmonics, round_settings)
    )
    return HarmonicRoundEvidence(model, plans, scanner_harmonics, scanner_plan)


def _line_model_from_detection(raw, result) -> lines.LineModel:
    """Attach recording channel identities to line detections."""
    return lines.build_line_model(
        result,
        channel_names=eeg_channel_names(raw),
    )


def clean_until_no_supported_lines(
    raw,
    settings: HarmonicNotchSettings,
    *,
    n_jobs: int = -1,
) -> HarmonicCleaningResult:
    """Apply FIR rounds until line and trigger-anchored comb tests are null."""
    return _clean_until_model_null(raw, settings, n_jobs=n_jobs)


def _clean_until_model_null(raw, settings, *, n_jobs: int = -1) -> HarmonicCleaningResult:
    """Iterate line and scanner-comb evidence to their joint terminal null."""
    cleaned = raw.copy()
    rounds = []

    while True:
        round_index = len(rounds) + 1
        round_settings = settings.for_round(round_index)
        evidence = fit_harmonic_round(
            cleaned,
            settings,
            round_index=round_index,
        )
        if evidence.filter_plan is None:
            return HarmonicCleaningResult(
                cleaned,
                tuple(rounds),
                evidence.model,
                evidence.scanner_harmonics,
            )

        filter_plan = evidence.filter_plan
        filtered = apply_harmonic_notches(cleaned, filter_plan, n_jobs=n_jobs)
        if np.array_equal(filtered.get_data(), cleaned.get_data()):
            raise RuntimeError(
                "A supported residual line remains, but its FIR round changed no samples."
            )
        changes_db = _measure_channel_stopband_changes(
            cleaned,
            filtered,
            evidence.plans,
            round_settings,
        )
        if evidence.scanner_plan is not None:
            changes_db += _measure_scanner_stopband_changes(
                cleaned,
                filtered,
                evidence.scanner_plan,
                round_settings,
            )
        rounds.append(
            HarmonicRemovalRound(
                evidence.model,
                evidence.plans,
                filter_plan,
                changes_db,
                evidence.scanner_harmonics,
                evidence.scanner_plan,
            )
        )
        cleaned = filtered


def apply_harmonic_notches(raw, plan: HarmonicNotchPlan, *, n_jobs: int = -1):
    """Return a copy with the evidence-bounded line intervals removed from EEG."""
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    if len(picks) == 0:
        raise ValueError("Harmonic notching requires at least one EEG channel.")
    filtered = raw.copy()
    _apply_harmonic_notches(filtered, plan, picks, n_jobs=n_jobs)
    return filtered


def apply_channel_notches(
    raw,
    plans: Sequence[ChannelNotchPlan],
    *,
    n_jobs: int = -1,
):
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
        _apply_harmonic_notches(filtered, geometry, names, n_jobs=n_jobs)
    return filtered


def _apply_harmonic_notches(raw, plan, picks, *, n_jobs: int = -1):
    """Apply one validated FIR geometry to selected channels in place."""
    sampling_frequency_hz = float(raw.info["sfreq"])
    nyquist_hz = sampling_frequency_hz / 2.0
    unavailable_edges = plan.unavailable_edges()
    if unavailable_edges[0][0] <= 0.0:
        raise ValueError("The first line-notch transition reaches 0 Hz.")
    if unavailable_edges[-1][1] >= nyquist_hz:
        raise ValueError(
            f"The last line-notch transition reaches the {nyquist_hz:g} Hz Nyquist limit."
        )
    filter_length = characterize_harmonic_filter(
        sampling_frequency_hz,
        plan,
    ).length_samples
    short_segments = tuple(
        stop - start
        for start, stop in recordings.acquisition_segments(raw)
        if stop - start < filter_length
    )
    if short_segments:
        raise ValueError(
            f"A continuous acquisition span has {min(short_segments)} samples, shorter "
            f"than the {filter_length}-sample FIR. MNE warns that this geometry is "
            "likely to distort the signal."
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
        n_jobs=n_jobs,
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
    availability = _band_availability_fields(plan, analysed_bands)

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
            **availability,
        }
        rows.append(row)
    return rows


def band_availability_from_intervals(
    intervals: Sequence[tuple[float, float]],
    analysed_bands: tuple[tuple[str, float, float], ...],
) -> dict[str, float]:
    """Unavailable and retained shares per study band, from bare intervals."""
    shares = {
        name: sum(
            _interval_overlap_hz(interval, (low_hz, high_hz)) for interval in intervals
        )
        / (high_hz - low_hz)
        for name, low_hz, high_hz in analysed_bands
    }
    return {
        field: value
        for name, share in shares.items()
        for field, value in (
            (f"{name}_unavailable_share", share),
            (f"{name}_retained_share", 1.0 - share),
        )
    }


def _band_availability_fields(
    plan: HarmonicNotchPlan,
    analysed_bands: tuple[tuple[str, float, float], ...],
) -> dict[str, float]:
    """Recording-wide unavailable and retained shares for each study band."""
    return band_availability_from_intervals(plan.unavailable_edges(), analysed_bands)


def line_manifest_rows(
    recording: str,
    model: lines.LineModel,
    plans: Sequence[ChannelNotchPlan],
    analysed_bands: tuple[tuple[str, float, float], ...],
    settings: HarmonicNotchSettings,
    *,
    round_index: int = 1,
) -> list[dict[str, float | int | str]]:
    """Attach channel-local statistical evidence to every channel stopband."""
    channel_models = {channel.channel_name: channel for channel in model.channels}
    channel_plans = tuple(plans)
    if {plan.channel_name for plan in channel_plans} != set(channel_models):
        raise ValueError("Channel plans must cover exactly the affected EEG channels.")
    if not channel_plans:
        return [
            _null_line_manifest_row(
                recording,
                model,
                settings,
                round_index=round_index,
            )
        ]

    manifest_rows = []
    for channel_plan in channel_plans:
        channel = channel_models[channel_plan.channel_name]
        rows = harmonic_exclusion_rows(
            recording,
            channel_plan.geometry,
            analysed_bands,
        )
        assigned_positions = []
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
                    "outcome": "line_detected",
                    "channel": channel.channel_name,
                    "detected_line_frequencies_hz": ";".join(
                        f"{line.position_hz:.17g}" for line in supported
                    ),
                    "detected_line_input_p_values": ";".join(
                        f"{line.raw_p_value:.17g}" for line in supported
                    ),
                    "detected_line_corrected_p_values": ";".join(
                        f"{line.corrected_p_value:.17g}" for line in supported
                    ),
                    "detected_line_window_indices": ";".join(
                        ",".join(str(index) for index in line.window_indices)
                        for line in supported
                    ),
                    "fundamental_hz": "",
                    "scanner_family_corrected_p_value": "",
                    "scanner_supporting_harmonics": "",
                    "scanner_repetition_time_s": settings.scanner_repetition_time_s,
                    "scanner_trigger_event_name": settings.scanner_trigger_event_name,
                    "multiple_testing_method": _line_method_for_round(
                        round_index
                    ),
                    "multiple_testing_scope": (
                        "as_recorded_non_bad_eeg_recording_removal_sequence"
                    ),
                    "familywise_error_rate": settings.familywise_error_rate,
                    "round_familywise_error_rate": (
                        settings.error_rate_for_round(round_index) / 2.0
                    ),
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


def scanner_harmonic_manifest_rows(
    recording: str,
    evidence: ScannerHarmonicEvidence,
    plan: HarmonicNotchPlan,
    analysed_bands: tuple[tuple[str, float, float], ...],
    settings: HarmonicNotchSettings,
    *,
    round_index: int,
) -> list[dict[str, float | int | str]]:
    """Attach prespecified scanner-harmonic evidence to its authorized plan."""
    rows = harmonic_exclusion_rows(recording, plan, analysed_bands)
    supporting = ";".join(str(value) for value in evidence.supporting_harmonics)
    for row in rows:
        row.update(
            {
                "outcome": "scanner_harmonics_detected",
                "channel": "",
                "detected_line_frequencies_hz": "",
                "detected_line_input_p_values": "",
                "detected_line_corrected_p_values": "",
                "detected_line_window_indices": "",
                "fundamental_hz": evidence.fundamental_hz,
                "scanner_family_corrected_p_value": evidence.corrected_p_value,
                "scanner_supporting_harmonics": supporting,
                "scanner_repetition_time_s": settings.scanner_repetition_time_s,
                "scanner_trigger_event_name": settings.scanner_trigger_event_name,
                "multiple_testing_method": "bonferroni",
                "multiple_testing_scope": (
                    "trigger_prespecified_harmonics_across_windows_channels_and_recording"
                ),
                "familywise_error_rate": settings.familywise_error_rate,
                "round_familywise_error_rate": (
                    settings.error_rate_for_round(round_index) / 2.0
                ),
                "estimation_window_count": evidence.window_count,
                "tested_eeg_channel_count": evidence.channel_count,
                "detection_test_count_per_channel": (
                    evidence.window_count * evidence.frequency_count
                ),
                "total_detection_test_count": (
                    evidence.window_count
                    * evidence.channel_count
                    * evidence.frequency_count
                ),
            }
        )
    return rows


def cleaning_manifest_rows(
    recording: str,
    result: HarmonicCleaningResult,
    analysed_bands: tuple[tuple[str, float, float], ...],
    settings: HarmonicNotchSettings,
) -> list[dict[str, float | int | str]]:
    """Record every supported FIR round followed by its terminal Holm null."""
    sampling_frequency_hz = float(result.cleaned.info["sfreq"])
    cumulative_availability = (
        _band_availability_fields(
            merge_recording_plans(
                tuple(round_.filter_plan for round_ in result.rounds)
            ),
            analysed_bands,
        )
        if result.rounds
        else {
            field: value
            for name, _, _ in analysed_bands
            for field, value in (
                (f"{name}_unavailable_share", 0.0),
                (f"{name}_retained_share", 1.0),
            )
        }
    )
    manifest_rows = []
    for round_index, removal_round in enumerate(result.rounds, start=1):
        rows = []
        if removal_round.model.channels:
            rows.extend(
                line_manifest_rows(
                    recording,
                    removal_round.model,
                    removal_round.plans,
                    analysed_bands,
                    settings,
                    round_index=round_index,
                )
            )
        if removal_round.scanner_harmonics is not None:
            rows.extend(
                scanner_harmonic_manifest_rows(
                    recording,
                    removal_round.scanner_harmonics,
                    removal_round.scanner_plan,
                    analysed_bands,
                    settings,
                    round_index=round_index,
                )
            )
        design = characterize_harmonic_filter(
            sampling_frequency_hz,
            removal_round.filter_plan,
        )
        for row, change_db in zip(
            rows,
            removal_round.in_stopband_changes_db,
            strict=True,
        ):
            row["removal_round"] = round_index
            row.update(design.manifest_fields())
            row.update(cumulative_availability)
            row["in_stopband_change_db"] = change_db
        manifest_rows.extend(rows)

    terminal_rows = line_manifest_rows(
        recording,
        result.residual_model,
        (),
        analysed_bands,
        settings,
        round_index=len(result.rounds) + 1,
    )
    terminal = terminal_rows[0]
    terminal.update(
        {
            "removal_round": len(result.rounds) + 1,
            "fir_filter_length_samples": "",
            "fir_filter_length_s": "",
            "fir_minimum_stopband_attenuation_db": "",
            "fir_maximum_passband_deviation_db": "",
            "in_stopband_change_db": "",
        }
    )
    terminal.update(cumulative_availability)
    manifest_rows.append(terminal)
    return manifest_rows


def _null_line_manifest_row(
    recording: str,
    model: lines.LineModel,
    settings: HarmonicNotchSettings,
    *,
    round_index: int,
) -> dict[str, float | int | str]:
    """Represent a valid null result without inventing filter geometry."""
    return {
        "recording": recording,
        "outcome": "no_line_detected",
        "channel": "",
        "kind": "",
        "harmonics": "",
        "stopband_low_hz": "",
        "stopband_high_hz": "",
        "unavailable_low_hz": "",
        "unavailable_high_hz": "",
        "transition_bandwidth_hz": "",
        "detected_line_frequencies_hz": "",
        "detected_line_input_p_values": "",
        "detected_line_corrected_p_values": "",
        "detected_line_window_indices": "",
        "fundamental_hz": "",
        "scanner_family_corrected_p_value": "",
        "scanner_supporting_harmonics": "",
        "scanner_repetition_time_s": settings.scanner_repetition_time_s,
        "scanner_trigger_event_name": settings.scanner_trigger_event_name,
        "multiple_testing_method": (
            f"{_line_method_for_round(round_index)}_and_scanner_bonferroni"
        ),
        "multiple_testing_scope": (
            "joint_as_recorded_line_and_trigger_anchored_scanner_families"
        ),
        "familywise_error_rate": settings.familywise_error_rate,
        "round_familywise_error_rate": (
            settings.error_rate_for_round(round_index) / 2.0
        ),
        "estimation_window_count": model.window_count,
        "tested_eeg_channel_count": model.channel_count,
        "detection_test_count_per_channel": model.test_count_per_channel,
        "total_detection_test_count": (
            model.test_count_per_channel * model.channel_count
        ),
    }


def _line_method_for_round(round_index: int) -> str:
    """Statistical line-shape family fitted at one pre-allocated round."""
    if round_index == 1:
        return "bonferroni_two_shape_union_then_holm"
    return "holm"


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
    if outcomes == {"no_line_detected"}:
        if len(rows) != 1:
            raise ValueError("A null recording must have exactly one manifest row.")
        return ()
    if outcomes != {"line_detected"}:
        raise ValueError("A recording cannot mix line and null manifest outcomes.")
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


def removal_rounds_from_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[HarmonicNotchPlan, ...]:
    """Reconstruct the ordered FIR cascade and require a terminal null round."""
    manifest_rows = tuple(rows)
    if not manifest_rows:
        raise ValueError("A recording's line-notch manifest is empty.")
    try:
        round_indices = tuple(
            _removal_round_index(row["removal_round"])
            for row in manifest_rows
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Every manifest row requires an integer removal_round.") from error
    unique_rounds = tuple(sorted(set(round_indices)))
    if unique_rounds != tuple(range(1, unique_rounds[-1] + 1)):
        raise ValueError("Manifest removal rounds must be contiguous from one.")

    plan_rounds = []
    for round_index in unique_rounds:
        block = [
            row
            for row, row_round in zip(manifest_rows, round_indices, strict=True)
            if row_round == round_index
        ]
        outcomes = {str(row["outcome"]) for row in block}
        is_terminal = round_index == unique_rounds[-1]
        if is_terminal:
            if outcomes != {"no_line_detected"} or len(block) != 1:
                raise ValueError("The final removal round must be one terminal null row.")
            continue
        if not outcomes <= {"line_detected", "scanner_harmonics_detected"}:
            raise ValueError(
                "Every non-terminal removal round must contain line or scanner-comb "
                "evidence."
            )
        geometries = []
        line_rows = [row for row in block if str(row["outcome"]) == "line_detected"]
        if line_rows:
            geometries.extend(
                plan.geometry for plan in channel_plans_from_rows(line_rows)
            )
        scanner_rows = [
            row for row in block if str(row["outcome"]) == "scanner_harmonics_detected"
        ]
        if scanner_rows:
            geometries.append(harmonic_plan_from_rows(scanner_rows))
        plan_rounds.append(merge_recording_plans(tuple(geometries)))
    return tuple(plan_rounds)


def _removal_round_index(value: object) -> int:
    """Return one strictly positive integer manifest round index."""
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 1.0 or not numeric.is_integer():
        raise ValueError("removal_round must be a positive integer.")
    return int(numeric)


def validate_residual_postcondition(
    raw,
    settings,
    *,
    round_index: int = 1,
) -> lines.LineModel:
    """Require a joint line and trigger-anchored null at the next test level."""
    evidence = fit_harmonic_round(
        raw,
        settings,
        round_index=round_index,
    )
    residual_model = evidence.model
    if residual_model.channels:
        raise RuntimeError(
            "The cleaned derivative contains "
            f"{residual_model.line_count} Holm-significant residual line(s) across "
            f"{len(residual_model.channels)} EEG channel(s)."
        )

    if evidence.scanner_harmonics is not None:
        raise RuntimeError(
            "The cleaned derivative contains a statistically authorized "
            "trigger-anchored scanner-comb residual."
        )
    return residual_model


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


def _measure_scanner_stopband_changes(
    raw_before,
    raw_after,
    plan: HarmonicNotchPlan,
    settings: HarmonicNotchSettings,
) -> tuple[float, ...]:
    """Measure trigger-anchored stopbands in the equal-channel mean spectrum."""
    import mne

    picks = mne.pick_types(raw_before.info, eeg=True, exclude="bads")
    frequencies_hz, before_psd = recordings.psd(raw_before, picks, settings)
    after_frequencies_hz, after_psd = recordings.psd(raw_after, picks, settings)
    if not np.array_equal(frequencies_hz, after_frequencies_hz):
        raise ValueError("Before and after spectra use different frequency grids.")
    before_mean = before_psd.mean(axis=0)
    after_mean = after_psd.mean(axis=0)
    return tuple(
        _change_db(
            band_power(
                frequencies_hz,
                before_mean,
                stopband.low_hz,
                stopband.high_hz,
            ),
            band_power(
                frequencies_hz,
                after_mean,
                stopband.low_hz,
                stopband.high_hz,
            ),
        )
        for stopband in plan.stopbands
    )


def clean_harmonic_run(
    vhdr: Path,
    output_root: Path,
    source_root: Path,
    settings,
    analysed_bands: tuple[tuple[str, float, float], ...],
    *,
    n_jobs: int = -1,
) -> list[dict[str, float | str]]:
    """Fit supported residual rounds, write, and require a terminal null."""
    from decomb import subtraction

    raw = recordings.read_bids_raw(vhdr)
    evidence = fit_harmonic_round(raw, settings, round_index=1)
    recovered, record = subtraction.subtract_authorized(
        raw,
        evidence,
        settings,
        n_jobs=n_jobs,
    )
    result = clean_until_no_supported_lines(recovered, settings, n_jobs=n_jobs)
    filtered = result.cleaned

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
    validate_residual_postcondition(
        written,
        settings,
        round_index=len(result.rounds) + 1,
    )

    rows = record.manifest_rows(vhdr.stem, analysed_bands, settings)
    rows.extend(
        cleaning_manifest_rows(
            vhdr.stem,
            result,
            analysed_bands,
            settings,
        )
    )
    for row in rows:
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
                "Complementary Thomson sinusoid and persistent narrowband-power tests "
                "identified source components in the as-recorded non-bad EEG channels. "
                "Their p-values were Bonferroni-combined before recording-family Holm "
                "correction; residual rounds used Thomson tests because prior notches "
                "shape local power. Each pre-allocated "
                "round split its error rate equally between (1) Holm correction across "
                "every channel, continuous estimation window, and tested frequency and "
                "(2) a trigger-anchored scanner-harmonic test. The scanner test used the "
                "configured TR and exact event name, Bonferroni-corrected each expected "
                "harmonic across windows, channels, and the prespecified harmonic grid. "
                "Each supported harmonic authorized its own local-background "
                "envelope; no unsupported tooth on the prespecified grid was "
                "notched. A "
                "summable alpha-spending sequence pre-allocated error rates across "
                "adaptive removal rounds. Supported components were "
                "merged into one recording plan and removed from every EEG channel with "
                "zero-phase MNE FIR notches, then the "
                "complete residual families were tested again at their pre-allocated "
                "rates. Filtering continued until both fresh tests were null. "
                "Every removal round, stopband, transition, and terminal null is listed "
                f"with its channel evidence in {MANIFEST_NAME}."
            ),
            "Parameters": {
                "multiple_testing_method": MULTIPLE_TESTING_METHOD,
                "familywise_error_unit": (
                    "as_recorded_non_bad_eeg_recording_removal_sequence"
                ),
                "detection_reference": "as_recorded_non_bad_eeg_channels",
                "filter_scope": "all_eeg_channels",
                "spatial_invariance": "identical_recording_plan_for_every_eeg_channel",
                "convergence_rule": "fresh_joint_line_and_scanner_harmonics_null",
                "multiple_testing_scope": (
                    "recording_wide_alpha_spending_split_equally_between_test_families"
                ),
                "alpha_spending_rule": settings.alpha_spending_rule,
                "scanner_harmonics_estimation_window_s": (
                    SCANNER_HARMONIC_ESTIMATION_WINDOW_S
                ),
                "scanner_harmonics_local_supporting_harmonics": 1,
                "scanner_harmonics_complete_comb_supporting_harmonics": 2,
                "scanner_harmonics_target_rule": (
                    "one_supported_harmonic_targets_its_tooth_"
                    "two_target_complete_comb"
                ),
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
                "frequency_bin_width_hz": settings.frequency_bin_width_hz,
                "ordinary_line_stopband_width_hz": (
                    settings.ordinary_line_stopband_width_hz
                ),
                "supported_scanner_harmonic_stopband_width_hz": (
                    settings.supported_scanner_harmonic_stopband_width_hz
                ),
                "filter_resolution_window_s": (
                    settings.filter_resolution_window_s
                ),
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
            scanner_repetition_time_s=float(
                parameters["scanner_repetition_time_s"]
            ),
            scanner_trigger_event_name=str(
                parameters["scanner_trigger_event_name"]
            ),
            comb_fundamental_hz=(
                None
                if parameters.get("comb_fundamental_hz") is None
                else float(parameters["comb_fundamental_hz"])
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "The decomb derivative does not contain valid apply-time settings."
        ) from error

    expected_provenance = {
        "multiple_testing_method": MULTIPLE_TESTING_METHOD,
        "familywise_error_unit": "as_recorded_non_bad_eeg_recording_removal_sequence",
        "detection_reference": "as_recorded_non_bad_eeg_channels",
        "filter_scope": "all_eeg_channels",
        "spatial_invariance": "identical_recording_plan_for_every_eeg_channel",
        "convergence_rule": "fresh_joint_line_and_scanner_harmonics_null",
        "multiple_testing_scope": (
            "recording_wide_alpha_spending_split_equally_between_test_families"
        ),
        "alpha_spending_rule": applied_settings.alpha_spending_rule,
        "scanner_harmonics_estimation_window_s": SCANNER_HARMONIC_ESTIMATION_WINDOW_S,
        "scanner_harmonics_local_supporting_harmonics": 1,
        "scanner_harmonics_complete_comb_supporting_harmonics": 2,
        "scanner_harmonics_target_rule": (
            "one_supported_harmonic_targets_its_tooth_two_target_complete_comb"
        ),
        "supported_scanner_harmonic_stopband_width_hz": (
            applied_settings.supported_scanner_harmonic_stopband_width_hz
        ),
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
        "frequency_bin_width_hz": applied_settings.frequency_bin_width_hz,
        "ordinary_line_stopband_width_hz": (
            applied_settings.ordinary_line_stopband_width_hz
        ),
        "filter_resolution_window_s": (
            applied_settings.filter_resolution_window_s
        ),
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

    from decomb import effective, subtraction
    from decomb.config import load_config

    mne.set_log_level("ERROR")
    config = load_config(getattr(args, "config", None))
    source_root = config.path("bids_root", override=getattr(args, "bids_root", None))
    output_root = config.path("output_root", override=getattr(args, "output_root", None))
    report_dir = config.path("removal_dir", override=getattr(args, "report_dir", None))
    settings = HarmonicNotchSettings.from_config(config)
    analysed_bands = analysed_bands_from_config(config)
    n_jobs = (
        recordings.n_jobs_from_config(config)
        if getattr(args, "n_jobs", None) is None
        else recordings.validated_n_jobs(args.n_jobs)
    )
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
    print(f"Applying automatic line and scanner-comb notches to {len(runs)} recordings")
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
            n_jobs=n_jobs,
        )
        rows.extend(measured)
        notched = subtraction.notch_rows(measured)
        detected_rows = [
            row for row in notched if row["outcome"] != "no_line_detected"
        ]
        if not detected_rows:
            outcome = (
                "no residual line after subtraction"
                if subtraction.subtraction_rows(measured)
                else "no authorized line or scanner comb; copied unchanged"
            )
            print(
                f"[{index}/{len(runs)}] {vhdr.stem[:44]:44s} "
                f"{outcome} ({time.time() - started:.0f}s)"
            )
            continue
        filter_plans = removal_rounds_from_rows(notched)
        filter_stopband_count = sum(
            len(plan.stopbands) for plan in filter_plans
        )
        stopband_width_hz = sum(
            stopband.width_hz
            for plan in filter_plans
            for stopband in plan.stopbands
        )
        median_change_db = float(
            np.median(
                [float(row["in_stopband_change_db"]) for row in detected_rows]
            )
        )
        print(
            f"[{index}/{len(runs)}] {vhdr.stem[:44]:44s} "
            f"{filter_stopband_count} recording stopbands, "
            f"{stopband_width_hz:.3f} Hz, "
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
    """Require an ordered supported cascade ending in an explicit Holm null."""
    rows = tuple(manifest_rows)
    has_rounds = ["removal_round" in row for row in rows]
    if not rows or any(has_rounds) != all(has_rounds):
        raise ValueError("Manifest rows must consistently declare removal_round.")
    if not any(has_rounds):
        _validate_round_manifest_evidence(rows, settings)
        return

    removal_rounds_from_rows(rows)
    round_indices = tuple(int(row["removal_round"]) for row in rows)
    for round_index in sorted(set(round_indices)):
        block = tuple(
            row
            for row, row_round in zip(rows, round_indices, strict=True)
            if row_round == round_index
        )
        _validate_round_manifest_evidence(block, settings)

def _validate_round_manifest_evidence(
    manifest_rows: Sequence[Mapping[str, object]],
    settings: HarmonicNotchSettings,
) -> None:
    """Require one round to carry valid line, scanner-comb, or joint-null evidence."""
    outcomes = {str(row["outcome"]) for row in manifest_rows}
    allowed = {"line_detected", "scanner_harmonics_detected", "no_line_detected"}
    if not outcomes <= allowed:
        raise ValueError("Manifest contains an unknown statistical outcome.")
    if "no_line_detected" in outcomes and len(outcomes) != 1:
        raise ValueError("A recording cannot mix detected and null outcomes.")
    error_rates = {float(row["familywise_error_rate"]) for row in manifest_rows}
    if error_rates != {settings.familywise_error_rate}:
        raise ValueError("Manifest family-wise error rate differs from apply settings.")
    round_indices = {
        1
        if "removal_round" not in row
        else _removal_round_index(row["removal_round"])
        for row in manifest_rows
    }
    if len(round_indices) != 1:
        raise ValueError("A manifest block must contain one removal round.")
    round_index = round_indices.pop()
    expected_round_error_rate = (
        settings.error_rate_for_round(round_index) / 2.0
    )
    round_error_rates = {
        float(row["round_familywise_error_rate"])
        for row in manifest_rows
    }
    if round_error_rates != {expected_round_error_rate}:
        raise ValueError("Manifest round error rate does not match alpha spending.")

    if outcomes == {"no_line_detected"}:
        if len(manifest_rows) != 1:
            raise ValueError("A null recording must have exactly one manifest row.")
        row = manifest_rows[0]
        expected_method = (
            f"{_line_method_for_round(round_index)}_and_scanner_bonferroni"
        )
        if str(row["multiple_testing_method"]) != expected_method:
            raise ValueError("A terminal null must record both statistical families.")
        if str(row["multiple_testing_scope"]) != (
            "joint_as_recorded_line_and_trigger_anchored_scanner_families"
        ):
            raise ValueError("A terminal null must declare its joint recording scope.")
        _manifest_test_counts(manifest_rows)
        null_fields = (
            "channel",
            "kind",
            "harmonics",
            "stopband_low_hz",
            "stopband_high_hz",
            "detected_line_frequencies_hz",
            "detected_line_input_p_values",
            "detected_line_corrected_p_values",
            "detected_line_window_indices",
            "fundamental_hz",
            "scanner_family_corrected_p_value",
            "scanner_supporting_harmonics",
        )
        if any(not _missing(row[name]) for name in null_fields):
            raise ValueError("A null result cannot contain line or filter evidence.")
        return

    line_rows = tuple(
        row for row in manifest_rows if str(row["outcome"]) == "line_detected"
    )
    if line_rows:
        if {str(row["multiple_testing_method"]) for row in line_rows} != {
            _line_method_for_round(round_index)
        }:
            raise ValueError("Line targets record the wrong shape-test correction.")
        if {str(row["multiple_testing_scope"]) for row in line_rows} != {
            "as_recorded_non_bad_eeg_recording_removal_sequence"
        }:
            raise ValueError("Line targets must declare their recording scope.")
        counts = _manifest_test_counts(line_rows)
        channel_names = [str(row["channel"]) for row in line_rows]
        if any(not name.strip() for name in channel_names):
            raise ValueError("Line-manifest channel names must not be empty.")
        for channel_name in dict.fromkeys(channel_names):
            channel_rows = [
                row for row in line_rows if str(row["channel"]) == channel_name
            ]
            _validate_channel_manifest_evidence(channel_rows, settings, counts)

    scanner_rows = tuple(
        row
        for row in manifest_rows
        if str(row["outcome"]) == "scanner_harmonics_detected"
    )
    if scanner_rows:
        _validate_scanner_harmonic_manifest_evidence(
            scanner_rows,
            settings,
            expected_round_error_rate,
        )


def _manifest_test_counts(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    """Return one internally consistent positive hypothesis-family shape."""
    count_fields = (
        "estimation_window_count",
        "tested_eeg_channel_count",
        "detection_test_count_per_channel",
        "total_detection_test_count",
    )
    counts = {}
    for name in count_fields:
        values = {float(row[name]) for row in rows}
        if len(values) != 1 or next(iter(values)) <= 0.0:
            raise ValueError(f"Manifest {name} must be one positive count.")
        if not next(iter(values)).is_integer():
            raise ValueError(f"Manifest {name} must be an integer.")
        counts[name] = int(next(iter(values)))
    if counts["total_detection_test_count"] != (
        counts["tested_eeg_channel_count"]
        * counts["detection_test_count_per_channel"]
    ):
        raise ValueError("Manifest total test count does not match its channel families.")
    return counts


def _validate_scanner_harmonic_manifest_evidence(
    rows: Sequence[Mapping[str, object]],
    settings: HarmonicNotchSettings,
    round_error_rate: float,
) -> None:
    """Validate a trigger-prespecified harmonic family and its exact plan."""
    if {str(row["multiple_testing_method"]) for row in rows} != {
        "bonferroni"
    }:
        raise ValueError("Scanner-harmonic targets require Bonferroni correction.")
    if {str(row["multiple_testing_scope"]) for row in rows} != {
        "trigger_prespecified_harmonics_across_windows_channels_and_recording"
    }:
        raise ValueError(
            "Scanner-harmonic targets must declare their trigger-prespecified scope."
        )
    _manifest_test_counts(rows)
    if any(str(row["channel"]).strip() for row in rows):
        raise ValueError(
            "Scanner-harmonic evidence is recording-level, not channel-level."
        )
    if {float(row["scanner_repetition_time_s"]) for row in rows} != {
        settings.scanner_repetition_time_s
    } or {str(row["scanner_trigger_event_name"]) for row in rows} != {
        settings.scanner_trigger_event_name
    }:
        raise ValueError("Manifest scanner timing differs from apply settings.")
    fundamentals = {float(row["fundamental_hz"]) for row in rows}
    if fundamentals != {settings.comb_fundamental}:
        raise ValueError(
            "Manifest comb fundamental does not equal the configured fundamental."
        )
    p_values = {float(row["scanner_family_corrected_p_value"]) for row in rows}
    if len(p_values) != 1 or not 0.0 <= next(iter(p_values)) < round_error_rate:
        raise ValueError("Manifest scanner harmonics are not statistically supported.")
    support = {
        _semicolon_ints(row["scanner_supporting_harmonics"])
        for row in rows
    }
    if len(support) != 1 or not next(iter(support)):
        raise ValueError("Scanner-harmonic evidence requires supporting harmonics.")
    supported_harmonics = next(iter(support))
    planned = tuple(
        harmonic for row in rows for harmonic in _semicolon_ints(row["harmonics"])
    )
    if planned != supported_harmonics:
        raise ValueError("The scanner-harmonic plan does not match its evidence.")
    empty_line_fields = (
        "detected_line_frequencies_hz",
        "detected_line_input_p_values",
        "detected_line_corrected_p_values",
        "detected_line_window_indices",
    )
    if any(not _missing(row[name]) for row in rows for name in empty_line_fields):
        raise ValueError("Scanner-comb rows cannot claim individual line evidence.")


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
                _semicolon_floats(row["detected_line_input_p_values"]),
                _semicolon_floats(row["detected_line_corrected_p_values"]),
                _window_index_groups(row["detected_line_window_indices"]),
                _optional_float(row["fundamental_hz"]),
                _optional_float(row["scanner_family_corrected_p_value"]),
                _semicolon_ints(row["scanner_supporting_harmonics"]),
                float(row["scanner_repetition_time_s"]),
                _text(row["scanner_trigger_event_name"]),
                _text(row["multiple_testing_method"]),
                _text(row["multiple_testing_scope"]),
                float(row["familywise_error_rate"]),
                float(row["round_familywise_error_rate"]),
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
    if any(
        not _missing(row[field])
        for row in rows
        for field in (
            "scanner_family_corrected_p_value",
            "scanner_supporting_harmonics",
        )
    ):
        raise ValueError("Channel-line rows cannot claim scanner-harmonic evidence.")
    round_error_rates = {float(row["round_familywise_error_rate"]) for row in rows}
    if len(round_error_rates) != 1:
        raise ValueError("Manifest channel evidence must use one round error rate.")
    round_error_rate = round_error_rates.pop()
    if any(not _missing(row["fundamental_hz"]) for row in rows):
        raise ValueError("Ordinary line rows cannot claim a scanner fundamental.")

    positions = []
    for row in rows:
        frequencies_hz = _semicolon_floats(row["detected_line_frequencies_hz"])
        raw_p_values = _semicolon_floats(row["detected_line_input_p_values"])
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
                raise ValueError(
                    "Manifest Holm-input and adjusted p-values are invalid."
                )
            if corrected_p_value >= round_error_rate:
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
    plan_rounds: Sequence[HarmonicNotchPlan],
) -> float:
    """Reapply the declared FIR cascade and require exact BrainVision samples."""
    _validate_matching_recordings(original, cleaned)
    filtered = original.copy()
    for plan in plan_rounds:
        filtered = apply_harmonic_notches(filtered, plan)
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
    """Refit and replay every removal round, including the terminal null."""
    original = recordings.read_bids_raw(source_vhdr)
    cleaned = recordings.read_bids_raw(cleaned_vhdr)
    _validate_matching_recordings(original, cleaned)
    _validate_manifest_evidence(manifest_rows, settings)

    indexed_rows = tuple(
        (int(row["removal_round"]), row)
        for row in manifest_rows
    )
    round_indices = tuple(sorted({index for index, _ in indexed_rows}))
    current = original
    plan_rounds = []
    verification_rows = []
    sampling_frequency_hz = float(original.info["sfreq"])
    for round_index in round_indices:
        block = tuple(row for index, row in indexed_rows if index == round_index)
        round_settings = settings.for_round(round_index)
        evidence = fit_harmonic_round(
            current,
            settings,
            round_index=round_index,
        )
        refitted_model = evidence.model
        refitted_plans = evidence.plans
        scanner_harmonics = evidence.scanner_harmonics
        scanner_plan = evidence.scanner_plan

        refitted_rows = []
        if refitted_plans:
            refitted_rows.extend(
                line_manifest_rows(
                    source_vhdr.stem,
                    refitted_model,
                    refitted_plans,
                    (),
                    settings,
                    round_index=round_index,
                )
            )
        if scanner_harmonics is not None:
            refitted_rows.extend(
                scanner_harmonic_manifest_rows(
                    source_vhdr.stem,
                    scanner_harmonics,
                    scanner_plan,
                    (),
                    settings,
                    round_index=round_index,
                )
            )
        if not refitted_rows:
            refitted_rows = line_manifest_rows(
                source_vhdr.stem,
                refitted_model,
                (),
                (),
                settings,
                round_index=round_index,
            )
        _validate_refitted_evidence(block, refitted_rows)

        if not refitted_plans and scanner_plan is None:
            verification_rows.append(
                {
                    "recording": source_vhdr.stem,
                    "removal_round": round_index,
                    "outcome": "no_line_detected",
                    "channel": "",
                    "kind": "",
                    "harmonics": "",
                    "stopband_low_hz": "",
                    "stopband_high_hz": "",
                    "unavailable_low_hz": "",
                    "unavailable_high_hz": "",
                    "verified_stopband_change_db": "",
                }
            )
            continue

        geometries = tuple(plan.geometry for plan in refitted_plans)
        if scanner_plan is not None:
            geometries = (*geometries, scanner_plan)
        filter_plan = merge_recording_plans(geometries)
        design = characterize_harmonic_filter(
            sampling_frequency_hz,
            filter_plan,
        )
        _validate_filter_design(block, design)

        filtered = apply_harmonic_notches(current, filter_plan)
        changes_db = _measure_channel_stopband_changes(
            current,
            filtered,
            refitted_plans,
            round_settings,
        )
        if scanner_plan is not None:
            changes_db += _measure_scanner_stopband_changes(
                current,
                filtered,
                scanner_plan,
                round_settings,
            )
        change_index = 0
        for channel_plan in refitted_plans:
            for stopband, unavailable in zip(
                channel_plan.geometry.stopbands,
                channel_plan.geometry.unavailable_edges(),
                strict=True,
            ):
                verification_rows.append(
                    {
                        "recording": source_vhdr.stem,
                        "removal_round": round_index,
                        "outcome": "line_detected",
                        "channel": channel_plan.channel_name,
                        "kind": stopband.kind,
                        "harmonics": ";".join(
                            str(value) for value in stopband.harmonics
                        ),
                        "stopband_low_hz": stopband.low_hz,
                        "stopband_high_hz": stopband.high_hz,
                        "unavailable_low_hz": unavailable[0],
                        "unavailable_high_hz": unavailable[1],
                        "verified_stopband_change_db": changes_db[change_index],
                    }
                )
                change_index += 1
        if scanner_plan is not None:
            for stopband, unavailable in zip(
                scanner_plan.stopbands,
                scanner_plan.unavailable_edges(),
                strict=True,
            ):
                verification_rows.append(
                    {
                        "recording": source_vhdr.stem,
                        "removal_round": round_index,
                        "outcome": "scanner_harmonics_detected",
                        "channel": "",
                        "kind": stopband.kind,
                        "harmonics": ";".join(
                            str(value) for value in stopband.harmonics
                        ),
                        "stopband_low_hz": stopband.low_hz,
                        "stopband_high_hz": stopband.high_hz,
                        "unavailable_low_hz": unavailable[0],
                        "unavailable_high_hz": unavailable[1],
                        "verified_stopband_change_db": changes_db[change_index],
                    }
                )
                change_index += 1
        plan_rounds.append(filter_plan)
        current = filtered

    maximum_sample_deviation_v = _validate_exact_derivative(
        original,
        cleaned,
        cleaned_vhdr,
        plan_rounds,
    )
    validate_residual_postcondition(
        cleaned,
        settings,
        round_index=len(plan_rounds) + 1,
    )
    for row in verification_rows:
        row["maximum_sample_deviation_v"] = maximum_sample_deviation_v
    return verification_rows


def _read_manifest(path: Path) -> pd.DataFrame:
    """Read an authored manifest without converting empty fields to NaN."""
    return pd.read_csv(
        path,
        sep="\t",
        float_precision="round_trip",
        keep_default_na=False,
    )


def run_verify(args: argparse.Namespace) -> None:
    """Refit and audit the written converged line-notch derivative."""
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
    manifest = _read_manifest(manifest_path)
    missing = MANIFEST_REQUIRED_COLUMNS - set(manifest.columns)
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
