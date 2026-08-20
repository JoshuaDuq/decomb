"""Notch comb residue that survives subtraction but that no test authorizes.

This stage is deliberately heuristic: it removes material the converged statistical rounds
would not remove, which is where its comb advantage comes from. The bandwidth it costs is
declared like any other, so the departure is from minimality, not from honesty. Rationale
and the measurements behind every constant are in the 2026-08-19 tuned-removal spec.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from decomb import notch

RESIDUAL_FLOOR_DB = 2.0
TOOTH_CANDIDATE_DB = 1.0
CLUSTER_GAP_BINS = 3.0
STOPBAND_MARGIN_BINS = 1.25
TOOTH_LOWEST_HZ = 20.0
TOOTH_HIGHEST_HZ = 95.0
PEAK_HALF_WIDTH_HZ = 0.11
REFERENCE_LOW_HZ = 1.0
REFERENCE_HIGH_HZ = 4.0
TOOTH_DEDUPLICATION_HZ = 0.05


def cluster_gap_hz(settings) -> float:
    """Bins closer than this belong to one physical line."""
    return CLUSTER_GAP_BINS * settings.frequency_bin_width_hz


def stopband_margin_hz(settings) -> float:
    """Half-width added outside a cluster when it becomes a stopband."""
    return STOPBAND_MARGIN_BINS * settings.frequency_bin_width_hz


def channel_mean_db(raw, settings) -> tuple[np.ndarray, np.ndarray]:
    """Welch spectrum averaged over EEG channels in power, returned in dB."""
    import mne

    from decomb import recordings

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    sampling_frequency_hz = float(raw.info["sfreq"])
    samples = recordings.estimation_window_samples(
        sampling_frequency_hz,
        settings.estimation_window_s,
    )
    power, frequencies = mne.time_frequency.psd_array_welch(
        raw.get_data(picks=picks),
        sampling_frequency_hz,
        fmin=1.0,
        fmax=min(100.0, np.nextafter(sampling_frequency_hz / 2.0, 0.0)),
        n_fft=samples,
        n_per_seg=samples,
        n_overlap=samples // 2,
        average="mean",
        window="hamming",
        remove_dc=True,
        verbose="ERROR",
    )
    return 10.0 * np.log10(power.mean(axis=0) * 1e12), frequencies


def prominence_db(
    decibels: np.ndarray,
    frequencies: np.ndarray,
    centre_hz: float,
) -> float:
    """Peak decibels at a frequency above the median of its 1-4 Hz neighbourhood."""
    offsets = np.abs(frequencies - centre_hz)
    peak = offsets <= PEAK_HALF_WIDTH_HZ
    reference = (offsets > REFERENCE_LOW_HZ) & (offsets <= REFERENCE_HIGH_HZ)
    if not peak.any() or reference.sum() < 3:
        return float("nan")
    return float(decibels[peak].max() - np.median(decibels[reference]))


def comb_teeth(settings, sampling_frequency_hz: float) -> np.ndarray:
    """Every comb harmonic inside the band the residual stage examines."""
    fundamental_hz = float(settings.comb_fundamental)
    nyquist_hz = np.nextafter(sampling_frequency_hz / 2.0, 0.0)
    highest_hz = min(TOOTH_HIGHEST_HZ, nyquist_hz)
    count = int(highest_hz / fundamental_hz)
    teeth = np.arange(1, count + 1) * fundamental_hz
    return teeth[(teeth >= TOOTH_LOWEST_HZ) & (teeth < nyquist_hz)]


def cluster(frequencies: Sequence[float], gap_hz: float) -> list[list[float]]:
    """Group frequencies whose neighbours lie within one gap."""
    groups: list[list[float]] = []
    for frequency in sorted(float(value) for value in frequencies):
        if groups and frequency - groups[-1][-1] <= gap_hz:
            groups[-1].append(frequency)
        else:
            groups.append([frequency])
    return groups


def threshold_stopbands(
    decibels: np.ndarray,
    frequencies: np.ndarray,
    candidates: Sequence[float],
    settings,
) -> tuple[tuple[float, float], ...]:
    """Merged stopbands for every candidate cluster still proud of the residual floor."""
    margin_hz = stopband_margin_hz(settings)
    spans = []
    for group in cluster(candidates, cluster_gap_hz(settings)):
        proudest = np.nanmax([prominence_db(decibels, frequencies, f) for f in group])
        if np.isfinite(proudest) and proudest > RESIDUAL_FLOOR_DB:
            spans.append((min(group) - margin_hz, max(group) + margin_hz))
    merged: list[list[float]] = []
    for low_hz, high_hz in sorted(spans):
        if merged and low_hz <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high_hz)
        else:
            merged.append([low_hz, high_hz])
    return tuple((low_hz, high_hz) for low_hz, high_hz in merged)


def ordinary_line_frequencies(evidence) -> tuple[float, ...]:
    """Round-one stopband centres, which is what the arm study treated as ordinary lines."""
    return tuple(
        sorted(
            {
                (stopband.low_hz + stopband.high_hz) / 2.0
                for plan in evidence.plans
                for stopband in plan.geometry.stopbands
            }
        )
    )


def subtraction_targets(raw, evidence, settings) -> tuple[float, ...]:
    """Ordinary lines, plus comb teeth standing proud enough to be worth fitting."""
    ordinary = ordinary_line_frequencies(evidence)
    decibels, frequencies = channel_mean_db(raw, settings)
    known = np.array(ordinary) if ordinary else np.array([])
    targets = list(ordinary)
    for tooth in comb_teeth(settings, float(raw.info["sfreq"])):
        proud = prominence_db(decibels, frequencies, float(tooth))
        if not np.isfinite(proud) or proud <= TOOTH_CANDIDATE_DB:
            continue
        if known.size and np.min(np.abs(known - tooth)) <= TOOTH_DEDUPLICATION_HZ:
            continue
        targets.append(float(tooth))
    return tuple(sorted(set(targets)))


@dataclass(frozen=True)
class ThresholdRecord:
    """The stopbands the residual stage notched, and the prominence that authorized each."""

    stopbands: tuple[tuple[float, float], ...]
    prominences_db: tuple[float, ...]

    def plan(self, settings) -> notch.HarmonicNotchPlan | None:
        if not self.stopbands:
            return None
        return notch.HarmonicNotchPlan(
            tuple(
                notch.HarmonicStopband((), low_hz, high_hz, "isolated")
                for low_hz, high_hz in self.stopbands
            ),
            settings.transition_bandwidth_hz,
        )

    def manifest_rows(
        self,
        recording: str,
        analysed_bands: tuple[tuple[str, float, float], ...],
        settings,
    ) -> list[dict[str, float | str]]:
        """One row per notched cluster, in the same contract as a stopband."""
        from decomb import subtraction

        plan = self.plan(settings)
        if plan is None:
            return []
        edges = plan.unavailable_edges()
        shares = notch.band_availability_from_intervals(edges, analysed_bands)
        return [
            {
                **subtraction._inapplicable_manifest_fields(settings),
                "recording": recording,
                "outcome": "residual_notched",
                "kind": "threshold_notched",
                "stopband_low_hz": stopband.low_hz,
                "stopband_high_hz": stopband.high_hz,
                "transition_bandwidth_hz": plan.transition_bandwidth_hz,
                "unavailable_low_hz": unavailable[0],
                "unavailable_high_hz": unavailable[1],
                "authorizing_prominence_db": prominence,
                **shares,
            }
            for stopband, unavailable, prominence in zip(
                plan.stopbands, edges, self.prominences_db, strict=True
            )
        ]


def fit_threshold_stage(raw, targets, settings) -> ThresholdRecord:
    """Choose stopbands for whatever still stands proud after subtraction."""
    decibels, frequencies = channel_mean_db(raw, settings)
    teeth = comb_teeth(settings, float(raw.info["sfreq"]))
    candidates = np.unique(np.concatenate([np.asarray(targets, dtype=float), teeth]))
    stopbands = threshold_stopbands(decibels, frequencies, candidates, settings)
    # Every emitted stopband spans at least one cluster whose proudest candidate cleared
    # the floor, so the nanmax below always sees a finite value. `test_residual.py` pins
    # that invariant; np.nanmax raises on an all-NaN input if it is ever broken.
    prominences = tuple(
        float(
            np.nanmax(
                [
                    prominence_db(decibels, frequencies, float(candidate))
                    for candidate in candidates
                    if low_hz <= candidate <= high_hz
                ]
            )
        )
        for low_hz, high_hz in stopbands
    )
    return ThresholdRecord(stopbands, prominences)


def threshold_rows(rows: Sequence[dict]) -> list[dict]:
    """The residual-stage rows of a manifest, empty for manifests written before it."""
    return [row for row in rows if str(row.get("kind", "")) == "threshold_notched"]


def recorded_stopbands(rows: Sequence[dict]) -> tuple[tuple[float, float], ...]:
    """Every stopband a manifest's residual rows say was notched."""
    return tuple(
        sorted(
            (float(row["stopband_low_hz"]), float(row["stopband_high_hz"]))
            for row in rows
        )
    )


def comb_analysis_mask(settings, sampling_frequency_hz: float):
    """Every comb tooth's damage interval, whether or not this recording removed it.

    A conservative mask for downstream analysis: it covers the band where the comb was
    measured, at the width a subtracted tooth already declares. It removes nothing and is
    not part of the derivative's provenance -- see the manifest for what was destroyed.
    """
    from decomb import subtraction

    return subtraction.damage_intervals(
        comb_teeth(settings, sampling_frequency_hz),
        subtraction.fit_window_s(settings),
    )
