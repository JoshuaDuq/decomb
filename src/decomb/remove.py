"""Detect the line comb in a BIDS EEG dataset, prove the removal is safe, then apply it.

    decomb benchmark
    decomb apply

``benchmark`` injects known signals into every recording, removes the lines, and reports
what survived against the criteria in :class:`decomb.estimators.PreservationGate`.
``apply`` writes a cleaned copy of the dataset, and refuses unless a passing benchmark for
the same data and the same settings is on disk. Run the benchmark first: its criteria are
stated before the measurement, so a failure means the settings are wrong, not that the
criteria should move.

The cleaned dataset keeps every sidecar byte-identical and rewrites only the ``.eeg``
binaries. Sampling rate, channel set, length and annotations are untouched, so the BIDS
contract downstream tooling relies on -- including any marker names read from
``events.tsv`` -- cannot drift.

Nothing here reads events. A resting or baseline acquisition, or any other continuous
recording, is a valid input.
"""

from __future__ import annotations

import argparse
import re
import shutil
import time
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

from decomb import catalogue, estimators, spectral

NOTCH_WIDTH_RATIO = 450.0
"""Width around each target within which bins are subtracted: ``freq / ratio``.

This, not ``mt_bandwidth``, decides how much spectrum the removal takes, because
spectrum_fit subtracts a sinusoid at every bin inside it. The width scales with frequency
because the uncertainty does: a mains-locked comb has harmonic *k* inheriting *k* times
the fundamental's wander.

The right ratio is the largest one that still pushes every line below its local
background: too small and the removal becomes a band removal, too large and the high
harmonics escape the window their own wander needs. Sweep it on a development subset and
read the cost off ``decomb benchmark``, which reports both the share of ``cost_band_hz``
lost and the worst line left standing. ``notch_width_min_hz`` sets a floor under the
result, because the lowest harmonics need more than one bin whatever the ratio says.
"""
MT_BANDWIDTH = 0.6
"""Multitaper bandwidth for the sinusoid estimate, in Hz.

The estimation band reaches half this either side of a target, so set it to the spacing of
your comb: no line's amplitude is then estimated from a band containing another.

``filter_length`` is chosen alongside it. A window too short to resolve the comb spacing
cannot work at all; one merely long enough still leaves a bin-quantised shoulder beside a
validated harmonic. Half the estimation window is a good default -- it avoids the
irregular tail window a longer choice creates inside MNE.
"""


def adaptive_window_bounds(
    *,
    n_times: int,
    window_samples: int,
    hop_samples: int,
) -> tuple[tuple[int, int], ...]:
    """Overlapping fixed-length windows that cover a run exactly once reconstructed."""
    if window_samples <= 0:
        raise ValueError("window_samples must be positive.")
    if not 0 < hop_samples < window_samples:
        raise ValueError("hop_samples must lie between zero and window_samples.")
    if n_times < window_samples:
        raise ValueError(
            f"The recording holds {n_times} samples, fewer than the "
            f"{window_samples} of one adaptive estimation window. Lower "
            "`removal.estimation_window_s`, at the cost of coarser frequency "
            "resolution, or use a longer recording."
        )

    tail_start = n_times - window_samples
    starts = list(range(0, tail_start + 1, hop_samples))
    if starts[-1] != tail_start:
        if tail_start - starts[-1] < hop_samples / 2.0:
            starts[-1] = tail_start
        else:
            starts.append(tail_start)
    bounds = tuple((start, start + window_samples) for start in starts)
    if any(
        right_start >= left_stop for (_, left_stop), (right_start, _) in zip(bounds, bounds[1:])
    ):
        raise ValueError("Adaptive estimation windows must overlap.")
    return bounds


def squared_sine_weights(
    bounds: tuple[tuple[int, int], ...],
    *,
    n_times: int,
) -> tuple[np.ndarray, ...]:
    """Positive squared-sine synthesis weights normalized to a partition of unity."""
    if not bounds:
        raise ValueError("At least one adaptive window is required.")
    raw_weights = []
    total = np.zeros(n_times, dtype=float)
    for start, stop in bounds:
        if not 0 <= start < stop <= n_times:
            raise ValueError("Adaptive window bounds lie outside the recording.")
        length = stop - start
        phase = np.pi * (np.arange(length, dtype=float) + 0.5) / length
        weight = np.sin(phase) ** 2
        raw_weights.append(weight)
        total[start:stop] += weight
    if np.any(total <= 0.0):
        raise ValueError("Adaptive windows do not cover every sample.")
    return tuple(weight / total[start:stop] for weight, (start, stop) in zip(raw_weights, bounds))


def overlap_add_segments(
    segments: tuple[np.ndarray, ...],
    bounds: tuple[tuple[int, int], ...],
    n_times: int,
) -> np.ndarray:
    """Reconstruct channel-by-time data with exact normalized overlap-add."""
    if len(segments) != len(bounds) or not segments:
        raise ValueError("segments and bounds must have the same non-zero length.")
    arrays = tuple(np.asarray(segment, dtype=float) for segment in segments)
    n_channels = arrays[0].shape[0]
    for segment, (start, stop) in zip(arrays, bounds):
        if segment.shape != (n_channels, stop - start):
            raise ValueError("Each segment must match its adaptive window bounds.")
        if not np.all(np.isfinite(segment)):
            raise ValueError("Adaptive filtered segments must be finite.")

    reconstructed = np.zeros((n_channels, n_times), dtype=float)
    for segment, weight, (start, stop) in zip(
        arrays,
        squared_sine_weights(bounds, n_times=n_times),
        bounds,
    ):
        reconstructed[:, start:stop] += segment * weight
    return reconstructed


def _coerce_setting(name: str, annotation, value):
    """Turn one YAML value into the type its setting is declared with.

    ``from __future__ import annotations`` makes every annotation a string, so this reads
    them as text. YAML gives lists where the settings want tuples, and gives integers
    where a float is wanted; both are ordinary and are converted rather than refused.
    """
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    if "None" in text and value is None:
        return None
    if text.startswith("tuple[tuple["):
        return tuple(tuple(float(edge) for edge in band) for band in value)
    if text.startswith("tuple[int"):
        return tuple(int(item) for item in value)
    if text.startswith("tuple["):
        return tuple(float(item) for item in value)
    if text.startswith("bool"):
        return bool(value)
    if text.startswith("int"):
        return int(value)
    if text.startswith("str"):
        return str(value)
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"removal.{name} must be a number; got {value!r}.") from error


@dataclass(frozen=True)
class RemovalSettings:
    """Everything the removal needs, resolved from configuration."""

    task: str = "*"
    """BIDS task label to process. ``*`` takes every task in the dataset."""
    estimation_window_s: float = 54.0
    """Length of each adaptive estimation window, in seconds.

    Sets the frequency resolution the fit works at -- ``1 / estimation_window_s`` -- and so
    the shortest recording that can be processed. It must resolve the spacing of the lines
    being removed, and the recording must hold at least one whole window.

    Where the artifact comes from a periodic acquisition, a window that is a whole number of
    those periods puts the bins in a fixed relationship to it, which is worth preferring.
    """
    max_band_cost: float | None = None
    """Optional ceiling on the share of ``cost_band_hz`` a broadband signal may lose.

    ``None`` by default, and deliberately so. The cost is already determined by the
    evidence -- the notch width times the number of targets, with each target admitted by
    the replication rules -- so any number shipped here could only be one chosen after
    seeing the answer.

    A study that wants a stated budget declares it. That declaration is its own
    scientific decision, is recorded in the derivative's provenance, and ``apply`` refuses
    against it. What the cost actually was is measured and reported either way.
    """
    cost_band_hz: tuple[float, float] = (28.0, 95.0)
    """Band every cost measurement is made over: what a broadband signal loses to the
    removal, and what the removal touched. Set it to the span your analyses use."""
    band_cost_thresholds_db: tuple[float, float] = (1.0, 3.0)
    """Loss thresholds reported for the broadband cost measurement, in dB."""
    mains_notch_hz: tuple[float, float] = estimators.MAINS_NOTCH_HZ
    """Band left to a downstream wide notch rather than taken here.

    59.5-60.5 Hz for a 60 Hz region. Set 49.5-50.5 where mains is 50 Hz. Only consulted
    when ``exclude_mains`` is true.
    """
    nominal_fundamental_hz: float = estimators.NOMINAL_FUNDAMENTAL_HZ
    harmonic_range: tuple[int, int] = estimators.COMB_HARMONIC_RANGE
    removal_harmonic_range: tuple[int, int] = estimators.REMOVAL_HARMONIC_RANGE
    search_hz: float = 0.25
    """Half-width of the window each comb harmonic is looked for in. Keep it below half
    the comb spacing, or one harmonic's search reaches its neighbour."""
    min_prominence_db: float = 1.0
    """Prominence a harmonic peak needs to take part in the fundamental fit."""
    filter_length: str = "27s"
    filter_jobs: int = 4
    mt_bandwidth: float = MT_BANDWIDTH
    notch_width_ratio: float = NOTCH_WIDTH_RATIO
    notch_width_min_hz: float = 0.05
    uncertainty_confidence_z: float = 2.0
    """Each window's fundamental uncertainty is propagated to harmonic *k* as
    ``z * k * SE(f0)`` and added to that harmonic's removal width."""
    low_hz: float = 3.0
    high_hz: float = 95.0
    background_half_width_hz: float = 100.0 / 21.6
    """Half-width of the window a bin's local background is taken from, in Hz.

    Wide enough that a line cannot raise its own background -- several comb spacings -- and
    narrow enough to follow the 1/f slope. Every prominence in this workflow is measured
    against it, so widening it raises every prominence reported.

    Whatever value you set is used exactly as given: the residual audit selects background
    bins with a continuous test, so rounding changes which bins qualify at some targets and
    moves the reported null slightly.
    """
    min_harmonics_for_fit: int = estimators.MIN_HARMONICS_FOR_FIT

    # The three tolerances below are widths in the spectrum the fit is read from, so they
    # are stated in units of that spectrum's resolution rather than in hertz. A Hann-
    # windowed pure tone has a half-power width of 1.4382 / T, which is the narrowest peak
    # the analysis can produce and therefore the natural unit for "how far is far".
    #
    # In hertz they were only right for one window length. `estimation_window_s` sets the
    # resolution -- 26.6 mHz at the shipped 54 s -- so lengthening the window to 108 s
    # halves it while a fixed 0.06 Hz stays put, silently doubling the tolerance in the
    # units that matter. Measured on a 15-participant cohort, one recording's supported
    # harmonic count fell from 24 to 19 across exactly that change while another's rose,
    # which is not a property any of these settings were meant to have.
    #
    # The multipliers reproduce the previous hertz values at 54 s to within 0.14%, so the
    # shipped behaviour is unchanged and only its response to a different window is.
    max_harmonic_residual_resolutions: float = 2.25
    max_fit_residual_rms_resolutions: float = 1.5
    max_line_width_resolutions: float = 9.4

    line_claim_hz: float = estimators.LINE_CLAIM_HZ
    residual_search_hz: float = estimators.RESIDUAL_SEARCH_HZ
    """How far either side of a target a residual is still that target's responsibility.

    The notch's own width is the wrong region to search: the failure being looked for is a
    target that missed, and a missed line then sits just outside what the notch claimed.
    This is the frequency-uncertainty scale of the estimate instead. Keep it well inside
    half the comb spacing so one target is never charged with the next harmonic's line.
    """
    residual_family_alpha: float = 0.05
    """Family-wise error rate of the Thomson F search that authorises a second-pass
    removal. The family is one channel's complete frequency grid."""
    false_discovery_rate: float = 0.05
    """FDR the residual criteria are decided at, over the recordings jointly."""
    seam_alpha: float = 0.05
    """Two-sided level of the synchronised-shift test on the overlap-add seams."""
    n_seam_controls: int = 40
    """Blind control placements the seam maximum is judged against. The smallest p-value
    attainable is ``1/(n+1)``, so this sets the resolution of that test."""
    benchmark: estimators.BenchmarkSettings = field(default_factory=estimators.BenchmarkSettings)
    """What ``decomb benchmark`` injects and what it accepts. Read from the config's
    top-level ``benchmark`` block; nested here so the fingerprint covers it."""
    roundtrip_relative_tolerance: float = 1e-6
    """Largest round-trip error accepted when reading a written binary back.

    The binaries are float32, whose 24-bit mantissa gives a relative precision near 6e-8
    of full scale. A decade of headroom above that distinguishes quantisation from
    corruption.
    """
    detection_fdr_alpha: float | None = estimators.DETECTION_FDR_ALPHA
    """Empirical-null screening level a peak must clear to enter the candidate pool."""
    detection_null_min_bins: int = 32
    detection_null_lower_percentile: float = 15.865525393145702
    """Empirical-null fitting geometry for per-recording line screening."""
    detection_min_prominence_db: float | None = estimators.LINE_PROMINENCE_FLOOR_DB
    """Optional prominence floor on top of the calibrated test. ``null`` disables it.

    A decibel bar is not invariant to the analysis -- a line gains about 3 dB per doubling
    of ``estimation_window_s`` -- so one cannot serve as the criterion. Declare one only if
    your site wants a stated minimum amplitude, and it is recorded in the provenance as the
    choice it is. It applies wherever a line is looked for, including beside a harmonic.
    """
    detection_adjacent_min_prominence_db: float = 10.0
    """Prominence a summit beside a validated harmonic needs to count as a distinct source.

    A fixed decibel bar, unlike the rest of detection, and a known limitation: prominence is
    not invariant to window length, so this means something different at every
    ``estimation_window_s``. Raise it if narrow features of your own sit beside harmonics
    and are being taken; lower it if a real neighbouring line is being missed.

    A bar cannot separate a narrow source from a narrow noise summit riding on a broad
    rhythm that happens to cross a harmonic. Where that matters, keep the affected span in
    ``notch_bands`` so this stage leaves it alone.
    """
    support_margin_hz: float = 0.0
    """Extra spectrum left either side of a peak's measured support when widening a notch.

    Zero by default: the notch then covers exactly the bins the support was observed over.
    Raise it if your line positions move between the estimation window and the recording it
    is applied to, and the extra is charged to the band cost the benchmark reports.
    """
    support_min_prominence_db: float = 10.0
    """Prominence a peak needs before its observed extent may widen a target's notch.

    Not a detection setting: it asks how far a peak already being removed reaches, and the
    answer sets how much spectrum the notch empties. Admitting more peaks here does not find
    more artifact, it removes more band, so this is deliberately conservative.
    """
    detection_low_hz: float = 20.0
    detection_high_hz: float = 100.0
    detection_search_hz: float = 0.05
    """Refinement window for a nominal that came from detection.

    A detected nominal already sits on its summit, so it needs only enough room to refine
    sub-bin, and the window has to stay narrow: an isolated line can sit well inside a
    tenth of a hertz of a harmonic, and a wider window then walks the refinement onto the
    comb member rather than the line it was given.

    Kept below ``line_claim_hz``, so a line the detector admits can never be one the
    estimator refuses.
    """
    min_runs_per_line: int = 3
    """Recordings a line must appear in before it becomes a session-wide target.

    Separate recordings of one session are replication already in hand, and they separate
    a persistent line from a one-recording fluctuation. A strong line confined to a single
    recording takes the independent within-recording route instead.
    """
    min_runs_per_block_line: int = 2
    min_independent_windows_per_line: int = 3
    """Non-overlapping windows needed to support a recording-specific line."""
    exclude_mains: bool = True
    """Leave ``mains_notch_hz`` to a wide notch elsewhere rather than taking it here.

    False moves mains into this pass, which subtracts a far narrower band: roughly 0.13 Hz
    at freq/450 against the ~1 Hz an FIR notch takes. That is worth having only if mains is
    a resolvable line in your data. It frequently is not -- a mains peak that is really a
    dense cluster of tens of non-stationary peaks cannot be reached by sinusoid subtraction
    at all, and aiming at it promotes the neighbour instead.

    Exactly one stage may remove mains. If you set this False, make sure whatever notch
    runs downstream is no longer also taking it.
    """
    excluded_bands_hz: tuple[tuple[float, float], ...] = ()
    """Bands ``decomb notch`` takes wholesale, read from the config's ``notch_bands``.

    Same division of labour as ``exclude_mains``, for the same reason and against a
    different stage. A band is declared there precisely when the contamination is a
    *cluster* -- scores of distinct non-stationary peaks packed into a hertz or less --
    and subtracting sinusoids from a cluster cannot clear it: the summit aimed at goes and
    its neighbour becomes the new summit.

    Leaving those bands targeted deadlocks the workflow rather than merely wasting effort.
    The surviving peak fails the residual criterion, which refuses ``apply``; the notch
    stage that removes the band outright reads what ``apply`` wrote, so it could never run.
    The failure scales with how much cluster power a recording carries, so it presents as
    most recordings failing and the cleanest ones passing.
    """

    @property
    def protected_bands_hz(self) -> tuple[tuple[float, float], ...]:
        """Every band some other stage owns, so this pass must leave all of it alone."""
        bands = list(self.excluded_bands_hz)
        if self.exclude_mains:
            bands.append(tuple(self.mains_notch_hz))
        return tuple(sorted(bands))

    @property
    def spectral_resolution_hz(self) -> float:
        """Narrowest peak the fit's own spectrum can produce: a Hann tone's half-power width.

        The window spectra the comb is fitted from are Hann periodograms over
        ``estimation_window_s``, and the whole-run spectrum is their mean, which averages
        the noise without sharpening the line. One resolution therefore describes both.
        """
        return spectral.hann_resolution_hz(self.estimation_window_s)

    @property
    def max_harmonic_residual_hz(self) -> float:
        """How far a peak may sit from the fitted grid and still count as a member."""
        return self.max_harmonic_residual_resolutions * self.spectral_resolution_hz

    @property
    def max_fit_residual_rms_hz(self) -> float:
        """How much the kept harmonics may scatter about the grid they fitted."""
        return self.max_fit_residual_rms_resolutions * self.spectral_resolution_hz

    @property
    def max_line_width_hz(self) -> float:
        """How wide a peak may be and still be a line rather than a rhythm."""
        return self.max_line_width_resolutions * self.spectral_resolution_hz

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("task must name a BIDS task label.")
        if not np.isfinite(self.estimation_window_s) or self.estimation_window_s <= 0.0:
            raise ValueError("estimation_window_s must be finite and positive.")
        if self.max_band_cost is not None and not 0.0 < self.max_band_cost <= 1.0:
            raise ValueError("max_band_cost must be a share of the band, or null for none.")
        low_hz, high_hz = self.mains_notch_hz
        if not np.all(np.isfinite((low_hz, high_hz))) or not 0.0 < low_hz < high_hz:
            raise ValueError("mains_notch_hz must be an increasing positive band.")
        for band in self.excluded_bands_hz:
            low, high = band
            if not np.all(np.isfinite((low, high))) or not 0.0 < low < high:
                raise ValueError(
                    f"excluded_bands_hz must hold increasing positive bands; got {band}."
                )
        low, high = self.cost_band_hz
        if not np.all(np.isfinite((low, high))) or not 0.0 <= low < high:
            raise ValueError("cost_band_hz must be an increasing non-negative band.")
        if (
            len(self.band_cost_thresholds_db) != 2
            or not np.all(np.isfinite(self.band_cost_thresholds_db))
            or not 0.0 < self.band_cost_thresholds_db[0] < self.band_cost_thresholds_db[1]
        ):
            raise ValueError("band_cost_thresholds_db must contain two increasing positives.")
        for name in (
            "uncertainty_confidence_z",
            "background_half_width_hz",
            "line_claim_hz",
            "residual_search_hz",
            "max_harmonic_residual_resolutions",
            "max_fit_residual_rms_resolutions",
            "max_line_width_resolutions",
            "roundtrip_relative_tolerance",
            "detection_null_lower_percentile",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("residual_family_alpha", "false_discovery_rate", "seam_alpha"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one.")
        if self.min_harmonics_for_fit < 3:
            raise ValueError("min_harmonics_for_fit must be at least three.")
        if self.detection_null_min_bins < 2:
            raise ValueError("detection_null_min_bins must be at least two.")
        if self.max_fit_residual_rms_resolutions > self.max_harmonic_residual_resolutions:
            raise ValueError(
                "max_fit_residual_rms_resolutions cannot exceed "
                "max_harmonic_residual_resolutions: harmonics further from the grid than "
                "the residual bound are dropped before the RMS is taken, so a larger RMS "
                "bound than that can never bind."
            )
        if self.n_seam_controls < 2:
            raise ValueError("n_seam_controls must be at least two.")
        if self.detection_fdr_alpha is None and self.detection_min_prominence_db is None:
            raise ValueError(
                "Nothing would decide which peaks are lines: set detection_fdr_alpha, "
                "detection_min_prominence_db, or both."
            )
        if self.detection_fdr_alpha is not None and not 0.0 < self.detection_fdr_alpha < 1.0:
            raise ValueError("detection_fdr_alpha must lie strictly between zero and one.")
        if not 0.0 < self.detection_search_hz < self.line_claim_hz:
            raise ValueError(
                f"detection_search_hz must lie between zero and line_claim_hz "
                f"({self.line_claim_hz} Hz)."
            )
        if self.min_runs_per_line < 2:
            raise ValueError("min_runs_per_line must require at least two independent runs.")
        if not 2 <= self.min_runs_per_block_line <= self.min_runs_per_line:
            raise ValueError(
                "min_runs_per_block_line must be at least two and no larger than min_runs_per_line."
            )
        if self.min_independent_windows_per_line < self.min_runs_per_block_line:
            raise ValueError(
                "min_independent_windows_per_line cannot be smaller than its run requirement."
            )
        if self.filter_jobs < 1:
            raise ValueError("filter_jobs must be positive.")

    @classmethod
    def from_config(cls, config) -> RemovalSettings:
        """Read ``removal`` from the workflow configuration.

        Every field above is settable by its own name, coerced to its annotated type.
        Three come from elsewhere in the file: ``task`` from ``dataset``,
        ``excluded_bands_hz`` from the top-level ``notch_bands`` -- which belongs to the
        notch stage and is read here only so this pass stays out of it -- and
        ``benchmark`` from the top-level ``benchmark`` block.
        """
        block = dict(config.get("removal") or {})
        # A key nobody reads is a setting the author believes is in force. Refuse rather
        # than ignore it, and name the misspelling.
        derived = {"task", "excluded_bands_hz", "benchmark"}
        known = {entry.name for entry in fields(cls)} - derived
        unknown = set(block) - known
        if unknown:
            raise ValueError(
                f"Unknown `removal` setting(s): {sorted(unknown)}. Known settings are "
                f"{sorted(known)}."
            )

        notch_bands = config.get("notch_bands") or ()
        for band in notch_bands:
            if not isinstance(band, Sequence) or isinstance(band, str) or len(band) != 2:
                raise ValueError(
                    f"notch_bands must hold [low, high] edge pairs; got {band!r}. The "
                    "removal reads them to stay out of the bands `decomb notch` takes."
                )

        values = {
            "task": str((config.get("dataset") or {}).get("task", cls.task)),
            "excluded_bands_hz": tuple((float(low), float(high)) for low, high in notch_bands),
            "benchmark": estimators.BenchmarkSettings.from_config(config.get("benchmark")),
        }
        for entry in fields(cls):
            if entry.name in derived or entry.name not in block:
                continue
            values[entry.name] = _coerce_setting(entry.name, entry.type, block[entry.name])
        return cls(**values)


@dataclass(frozen=True)
class AdaptiveWindowRemovalPlan:
    """One window's independently estimated transformation."""

    bounds: tuple[int, int]
    estimate: estimators.CombEstimate
    targets_hz: tuple[float, ...]
    notch_widths_hz: tuple[float, ...]
    narrow_targets_hz: tuple[float, ...]
    channel_targets_hz: tuple[tuple[float, ...], ...] | None = None
    channel_target_widths_hz: tuple[tuple[float, ...], ...] | None = None
    aggregate_residual_targets_hz: tuple[float, ...] = ()
    aggregate_residual_widths_hz: tuple[float, ...] = ()
    channel_residual_targets_hz: tuple[tuple[float, ...], ...] = ()
    channel_residual_widths_hz: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        start, stop = self.bounds
        if not 0 <= start < stop:
            raise ValueError("Adaptive-window bounds must be positive and stop-exclusive.")
        if len(self.targets_hz) != len(self.notch_widths_hz):
            raise ValueError("Adaptive-window targets and widths must match.")
        if not self.targets_hz and self.channel_targets_hz is None:
            raise ValueError("An unauthorized adaptive window cannot be cleaned.")
        if not all(np.isfinite(value) for value in (*self.targets_hz, *self.notch_widths_hz)):
            raise ValueError("Adaptive-window targets and widths must be finite.")
        if any(width <= 0.0 for width in self.notch_widths_hz):
            raise ValueError("Adaptive-window notch widths must be positive.")
        if (self.channel_targets_hz is None) != (self.channel_target_widths_hz is None):
            raise ValueError("Channel targets and widths must be supplied together.")
        if self.channel_targets_hz is not None:
            if len(self.channel_targets_hz) != len(self.channel_target_widths_hz):
                raise ValueError("Every channel target list requires matching widths.")
            if any(
                len(targets) != len(widths)
                for targets, widths in zip(
                    self.channel_targets_hz,
                    self.channel_target_widths_hz,
                )
            ):
                raise ValueError("Channel targets and widths must match.")
            values = (
                *(target for channel in self.channel_targets_hz for target in channel),
                *(width for channel in self.channel_target_widths_hz for width in channel),
            )
            if not all(np.isfinite(value) for value in values):
                raise ValueError("Channel targets and widths must be finite.")
            if any(
                width <= 0.0
                for widths in self.channel_target_widths_hz
                for width in widths
            ):
                raise ValueError("Channel target widths must be positive.")
        _validate_residual_targets(self, "Adaptive-window")

    @property
    def channel_target_plans(self) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
        """Base target plans in the channel order used by the cleaner."""
        if self.channel_targets_hz is None:
            return ((self.targets_hz, self.notch_widths_hz),)
        return tuple(zip(self.channel_targets_hz, self.channel_target_widths_hz))

    @property
    def applied_target_spans(self) -> tuple[tuple[float, float], ...]:
        """Distinct base targets and widest width actually authorized by this window."""
        spans = {}
        for targets, widths in self.channel_target_plans:
            for target, width in zip(targets, widths):
                spans[float(target)] = max(spans.get(float(target), 0.0), float(width))
        return tuple((target, spans[target]) for target in sorted(spans))


def _validate_residual_targets(window, label: str) -> None:
    """Validate aggregate and channel-local residual transforms."""
    if len(window.aggregate_residual_targets_hz) != len(window.aggregate_residual_widths_hz):
        raise ValueError("Aggregate residual targets and widths must match.")
    if len(window.channel_residual_targets_hz) != len(window.channel_residual_widths_hz):
        raise ValueError("Every channel residual-target list requires matching widths.")
    if any(
        len(targets) != len(widths)
        for targets, widths in zip(
            window.channel_residual_targets_hz,
            window.channel_residual_widths_hz,
        )
    ):
        raise ValueError("Channel residual targets and widths must match.")
    residual_targets = (
        *window.aggregate_residual_targets_hz,
        *(target for channel in window.channel_residual_targets_hz for target in channel),
    )
    residual_widths = (
        *window.aggregate_residual_widths_hz,
        *(width for channel in window.channel_residual_widths_hz for width in channel),
    )
    if not all(np.isfinite(value) for value in (*residual_targets, *residual_widths)):
        raise ValueError(f"{label} residual targets and widths must be finite.")
    if any(width <= 0.0 for width in residual_widths):
        raise ValueError(f"{label} residual targets and widths must be positive.")


@dataclass(frozen=True)
class RunRemovalPlan:
    """The immutable adaptive transformation benchmarked and applied to one run."""

    model: estimators.AdaptiveCombModel
    windows: tuple[AdaptiveWindowRemovalPlan, ...]

    @property
    def all_targets_hz(self) -> tuple[float, ...]:
        """Every distinct frequency used by any adaptive window."""
        return tuple(
            sorted(
                {
                    *(
                        target
                        for window in self.windows
                        for target, _ in window.applied_target_spans
                    ),
                    *(
                        target
                        for window in self.windows
                        for target in window.aggregate_residual_targets_hz
                    ),
                    *(
                        target
                        for window in self.windows
                        for channel in window.channel_residual_targets_hz
                        for target in channel
                    ),
                }
            )
        )

    @property
    def all_narrow_targets_hz(self) -> tuple[float, ...]:
        """Every distinct comb-adjacent source authorised by raw-data evidence."""
        return tuple(
            sorted({target for window in self.windows for target in window.narrow_targets_hz})
        )


@dataclass(frozen=True)
class SessionRunSpectra:
    """Whole-run and block spectra supplying independent isolated-line evidence."""

    whole: tuple[np.ndarray, np.ndarray, np.ndarray]
    windows: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]
    bounds: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.windows or len(self.windows) != len(self.bounds):
            raise ValueError("SessionRunSpectra requires one bound per non-empty window list.")


@dataclass(frozen=True)
class RunIsolatedLinePlan:
    """Automatically supported isolated-line targets for one recording."""

    whole_hz: tuple[float, ...]
    window_hz: tuple[tuple[float, ...], ...]
    narrow_window_hz: tuple[tuple[float, ...], ...]
    source_count: int

    def __post_init__(self) -> None:
        if not self.window_hz:
            raise ValueError("An isolated-line plan requires at least one adaptive window.")
        if len(self.window_hz) != len(self.narrow_window_hz):
            raise ValueError("Every adaptive window requires one narrow-target list.")
        values = (
            *self.whole_hz,
            *(value for window in self.window_hz for value in window),
            *(value for window in self.narrow_window_hz for value in window),
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Isolated-line plan frequencies must be finite.")
        if self.source_count < 0:
            raise ValueError("Isolated-line source_count must not be negative.")

    @property
    def all_hz(self) -> tuple[float, ...]:
        """Every nominal used by any spectrum in the recording."""
        return tuple(
            sorted(
                {
                    *self.whole_hz,
                    *(value for row in self.window_hz for value in row),
                    *(value for row in self.narrow_window_hz for value in row),
                }
            )
        )


def read_bids_raw(vhdr: Path):
    """Read one BIDS recording with strict sidecar-derived metadata."""
    from mne_bids import get_bids_path_from_fname, read_raw_bids

    bids_path = get_bids_path_from_fname(vhdr)
    return read_raw_bids(
        bids_path,
        extra_params={"preload": True},
        on_ch_mismatch="raise",
        verbose="ERROR",
    )


def _window_removal_plan(
    bounds: tuple[int, int],
    estimate: estimators.CombEstimate,
    narrow_targets_hz: Sequence[float],
    settings: RemovalSettings,
    *,
    spectrum_resolution_hz: float,
) -> AdaptiveWindowRemovalPlan:
    """Resolve one independently supported target set and its physical widths."""
    model_targets = estimators.removal_frequencies(
        estimate,
        harmonic_range=settings.removal_harmonic_range,
        low_hz=settings.low_hz,
        high_hz=settings.high_hz,
        excluded_hz=settings.protected_bands_hz,
    )
    model_widths = estimators.uncertainty_aware_notch_widths(
        estimate,
        model_targets,
        ratio=settings.notch_width_ratio,
        minimum_hz=settings.notch_width_min_hz,
        confidence_z=settings.uncertainty_confidence_z,
        isolated_minimum_hz=spectrum_resolution_hz,
    )
    narrow_array = np.asarray(narrow_targets_hz, dtype=float)
    if narrow_array.ndim != 1 or not np.all(np.isfinite(narrow_array)):
        raise ValueError("Narrow targets must be finite one-dimensional sequences.")
    narrow_widths = estimators.notch_widths_for(
        narrow_array,
        ratio=settings.notch_width_ratio,
        minimum_hz=settings.notch_width_min_hz,
    )
    target_widths = {
        float(target): float(width) for target, width in zip(model_targets, model_widths)
    }
    retained_narrow_targets = []
    for target, width in zip(narrow_array, narrow_widths):
        # Narrow targets arrive beside the comb model rather than through it, so
        # `removal_frequencies` never saw them and the band exclusion has to be repeated.
        if any(low <= float(target) <= high for low, high in settings.protected_bands_hz):
            continue
        covered_by_model = any(
            abs(float(target) - float(model_target)) + float(width) / 2.0
            <= float(model_width) / 2.0
            for model_target, model_width in zip(model_targets, model_widths)
        )
        if covered_by_model:
            continue
        target_widths[float(target)] = float(width)
        retained_narrow_targets.append(float(target))
    targets = tuple(sorted(target_widths))
    return AdaptiveWindowRemovalPlan(
        bounds=bounds,
        estimate=estimate,
        targets_hz=targets,
        notch_widths_hz=tuple(target_widths[target] for target in targets),
        narrow_targets_hz=tuple(retained_narrow_targets),
    )


def build_removal_plan(
    model: estimators.AdaptiveCombModel,
    *,
    bounds: tuple[tuple[int, int], ...],
    narrow_targets_hz: tuple[tuple[float, ...], ...],
    settings: RemovalSettings,
) -> RunRemovalPlan:
    """Resolve model-supported targets and widths once for benchmark and apply."""
    if not len(bounds) == len(model.window_estimates) == len(narrow_targets_hz):
        raise ValueError("Window bounds, estimates and narrow targets must have the same length.")
    resolution_hz = spectrum_fit_nominal_resolution_hz(settings.filter_length)
    windows = tuple(
        _window_removal_plan(
            window_bounds,
            estimate,
            narrow_targets,
            settings,
            spectrum_resolution_hz=resolution_hz,
        )
        for window_bounds, estimate, narrow_targets in zip(
            bounds,
            model.window_estimates,
            narrow_targets_hz,
        )
    )
    return RunRemovalPlan(model=model, windows=windows)


def parse_channel_scaling(vhdr_path: Path) -> tuple[list[str], np.ndarray]:
    """Channel names and their binary resolution, in the file's own unit."""
    text = vhdr_path.read_text(encoding="utf-8", errors="replace")
    binary_format = re.search(r"BinaryFormat=(\S+)", text)
    orientation = re.search(r"DataOrientation=(\S+)", text)
    if binary_format is None or binary_format.group(1) != "IEEE_FLOAT_32":
        raise ValueError(f"{vhdr_path.name}: expected IEEE_FLOAT_32 binary data.")
    if orientation is None or orientation.group(1) != "MULTIPLEXED":
        raise ValueError(f"{vhdr_path.name}: expected MULTIPLEXED data orientation.")

    # Channel definitions carry four comma-separated fields: name, reference, resolution,
    # unit. The classes exclude newlines so a `[Coordinates]` line, which holds only three
    # numbers, cannot be run into the one below it and parsed as `"-72\nCh2=1"`.
    names, resolutions = [], []
    for match in re.finditer(r"^Ch(\d+)=([^,\n]*),([^,\n]*),([^,\n]*),", text, flags=re.MULTILINE):
        names.append(match.group(2))
        resolutions.append(float(match.group(4)))
    if not names:
        raise ValueError(f"{vhdr_path.name}: no channel definitions found.")
    return names, np.asarray(resolutions, dtype=float)


def write_eeg_binary(vhdr_path: Path, destination: Path, data_volts: np.ndarray) -> None:
    """Write one ``.eeg`` binary in the layout its existing header already describes."""

    array = np.asarray(data_volts, dtype=float)
    if not np.all(np.isfinite(array)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        raise ValueError(
            f"Refusing to write {destination}: {bad} non-finite sample(s). The round-trip "
            "check cannot catch this -- a NaN makes the deviation NaN, and NaN > tolerance "
            "is False -- so it is caught here instead."
        )

    names, resolutions = parse_channel_scaling(vhdr_path)
    if array.shape[0] != len(names):
        raise ValueError(
            f"{vhdr_path.name}: header describes {len(names)} channels, got {array.shape[0]}."
        )
    scaled = (array * 1e6) / resolutions[:, None]
    scaled.T.astype("<f4").tofile(destination)


def write_derivative_description(
    output_root: Path,
    source_root: Path,
    settings: RemovalSettings,
    source_version: str,
    band_cost: dict[str, float] | None = None,
) -> Path:
    """Declare the cleaned root a derivative and record what produced it.

    ``mirror_sidecars`` copies every sidecar byte-for-byte, so without this the cleaned
    dataset carried the raw one's description: DatasetType "raw", credit to MNE-BIDS alone,
    and nothing tying it to the removal, its settings or the code revision. BIDS asks
    derivatives to carry ``GeneratedBy`` for that reason -- otherwise the delivered data
    cannot be traced to the transformation that made it, which is the whole question an
    audit asks first.
    """
    import json
    from dataclasses import asdict

    path = Path(output_root) / "dataset_description.json"
    if not path.is_file():
        raise FileNotFoundError(f"Source dataset description was not mirrored to {path}.")
    described = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(described, dict):
        raise ValueError("BIDS dataset_description.json must contain a JSON object.")

    described["DatasetType"] = "derivative"
    described.setdefault("Name", "decomb cleaned EEG")
    described.setdefault("BIDSVersion", "1.8.0")
    existing_generated = described.get("GeneratedBy", [])
    if not isinstance(existing_generated, list) or not all(
        isinstance(entry, dict) for entry in existing_generated
    ):
        raise ValueError("BIDS GeneratedBy must be a list of objects.")
    generated = [
        entry for entry in existing_generated if "decomb" not in str(entry.get("Name", ""))
    ]
    generated.append(
        {
            "Name": "decomb",
            "Version": _code_revision(),
            "Description": (
                "Projection onto sinusoids at the measured comb and isolated-line "
                "frequencies, estimated in overlapping windows and reconstructed by "
                "normalized squared-sine overlap-add. A Thomson-F detector authorizes "
                "sliding sub-bin sinusoid regression only inside established artifact "
                "regions, per channel. Sidecars are byte-identical to the source; only "
                "the .eeg binaries differ."
            ),
            "Parameters": {
                "settings_fingerprint": settings_fingerprint(settings),
                # What the removal actually cost, so the delivered data carries it rather
                # than the reader having to find the benchmark that produced it.
                **({"band_cost": band_cost} if band_cost else {}),
                **{
                    k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(settings).items()
                },
            },
        }
    )
    described["GeneratedBy"] = generated
    described["SourceDatasets"] = [
        {"URL": f"../{Path(source_root).name}", "Version": source_version}
    ]

    path.write_text(json.dumps(described, indent=2) + "\n", encoding="utf-8")
    return path


def mirror_sidecars(source_root: Path, output_root: Path) -> int:
    """Copy every BIDS file except the binaries, which get rewritten."""
    copied = 0
    for path in sorted(source_root.rglob("*")):
        if path.is_dir() or path.suffix in {".eeg", ".lock"}:
            continue
        target = output_root / path.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def estimation_window_samples(sampling_frequency_hz: float, settings: RemovalSettings) -> int:
    """Samples in one adaptive estimation window, rounded to a whole sample."""
    samples = int(round(settings.estimation_window_s * float(sampling_frequency_hz)))
    if samples < 2:
        raise ValueError(
            f"estimation_window_s={settings.estimation_window_s:g} s is under two samples "
            f"at {sampling_frequency_hz:g} Hz."
        )
    return samples


def _block_psd(raw, settings: RemovalSettings):
    """EEG channel-by-window spectra on the adaptive estimator's grid."""
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    sfreq = float(raw.info["sfreq"])
    window_samples = estimation_window_samples(sfreq, settings)
    hop_samples = window_samples // 2
    data = raw.get_data(picks=picks)
    bounds = adaptive_window_bounds(
        n_times=data.shape[-1],
        window_samples=window_samples,
        hop_samples=hop_samples,
    )
    windows = np.stack([data[:, start:stop] for start, stop in bounds], axis=1)
    freqs, psd = spectral.hann_periodogram(windows, sfreq)
    return freqs, psd, bounds


def run_spectra(raw, settings: RemovalSettings):
    """Whole-run and equal-duration block spectra on the same frequency grid."""
    freqs, psd, bounds = _block_psd(raw, settings)
    half_width = int(round(settings.background_half_width_hz / float(freqs[1])))
    whole_db = spectral.to_db(np.median(psd.mean(axis=1), axis=0))
    whole = (freqs, whole_db, spectral.prominence_db(whole_db, half_width_bins=half_width))
    per_block = []
    for block_psd in np.moveaxis(psd, 1, 0):
        block_db = spectral.to_db(np.median(block_psd, axis=0))
        per_block.append(
            (freqs, block_db, spectral.prominence_db(block_db, half_width_bins=half_width))
        )
    return whole, tuple(per_block), bounds


def session_run_spectra(raw, settings: RemovalSettings) -> SessionRunSpectra:
    """All raw-data evidence scopes used to plan one continuous recording."""
    whole, windows, bounds = run_spectra(raw, settings)
    return SessionRunSpectra(whole=whole, windows=windows, bounds=bounds)


def run_spectrum(raw, settings: RemovalSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Channel-median whole-run spectrum on a TR-commensurate grid."""
    whole, _, _ = run_spectra(raw, settings)
    return whole


def spatiotemporal_line_metrics(
    raw_before,
    raw_after,
    plan: RunRemovalPlan,
    settings: RemovalSettings,
) -> dict:
    """Focal residual excess relative to the unchanged pre-clean background."""
    import mne

    freqs, before_psd, before_bounds = _block_psd(raw_before, settings)
    after_freqs, after_psd, after_bounds = _block_psd(raw_after, settings)
    plan_bounds = tuple(window.bounds for window in plan.windows)
    if before_bounds != plan_bounds or after_bounds != plan_bounds:
        raise ValueError("The adaptive plan window geometry does not match the recording.")
    if not np.array_equal(freqs, after_freqs):
        raise ValueError("Before and after adaptive spectra use different frequency grids.")
    target_spans = tuple(window.applied_target_spans for window in plan.windows)
    active = np.array([bool(spans) for spans in target_spans], dtype=bool)
    if not np.any(active):
        return {
            "max_channel_block_residual_prominence_db": 0.0,
            "p99_channel_block_residual_prominence_db": 0.0,
            "focal_null_max_95_db": 0.0,
            "focal_residual_excess_db": 0.0,
            "focal_residual_null_p": 1.0,
            "worst_focal_window": -1.0,
            "worst_focal_channel_index": -1.0,
            "worst_focal_target_hz": np.nan,
            "worst_focal_frequency_hz": np.nan,
        }
    active_spans = tuple(spans for spans in target_spans if spans)
    metrics = estimators.adaptive_spatiotemporal_suppression(
        freqs,
        spectral.to_db(before_psd[:, active]),
        spectral.to_db(after_psd[:, active]),
        tuple(tuple(target for target, _ in spans) for spans in active_spans),
        tuple(tuple(width for _, width in spans) for spans in active_spans),
        background_half_width_hz=settings.background_half_width_hz,
        search_hz=settings.residual_search_hz,
    )
    picks = mne.pick_types(raw_before.info, eeg=True, exclude=())
    channel_index = int(metrics["worst_focal_channel_index"])
    metrics["worst_focal_channel_name"] = raw_before.ch_names[int(picks[channel_index])]
    return metrics


def adaptive_spectrum_db(
    raw, settings: RemovalSettings
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]:
    """Channel-median spectra for every adaptive estimation window."""
    freqs, psd, bounds = _block_psd(raw, settings)
    values = []
    for window_psd in np.moveaxis(psd, 1, 0):
        values.append(spectral.to_db(np.median(window_psd, axis=0)))
    return freqs, np.stack(values), bounds


def _reference_prominence(
    background_spectrum_db: np.ndarray,
    peak_spectrum_db: np.ndarray,
    *,
    half_width_bins: int,
) -> np.ndarray:
    """Prominence whose local floor cannot be changed by the cleaner."""
    background = np.asarray(background_spectrum_db, dtype=float)
    peaks = np.asarray(peak_spectrum_db, dtype=float)
    if background.shape != peaks.shape or background.ndim != 2:
        raise ValueError("Background and peak spectra must be matching two-dimensional arrays.")
    floors = np.stack(
        [spectral.local_background_db(row, half_width_bins=half_width_bins) for row in background]
    )
    return peaks - floors


def adaptive_suppression_metrics(
    raw_before, raw_after, plan: RunRemovalPlan, settings: RemovalSettings
) -> dict[str, float]:
    """Aggregate residual evidence over the model-supported target positions."""
    freqs, before_db, before_bounds = adaptive_spectrum_db(raw_before, settings)
    after_freqs, after_db, after_bounds = adaptive_spectrum_db(raw_after, settings)
    plan_bounds = tuple(window.bounds for window in plan.windows)
    if before_bounds != plan_bounds or after_bounds != plan_bounds:
        raise ValueError("Adaptive spectra and fitted plan use different window geometry.")
    if not np.array_equal(freqs, after_freqs):
        raise ValueError("Before and after adaptive spectra use different frequency grids.")
    half_width = int(round(settings.background_half_width_hz / float(freqs[1])))
    before = _reference_prominence(before_db, before_db, half_width_bins=half_width)
    after = _reference_prominence(before_db, after_db, half_width_bins=half_width)
    target_spans = tuple(window.applied_target_spans for window in plan.windows)
    active = np.array([bool(spans) for spans in target_spans], dtype=bool)
    if not np.any(active):
        return {
            "n_targets": 0.0,
            "median_prominence_before_db": 0.0,
            "median_residual_prominence_db": 0.0,
            "max_residual_prominence_db": 0.0,
            "null_max_95_db": 0.0,
            "residual_excess_db": 0.0,
            "residual_null_p": 1.0,
            "median_suppression_db": 0.0,
        }
    active_spans = tuple(spans for spans in target_spans if spans)
    metrics = estimators.adaptive_line_suppression(
        freqs,
        before[active],
        after[active],
        tuple(tuple(target for target, _ in spans) for spans in active_spans),
        tuple(tuple(width for _, width in spans) for spans in active_spans),
        search_hz=settings.residual_search_hz,
        max_line_width_hz=settings.max_line_width_hz,
    )
    return metrics


def continuous_refinement_metrics(
    plan: RunRemovalPlan,
    eeg_names: Sequence[str],
) -> dict[str, int | str]:
    """Summarise adaptive residual refinements with window/channel provenance."""
    channel_names = tuple(str(name) for name in eeg_names)
    aggregate_details = []
    focal_details = []
    focal_channel_windows = 0
    for window_index, window in enumerate(plan.windows):
        if window.channel_residual_targets_hz and len(window.channel_residual_targets_hz) != len(
            channel_names
        ):
            raise ValueError("The adaptive residual plan does not match the EEG channel names.")
        aggregate_details.extend(
            f"{window_index}:{frequency_hz:.6f}"
            for frequency_hz in window.aggregate_residual_targets_hz
        )
        for channel_index, targets in enumerate(window.channel_residual_targets_hz):
            focal_channel_windows += bool(targets)
            focal_details.extend(
                f"{window_index}:{channel_names[channel_index]}:{frequency_hz:.6f}"
                for frequency_hz in targets
            )
    return {
        "n_continuous_common_targets": sum(
            len(window.applied_target_spans) for window in plan.windows
        ),
        "n_continuous_aggregate_refinement_targets": len(aggregate_details),
        "n_continuous_focal_refinement_targets": len(focal_details),
        "n_continuous_focal_refinement_channel_windows": focal_channel_windows,
        "continuous_aggregate_refinement_hz": ";".join(aggregate_details),
        "continuous_focal_refinement_hz": ";".join(focal_details),
    }


def plan_target_spans(
    windows: Sequence[AdaptiveWindowRemovalPlan],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Each distinct target and the widest span any window removed around it.

    ``RunRemovalPlan.all_targets_hz`` deduplicates across windows while the widths are
    per-window, so the two cannot be zipped. Taking the maximum width per frequency gives
    the span the transform could have reached at that frequency anywhere in the run, which
    is what a measurement excluding the removals has to exclude.
    """
    spans: dict[float, float] = {}
    for window in windows:
        base_groups = window.channel_target_plans
        groups = (
            *base_groups,
            (window.aggregate_residual_targets_hz, window.aggregate_residual_widths_hz),
            *zip(window.channel_residual_targets_hz, window.channel_residual_widths_hz),
        )
        for targets, widths in groups:
            for target, width in zip(targets, widths):
                key = float(target)
                spans[key] = max(spans.get(key, 0.0), float(width))
    ordered = tuple(sorted(spans))
    return ordered, tuple(spans[target] for target in ordered)


def spectrum_fit_nominal_resolution_hz(filter_length: str) -> float:
    """Nominal FFT-bin resolution implied by MNE's spectrum-fit duration."""
    duration = "10s" if filter_length.lower() == "auto" else filter_length.lower()
    match = re.fullmatch(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>ms|s)", duration)
    if match is None:
        raise ValueError("filter_length must be a positive duration such as '20s'.")
    seconds = float(match.group("value"))
    if match.group("unit") == "ms":
        seconds /= 1_000.0
    if not np.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("filter_length must be positive.")
    return 1.0 / seconds


def spectrum_fit_frequency_grids(
    *,
    sampling_frequency_hz: float,
    filter_length: str,
    window_samples: int,
) -> tuple[np.ndarray, ...]:
    """Frequency grids used by MNE's inner spectrum-fit overlap-add windows."""
    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0.0:
        raise ValueError("sampling_frequency_hz must be finite and positive.")
    if window_samples < 1:
        raise ValueError("window_samples must be positive.")
    seconds = 1.0 / spectrum_fit_nominal_resolution_hz(filter_length)
    filter_samples = min(
        max(int(np.ceil(seconds * sampling_frequency_hz)), 1),
        window_samples,
    )
    if filter_samples < 2:
        raise ValueError("filter_length must contain at least two samples.")
    overlap_samples = (filter_samples + 1) // 2
    hop_samples = filter_samples - overlap_samples
    starts = np.arange(0, window_samples - filter_samples + 1, hop_samples)
    stops = starts + filter_samples
    stops[-1] = window_samples
    sample_counts = tuple(dict.fromkeys((stops - starts).tolist()))
    return tuple(
        np.fft.rfftfreq(sample_count, d=1.0 / sampling_frequency_hz)
        for sample_count in sample_counts
    )


def adaptive_band_metrics(
    *,
    sampling_frequency_hz: float,
    plan: RunRemovalPlan,
    settings: RemovalSettings,
) -> dict[str, float]:
    """Worst channel-level spectral cost of the transform that is actually applied.

    Only the continuous overlap-add is measured, because only it is applied. A cost
    charged for spectrum no filter ever touches would describe something other than the
    delivered data.
    """
    window_samples = {stop - start for start, stop in (window.bounds for window in plan.windows)}
    if len(window_samples) != 1:
        raise ValueError("Adaptive removal windows must have one fixed sample length.")
    grids = spectrum_fit_frequency_grids(
        sampling_frequency_hz=sampling_frequency_hz,
        filter_length=settings.filter_length,
        window_samples=window_samples.pop(),
    )

    def maximum_fraction(target_width_plans_for):
        measurements = []
        for freqs in grids:
            band_bin_count = int(
                np.count_nonzero(
                    (freqs >= settings.cost_band_hz[0]) & (freqs <= settings.cost_band_hz[1])
                )
            )
            for window in plan.windows:
                for targets, widths in target_width_plans_for(window):
                    if len(targets):
                        fraction = estimators.removed_band_fraction(
                            freqs, targets, widths, band_hz=settings.cost_band_hz
                        )
                        measurements.append((fraction, 1.0 / band_bin_count))
        return (
            max(measurements, key=lambda item: (item[0], item[1]))
            if measurements
            else (0.0, 1.0 / band_bin_count)
        )

    def base_plans(window):
        return window.channel_target_plans

    def continuous_plans(window):
        focal_targets = window.channel_residual_targets_hz or tuple(
            () for _ in base_plans(window)
        )
        focal_widths = window.channel_residual_widths_hz or tuple(
            () for _ in base_plans(window)
        )
        return tuple(
            (
                (*base_targets, *window.aggregate_residual_targets_hz, *targets),
                (*base_widths, *window.aggregate_residual_widths_hz, *widths),
            )
            for (base_targets, base_widths), targets, widths in zip(
                base_plans(window), focal_targets, focal_widths
            )
        )

    expanded_fraction, expanded_bin_size = maximum_fraction(continuous_plans)
    base_fraction, base_bin_size = maximum_fraction(base_plans)
    return {
        "base_removed_band_fraction": base_fraction,
        "base_band_fraction_bin_size": base_bin_size,
        "width_expansion_band_fraction": expanded_fraction - base_fraction,
        "continuous_removed_band_fraction": expanded_fraction,
        "removed_band_fraction": expanded_fraction,
        "band_fraction_bin_size": expanded_bin_size,
    }


def clean_raw(
    raw,
    targets,
    *,
    filter_length: str,
    filter_jobs: int,
    mt_bandwidth: float,
    notch_widths,
    picks="eeg",
):
    """Project the listed frequencies out of the EEG channels.

    ``notch_widths`` is always passed explicitly. Left to its default it becomes
    ``freq / 200``, which turns a line removal into a band removal.
    """
    import warnings

    from joblib import parallel_backend

    with warnings.catch_warnings(), parallel_backend("threading", n_jobs=filter_jobs):
        # scipy's DPSS eigenvalue side-computation overflows on long windows. The tapers
        # themselves are finite and orthonormal to 5e-4, and spectrum_fit uses only the
        # tapers, never the eigenvalues.
        warnings.filterwarnings("ignore", message=".*matmul", category=RuntimeWarning)
        return raw.notch_filter(
            freqs=list(targets),
            picks=picks,
            method="spectrum_fit",
            filter_length=filter_length,
            mt_bandwidth=mt_bandwidth,
            notch_widths=notch_widths,
            n_jobs=filter_jobs,
            verbose="ERROR",
        )


def _refine_regression_frequencies(
    data: np.ndarray,
    times: np.ndarray,
    targets_hz: Sequence[float],
    widths_hz: Sequence[float],
) -> tuple[float, ...]:
    """Locate each authorised sinusoid within one regression window."""
    centered = np.asarray(data, dtype=float) - float(np.mean(data))
    duration_s = times.size * float(times[1] - times[0])
    resolution_hz = 1.0 / duration_s
    refined = []
    for target_hz, width_hz in zip(targets_hz, widths_hz):
        search_half_width_hz = max(float(width_hz) / 2.0, resolution_hz)
        search_step_hz = resolution_hz / 10.0
        offsets = np.arange(
            -search_half_width_hz,
            search_half_width_hz + search_step_hz / 2.0,
            search_step_hz,
        )
        candidates_hz = float(target_hz) + offsets
        phases = np.exp(-2j * np.pi * times[:, np.newaxis] * candidates_hz)
        coefficients = np.einsum("i,ij->j", centered, phases, optimize=True)
        refined.append(float(candidates_hz[int(np.argmax(np.abs(coefficients)))]))
    return tuple(refined)


def _clean_channel_residuals(
    data: np.ndarray,
    picked_info,
    targets_hz: Sequence[Sequence[float]],
    widths_hz: Sequence[Sequence[float]],
    settings: RemovalSettings,
) -> np.ndarray:
    """Regress detected sinusoids jointly in overlapping spectrum-fit-length windows."""
    values = np.asarray(data, dtype=float)
    target_plans = tuple(tuple(float(value) for value in plan) for plan in targets_hz)
    width_plans = tuple(tuple(float(value) for value in plan) for plan in widths_hz)
    if values.ndim != 2 or values.shape[0] != len(picked_info["ch_names"]):
        raise ValueError("Channel-local cleaning requires channel-by-time data matching info.")
    if len(target_plans) != values.shape[0] or len(width_plans) != values.shape[0]:
        raise ValueError("Every channel requires one residual target and width plan.")
    if any(len(targets) != len(widths) for targets, widths in zip(target_plans, width_plans)):
        raise ValueError("Every channel's residual targets and widths must match.")
    planned_values = (
        *(target for plan in target_plans for target in plan),
        *(width for plan in width_plans for width in plan),
    )
    if not all(np.isfinite(value) for value in planned_values):
        raise ValueError("Channel-local residual targets and widths must be finite.")
    if any(width <= 0.0 for plan in width_plans for width in plan):
        raise ValueError("Channel-local residual widths must be positive.")

    sampling_frequency_hz = float(picked_info["sfreq"])
    if any(
        not 0.0 < target < sampling_frequency_hz / 2.0 for plan in target_plans for target in plan
    ):
        raise ValueError("Channel-local residual targets must lie between DC and Nyquist.")
    filter_seconds = 0.5 / spectrum_fit_nominal_resolution_hz(settings.filter_length)
    filter_samples = min(
        max(int(np.ceil(filter_seconds * sampling_frequency_hz)), 1),
        values.shape[1],
    )
    if filter_samples < 2:
        raise ValueError("Channel-local regression requires at least two samples.")
    bounds = (
        ((0, values.shape[1]),)
        if filter_samples == values.shape[1]
        else adaptive_window_bounds(
            n_times=values.shape[1],
            window_samples=filter_samples,
            hop_samples=filter_samples // 2,
        )
    )
    channel_groups: dict[
        tuple[tuple[float, ...], tuple[float, ...]],
        list[int],
    ] = {}
    for channel_index, targets in enumerate(target_plans):
        if targets:
            channel_groups.setdefault(
                (targets, width_plans[channel_index]),
                [],
            ).append(channel_index)

    # Locate each sinusoid once, over the whole window that evidenced it, and hold that
    # frequency fixed while amplitude and phase are re-fitted in every sub-window.
    #
    # Re-searching inside each sub-window would make the subtraction unaccountable. The
    # search spans +/-width/2, barely more than one sub-window frequency bin, so its argmax
    # is a maximum over noise: where the line is absent from a sub-window it would select
    # the largest local fluctuation and subtract it. Taking a maximum before subtracting
    # also removes more than the two degrees of freedom the regression is charged for.
    #
    # Nothing is lost by fixing it, because the search could not have been tracking drift.
    # A fundamental wandering by under a millihertz across a whole recording moves even a
    # high harmonic by a few millihertz within one window -- orders below the sub-window
    # resolution. Amplitude modulation is real and does vary sub-window to sub-window, and
    # the per-sub-window regression still follows it.
    whole_window_times = np.arange(values.shape[1], dtype=float) / sampling_frequency_hz
    refined_by_channel = {
        channel_index: _refine_regression_frequencies(
            values[channel_index],
            whole_window_times,
            targets,
            widths,
        )
        for (targets, widths), channel_indices in channel_groups.items()
        for channel_index in channel_indices
    }

    segments = []
    for start, stop in bounds:
        segment = values[:, start:stop].copy()
        times = np.arange(start, stop, dtype=float) / sampling_frequency_hz
        for _target_width, channel_indices in channel_groups.items():
            for channel_index in channel_indices:
                refined_targets = refined_by_channel[channel_index]
                angular_phase = 2.0 * np.pi * times[:, np.newaxis] * np.asarray(refined_targets)
                sinusoid_basis = np.column_stack((np.sin(angular_phase), np.cos(angular_phase)))
                design = np.column_stack((np.ones(times.size), sinusoid_basis))
                coefficients, _, rank, _ = np.linalg.lstsq(
                    design,
                    segment[channel_index],
                    rcond=None,
                )
                if rank != design.shape[1]:
                    raise ValueError("Channel-local sinusoid regression is rank deficient.")
                fitted_lines = np.einsum(
                    "ij,j->i",
                    sinusoid_basis,
                    coefficients[1:],
                    optimize=True,
                )
                if not np.all(np.isfinite(fitted_lines)):
                    raise ValueError(
                        "Channel-local sinusoid regression produced non-finite values."
                    )
                segment[channel_index] -= fitted_lines
        segments.append(segment)
    return overlap_add_segments(tuple(segments), bounds, values.shape[1])


def clean_continuous_raw(
    raw,
    plan: RunRemovalPlan,
    settings: RemovalSettings,
    *,
    eeg_plan_indices: Sequence[int] | None = None,
):
    """Apply and overlap-add only the independently fitted continuous windows."""
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    if len(picks) == 0:
        raise ValueError("Adaptive comb removal requires at least one EEG channel.")
    planned_channel_counts = {
        len(window.channel_targets_hz)
        for window in plan.windows
        if window.channel_targets_hz is not None
    }
    if len(planned_channel_counts) > 1:
        raise ValueError("Residual plans disagree about the EEG channel count.")
    planned_channel_count = next(iter(planned_channel_counts), 0)
    if eeg_plan_indices is None:
        if planned_channel_count not in (0, len(picks)):
            raise ValueError("The residual plan does not match the EEG channel count.")
        channel_plan_indices = tuple(range(len(picks)))
    else:
        channel_plan_indices = tuple(int(index) for index in eeg_plan_indices)
        if len(channel_plan_indices) != len(picks):
            raise ValueError("eeg_plan_indices must map every filtered EEG channel.")
        if planned_channel_count and any(
            not 0 <= index < planned_channel_count for index in channel_plan_indices
        ):
            raise ValueError("eeg_plan_indices contains an out-of-range plan channel.")
    bounds = tuple(window.bounds for window in plan.windows)
    squared_sine_weights(bounds, n_times=raw.n_times)
    picked_info = mne.pick_info(raw.info, picks, copy=True)
    segments = []
    for window in plan.windows:
        start, stop = window.bounds
        cleaned = _clean_planned_segment(
            raw.get_data(picks=picks, start=start, stop=stop),
            picked_info,
            window,
            settings,
            channel_plan_indices=channel_plan_indices,
        )
        segments.append(cleaned)

    output = raw.copy()
    output._data[picks] = overlap_add_segments(
        tuple(segments),
        bounds,
        raw.n_times,
    )
    return output


def _clean_planned_segment(
    data: np.ndarray,
    picked_info,
    window: AdaptiveWindowRemovalPlan,
    settings: RemovalSettings,
    *,
    channel_plan_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Apply common, aggregate-residual, then channel-local line transforms.

    The input is copied because ``RawArray`` may wrap the caller's buffer and MNE filters
    it in place. Without the copy the caller's data is cleaned as a side effect, which is
    silent and wrong: residual detection holds the pre-clean segment to compare against,
    and would compare the cleaned data with itself.
    """
    import mne

    segment = mne.io.RawArray(
        np.array(data, dtype=float, copy=True),
        picked_info.copy(),
        verbose="ERROR",
    )
    if window.channel_targets_hz is None or window.channel_target_widths_hz is None:
        raise ValueError("The adaptive window has not been authorized per channel.")
    plan_indices = (
        tuple(range(segment.get_data().shape[0]))
        if channel_plan_indices is None
        else tuple(channel_plan_indices)
    )
    if len(plan_indices) != segment.get_data().shape[0]:
        raise ValueError("The removal plan does not map every supplied EEG channel.")
    if any(
        not 0 <= index < len(window.channel_targets_hz)
        for index in plan_indices
    ):
        raise ValueError("The removal plan contains an out-of-range channel index.")

    grouped: dict[tuple[tuple[float, ...], tuple[float, ...]], list[int]] = {}
    for local_index, plan_index in enumerate(plan_indices):
        targets = window.channel_targets_hz[plan_index]
        widths = window.channel_target_widths_hz[plan_index]
        if targets:
            grouped.setdefault((targets, widths), []).append(local_index)
    for (targets, widths), picks in grouped.items():
        clean_raw(
            segment,
            targets,
            filter_length=settings.filter_length,
            filter_jobs=settings.filter_jobs,
            mt_bandwidth=settings.mt_bandwidth,
            notch_widths=np.asarray(widths),
            picks=picks,
        )
    cleaned = segment.get_data()
    if not window.aggregate_residual_targets_hz and not window.channel_residual_targets_hz:
        return cleaned
    channel_plans = []
    for index in plan_indices:
        target_width_pairs = list(
            zip(
                window.aggregate_residual_targets_hz,
                window.aggregate_residual_widths_hz,
            )
        )
        if window.channel_residual_targets_hz:
            target_width_pairs.extend(
                zip(
                    window.channel_residual_targets_hz[index],
                    window.channel_residual_widths_hz[index],
                )
            )
        channel_plans.append(_merge_residual_support(target_width_pairs))
    channel_plans = tuple(channel_plans)
    targets = tuple(plan[0] for plan in channel_plans)
    widths = tuple(plan[1] for plan in channel_plans)
    return _clean_channel_residuals(cleaned, picked_info, targets, widths, settings)


def _source_digest() -> str:
    """Content hash of the modules that decide what the removal does.

    A commit id is not enough: on a dirty tree it reads the same for every uncommitted
    state, so two different implementations would share a fingerprint, which is the one
    thing the fingerprint exists to prevent. Untracked files are invisible to it as well.
    Hashing the sources themselves has neither problem.
    """
    import hashlib

    package = Path(__file__).resolve().parent
    sources = sorted(package.glob("*.py"))
    if not sources:
        return "unknown"
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.name.encode("utf-8"))
        digest.update(source.read_bytes())
    return digest.hexdigest()[:16]


def _code_revision() -> str:
    """Short git revision of the installed package, when there is one.

    Supplementary provenance only. An installed package is usually not inside a checkout,
    so this returns ``"unknown"`` rather than raising -- what actually ties a benchmark to
    the code that produced it is :func:`_source_digest`, which hashes the sources
    themselves and cannot be absent.

    Run against the package directory, not the working directory: the revision should
    describe this code, not whichever repository the user happens to be standing in.
    """
    import subprocess

    package = str(Path(__file__).resolve().parent)

    def git(*arguments: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", package, *arguments],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
        except (subprocess.SubprocessError, OSError):
            return None
        return out.stdout.strip()

    revision = git("rev-parse", "--short", "HEAD")
    if revision is None:
        return "unknown"
    dirty = git("status", "--porcelain", "--untracked-files=no")
    return revision + ("+dirty" if dirty else "")


def settings_fingerprint(settings: RemovalSettings) -> str:
    """A short stable hash of every setting that changes what the removal does.

    Binds a benchmark to the configuration it measured, so a stale or mismatched
    benchmark.tsv cannot stand in for one describing the settings about to be applied.
    """
    import hashlib
    import importlib.metadata
    import platform
    from dataclasses import asdict

    digest = _source_digest()
    if digest == "unknown":
        raise RuntimeError(
            "Cannot identify the removal source, so a benchmark cannot be bound to it. "
            "Refusing to fingerprint rather than certify data against unknown code."
        )
    runtime = {
        "python": platform.python_version(),
        **{package: importlib.metadata.version(package) for package in ("mne", "numpy", "scipy")},
    }
    payload = repr((sorted(asdict(settings).items()), digest, sorted(runtime.items()))).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def recording_digest(vhdr: Path) -> str:
    """Hash the complete BrainVision recording consumed by the transformation."""
    import hashlib

    vhdr = Path(vhdr)
    references = {}
    for line in vhdr.read_text(encoding="utf-8", errors="strict").splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name in {"DataFile", "MarkerFile"}:
            references[name] = value.strip()
    missing = {"DataFile", "MarkerFile"} - references.keys()
    if missing:
        raise ValueError(f"{vhdr}: missing BrainVision reference(s): {sorted(missing)}")

    paths = (vhdr, *(vhdr.parent / references[name] for name in ("DataFile", "MarkerFile")))
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"BrainVision component does not exist: {path}")
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def dataset_digest(recordings: dict[str, str], source_root: Path) -> str:
    """Content identity for recording bytes and every mirrored source sidecar."""
    import hashlib

    digest = hashlib.sha256(repr(sorted(recordings.items())).encode("utf-8"))
    for path in sorted(Path(source_root).rglob("*")):
        if not path.is_file() or path.suffix in {".eeg", ".lock"}:
            continue
        digest.update(str(path.relative_to(source_root)).encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def source_input_digests(
    runs: Sequence[Path],
    source_root: Path,
) -> tuple[dict[str, str], str]:
    """Bind each recording to its raw bytes and the complete BIDS source metadata."""
    recording_digests = {run.stem: recording_digest(run) for run in runs}
    source_digest = dataset_digest(recording_digests, source_root)
    return {
        recording: f"{recording_digest}:{source_digest}"
        for recording, recording_digest in recording_digests.items()
    }, source_digest


def removal_plan_digest(plan: RunRemovalPlan) -> str:
    """Stable identity for the exact fitted transformation of one recording."""
    import hashlib
    from dataclasses import asdict

    return hashlib.sha256(repr(asdict(plan)).encode("utf-8")).hexdigest()


def partial_benchmark_path(report_dir) -> Path:
    """Where an in-progress benchmark journals the recordings it has already measured.

    Deliberately not ``benchmark.tsv``. A benchmark that dies partway through must not
    leave a file that reads as a complete pass, and must not overwrite the previous
    complete one either. Nothing reads this file except a resuming benchmark.
    """
    return Path(report_dir) / "benchmark_partial.tsv"


def resumable_benchmark_rows(
    path,
    fingerprint: str,
    recordings: dict[str, str],
    plans: dict[str, str],
) -> dict[str, dict]:
    """Rows from an interrupted benchmark that still describe exactly this work.

    A row survives only if it was produced under these settings, from this recording's
    content, and from this run's fitted plan. Anything else is measurement of something
    other than what is about to be applied, and is thrown away rather than trusted.

    Failure to read the journal is never an error: recomputing costs time, and time is the
    thing this function exists to save, not the thing it exists to protect.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path, sep="\t")
    except Exception as error:  # a damaged journal is worth redoing, never worth trusting
        print(f"  ignoring unreadable partial benchmark {path.name}: {error}")
        return {}
    required = {"recording", "settings_fingerprint", "input_digest", "plan_digest"}
    if not required.issubset(frame.columns) or frame["recording"].duplicated().any():
        print(f"  ignoring partial benchmark {path.name}: not a usable journal")
        return {}

    reusable = {}
    for row in frame.to_dict("records"):
        recording = row["recording"]
        if row.get("settings_fingerprint") != fingerprint:
            continue
        if recordings.get(recording) != row.get("input_digest"):
            continue
        if plans.get(recording) != row.get("plan_digest"):
            continue
        reusable[recording] = row
    return reusable


def require_passing_benchmark(
    path,
    settings: RemovalSettings,
    *,
    recordings: dict[str, str] | None = None,
    plans: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Refuse to write derived data without a passing benchmark of these settings.

    Returns the benchmark it accepted, so the caller can carry its measurements into the
    delivered dataset's provenance. The measured band cost is a benchmark quantity -- it
    needs the broadband probe, which only the benchmark injects -- so this is the only
    place ``apply`` can get it from.

    Every clause is a way a benchmark can silently fail to describe the data about to be
    written: produced under different settings, produced from different bytes, produced
    from a different fitted plan, or incomplete. Each is checked by identity rather than
    by trust.
    """
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            f"Refusing to apply: no benchmark at {path}. Run `decomb benchmark` first; "
            "the criteria are stated before the measurement for a reason."
        )
    frame = pd.read_csv(path, sep="	")
    expected = settings_fingerprint(settings)
    recorded = set(frame.get("settings_fingerprint", pd.Series(dtype=str)).dropna().unique())
    if recorded != {expected}:
        raise RuntimeError(
            f"Refusing to apply: {path} was produced under different settings "
            f"({recorded or 'none recorded'} against {expected}). Re-run the benchmark."
        )
    if not bool(frame["gate_passed"].all()):
        failed = int((~frame["gate_passed"].astype(bool)).sum())
        raise RuntimeError(
            f"Refusing to apply: {failed} of {len(frame)} benchmarked runs did not pass. "
            "A failure means the settings are wrong, not that the criteria should move."
        )
    if recordings is not None:
        if frame["recording"].duplicated().any():
            raise RuntimeError("Refusing to apply: benchmark contains duplicate recordings.")
        covered = set(frame["recording"])
        required = set(recordings)
        if covered != required:
            missing = sorted(required - covered)
            unexpected = sorted(covered - required)
            raise RuntimeError(
                "Refusing to apply: benchmark recordings differ from the apply set; "
                f"missing={missing}, unexpected={unexpected}."
            )
        benchmark_digests = frame.set_index("recording")["input_digest"].to_dict()
        changed = sorted(
            recording
            for recording, digest in recordings.items()
            if benchmark_digests.get(recording) != digest
        )
        if changed:
            raise RuntimeError(
                f"Refusing to apply: input digest changed since benchmarking for {changed}."
            )
    if plans is not None:
        benchmark_plans = frame.set_index("recording")["plan_digest"].to_dict()
        changed = sorted(
            recording
            for recording, digest in plans.items()
            if benchmark_plans.get(recording) != digest
        )
        if changed:
            raise RuntimeError(
                f"Refusing to apply: fitted removal plan changed since benchmarking for {changed}."
            )
    # Last, because a cohort statistic means nothing until the cohort is known to be the
    # right one. The seam criterion is absent from gate_passed -- a 2/41 per-run test
    # cannot be required of 90 runs at once -- so every row passing says nothing about it
    # and apply has to ask the cohort question itself.
    required_seam_columns = {
        "boundary_discontinuity_max_v",
        "boundary_control_maxima_v",
    }
    if not required_seam_columns.issubset(frame.columns):
        raise RuntimeError(
            f"Refusing to apply: {path} carries no seam measurements, so the cohort seam "
            "criterion cannot be evaluated. Re-run the benchmark."
        )
    seam = estimators.seam_randomization_verdict(
        _seam_evidence_from_frame(frame), alpha=settings.seam_alpha
    )
    if not seam["passed"]:
        raise RuntimeError(
            "Refusing to apply: the cohort seam criterion failed -- "
            f"{int(seam['n_exceeding'])} of {int(seam['n_runs'])} runs exceeded their control "
            f"scale (count p={seam['count_p_value']:.4f}, maximum p="
            f"{seam['max_p_value']:.4f}), worst ratio {seam['max_ratio']:.2f}."
        )
    if settings.max_band_cost is not None:
        if "measured_band_attenuated_1db" not in frame.columns:
            raise RuntimeError(
                f"Refusing to apply: {path} carries no measured band cost, so the declared "
                f"budget of {settings.max_band_cost:.3f} cannot be checked. Re-run the benchmark."
            )
        worst = float(frame["measured_band_attenuated_1db"].max())
        if worst > settings.max_band_cost:
            raise RuntimeError(
                f"Refusing to apply: a broadband signal loses {worst:.3f} of the analysed "
                f"band on the worst recording, above the {settings.max_band_cost:.3f} declared "
                "in `removal.max_band_cost`."
            )
    for scope, column in (
        ("whole-run residual", "residual_null_p"),
        ("focal residual", "focal_residual_null_p"),
    ):
        if column not in frame.columns:
            raise RuntimeError(
                f"Refusing to apply: {path} carries no {column}, so the {scope} "
                "criterion cannot be evaluated. Re-run the benchmark."
            )
        verdict = estimators.residual_randomization_verdict(
            frame[column].to_numpy(), false_discovery_rate=settings.false_discovery_rate
        )
        if not verdict["passed"]:
            raise RuntimeError(
                f"Refusing to apply: the cohort {scope} criterion failed -- "
                f"{int(verdict['n_discoveries'])} of {int(verdict['n_runs'])} recordings "
                f"exceed what their own matched controls reach "
                f"(smallest p={verdict['min_run_p_value']:.3g})."
            )
    # The residual-sinusoid probabilities are deliberately NOT consulted here. Residual
    # targets are selected by Thomson's F test on each exact epoch, and this criterion
    # repeats that same test on the same epochs after removing exactly what it found. The
    # multiplicity arithmetic is sound, but the inference is post-selection: it measures
    # whether the selected sinusoids were subtracted, not whether the detector missed one,
    # so it cannot certify detection completeness. The symptom is a comfortable p value
    # reported while a large residual still stands in the spectrum.
    #
    # The independent acceptance test is the PSD matched-control gate, which scores a
    # different statistic that the detector does not optimise. The probabilities stay in
    # the benchmark and the cohort verdict is still printed, as provenance.
    return frame


def measured_band_cost(benchmark: pd.DataFrame) -> dict[str, float] | None:
    """What a broadband signal actually lost, taken from a passing benchmark.

    ``None`` when the benchmark predates the measurement, so an older file still applies
    rather than being rejected for a column it could not have had.
    """
    column = "measured_band_attenuated_1db"
    if benchmark is None or column not in getattr(benchmark, "columns", ()):
        return None
    values = benchmark[column].dropna()
    if values.empty:
        return None
    return {
        "measured_band_attenuated_1db_median": float(values.median()),
        "measured_band_attenuated_1db_worst": float(values.max()),
    }


def _boundary_metrics(
    original: np.ndarray,
    cleaned: np.ndarray,
    boundaries: Sequence[int],
    settings: RemovalSettings,
) -> dict[str, float | str]:
    evidence = estimators.boundary_discontinuity_evidence(
        original, cleaned, boundaries, n_controls=settings.n_seam_controls
    )
    return {
        "max_boundary_discontinuity_ratio": evidence.ratio,
        "boundary_discontinuity_max_v": evidence.observed_max,
        "boundary_control_maxima_v": ";".join(f"{value:.17g}" for value in evidence.control_maxima),
    }


def _plan_transition_boundaries(plan: RunRemovalPlan, n_times: int) -> tuple[int, ...]:
    """Every interior sample where the set of contributing windows changes.

    A seam can only occur where one estimate starts or stops contributing, so the
    boundaries are the adaptive windows' own starts and stops. Both are included: with a
    half-window hop most stops coincide with a later start, but the tail window is placed
    to end exactly at ``n_times`` and so contributes a stop that no start repeats.

    Only real seams belong here. ``boundary_discontinuity_evidence`` matches its controls
    to the *count* of boundaries, so adding samples where no estimate changes would bias
    the test toward passing: the observed maximum would come from the real seams alone
    while each control maximum was taken over more shifted indices.
    """
    boundaries = {
        boundary for window in plan.windows for boundary in window.bounds if 0 < boundary < n_times
    }
    if not boundaries:
        raise ValueError("The plan has no interior adaptive boundary to test for seams.")
    return tuple(sorted(boundaries))


def _seam_evidence_from_frame(
    frame: pd.DataFrame,
) -> tuple[estimators.BoundaryDiscontinuityEvidence, ...]:
    evidence = []
    for row in frame.itertuples(index=False):
        controls = tuple(float(value) for value in row.boundary_control_maxima_v.split(";"))
        evidence.append(
            estimators.BoundaryDiscontinuityEvidence(
                float(row.boundary_discontinuity_max_v),
                controls,
            )
        )
    return tuple(evidence)


def _detection_scaffold(
    freqs,
    spectrum_db,
    prominence,
    settings: RemovalSettings,
    *,
    fallback: estimators.CombEstimate | None = None,
):
    """Fit the comb-only model that defines isolated-line clearance.

    Takes the same ``fallback`` as the planning fit, for the same reason: this runs per
    window too, so a window that cannot establish a grid would otherwise end the run
    before a plan is ever built.
    """
    return estimators.estimate_comb(
        freqs,
        spectrum_db,
        prominence,
        nominal_hz=settings.nominal_fundamental_hz,
        harmonic_range=settings.harmonic_range,
        isolated_nominal_hz=(),
        search_hz=settings.search_hz,
        isolated_search_hz=settings.detection_search_hz,
        min_prominence_db=settings.min_prominence_db,
        min_harmonics=settings.min_harmonics_for_fit,
        max_harmonic_residual_hz=settings.max_harmonic_residual_hz,
        max_residual_rms_hz=settings.max_fit_residual_rms_hz,
        fallback=fallback,
    )


def detect_comb_adjacent_lines(
    freqs,
    spectrum_db,
    prominence,
    *,
    estimate: estimators.CombEstimate,
    settings: RemovalSettings,
) -> tuple[float, ...]:
    """Detect distinct narrow sources beside an already supported comb target."""
    targets, widths = _comb_detection_support(estimate, settings)
    return _detect_lines_adjacent_to_targets(
        freqs,
        spectrum_db,
        prominence,
        targets=targets,
        widths=widths,
        settings=settings,
    )


def _comb_detection_support(
    estimate: estimators.CombEstimate,
    settings: RemovalSettings,
) -> tuple[tuple[float, ...], np.ndarray]:
    """Comb targets and physical support used by adjacent-line detection."""
    targets = estimators.removal_frequencies(
        estimate,
        harmonic_range=settings.removal_harmonic_range,
        low_hz=settings.low_hz,
        high_hz=settings.high_hz,
        excluded_hz=settings.protected_bands_hz,
    )
    widths = estimators.uncertainty_aware_notch_widths(
        estimate,
        targets,
        ratio=settings.notch_width_ratio,
        minimum_hz=settings.notch_width_min_hz,
        confidence_z=settings.uncertainty_confidence_z,
        isolated_minimum_hz=spectrum_fit_nominal_resolution_hz(settings.filter_length),
    )
    return targets, widths


def _comb_annulus(
    frequency_array: np.ndarray,
    target_array: np.ndarray,
    covered_reaches: np.ndarray,
    settings: RemovalSettings,
) -> np.ndarray:
    """Where a new source beside the comb could still be found.

    Close enough to a target to be its business, and outside what that target's own notch
    already covers -- inside the notch there is nothing left to find, because it is already
    being removed.
    """
    inside = (frequency_array >= settings.detection_low_hz) & (frequency_array <= settings.high_hz)
    distance = np.abs(frequency_array[:, None] - target_array[None, :])
    return (
        inside
        & (distance.min(axis=1) <= settings.residual_search_hz)
        & np.all(distance > covered_reaches[None, :], axis=1)
    )


def _detect_lines_adjacent_to_targets(
    freqs,
    spectrum_db,
    prominence,
    *,
    targets: Sequence[float],
    widths: Sequence[float],
    settings: RemovalSettings,
) -> tuple[float, ...]:
    """Narrow summits connected to, but not contained by, supported targets."""
    frequency_array = np.asarray(freqs, dtype=float)
    spectrum = np.asarray(spectrum_db, dtype=float)
    prominence_array = np.asarray(prominence, dtype=float)
    if not frequency_array.shape == spectrum.shape == prominence_array.shape:
        raise ValueError("freqs, spectrum_db and prominence must have the same shape.")
    if frequency_array.ndim != 1 or frequency_array.size < 3:
        raise ValueError("Comb-adjacent detection requires a one-dimensional spectrum.")
    frequency_steps = np.diff(frequency_array)
    if np.any(frequency_steps <= 0.0):
        raise ValueError("Comb-adjacent detection requires increasing frequencies.")
    frequency_resolution_hz = float(np.median(frequency_steps))

    target_array = np.asarray(targets, dtype=float)
    width_array = np.asarray(widths, dtype=float)
    if target_array.ndim != 1 or target_array.size == 0 or width_array.shape != target_array.shape:
        raise ValueError("Adjacent-line targets and widths must be matching non-empty vectors.")
    if not np.all(np.isfinite(target_array)) or np.any(width_array <= 0.0):
        raise ValueError("Adjacent-line targets and widths must be finite and positive.")
    covered_reaches = np.maximum(width_array / 2.0, frequency_resolution_hz)

    summits = np.zeros(prominence_array.shape, dtype=bool)
    summits[1:-1] = (
        np.isfinite(prominence_array[1:-1])
        & (prominence_array[1:-1] > prominence_array[:-2])
        & (prominence_array[1:-1] >= prominence_array[2:])
    )
    annulus = _comb_annulus(frequency_array, target_array, covered_reaches, settings)
    # A decibel floor. See `detection_adjacent_min_prominence_db` for the calibrated test
    # that was built to replace it, and the measurement that sent it back.
    candidate_indices = np.flatnonzero(
        summits & annulus & (prominence_array >= settings.detection_adjacent_min_prominence_db)
    )

    candidates = []
    for index in candidate_indices:
        peak_width_hz = estimators._peak_width_hz(frequency_array, prominence_array, int(index))
        if peak_width_hz > settings.max_line_width_hz:
            continue
        candidates.append((float(frequency_array[index]), float(prominence_array[index])))

    # Two summits closer together than a line's own width are one source seen twice, not
    # two sources -- which is what `line_claim_hz` means and why `detect_isolated_lines`
    # separates on it. The filter's resolution is a floor under that, not a substitute for
    # it: it asks whether the transform could separate them, not whether they are distinct.
    minimum_separation_hz = max(
        settings.line_claim_hz,
        spectrum_fit_nominal_resolution_hz(settings.filter_length),
    )
    accepted = []
    for position_hz, _strength_db in sorted(
        candidates,
        key=lambda item: (-item[1], item[0]),
    ):
        if any(abs(position_hz - taken) <= minimum_separation_hz for taken in accepted):
            continue
        accepted.append(position_hz)
    return tuple(sorted(accepted))


def _estimate_spectrum(
    spectrum,
    settings: RemovalSettings,
    isolated_hz,
    *,
    fallback: estimators.CombEstimate | None = None,
) -> estimators.CombEstimate:
    freqs, spectrum_db, prominence = spectrum
    return estimators.estimate_comb(
        freqs,
        spectrum_db,
        prominence,
        nominal_hz=settings.nominal_fundamental_hz,
        harmonic_range=settings.harmonic_range,
        isolated_nominal_hz=isolated_hz,
        search_hz=settings.search_hz,
        isolated_search_hz=settings.detection_search_hz,
        min_prominence_db=settings.min_prominence_db,
        min_harmonics=settings.min_harmonics_for_fit,
        max_harmonic_residual_hz=settings.max_harmonic_residual_hz,
        max_residual_rms_hz=settings.max_fit_residual_rms_hz,
        fallback=fallback,
    )


def build_run_plan_from_spectra(
    spectra: SessionRunSpectra,
    settings: RemovalSettings,
    isolated_lines: RunIsolatedLinePlan,
) -> RunRemovalPlan:
    """Fit independently supported targets for every overlapping run window."""
    if len(spectra.windows) != len(isolated_lines.window_hz):
        raise ValueError("Each adaptive spectrum requires its own isolated-line target list.")
    # The whole-run estimate is fitted first and without a fallback: it has the whole
    # recording behind it, so if a comb cannot be established there, there is nothing for a
    # window to inherit and the recording genuinely has no grid to remove on.
    whole_estimate = _estimate_spectrum(spectra.whole, settings, isolated_lines.whole_hz)
    window_estimates = tuple(
        _estimate_spectrum(window, settings, nominals, fallback=whole_estimate)
        for window, nominals in zip(spectra.windows, isolated_lines.window_hz)
    )
    model = estimators.build_adaptive_comb_model(
        whole_estimate,
        window_estimates,
        min_harmonics=settings.min_harmonics_for_fit,
    )
    plan = build_removal_plan(
        model,
        bounds=spectra.bounds,
        narrow_targets_hz=isolated_lines.narrow_window_hz,
        settings=settings,
    )
    plan = _ensure_routed_isolated_targets(
        plan,
        isolated_lines.window_hz,
        settings,
    )
    return _expand_widths_to_observed_line_support(plan, spectra, settings)


def _ensure_routed_isolated_targets(
    plan: RunRemovalPlan,
    routed_targets_hz: tuple[tuple[float, ...], ...],
    settings: RemovalSettings,
) -> RunRemovalPlan:
    """Keep exact-window evidence even when a 54-second fit dilutes its source."""
    if len(plan.windows) != len(routed_targets_hz):
        raise ValueError("Every removal window requires one routed isolated-target list.")
    minimum_width_hz = spectrum_fit_nominal_resolution_hz(settings.filter_length)
    windows = []
    for window, routed_targets in zip(plan.windows, routed_targets_hz):
        target_widths = dict(zip(window.targets_hz, window.notch_widths_hz))
        for target in routed_targets:
            width = max(
                target / settings.notch_width_ratio,
                settings.notch_width_min_hz,
                minimum_width_hz,
            )
            covered = any(
                abs(target - existing_target) + width / 2.0 <= existing_width / 2.0
                for existing_target, existing_width in target_widths.items()
            )
            if not covered:
                target_widths[target] = width
        targets = tuple(sorted(target_widths))
        widths = tuple(target_widths[target] for target in targets)
        windows.append(
            replace(
                window,
                targets_hz=targets,
                notch_widths_hz=widths,
            )
        )
    return replace(plan, windows=tuple(windows))


def _expand_widths_to_observed_line_support(
    plan: RunRemovalPlan,
    spectra: SessionRunSpectra,
    settings: RemovalSettings,
) -> RunRemovalPlan:
    """Cover narrow line support observed around each independently valid target."""
    expanded_windows = []
    for window_index, window in enumerate(plan.windows):
        evidence = (spectra.windows[window_index],)
        expanded_windows.append(
            _expand_window_to_observed_support(
                window, evidence, settings, localization_margin_hz=settings.support_margin_hz
            )
        )
    return replace(plan, windows=tuple(expanded_windows))


def _expand_window_to_observed_support(
    window: AdaptiveWindowRemovalPlan,
    evidence: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    settings: RemovalSettings,
    *,
    localization_margin_hz: float = 0.0,
) -> AdaptiveWindowRemovalPlan:
    """Cover strong narrow support beside a validated target, where it was observed.

    A notch is symmetric about its target, so widening the target itself to reach an
    asymmetric peak also empties the mirror image of that peak, where nothing was ever
    seen: support running from ``target + 0.025`` to ``target + 0.275`` would take
    0.55 Hz to cover 0.25 Hz of evidence, and half of what it removed would be chosen by
    arithmetic rather than observed.

    The support gets its own notch instead, centred on the interval that was measured, so
    the cost is the width of the evidence. Where the support does sit symmetrically the
    two formulations agree, and where the existing notch already covers it nothing is
    added.
    """
    if not np.isfinite(localization_margin_hz) or localization_margin_hz < 0.0:
        raise ValueError("localization_margin_hz must be finite and non-negative.")
    targets = np.asarray(window.targets_hz, dtype=float)
    widths = np.asarray(window.notch_widths_hz, dtype=float).copy()
    covers: list[tuple[float, float]] = []
    for freqs, _, prominence in evidence:
        frequency_array = np.asarray(freqs, dtype=float)
        prominence_array = np.asarray(prominence, dtype=float)
        if frequency_array.size < 2:
            raise ValueError("Support evidence needs a frequency grid of at least two bins.")
        # Support is read off a discrete grid, so a peak confined to one bin still occupies
        # that bin's full span. Adding one resolution step states exactly the width of the
        # bins observed, and keeps a single-bin support from asking for a zero-width notch.
        resolution_hz = float(frequency_array[1] - frequency_array[0])
        # A decibel floor, and the one place in the workflow that still has one. See
        # `support_min_prominence_db` for what was tried instead and what it measured.
        summits = np.zeros(prominence_array.shape, dtype=bool)
        summits[1:-1] = (
            np.isfinite(prominence_array[1:-1])
            & (prominence_array[1:-1] > prominence_array[:-2])
            & (prominence_array[1:-1] >= prominence_array[2:])
        )
        candidate_indices = np.flatnonzero(
            summits & (prominence_array >= settings.support_min_prominence_db)
        )
        for index in candidate_indices:
            if (
                estimators._peak_width_hz(frequency_array, prominence_array, int(index))
                > settings.max_line_width_hz
            ):
                continue
            position_hz = float(frequency_array[index])
            target_index = int(np.argmin(np.abs(targets - position_hz)))
            target_hz = float(targets[target_index])
            # Only peaks a target is answerable for. Without this a summit anywhere in the
            # band gets its own notch, half a hertz from the nearest harmonic, and the
            # widening stops being widening.
            if abs(target_hz - position_hz) > settings.residual_search_hz:
                continue
            left_hz, right_hz = estimators._peak_support_bounds_hz(
                frequency_array,
                prominence_array,
                int(index),
            )
            reach_hz = float(widths[target_index]) / 2.0
            if abs(left_hz - target_hz) <= reach_hz and abs(right_hz - target_hz) <= reach_hz:
                continue
            covers.append(
                (
                    (left_hz + right_hz) / 2.0,
                    (right_hz - left_hz) + resolution_hz + 2.0 * localization_margin_hz,
                )
            )

    if not covers:
        return window
    cover_targets, cover_widths = _merge_residual_support(tuple(covers))
    return replace(
        window,
        targets_hz=(*window.targets_hz, *cover_targets),
        notch_widths_hz=(*tuple(widths), *cover_widths),
    )


def _refine_window_residual_plan(
    original: np.ndarray,
    picked_info,
    window: AdaptiveWindowRemovalPlan,
    settings: RemovalSettings,
    clean_segment,
    *,
    cleaned_data: np.ndarray | None = None,
) -> AdaptiveWindowRemovalPlan:
    """Encode the sinusoids that survived one window's first pass.

    What licenses a subtraction here is ``estimators.ResidualDetection`` -- Thomson's F test on
    the first pass's own output -- and never the tolerance the benchmark will judge the
    result by. See that class for why the two have to stay apart.

    A window that authorized nothing is returned untouched. A residual is what a subtraction
    left behind, so a window that subtracted nothing has none to find, and searching its
    neighbourhoods would be searching around targets that were never taken.
    """
    if not window.targets_hz:
        return window
    sampling_frequency_hz = float(picked_info["sfreq"])
    residual_width_hz = 2.0 * max(
        sampling_frequency_hz / original.shape[1],
        2.0 * spectrum_fit_nominal_resolution_hz(settings.filter_length),
    )
    aggregate_targets = list(window.aggregate_residual_targets_hz)
    focal_targets = (
        [list(values) for values in window.channel_residual_targets_hz]
        if window.channel_residual_targets_hz
        else [[] for _ in range(original.shape[0])]
    )
    if len(focal_targets) != original.shape[0]:
        raise ValueError("The residual plan does not match the supplied EEG channels.")

    cleaned = (
        clean_segment(
            original,
            picked_info,
            _encode_residual_targets(window, aggregate_targets, focal_targets, residual_width_hz),
            settings,
        )
        if cleaned_data is None
        else np.asarray(cleaned_data, dtype=float)
    )
    if cleaned.shape != original.shape or not np.all(np.isfinite(cleaned)):
        raise ValueError("Residual refinement requires finite cleaned data matching the input.")

    shared_candidates, focal_candidates = _residual_line_candidates(
        (cleaned,),
        sampling_frequency_hz=sampling_frequency_hz,
        window=window,
        settings=settings,
    )

    for frequency_hz in shared_candidates:  # always empty; the shared route is gone
        if not _already_searched(frequency_hz, aggregate_targets, residual_width_hz):
            aggregate_targets.append(frequency_hz)
    for channel_index, channel_candidates in enumerate(focal_candidates):
        for frequency_hz in channel_candidates:
            if _already_searched(frequency_hz, aggregate_targets, residual_width_hz):
                continue
            if not _already_searched(frequency_hz, focal_targets[channel_index], residual_width_hz):
                focal_targets[channel_index].append(frequency_hz)
    return _encode_residual_targets(window, aggregate_targets, focal_targets, residual_width_hz)


def _residual_line_candidates(
    states: Sequence[np.ndarray],
    *,
    sampling_frequency_hz: float,
    window: AdaptiveWindowRemovalPlan,
    settings: RemovalSettings,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Sinusoids evidenced in one window's data, per channel.

    Every candidate is channel-local, and nothing is subtracted from a channel that did
    not evidence it. Routing a frequency carried by half the array into every channel's
    plan would be worse than useless: the regression searches each channel separately and
    subtracts whichever fluctuation is largest inside the width, so in a channel without
    the artifact it would remove whatever was there, possibly signal.

    The state passed is the first pass's own output, which is the raw data with the
    already-modelled component accounted for: that is how a line hidden under a stronger
    neighbour's skirt becomes visible, and it is the case this second pass exists for.
    Testing the raw segment as well finds far more candidates, but they are candidates the
    first pass already accounted for, and every one is another hole in the delivered
    spectrum.

    Whatever states are supplied, none of them is the tolerance the benchmark judges the
    result by. That is what keeps the acceptance gates able to fail.
    """
    if not states:
        raise ValueError("At least one state of the segment is required.")
    neighbourhood = {
        "targets_hz": window.targets_hz,
        "widths_hz": window.notch_widths_hz,
        "responsibility_hz": settings.residual_search_hz,
    }
    focal: list[list[float]] = [[] for _ in range(states[0].shape[0])]
    for state in states:
        frequencies, statistic, threshold, _ = estimators.thomson_f_statistics(
            state,
            sampling_frequency_hz=sampling_frequency_hz,
            bandwidth_hz=settings.mt_bandwidth,
            family_alpha=settings.residual_family_alpha,
        )
        channel_candidates = estimators.focal_residual_line_candidates(
            frequencies,
            statistic,
            threshold=threshold,
            **neighbourhood,
        )
        for channel_index, values in enumerate(channel_candidates):
            focal[channel_index].extend(values)
    return (), tuple(tuple(sorted(values)) for values in focal)


def _already_searched(
    frequency_hz: float,
    encoded_hz: Sequence[float],
    width_hz: float,
) -> bool:
    """Whether a residual search already encoded covers this frequency."""
    return any(abs(frequency_hz - existing_hz) <= width_hz / 2.0 for existing_hz in encoded_hz)


def _encode_residual_targets(
    window: AdaptiveWindowRemovalPlan,
    aggregate_targets: Sequence[float],
    focal_targets: Sequence[Sequence[float]],
    width_hz: float,
) -> AdaptiveWindowRemovalPlan:
    """Attach one window's residual searches, each the same measured width."""
    aggregate = tuple(sorted(aggregate_targets))
    focal = tuple(tuple(sorted(values)) for values in focal_targets)
    return replace(
        window,
        aggregate_residual_targets_hz=aggregate,
        aggregate_residual_widths_hz=(width_hz,) * len(aggregate),
        channel_residual_targets_hz=focal,
        channel_residual_widths_hz=tuple((width_hz,) * len(values) for values in focal),
    )


def _merge_residual_support(
    target_width_pairs: Sequence[tuple[float, float]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Merge overlapping frequency searches without duplicating a fitted source."""
    ordered = sorted((float(target), float(width)) for target, width in target_width_pairs)
    if not ordered:
        return (), ()
    clusters: list[list[tuple[float, float]]] = []
    cluster_right = float("-inf")
    for target, width in ordered:
        left = target - width / 2.0
        right = target + width / 2.0
        if clusters and left <= cluster_right:
            clusters[-1].append((target, width))
            cluster_right = max(cluster_right, right)
        else:
            clusters.append([(target, width)])
            cluster_right = right
    targets = []
    widths = []
    for cluster in clusters:
        if len(cluster) == 1:
            target, width = cluster[0]
        else:
            left = min(target - width / 2.0 for target, width in cluster)
            right = max(target + width / 2.0 for target, width in cluster)
            target = (left + right) / 2.0
            width = right - left
        targets.append(target)
        widths.append(width)
    return tuple(targets), tuple(widths)


def _route_continuous_residual_support(plan: RunRemovalPlan) -> RunRemovalPlan:
    """Give every overlapping synthesis window the residuals evidenced in its support."""
    channel_counts = {
        len(window.channel_residual_targets_hz)
        for window in plan.windows
        if window.channel_residual_targets_hz
    }
    if len(channel_counts) > 1:
        raise ValueError("Continuous residual plans disagree about the EEG channel count.")
    channel_count = next(iter(channel_counts), 0)
    routed = []
    for destination in plan.windows:
        sources = tuple(
            source
            for source in plan.windows
            if _intervals_overlap(destination.bounds, source.bounds)
        )
        aggregate_targets, aggregate_widths = _merge_residual_support(
            tuple(
                (target, width)
                for source in sources
                for target, width in zip(
                    source.aggregate_residual_targets_hz,
                    source.aggregate_residual_widths_hz,
                )
            )
        )
        focal_targets = []
        focal_widths = []
        for channel_index in range(channel_count):
            targets, widths = _merge_residual_support(
                tuple(
                    (target, width)
                    for source in sources
                    for target, width in zip(
                        source.channel_residual_targets_hz[channel_index],
                        source.channel_residual_widths_hz[channel_index],
                    )
                )
            )
            focal_targets.append(targets)
            focal_widths.append(widths)
        routed.append(
            replace(
                destination,
                aggregate_residual_targets_hz=aggregate_targets,
                aggregate_residual_widths_hz=aggregate_widths,
                channel_residual_targets_hz=tuple(focal_targets),
                channel_residual_widths_hz=tuple(focal_widths),
            )
        )
    return replace(plan, windows=tuple(routed))


def _grid_accuracy_note(plan: RunRemovalPlan, settings: RemovalSettings) -> str:
    """Say so when the fit places its harmonics less precisely than a bin.

    Authorization tests the one bin a target names, which is exact only while the target is
    where the fit says it is. The fit already measures how far its worst harmonic sits from
    the grid it derived, so the case where that assumption weakens is knowable in advance
    rather than after a line survives. It is reported, not acted on: whether a fit this
    loose should still be used is a question about a recording, and nothing here can answer
    it by refusing.
    """
    residuals = [
        float(window.estimate.max_abs_residual_hz)
        for window in plan.windows
        if window.estimate is not None and np.isfinite(window.estimate.max_abs_residual_hz)
    ]
    if not residuals:
        return ""
    worst = max(residuals)
    resolution = spectral.hann_resolution_hz(settings.estimation_window_s)
    if worst <= resolution:
        return ""
    return (
        f", fit places harmonics to {worst * 1e3:.0f} mHz against a "
        f"{resolution * 1e3:.0f} mHz resolution -- targets that far off carry no evidence "
        "at the bin they name"
    )


def _independent_window_indices(windows: Sequence[AdaptiveWindowRemovalPlan]) -> tuple[int, ...]:
    """A maximal set of windows that share no samples, by the same rule line support uses.

    `estimation_window_s` windows advance by half their length, so consecutive windows hold
    the same data twice and their statistics are not two observations of it. Pooling needs
    the subset that is.
    """
    chosen: list[int] = []
    previous_stop = -1
    for index in sorted(range(len(windows)), key=lambda item: windows[item].bounds[1]):
        start, stop = windows[index].bounds
        if start >= previous_stop:
            chosen.append(index)
            previous_stop = stop
    return tuple(sorted(chosen))


def _canonical_targets(
    plan: RunRemovalPlan,
    settings: RemovalSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """One entry per line the run plans against, and whether it is a comb member.

    Each window fits its own estimate, so the same physical line is named at slightly
    different frequencies from window to window. Grouping them within `line_claim_hz` --
    the spectrum a resolved line already claims for itself -- is what lets a line's evidence
    be pooled across the windows at all.
    """
    entries = sorted(
        (float(target), target in window.narrow_targets_hz)
        for window in plan.windows
        for target in window.targets_hz
    )
    centres: list[float] = []
    narrow: list[bool] = []
    group: list[float] = []
    group_narrow = False
    for frequency, is_narrow in (*entries, (np.inf, False)):
        if group and frequency - group[0] > settings.line_claim_hz:
            centres.append(float(np.mean(group)))
            narrow.append(group_narrow)
            group, group_narrow = [], False
        if np.isfinite(frequency):
            group.append(frequency)
            group_narrow = group_narrow or is_narrow
    return np.asarray(centres, dtype=float), np.asarray(narrow, dtype=bool)


def _authorize_channel_targets(
    raw,
    plan: RunRemovalPlan,
    settings: RemovalSettings,
) -> RunRemovalPlan:
    """Decide, per channel, which planned lines to subtract and in which windows.

    Two questions live here and they are not the same question. Whether a channel carries a
    line is a property of the run; whether the artifact was running at all is a property of
    a window. Asked together, in one test per (window x channel) cell, both are asked where
    the data has least power -- and a real 12 dB comb loses half its harmonics to windows
    that named them and then vetoed them.

    They are separate marginals of one array of statistics, so each is asked where its
    evidence is:

    * a window's verdict pools across the comb's harmonics. One generator drives all of
      them, which is the premise the comb fit already rests on, so a window carries dozens
      of observations of whether the artifact is running. A comb that starts partway
      through a recording is found by this and by nothing else here.
    * a line's verdict pools across the windows the artifact was running in, per channel.

    Ordering them is what makes a channel that carries nothing authorize nothing: no window
    of such a channel passes the first test, so there is nothing left to pool and no
    threshold is asked to notice.

    The statistic is the probability at the single bin the target names, and that is not a
    matter of taste. `thomson_f_statistics` gives an exact probability per bin; taking the
    smallest over a reach turns 16 near-independent bins into one number and inflates the
    false-positive rate 13x, measured on three recordings of a 63-channel EEG-fMRI cohort.
    A per-window threshold can absorb that. Pooling cannot: the bias compounds once per
    window while a Bonferroni correction divides once, so a widened reach places notches on
    demonstrably empty spectrum -- 160 of 423 such places at a three-bin reach against none
    at one bin. Refining the target first is the same error moved upstream, and doubles the
    null rate.

    What this gives up is lines that sit further from where the fit places them than one
    bin can reach: 0.3% of the lines standing 10 dB or more over background on that cohort.
    What it buys is that the places carrying nothing are left alone -- 102, 170 and 156
    notches on sub-threshold spectrum, per recording, become 0, 0 and 5.
    """
    import mne
    from scipy.stats import chi2

    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    if len(picks) == 0:
        raise ValueError("Channel target authorization requires at least one EEG channel.")
    sampling_frequency_hz = float(raw.info["sfreq"])
    n_channels = len(picks)
    n_windows = len(plan.windows)
    centres, is_narrow = _canonical_targets(plan, settings)
    alpha = settings.residual_family_alpha

    # [window, line, channel], at the one bin each line names.
    probabilities = np.ones((n_windows, centres.size, n_channels), dtype=float)
    for index, window in enumerate(plan.windows):
        start, stop = window.bounds
        data = raw.get_data(picks=picks, start=start, stop=stop)
        frequencies, _, _, per_bin = estimators.thomson_f_statistics(
            data,
            sampling_frequency_hz=sampling_frequency_hz,
            bandwidth_hz=settings.mt_bandwidth,
            family_alpha=alpha,
        )
        bins = [int(np.argmin(np.abs(frequencies - centre))) for centre in centres]
        probabilities[index] = per_bin[:, bins].T
    probabilities = np.clip(probabilities, np.finfo(float).tiny, 1.0)

    def fisher(values: np.ndarray, axis: int) -> np.ndarray:
        """Combined probability of independent tests, ordered as the remaining axes."""
        return chi2.sf(-2.0 * np.log(values).sum(axis=axis), 2 * values.shape[axis])

    # A window carries the comb if its harmonics say so together. Isolated lines are left
    # out: one of them standing is no evidence that the comb generator is running.
    comb = ~is_narrow if int((~is_narrow).sum()) >= 2 else np.ones(centres.size, dtype=bool)
    running = fisher(probabilities[:, comb, :], axis=1) < alpha / max(n_windows * n_channels, 1)

    # Pool only windows that share no samples, and only those the comb was running in.
    independent = set(_independent_window_indices(plan.windows))
    carries = np.zeros((centres.size, n_channels), dtype=bool)
    for channel in range(n_channels):
        live = [index for index in np.flatnonzero(running[:, channel]) if index in independent]
        if not live:
            continue
        pooled = fisher(probabilities[live][:, :, channel], axis=0)
        carries[:, channel] = pooled < alpha / max(centres.size, 1)

    authorized_windows = []
    for index, window in enumerate(plan.windows):
        lines = [int(np.argmin(np.abs(centres - target))) for target in window.targets_hz]
        channel_targets = []
        channel_widths = []
        for channel in range(n_channels):
            # An isolated line is not the comb's to switch off: it answers to its own
            # evidence across the run and to nothing the comb generator does.
            keep = [
                (float(target), float(width))
                for target, width, line in zip(window.targets_hz, window.notch_widths_hz, lines)
                if carries[line, channel] and (is_narrow[line] or running[index, channel])
            ]
            channel_targets.append(tuple(target for target, _ in keep))
            channel_widths.append(tuple(width for _, width in keep))
        active_widths = {
            target: width
            for targets, widths in zip(channel_targets, channel_widths)
            for target, width in zip(targets, widths)
        }
        authorized_windows.append(
            replace(
                window,
                targets_hz=tuple(sorted(active_widths)),
                notch_widths_hz=tuple(active_widths[target] for target in sorted(active_widths)),
                channel_targets_hz=tuple(channel_targets),
                channel_target_widths_hz=tuple(channel_widths),
            )
        )
    return replace(plan, windows=tuple(authorized_windows))


def _refine_continuous_residual_plans(
    raw,
    plan: RunRemovalPlan,
    settings: RemovalSettings,
) -> RunRemovalPlan:
    """Resolve residuals independently inside every continuous estimation window."""
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    if len(picks) == 0:
        raise ValueError("Continuous residual refinement requires at least one EEG channel.")
    picked_info = mne.pick_info(raw.info, picks, copy=True)
    plan = _authorize_channel_targets(raw, plan, settings)
    base_cleaned = clean_continuous_raw(
        raw.copy(),
        plan,
        settings,
    )
    refined = []
    for window in plan.windows:
        start, stop = window.bounds
        refined.append(
            _refine_window_residual_plan(
                raw.get_data(picks=picks, start=start, stop=stop),
                picked_info,
                window,
                settings,
                _clean_planned_segment,
                cleaned_data=base_cleaned.get_data(
                    picks=picks,
                    start=start,
                    stop=stop,
                ),
            )
        )
    return _route_continuous_residual_support(replace(plan, windows=tuple(refined)))


def isolated_line_summary(recording: str, plan: RunIsolatedLinePlan) -> str:
    """One recording's isolated-line evidence, with both of its counts named.

    A source is a cluster of nominals inside the spectral resolution, so the two numbers
    legitimately differ: two nominals a few tens of millihertz apart are one source
    observed twice. Naming both keeps a line that gets quoted in a methods section from
    reading as an arithmetic error.
    """
    nominals = plan.all_hz
    listed = ", ".join(f"{frequency:.4f}" for frequency in nominals) or "none"
    return (
        f"  {recording}: {plan.source_count} artifact source(s) as "
        f"{len(nominals)} nominal(s) at {listed}"
    )


def build_run_plans(runs: list[Path], settings: RemovalSettings) -> dict[str, RunRemovalPlan]:
    """Fit every run independently, sharing only replicated isolated-line nominals."""
    import mne

    mne.set_log_level("ERROR")
    by_subject: dict[str, list[Path]] = {}
    for vhdr in runs:
        by_subject.setdefault(_subject_of(vhdr), []).append(vhdr)

    plans = {}
    for _subject, subject_runs in by_subject.items():
        spectra = {}
        for vhdr in subject_runs:
            raw = read_bids_raw(vhdr)
            spectra[vhdr] = session_run_spectra(raw, settings)
        isolated_line_plans = automatic_line_plans(
            list(spectra.values()),
            settings,
        )
        for vhdr, isolated_lines in zip(subject_runs, isolated_line_plans):
            run_evidence = spectra[vhdr]
            print(isolated_line_summary(vhdr.stem, isolated_lines), flush=True)
            try:
                plan = build_run_plan_from_spectra(run_evidence, settings, isolated_lines)
                raw = read_bids_raw(vhdr)
                plan = _refine_continuous_residual_plans(raw, plan, settings)
            except ValueError as error:
                raise ValueError(f"{vhdr.stem}: {error}") from error
            plans[vhdr.stem] = plan
            print(
                f"  {vhdr.stem}: {len(plan.windows)} adaptive windows, "
                f"f0 range={plan.model.fundamental_range_hz * 1e6:.0f} uHz, "
                f"max step={plan.model.max_adjacent_shift_hz * 1e6:.0f} uHz"
                f"{_grid_accuracy_note(plan, settings)}",
                flush=True,
            )
    return plans


def _preservation_against_control(
    freqs,
    *,
    probe,
    probe_before,
    probe_after,
    probe_control,
    data_before,
    data_after,
    data_control,
    targets,
    widths,
    control_targets,
    control_widths,
    cost_band_hz,
) -> dict[str, float]:
    """Compare what the transform left alone with what a displaced one leaves alone.

    Both quantities should be zero and are not quite: the removal is local, but not
    perfectly, so a little of it reaches frequencies it never targeted. How little is
    "little enough" was a constant -- 0.5 dB on the probes and 0.2 dB across the band --
    with no derivation behind either, and margins of 3452x and 8x that made the first a
    formality and the second nearly one.

    Neither is replaced by a criterion, because neither admits a valid null. The control
    here is the whole transform displaced by a quarter of the comb spacing, and it is
    matched in size, width and window geometry -- but not in what it removes. Its targets
    land between harmonics where there is no line, so it subtracts almost nothing and
    therefore leaks almost nothing, while leakage from the real transform scales with the
    line power it took out. Counted against it the real transform "fails" at p=2e-16 on
    every recording, which says only that it removed something.

    No offset repairs that: a control that removes comparable power away from the lines
    cannot exist, because the power is only at the lines. So the control is reported beside
    the observation and nothing is decided from either. What the pair does show is the
    scale of the leakage -- both sit near 0.01 dB, four orders below the 0.2 dB that used
    to be the criterion.
    """
    observed_probe = np.abs(estimators.probe_deviations_db(freqs, probe_before, probe_after, probe))
    control_probe = np.abs(
        estimators.probe_deviations_db(freqs, probe_before, probe_control, probe)
    )
    observed_nonline = np.abs(
        estimators.nonline_change_db(
            freqs, data_before, data_after, targets, widths, band_hz=cost_band_hz
        )
    )
    control_nonline = np.abs(
        estimators.nonline_change_db(
            freqs,
            data_before,
            data_control,
            control_targets,
            control_widths,
            band_hz=cost_band_hz,
        )
    )
    return {
        # Four tones on one channel is four observations, and no test on four values can
        # reach 0.05 -- the best a sign test could return is 2^-4 = 0.0625. Reporting the
        # observation beside its control says what there is to say; inventing a criterion
        # that cannot fire is the defect this work removed, not a fix for it.
        "max_probe_deviation_db": float(np.max(observed_probe)),
        "control_probe_deviation_db": float(np.max(control_probe)),
        "max_nonline_change_db": float(np.max(observed_nonline)),
        "control_nonline_change_db": float(np.max(control_nonline)),
        "nonline_change_null_p": estimators.paired_excess_p_value(
            observed_nonline, control_nonline
        ),
    }


def matched_control_plan(plan: RunRemovalPlan, settings: RemovalSettings) -> RunRemovalPlan:
    """The same transform displaced to where no line is, as a null for the preservation checks.

    Same windows, same number of targets, same widths -- only the positions move, by a
    quarter of the comb spacing. That offset is not a free choice: harmonics sit at
    ``k * f0`` and the probe tones at ``(k + 0.5) * f0``, so the quarter point is the unique
    displacement equidistant from both, and a control target therefore lands where neither
    an artifact nor an injected signal is.

    What this buys is a threshold-free preservation check. A transform that leaves the
    probes and the untouched spectrum alone should do so no more and no less than an
    identical transform aimed somewhere harmless; asking whether it did needs a control of
    the same size, not a number in decibels chosen by hand.
    """
    offset_hz = settings.nominal_fundamental_hz / 4.0
    displaced = tuple(
        replace(
            window,
            targets_hz=tuple(target + offset_hz for target in window.targets_hz),
            aggregate_residual_targets_hz=tuple(
                target + offset_hz for target in window.aggregate_residual_targets_hz
            ),
            channel_targets_hz=(
                None
                if window.channel_targets_hz is None
                else tuple(
                    tuple(target + offset_hz for target in channel)
                    for channel in window.channel_targets_hz
                )
            ),
            channel_target_widths_hz=window.channel_target_widths_hz,
            channel_residual_targets_hz=tuple(
                tuple(target + offset_hz for target in channel)
                for channel in window.channel_residual_targets_hz
            ),
        )
        for window in plan.windows
    )
    return replace(plan, windows=displaced)


def benchmark_run(
    vhdr: Path,
    settings: RemovalSettings,
    plan: RunRemovalPlan,
) -> dict:
    """Inject probes, remove the lines, and measure what came back."""
    import mne

    mne.set_log_level("ERROR")
    raw = read_bids_raw(vhdr)
    estimate = plan.model.whole_estimate
    isolated = sorted(
        {
            float(frequency)
            for window in plan.windows
            for frequency in window.estimate.isolated_hz
            if np.isfinite(frequency)
        }
    )
    targets = plan.all_targets_hz
    span_targets, span_widths = plan_target_spans(plan.windows)
    probe = settings.benchmark.probe
    estimators.check_probe_clearance(
        probe,
        targets,
        min_separation_hz=settings.benchmark.min_probe_separation_hz,
    )

    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    eeg_names = tuple(raw.ch_names[int(pick)] for pick in picks)
    times = raw.times
    waveform = probe.waveform(times)

    n_eeg_channels = len(picks)
    eeg_plan_indices = tuple(range(n_eeg_channels))
    probe_data = np.broadcast_to(waveform, (n_eeg_channels, waveform.size)).copy()
    probe_only = mne.io.RawArray(
        probe_data,
        mne.pick_info(raw.info, picks, copy=True),
        verbose="ERROR",
    )
    background_probe = mne.io.RawArray(
        raw.get_data(picks=picks) + probe_data,
        mne.pick_info(raw.info, picks, copy=True),
        verbose="ERROR",
    )

    # The one probe placed where the removal does act. Reported, never gated: signal at an
    # artifact frequency is not separable from the artifact, so a loss here is the method's
    # cost rather than a defect. Positions are read off this recording's own plan.
    in_band_hz = estimators.in_band_probe_frequencies(
        targets, count=settings.benchmark.in_band_probe_count
    )
    in_band_waveform = estimators.sinusoid_waveform(
        times, in_band_hz, probe.sinusoid_amplitude_v
    )
    in_band_probe = mne.io.RawArray(
        np.broadcast_to(in_band_waveform, (n_eeg_channels, in_band_waveform.size)).copy(),
        mne.pick_info(raw.info, picks, copy=True),
        verbose="ERROR",
    )

    # Broadband probe: every EEG channel receives an independent realization and its own
    # channel-specific removal plan, so a configured band-cost ceiling measures the worst
    # channel rather than an arbitrary subset or an average over unrelated plans.
    broadband_probe = mne.io.RawArray(
        np.random.default_rng(zlib.crc32(vhdr.stem.encode("utf-8"))).normal(
            scale=probe.sinusoid_amplitude_v,
            size=(n_eeg_channels, times.size),
        ),
        mne.pick_info(raw.info, picks, copy=True),
        verbose="ERROR",
    )

    cleaned_continuous = clean_continuous_raw(raw.copy(), plan, settings)
    cleaned_bare = cleaned_continuous
    # The same transform aimed where no line is. Everything the preservation checks compare
    # against comes from here rather than from a decibel constant.
    control_plan = matched_control_plan(plan, settings)
    cleaned_control = clean_continuous_raw(raw.copy(), control_plan, settings)
    cleaned_probe_control = clean_continuous_raw(
        probe_only.copy(),
        control_plan,
        settings,
        eeg_plan_indices=eeg_plan_indices,
    )
    cleaned_background_probe = clean_continuous_raw(
        background_probe,
        plan,
        settings,
        eeg_plan_indices=eeg_plan_indices,
    )
    cleaned_probe = clean_continuous_raw(
        probe_only,
        plan,
        settings,
        eeg_plan_indices=eeg_plan_indices,
    )
    cleaned_in_band_probe = clean_continuous_raw(
        in_band_probe,
        plan,
        settings,
        eeg_plan_indices=eeg_plan_indices,
    )
    cleaned_broadband_probe = clean_continuous_raw(
        broadband_probe,
        plan,
        settings,
        eeg_plan_indices=eeg_plan_indices,
    )

    freqs, _, _ = run_spectrum(cleaned_bare, settings)
    _, probe_psd_before = _psd(probe_only, list(range(n_eeg_channels)), settings)
    _, probe_psd_after = _psd(cleaned_probe, list(range(n_eeg_channels)), settings)
    in_band_freqs, in_band_psd_before = _psd(
        in_band_probe, list(range(n_eeg_channels)), settings
    )
    _, in_band_psd_after = _psd(
        cleaned_in_band_probe, list(range(n_eeg_channels)), settings
    )
    broadband_picks = list(range(n_eeg_channels))
    broadband_freqs, broadband_psd_before = _psd(broadband_probe, broadband_picks, settings)
    _, broadband_psd_after = _psd(cleaned_broadband_probe, broadband_picks, settings)
    _, data_psd_before = _psd(raw, picks, settings)
    _, data_psd_after = _psd(cleaned_bare, picks, settings)
    _, data_psd_control = _psd(cleaned_control, picks, settings)
    _, probe_psd_control = _psd(cleaned_probe_control, list(range(n_eeg_channels)), settings)
    control_targets, control_widths = plan_target_spans(control_plan.windows)

    recovered = estimators.recover_probe(
        cleaned_background_probe.get_data(),
        cleaned_bare.get_data(picks=picks),
    )
    boundaries = _plan_transition_boundaries(plan, raw.n_times)
    metrics = {
        **adaptive_suppression_metrics(raw, cleaned_continuous, plan, settings),
        **spatiotemporal_line_metrics(raw, cleaned_continuous, plan, settings),
        **continuous_refinement_metrics(plan, eeg_names),
        **_boundary_metrics(
            raw.get_data(picks=picks),
            cleaned_bare.get_data(picks=picks),
            boundaries,
            settings,
        ),
        "min_probe_ratio": estimators.probe_preservation(
            freqs, probe_psd_before, probe_psd_after, probe
        )["min_probe_ratio"],
        **_preservation_against_control(
            freqs,
            probe=probe,
            probe_before=probe_psd_before,
            probe_after=probe_psd_after,
            probe_control=probe_psd_control,
            data_before=data_psd_before,
            data_after=data_psd_after,
            data_control=data_psd_control,
            targets=span_targets,
            widths=span_widths,
            control_targets=control_targets,
            control_widths=control_widths,
            cost_band_hz=settings.cost_band_hz,
        ),
        **adaptive_band_metrics(
            sampling_frequency_hz=float(raw.info["sfreq"]),
            plan=plan,
            settings=settings,
        ),
        **estimators.probe_recovery(recovered, cleaned_probe.get_data(), times, probe),
        **estimators.in_band_probe_survival(
            in_band_freqs,
            in_band_psd_before,
            in_band_psd_after,
            in_band_hz,
        ),
        "in_band_probe_hz": ";".join(f"{frequency:.4f}" for frequency in in_band_hz),
        **estimators.measured_band_attenuation(
            broadband_freqs,
            spectral.to_db(broadband_psd_before),
            spectral.to_db(broadband_psd_after),
            band_hz=settings.cost_band_hz,
            thresholds_db=settings.band_cost_thresholds_db,
        ),
    }
    verdict = settings.benchmark.gate.evaluate(metrics)
    return {
        "recording": vhdr.stem,
        "fundamental_hz": estimate.fundamental_hz,
        "n_adaptive_windows": len(plan.windows),
        "fundamental_range_hz": plan.model.fundamental_range_hz,
        "max_adjacent_shift_hz": plan.model.max_adjacent_shift_hz,
        "max_window_standard_error_hz": max(
            window.estimate.fundamental_jackknife_se_hz for window in plan.windows
        ),
        "n_harmonics": estimate.n_harmonics,
        "median_targets_per_window": float(
            np.median([len(window.targets_hz) for window in plan.windows])
        ),
        "n_adjacent_sources": len(plan.all_narrow_targets_hz),
        "n_isolated_sources": len(isolated),
        "isolated_hz": ";".join(f"{frequency:.4f}" for frequency in isolated),
        "adjacent_hz": ";".join(f"{frequency:.4f}" for frequency in plan.all_narrow_targets_hz),
        **metrics,
        **{f"gate_{name}": value for name, value in verdict.items()},
        "gate_passed": all(verdict.values()),
    }


def _psd(raw, picks, settings: RemovalSettings):
    sfreq = float(raw.info["sfreq"])
    block = estimation_window_samples(sfreq, settings)
    data = raw.get_data(picks=picks)
    n_blocks = data.shape[-1] // block
    blocks = data[..., : n_blocks * block].reshape(data.shape[0], n_blocks, block)
    freqs, psd = spectral.hann_periodogram(blocks, sfreq)
    return freqs, psd.mean(axis=1)


@dataclass(frozen=True)
class _LineObservation:
    run_index: int
    position_hz: float
    prominence_db: float
    bounds: tuple[int, int] | None


@dataclass
class _LineCluster:
    observations: list[_LineObservation] = field(default_factory=list)

    @property
    def centre_hz(self) -> float:
        per_run = []
        for run_index in sorted({item.run_index for item in self.observations}):
            positions = [
                item.position_hz for item in self.observations if item.run_index == run_index
            ]
            per_run.append(float(np.median(positions)))
        return float(np.median(per_run))


def _line_observations(
    spectra: Sequence[SessionRunSpectra],
    settings: RemovalSettings,
) -> tuple[tuple[_LineObservation, ...], tuple[float, ...]]:
    observations = []
    fundamentals = []
    for run_index, run in enumerate(spectra):
        # The whole-run scaffold is fitted first and without a fallback, so a recording
        # with no comb anywhere is still refused; the windows may then inherit it.
        whole_scaffold = _detection_scaffold(*run.whole, settings)
        window_scaffolds = tuple(
            _detection_scaffold(*spectrum, settings, fallback=whole_scaffold)
            for spectrum in run.windows
        )
        sources = (
            (run.whole, None, whole_scaffold),
            *(
                (spectrum, bounds, scaffold)
                for spectrum, bounds, scaffold in zip(
                    run.windows,
                    run.bounds,
                    window_scaffolds,
                )
            ),
        )
        for (freqs, spectrum_db, prominence), bounds, scaffold in sources:
            frequency_array = np.asarray(freqs, dtype=float)
            prominence_array = np.asarray(prominence, dtype=float)
            fundamentals.append(scaffold.fundamental_hz)
            positions = estimators.detect_isolated_lines(
                freqs,
                spectrum_db,
                prominence,
                fundamental_hz=scaffold.fundamental_hz,
                harmonic_range=settings.removal_harmonic_range,
                fdr_alpha=settings.detection_fdr_alpha,
                min_prominence_db=settings.detection_min_prominence_db,
                low_hz=settings.detection_low_hz,
                high_hz=settings.detection_high_hz,
                comb_clearance_hz=settings.residual_search_hz,
                claim_hz=settings.line_claim_hz,
                max_line_width_hz=settings.max_line_width_hz,
                excluded_bands_hz=settings.protected_bands_hz,
                null_min_bins=settings.detection_null_min_bins,
                null_lower_percentile=settings.detection_null_lower_percentile,
            )
            for position in positions:
                index = int(np.argmin(np.abs(frequency_array - position)))
                observations.append(
                    _LineObservation(
                        run_index=run_index,
                        position_hz=float(position),
                        prominence_db=float(prominence_array[index]),
                        bounds=bounds,
                    )
                )
    return tuple(observations), tuple(fundamentals)


def _cluster_line_observations(
    observations: Sequence[_LineObservation],
    settings: RemovalSettings,
) -> tuple[_LineCluster, ...]:
    clusters: list[_LineCluster] = []
    for observation in sorted(observations, key=lambda item: item.position_hz):
        eligible = [
            cluster
            for cluster in clusters
            if abs(observation.position_hz - cluster.centre_hz) <= settings.line_claim_hz
            and all(
                (item.run_index, item.bounds) != (observation.run_index, observation.bounds)
                for item in cluster.observations
            )
        ]
        if eligible:
            nearest = min(
                eligible,
                key=lambda cluster: abs(observation.position_hz - cluster.centre_hz),
            )
            nearest.observations.append(observation)
        else:
            clusters.append(_LineCluster([observation]))
    return tuple(clusters)


def _distinct_source_count(positions_hz: Sequence[float], resolution_hz: float) -> int:
    """Count only sources separable by the sinusoid-fit frequency grid."""
    distinct = []
    for position_hz in sorted(positions_hz):
        if distinct and position_hz - distinct[-1] <= resolution_hz:
            continue
        distinct.append(position_hz)
    return len(distinct)


def _intervals_overlap(
    left: tuple[int, int],
    right: tuple[int, int],
) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def _comb_adjacent_observations(
    spectra: Sequence[SessionRunSpectra],
    settings: RemovalSettings,
) -> tuple[_LineObservation, ...]:
    """Comb-adjacent summits from every block and epoch spectrum, tagged by recording.

    The mirror of :func:`_line_observations`, which collects the summits that clear the
    comb by ``RESIDUAL_SEARCH_HZ``. These are the ones that do not: close enough to a
    validated harmonic to be part of the same source, far enough out that the harmonic's
    own notch never reaches them. Between the two, every narrow summit in the band is
    observed exactly once.
    """
    observations = []
    for run_index, run in enumerate(spectra):
        # Same fallback as its mirror in `_line_observations`: this fit is per window, so a
        # window that cannot establish a grid takes the one the whole recording confirmed.
        whole_scaffold = _detection_scaffold(*run.whole, settings)
        window_estimates = tuple(
            _detection_scaffold(*spectrum, settings, fallback=whole_scaffold)
            for spectrum in run.windows
        )
        sources = (
            *(
                (spectrum, bounds, estimate)
                for spectrum, bounds, estimate in zip(
                    run.windows,
                    run.bounds,
                    window_estimates,
                )
            ),
        )
        for (freqs, spectrum_db, prominence), bounds, estimate in sources:
            frequency_array = np.asarray(freqs, dtype=float)
            prominence_array = np.asarray(prominence, dtype=float)
            positions = detect_comb_adjacent_lines(
                freqs,
                spectrum_db,
                prominence,
                estimate=estimate,
                settings=settings,
            )
            for position_hz in positions:
                index = int(np.argmin(np.abs(frequency_array - position_hz)))
                observations.append(
                    _LineObservation(
                        run_index=run_index,
                        position_hz=position_hz,
                        prominence_db=float(prominence_array[index]),
                        bounds=bounds,
                    )
                )
    return tuple(observations)


def _comb_adjacent_support(
    spectra: Sequence[SessionRunSpectra],
    settings: RemovalSettings,
) -> tuple[tuple[tuple[float, tuple[tuple[int, int], ...]], ...], ...]:
    """Comb-adjacent positions that clear the replication rules, with each run's support.

    Adjacency to a validated harmonic narrows *where* a false positive can land; it does
    not supply replication. These summits are read off short-window spectra, which make
    many more comparisons than a whole-recording scan and throw up recurrent local maxima
    that are not lines, so a single block summit must not become a target in every
    overlapping window.

    The clusters are therefore formed over the whole session and admitted by the routes an
    isolated line already has to pass -- cross-recording block replication first, then the
    single-recording route. No floor here that is not already configured for those.
    """
    clusters = _cluster_line_observations(_comb_adjacent_observations(spectra, settings), settings)
    per_run: list[list[tuple[float, tuple[tuple[int, int], ...]]]] = [[] for _ in spectra]
    for cluster in clusters:
        # No _clears_every_comb_grid test: sitting beside the grid is what defines this
        # set, and detect_comb_adjacent_lines has already refused anything a harmonic's
        # own notch covers.
        supported = _block_line_support(cluster, settings)
        if not supported:
            supported = tuple(
                observation
                for run_index in range(len(spectra))
                for observation in _single_run_block_support(cluster, run_index, settings)
            )
        if not supported:
            continue
        position_hz = _run_balanced_position(supported)
        for run_index in range(len(spectra)):
            support_bounds = tuple(
                observation.bounds
                for observation in supported
                if observation.run_index == run_index and observation.bounds is not None
            )
            if support_bounds:
                per_run[run_index].append((position_hz, support_bounds))
    return tuple(tuple(item) for item in per_run)


def _comb_adjacent_window_targets(
    run: SessionRunSpectra,
    support: Sequence[tuple[float, tuple[tuple[int, int], ...]]],
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    """Route each supported comb-adjacent position to the windows that evidenced it."""
    window_targets: list[list[float]] = [[] for _ in run.windows]
    positions = []
    for position_hz, support_bounds in support:
        positions.append(position_hz)
        for index, filter_bounds in enumerate(run.bounds):
            if any(
                _intervals_overlap(filter_bounds, source_bounds) for source_bounds in support_bounds
            ):
                window_targets[index].append(position_hz)
    return (
        tuple(tuple(sorted(set(targets))) for targets in window_targets),
        tuple(sorted(positions)),
    )


def _independent_window_count(observations: Sequence[_LineObservation]) -> int:
    count = 0
    run_indices = sorted({item.run_index for item in observations})
    for run_index in run_indices:
        bounds = sorted(
            (
                item.bounds
                for item in observations
                if item.run_index == run_index and item.bounds is not None
            ),
            key=lambda interval: interval[1],
        )
        previous_stop = -1
        for start, stop in bounds:
            if start >= previous_stop:
                count += 1
                previous_stop = stop
    return count


def _run_balanced_position(observations: Sequence[_LineObservation]) -> float:
    per_run = []
    for run_index in sorted({item.run_index for item in observations}):
        positions = [item.position_hz for item in observations if item.run_index == run_index]
        per_run.append(float(np.median(positions)))
    return float(np.median(per_run))


def _whole_line_support(
    cluster: _LineCluster,
    settings: RemovalSettings,
) -> tuple[_LineObservation, ...]:
    whole_observations = tuple(item for item in cluster.observations if item.bounds is None)
    whole_runs = {item.run_index for item in whole_observations}
    # Replication alone, because every observation in the cluster already cleared the
    # calibrated detector in its own recording. Requiring one of them to also reach a fixed
    # decibel bar made the same physical line removable for one participant and not another
    # -- measured across five participants, eleven replicated positions were refused that
    # way, including one present in six recordings of six.
    if len(whole_runs) >= settings.min_runs_per_line:
        return whole_observations
    return ()


def _block_line_support(
    cluster: _LineCluster,
    settings: RemovalSettings,
) -> tuple[_LineObservation, ...]:
    supported = tuple(item for item in cluster.observations if item.bounds is not None)
    runs = {item.run_index for item in supported}
    # A block scan makes many more comparisons than a whole-recording scan. That is handled
    # where it arises -- Benjamini-Hochberg counts the bins each scan actually searched --
    # rather than by a higher decibel bar standing in for the correction.
    if (
        len(runs) >= settings.min_runs_per_block_line
        and _independent_window_count(supported) >= settings.min_independent_windows_per_line
    ):
        return supported
    return ()


def _single_run_block_support(
    cluster: _LineCluster,
    run_index: int,
    settings: RemovalSettings,
) -> tuple[_LineObservation, ...]:
    """Evidence for a line independently repeated within one recording."""
    observations = tuple(
        item
        for item in cluster.observations
        if item.run_index == run_index and item.bounds is not None
    )
    if not observations:
        return ()
    if _independent_window_count(observations) < settings.min_independent_windows_per_line:
        return ()
    return observations


def _supported_line_position(
    cluster: _LineCluster,
    settings: RemovalSettings,
) -> float | None:
    observations = _whole_line_support(cluster, settings) or _block_line_support(cluster, settings)
    if not observations:
        return None
    return _run_balanced_position(observations)


def _clears_every_comb_grid(
    nominal_hz: float,
    fundamentals_hz: Sequence[float],
    settings: RemovalSettings,
) -> bool:
    clearance = settings.detection_search_hz
    return all(
        abs(nominal_hz - round(nominal_hz / fundamental) * fundamental) > clearance
        for fundamental in fundamentals_hz
    )


def _session_supported_positions(
    clusters: Sequence[_LineCluster],
    fundamentals_hz: Sequence[float],
    settings: RemovalSettings,
) -> dict[int, float]:
    """Cluster positions supported across recordings and clear of every comb grid."""
    return {
        cluster_index: position
        for cluster_index, cluster in enumerate(clusters)
        if (position := _supported_line_position(cluster, settings)) is not None
        and _clears_every_comb_grid(position, fundamentals_hz, settings)
    }


def automatic_line_plans(
    spectra: Sequence[SessionRunSpectra],
    settings: RemovalSettings,
) -> tuple[RunIsolatedLinePlan, ...]:
    """Resolve session-replicated and recording-specific artifact lines automatically.

    Session evidence supplies one stable nominal to every run. A line confined to one run
    is accepted only after it appears in at least three non-overlapping 54-second windows
    and one occurrence reaches 15 dB prominence. Those targets stay confined to the exact
    windows that detected them; absence in another run can therefore never authorize a
    notch there.

    Fewer recordings than ``min_runs_per_line`` is not an error. Cross-recording
    replication is simply unavailable, so only the single-recording route can fire -- and
    that route is the stricter of the two, wanting 15 dB and three non-overlapping windows
    where the cross-recording route wants 10 dB in three recordings. A session of one
    continuous acquisition, which is the usual shape of resting or baseline data, is
    therefore planned under a higher bar rather than refused.
    """
    observations, fundamentals = _line_observations(spectra, settings)
    clusters = _cluster_line_observations(observations, settings)
    session_positions = _session_supported_positions(clusters, fundamentals, settings)
    comb_adjacent_support = _comb_adjacent_support(spectra, settings)

    plans = []
    for run_index, run_spectra in enumerate(spectra):
        narrow_window_hz, narrow_positions = _comb_adjacent_window_targets(
            run_spectra,
            comb_adjacent_support[run_index],
        )
        whole_hz = []
        window_hz = [[] for _ in run_spectra.windows]
        routed_session_positions = []
        for cluster_index, position_hz in session_positions.items():
            cluster = clusters[cluster_index]
            if any(
                observation.run_index == run_index and observation.bounds is None
                for observation in cluster.observations
            ):
                whole_hz.append(position_hz)
            support_bounds = tuple(
                observation.bounds
                for observation in cluster.observations
                if observation.run_index == run_index and observation.bounds is not None
            )
            if not support_bounds:
                continue
            routed_session_positions.append(position_hz)
            for index, filter_bounds in enumerate(run_spectra.bounds):
                if any(
                    _intervals_overlap(filter_bounds, source_bounds)
                    for source_bounds in support_bounds
                ):
                    window_hz[index].append(position_hz)
        local_support = {
            cluster_index: support
            for cluster_index, cluster in enumerate(clusters)
            if cluster_index not in session_positions
            and (support := _single_run_block_support(cluster, run_index, settings))
        }
        source_positions = (
            *routed_session_positions,
            *(_run_balanced_position(support) for support in local_support.values()),
            *narrow_positions,
        )
        source_count = _distinct_source_count(
            source_positions,
            spectrum_fit_nominal_resolution_hz(settings.filter_length),
        )
        for cluster_index in local_support:
            cluster = clusters[cluster_index]
            supported = local_support[cluster_index]
            position_hz = _run_balanced_position(supported)
            whole_hz.extend(
                item.position_hz
                for item in cluster.observations
                if item.run_index == run_index and item.bounds is None
            )
            support_bounds = tuple(
                observation.bounds for observation in supported if observation.bounds is not None
            )
            for index, filter_bounds in enumerate(run_spectra.bounds):
                if any(
                    _intervals_overlap(filter_bounds, source_bounds)
                    for source_bounds in support_bounds
                ):
                    window_hz[index].append(position_hz)

        plans.append(
            RunIsolatedLinePlan(
                whole_hz=tuple(sorted(set(whole_hz))),
                window_hz=tuple(tuple(sorted(set(values))) for values in window_hz),
                narrow_window_hz=narrow_window_hz,
                source_count=source_count,
            )
        )
    return tuple(plans)


def apply_run(
    vhdr: Path,
    output_root: Path,
    bids_root: Path,
    settings: RemovalSettings,
    plan: RunRemovalPlan,
):
    """Apply the exact per-run plan that passed the benchmark."""
    import mne

    mne.set_log_level("ERROR")
    raw = read_bids_raw(vhdr)
    estimate = plan.model.whole_estimate
    cleaned = clean_continuous_raw(raw.copy(), plan, settings)

    destination = output_root / vhdr.relative_to(bids_root).with_suffix(".eeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_eeg_binary(vhdr, destination, cleaned.get_data())

    verify = read_bids_raw(output_root / vhdr.relative_to(bids_root))
    expected = cleaned.get_data()
    deviation = float(np.max(np.abs(verify.get_data() - expected)))
    scale = float(np.max(np.abs(expected)))
    # The binary is float32, so a round trip loses about 2^-24 of full scale. Anything
    # beyond a decade above that is corruption, not quantisation.
    tolerance = settings.roundtrip_relative_tolerance * scale
    if deviation > tolerance:
        raise RuntimeError(
            f"{vhdr.name}: written data differs by {deviation:.3e} V, "
            f"above the {tolerance:.3e} V float32 round-trip tolerance."
        )

    suppression = adaptive_suppression_metrics(raw, cleaned, plan, settings)
    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    eeg_names = tuple(raw.ch_names[int(pick)] for pick in picks)
    boundaries = _plan_transition_boundaries(plan, raw.n_times)
    isolated = sorted(
        {
            float(frequency)
            for window in plan.windows
            for frequency in window.estimate.isolated_hz
            if np.isfinite(frequency)
        }
    )
    return {
        **adaptive_band_metrics(
            sampling_frequency_hz=float(raw.info["sfreq"]),
            plan=plan,
            settings=settings,
        ),
        "recording": vhdr.stem,
        "fundamental_hz": estimate.fundamental_hz,
        "n_adaptive_windows": len(plan.windows),
        "fundamental_range_hz": plan.model.fundamental_range_hz,
        "max_adjacent_shift_hz": plan.model.max_adjacent_shift_hz,
        "max_window_standard_error_hz": max(
            window.estimate.fundamental_jackknife_se_hz for window in plan.windows
        ),
        "residual_rms_hz": estimate.residual_rms_hz,
        "n_harmonics": estimate.n_harmonics,
        "median_targets_per_window": float(
            np.median([len(window.targets_hz) for window in plan.windows])
        ),
        "isolated_hz": ";".join(f"{frequency:.4f}" for frequency in isolated),
        "adjacent_hz": ";".join(f"{frequency:.4f}" for frequency in plan.all_narrow_targets_hz),
        "n_adjacent_sources": len(plan.all_narrow_targets_hz),
        **suppression,
        **spatiotemporal_line_metrics(raw, cleaned, plan, settings),
        **continuous_refinement_metrics(plan, eeg_names),
        **_boundary_metrics(
            raw.get_data(picks=picks),
            cleaned.get_data(picks=picks),
            boundaries,
            settings,
        ),
        "roundtrip_max_deviation_v": deviation,
        "roundtrip_relative": deviation / scale if scale else 0.0,
    }


def record_manifest_provenance(
    metrics: dict,
    *,
    input_digest: str,
    plan_digest: str,
    fingerprint: str,
) -> dict:
    """Attach the identities needed to trace one applied transform."""
    return {
        **metrics,
        "input_digest": input_digest,
        "plan_digest": plan_digest,
        "settings_fingerprint": fingerprint,
    }


def verify_cohort(
    bids_root: Path,
    cleaned_root: Path,
    settings: RemovalSettings,
    runs,
    detection: catalogue.DetectionSettings | None = None,
):
    """Run the diagnosis's own line detector over cleaned and original data alike.

    The manifest reports each run against its own targets. This asks the question the
    diagnosis asked: sweeping the whole band with FDR control and no knowledge of where
    the lines were, what is still detectable?
    """
    import mne

    mne.set_log_level("ERROR")

    plans = build_run_plans(list(runs), settings)
    spectra = {"original": {}, "cleaned": {}}
    targeted = {"original": [], "cleaned": []}
    for vhdr in runs:
        subject = _subject_of(vhdr)
        original = read_bids_raw(vhdr)
        cleaned = read_bids_raw(cleaned_root / vhdr.relative_to(bids_root))
        for label, raw in (("original", original), ("cleaned", cleaned)):
            freqs, spectrum_db, _ = run_spectrum(raw, settings)
            spectra[label].setdefault(subject, []).append(10 ** (spectrum_db / 10.0))
        targeted["original"].append(
            spatiotemporal_line_metrics(original, original, plans[vhdr.stem], settings)
        )
        targeted["cleaned"].append(
            spatiotemporal_line_metrics(original, cleaned, plans[vhdr.stem], settings)
        )

    detection = detection or catalogue.DetectionSettings()
    grids = {
        label: catalogue.build_grid(
            freqs,
            np.stack([np.median(by_run[subject], axis=0) for subject in sorted(by_run)]),
            detection.background_half_width_hz,
        )
        for label, by_run in spectra.items()
    }
    report = []
    for label, grid in grids.items():
        try:
            lines = catalogue.detect_cohort_lines(
                grid,
                detection,
                exclude_hz=detection_exclusion_hz(settings),
            )
        except catalogue.NoLinesDetected:
            # A clean stage. Anything else -- no usable window, no usable background --
            # is the analysis failing and must not be written here as zero lines.
            summary = {"n_lines": 0, "n_comb_lines": 0, "max_prominence_db": float("nan")}
            report.append({"stage": label, **summary})
            continue
        classified = catalogue.classify_lines(
            lines, catalogue.comb_structure(lines, detection), detection
        )
        report.append(
            {
                "stage": label,
                "n_lines": int(len(classified)),
                "n_comb_lines": int(classified.kind.isin(("comb", "comb_wide")).sum()),
                "n_isolated": int((classified.kind == "isolated").sum()),
                "max_prominence_db": float(classified.cohort_median_prominence_db.max()),
                "median_prominence_db": float(classified.cohort_median_prominence_db.median()),
            }
        )
    frame = pd.DataFrame(report)
    for label, rows in targeted.items():
        maximum = max(row["max_channel_block_residual_prominence_db"] for row in rows)
        maximum_excess = max(row["focal_residual_excess_db"] for row in rows)
        count = focal_residual_discoveries(rows, settings.false_discovery_rate)
        selected = frame.stage == label
        frame.loc[selected, "max_channel_block_target_db"] = maximum
        frame.loc[selected, "max_focal_residual_excess_db"] = maximum_excess
        frame.loc[selected, "n_runs_with_focal_residual"] = count
    cleaned = frame.stage == "cleaned"
    frame.loc[cleaned, "verification_passed"] = (frame.loc[cleaned, "n_comb_lines"] == 0) & (
        frame.loc[cleaned, "n_runs_with_focal_residual"] == 0
    )
    return frame, grids


def focal_residual_discoveries(rows: Sequence[dict], false_discovery_rate: float = 0.05) -> int:
    """Recordings whose focal residual exceeds what their own matched controls reach.

    The same calibrated verdict ``benchmark`` prints and ``apply`` refuses on, rather than
    a second rule of verification's own. A verification scored against a different rule
    from the one that authorised the write can only produce a disagreement nobody can act
    on.

    A missing ``focal_residual_null_p`` raises rather than counting as clean: an absent
    measurement reported as a pass is the one direction a verification must not fail in.
    """
    return int(
        estimators.residual_randomization_verdict(
            [row["focal_residual_null_p"] for row in rows],
            false_discovery_rate=false_discovery_rate,
        )["n_discoveries"]
    )


def _resolve_probe_placement(
    settings: RemovalSettings,
    plans: dict[str, RunRemovalPlan],
) -> RemovalSettings:
    """Place the probe tones from the targets of every plan about to be benchmarked.

    Returns ``settings`` unchanged when ``probe.sinusoid_hz`` is already given, so an
    explicit list stays an override rather than a suggestion.

    One set of tones serves every recording, because the probes measure preservation across
    the cohort and moving them per recording would make those measurements incomparable.
    The union of every plan's targets is therefore what they have to clear.
    """
    import dataclasses

    probe = settings.benchmark.probe
    if probe.sinusoid_hz:
        return settings

    targets = sorted(
        {float(target) for plan in plans.values() for target in plan.all_targets_hz}
    )
    fundamental = float(
        np.median([plan.model.whole_estimate.fundamental_hz for plan in plans.values()])
    )
    placed = estimators.place_probes(
        targets,
        fundamental,
        count=probe.sinusoid_count,
        band_hz=settings.cost_band_hz,
        excluded_hz=settings.protected_bands_hz,
        min_separation_hz=settings.benchmark.min_probe_separation_hz,
    )
    clearances = [
        min(abs(target - position) for target in targets) for position in placed
    ]
    print(
        "  probes placed from "
        f"{len(targets)} target(s) over {len(plans)} plan(s), f0={fundamental:.6f} Hz: "
        + ", ".join(
            f"{position:.4f} Hz (clear {clearance:.3f})"
            for position, clearance in zip(placed, clearances)
        )
    )
    return dataclasses.replace(
        settings,
        benchmark=dataclasses.replace(
            settings.benchmark,
            probe=dataclasses.replace(probe, sinusoid_hz=placed),
        ),
    )


def _report_transient_cost(frame: pd.DataFrame) -> None:
    """Report what the removal cost the injected transient, and which part of it is a defect.

    Two different losses, and only one of them means anything is wrong. The share that does
    not survive is the price of projecting out this recording's targets, and it rises with
    how much artifact the recording carries. The departure of the recovered transient from
    the same transient cleaned alone is what the removal did by interacting with the data,
    and that is the part that would be a fault.

    Reporting them together is the point: a large cost beside a collateral figure of 1.0
    means the recording had a lot of artifact, not that the removal misbehaved, and the two
    numbers are what distinguish those cases.
    """
    ratios = frame["intrinsic_energy_ratio"]
    collateral = frame["burst_energy_ratio"]
    lines = frame["n_isolated_sources"]

    print(
        f"  {'transient cost (measurement)':32s} "
        f"median {1 - ratios.median():.1%} of the burst lost to the notches, "
        f"worst {1 - ratios.min():.1%} "
        f"({lines.min()}-{lines.max()} isolated lines per recording)"
    )
    print(
        f"  {'collateral damage (measurement)':32s} "
        f"recovered/reference {collateral.min():.4f}-{collateral.max():.4f}; "
        f"1.0 is a removal that took only what its notches specify"
    )


def discover_runs(
    bids_root: Path,
    subjects: list[str] | None,
    task: str = "*",
) -> list[Path]:
    """Every recording of ``task`` under a BIDS root, with or without run and session.

    ``task`` may be ``*``, which takes every task in the dataset.

    The ``run-`` entity is optional because BIDS omits it when a task was acquired once,
    which is the normal shape of a resting or baseline acquisition. Sessions are searched
    too, so ``sub-*/ses-*/eeg/`` datasets are found without a second call.
    """
    # ``_*eeg.vhdr`` rather than ``_*_eeg.vhdr``: with no run entity the name ends
    # ``_task-<task>_eeg.vhdr``, with nothing at all between the task and the suffix.
    patterns = (
        f"sub-*/eeg/sub-*_task-{task}_*eeg.vhdr",
        f"sub-*/ses-*/eeg/sub-*_task-{task}_*eeg.vhdr",
    )
    paths = sorted({path for pattern in patterns for path in bids_root.glob(pattern)})
    if subjects:
        wanted = set(subjects)
        paths = [path for path in paths if _subject_of(path) in wanted]
    if not paths:
        raise FileNotFoundError(
            f"No recordings of task {task!r} found under {bids_root}. decomb reads "
            "BrainVision recordings at sub-*/[ses-*/]eeg/*_eeg.vhdr; set `dataset.task` "
            "in the config to the BIDS task label to process, or `*` for every task."
        )
    return paths


def _subject_of(path: Path) -> str:
    """The ``sub-*`` directory owning a recording, whether or not a session sits between."""
    for parent in path.parents:
        if parent.name.startswith("sub-"):
            return parent.name
    raise ValueError(f"{path} does not lie under a BIDS subject directory.")


def detection_exclusion_hz(settings: RemovalSettings) -> tuple[float, float] | None:
    """Return the band owned by a downstream stage, if this pass excludes one."""
    return settings.mains_notch_hz if settings.exclude_mains else None


def _write_tsv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Publish a complete table or leave the previous table untouched."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        frame.to_csv(stream, sep="\t", index=False, float_format="%.6g")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> None:
    """Execute one stage. Split from the CLI so it can be called with any namespace."""
    from decomb.config import load_config

    config = load_config(getattr(args, "config", None))
    args.bids_root = config.path("bids_root", override=args.bids_root)
    args.output_root = config.path("output_root", override=args.output_root)
    args.report_dir = config.path("removal_dir", override=args.report_dir)

    settings = RemovalSettings.from_config(config)
    overrides = {}
    if args.filter_length is not None:
        overrides["filter_length"] = args.filter_length
    if args.mt_bandwidth is not None:
        overrides["mt_bandwidth"] = args.mt_bandwidth
    if overrides:
        settings = replace(settings, **overrides)
    runs = discover_runs(args.bids_root, subjects=None, task=settings.task)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    # A repr of the settings object was the whole record of what a run used. It omits every
    # value the workflow computes, and it says nothing about which of the rest the user
    # chose and which they inherited without knowing. Both belong in the outputs.
    from decomb import effective

    written = effective.write(
        config,
        settings,
        args.report_dir / f"effective_config_{args.stage}.txt",
        stage=args.stage,
    )
    print(effective.summarise(config, settings))
    print(f"  wrote {written}")

    if args.stage == "verify":
        print(f"Verifying all {len(runs)} runs")
        report, grids = verify_cohort(
            args.bids_root,
            args.output_root,
            settings,
            runs,
            catalogue.DetectionSettings.from_config(config),
        )
        _write_tsv_atomic(report, args.report_dir / "verification.tsv")
        print(report.to_string(index=False))
        _, source_digest = source_input_digests(runs, args.bids_root)
        np.savez_compressed(
            args.report_dir / "verification_spectra.npz",
            freqs=grids["original"].freqs,
            original=grids["original"].subject_psd,
            cleaned=grids["cleaned"].subject_psd,
            subjects=np.array(sorted({_subject_of(path) for path in runs})),
            settings_fingerprint=settings_fingerprint(settings),
            source_digest=source_digest,
        )
        print(f"  wrote {args.report_dir / 'verification.tsv'}")
        return

    if args.stage == "benchmark":
        print(f"Fitting immutable per-run plans for all {len(runs)} recordings")
        plans = build_run_plans(runs, settings)
        input_digests, source_digest = source_input_digests(runs, args.bids_root)
        plan_digests = {recording: removal_plan_digest(plan) for recording, plan in plans.items()}

        # Fingerprinted before the probes are resolved, and deliberately. The fingerprint
        # says which settings a benchmark was produced under, and `apply` recomputes it
        # from its own config to check they agree. Placed probes are a function of the data
        # as well, so folding them in would make the two disagree on identical settings.
        # What the probes actually were is recorded per row instead, and apply's separate
        # input and plan digest checks already tie the run to these recordings.
        fingerprint = settings_fingerprint(settings)
        settings = _resolve_probe_placement(settings, plans)

        # Journal each recording as it completes, so a raise late in the loop does not cost
        # every recording already measured. Resuming is only safe because a journalled row
        # is reused solely when the settings, the recording's content and its fitted plan
        # are all unchanged; anything else describes different work and is measured again.
        partial_path = partial_benchmark_path(args.report_dir)
        completed = resumable_benchmark_rows(partial_path, fingerprint, input_digests, plan_digests)
        if completed:
            print(f"Resuming: {len(completed)} of {len(runs)} recordings already measured")

        rows = []
        for index, vhdr in enumerate(runs, start=1):
            started = time.time()
            if vhdr.stem in completed:
                rows.append(completed[vhdr.stem])
                print(f"[{index}/{len(runs)}] {vhdr.stem[:44]:44s} reused from partial benchmark")
                continue
            plan = plans[vhdr.stem]
            row = benchmark_run(vhdr, settings, plan)
            row["input_digest"] = input_digests[vhdr.stem]
            row["plan_digest"] = plan_digests[vhdr.stem]
            row["settings_fingerprint"] = fingerprint
            rows.append(row)
            _write_tsv_atomic(pd.DataFrame(rows), partial_path)
            print(
                f"[{index}/{len(runs)}] {vhdr.stem[:44]:44s} "
                f"f0={row['fundamental_hz']:.6f} suppress={row['median_suppression_db']:5.1f} dB "
                f"probe={row['max_probe_deviation_db']:.3f} dB "
                f"burst={row['burst_energy_ratio']:.3f} "
                f"{'PASS' if row['gate_passed'] else 'FAIL'} ({time.time() - started:.0f}s)"
            )
        frame = pd.DataFrame(rows)
        frame["settings_fingerprint"] = fingerprint
        if set(frame.recording) != set(plans):
            raise RuntimeError("The benchmark does not contain exactly one result per plan.")
        _write_tsv_atomic(frame, args.report_dir / "benchmark.tsv")
        partial_path.unlink(missing_ok=True)
        gate_columns = [c for c in frame.columns if c.startswith("gate_") and c != "gate_passed"]
        print(f"\npassed {int(frame.gate_passed.sum())}/{len(frame)} runs")
        for column in gate_columns:
            print(f"  {column:32s} {int(frame[column].sum())}/{len(frame)}")
        _report_transient_cost(frame)
        seam = estimators.seam_randomization_verdict(
            _seam_evidence_from_frame(frame), alpha=settings.seam_alpha
        )
        print(
            f"  {'seam (cohort criterion)':32s} "
            f"{'PASS' if seam['passed'] else 'FAIL'}: {int(seam['n_exceeding'])} exceeded "
            f"(count p={seam['count_p_value']:.4f}, maximum p="
            f"{seam['max_p_value']:.4f}), worst ratio {seam['max_ratio']:.2f}"
        )
        for label, column in (
            ("residual (cohort criterion)", "residual_null_p"),
            ("focal residual (cohort)", "focal_residual_null_p"),
        ):
            verdict = estimators.residual_randomization_verdict(
                frame[column].to_numpy(), false_discovery_rate=settings.false_discovery_rate
            )
            print(
                f"  {label:32s} "
                f"{'PASS' if verdict['passed'] else 'FAIL'}: "
                f"{int(verdict['n_discoveries'])} of {int(verdict['n_runs'])} recordings "
                f"(smallest p={verdict['min_run_p_value']:.3g})"
            )
        print(
            f"  {'preservation (measurement)':32s} "
            f"probes {frame['max_probe_deviation_db'].max():.2g} dB against a control's "
            f"{frame['control_probe_deviation_db'].max():.2g}; "
            f"off-target band {frame['max_nonline_change_db'].max():.3f} dB against "
            f"{frame['control_nonline_change_db'].max():.3f}"
        )
        print(
            f"  {'band cost (measurement)':32s} "
            f"median {frame['measured_band_attenuated_1db'].median():.3f}, "
            f"worst {frame['measured_band_attenuated_1db'].max():.3f} of 28-95 Hz lost by a "
            f"broadband probe"
        )
        print(
            f"  {'in-band probe survival':32s} "
            f"median {frame['median_in_band_probe_survival'].median():.3f}, "
            f"worst {frame['min_in_band_probe_survival'].min():.3f} "
            f"(measurement, not a criterion)"
        )
        print(f"  wrote {args.report_dir / 'benchmark.tsv'}")
        return

    print(f"Re-fitting all {len(runs)} plans before authorising apply")
    plans = build_run_plans(runs, settings)
    input_digests, source_digest = source_input_digests(runs, args.bids_root)
    plan_digests = {recording: removal_plan_digest(plan) for recording, plan in plans.items()}
    benchmark = require_passing_benchmark(
        args.report_dir / "benchmark.tsv",
        settings,
        recordings=input_digests,
        plans=plan_digests,
    )
    fingerprint = settings_fingerprint(settings)
    print(f"Benchmark {fingerprint} passed on all {len(runs)} recordings; applying.")

    if args.output_root.exists():
        raise FileExistsError(
            f"Refusing to mix a new derivative with existing output: {args.output_root}"
        )
    staging = args.output_root.with_name(f".{args.output_root.name}.staging-{fingerprint}")
    if staging.exists():
        raise FileExistsError(
            f"Incomplete staging output exists at {staging}; inspect it before retrying."
        )
    staging.mkdir(parents=True)
    print(f"Staging a complete derivative in {staging}")
    print(f"  copied {mirror_sidecars(args.bids_root, staging)} sidecars")

    rows = []
    for index, vhdr in enumerate(runs, start=1):
        started = time.time()
        metrics = apply_run(vhdr, staging, args.bids_root, settings, plans[vhdr.stem])
        rows.append(
            record_manifest_provenance(
                metrics,
                input_digest=input_digests[vhdr.stem],
                plan_digest=plan_digests[vhdr.stem],
                fingerprint=fingerprint,
            )
        )
        print(
            f"[{index}/{len(runs)}] {vhdr.stem[:44]:44s} "
            f"suppress={rows[-1]['median_suppression_db']:5.1f} dB "
            f"max_resid={rows[-1]['max_residual_prominence_db']:6.2f} dB "
            f"({time.time() - started:.0f}s)"
        )
    frame = pd.DataFrame(rows)
    if set(frame.recording) != set(plans):
        raise RuntimeError("The staged derivative does not contain exactly one result per plan.")
    _write_tsv_atomic(frame, staging / "removal_manifest.tsv")
    described = write_derivative_description(
        staging,
        args.bids_root,
        settings,
        source_digest,
        # From the benchmark, not from this manifest: the measured cost needs the broadband
        # probe, which only the benchmark injects.
        band_cost=measured_band_cost(benchmark),
    )
    import os

    os.replace(staging, args.output_root)
    _write_tsv_atomic(frame, args.report_dir / "removal_manifest.tsv")
    print(
        f"\nmedian suppression {frame.median_suppression_db.median():.1f} dB; "
        f"worst residual line {frame.max_residual_prominence_db.max():.2f} dB"
    )
    print(f"  declared {(args.output_root / described.name)} a derivative of {args.bids_root}")
    print(f"  wrote {args.report_dir / 'removal_manifest.tsv'}")
