"""Detect the narrowband lines in a set of spectra and recover the comb they belong to.

The catalogue is what every other stage asks questions of. :func:`detect_cohort_lines`
sweeps a prominence spectrum under FDR control with no prior knowledge of where lines
should be; :func:`comb_structure` then asks whether the narrow ones share a single
fundamental, and :func:`classify_lines` labels each detection a comb member, an isolated
line, or neither.

Nothing here reads a file. The spectra come from :mod:`decomb.remove` -- either from the
plan it fits, or from its ``verify`` stage, which re-measures written data with this same
detector so the two answers are comparable.

"Cohort" means the units the prominence is averaged over -- participants, sessions, or
recordings. A single continuous acquisition is a valid input; only the
between-participant confidence interval is then undefined, and it is reported as such.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from decomb import spectral

BACKGROUND_HALF_WIDTH_HZ = 100.0 / 21.6
"""Default half-width of the window a bin's background is estimated from, in Hz.

Wide enough that a line cannot raise its own background, narrow enough to follow the
1/f slope. Prominence is measured against the lower half of this window, so a
neighbouring line does not inflate it either. Overridden by
``detection.background_half_width_hz``.
"""


@dataclass(frozen=True)
class DetectionSettings:
    """How the catalogue looks for lines. Read from the config's ``detection`` block.

    ``diagnose`` and ``verify`` both use these, so the sweep that finds the lines and the
    sweep that checks whether they went are the same measurement.
    """

    low_hz: float = 3.0
    high_hz: float = 95.0
    """Band swept for lines. Set it to the span your analyses read."""
    background_half_width_hz: float = BACKGROUND_HALF_WIDTH_HZ
    fdr_alpha: float = 0.05
    """Screening level after dependence-robust empirical-null correction."""
    null_min_bins: int = 32
    null_lower_percentile: float = 15.865525393145702
    comb_chance_sigma: float = 2.0
    tr_tolerance_bins: float = 1.0
    comb_tolerance_hz: float = 0.06
    """How near an integer multiple a line must sit to count as a comb member."""
    max_pair_spacing_hz: float = 12.0
    """Largest gap between two detections that may vote for the comb spacing. Above a few
    times the fundamental the votes are all multiples and add nothing."""
    narrow_linewidth_ratio: float = 3.0
    """Half-power width, in window widths, below which a detection is monochromatic.

    This decides only detections that are *not* comb members; see :func:`classify_lines`
    for why membership is the primary criterion. A Hann window imposes a floor of
    1.4382/T, so a true sinusoid measures a little above 1.0 and anything appreciably
    wider is not one.
    """
    wide_member_ratio: float = 10.0
    """Comb members wider than this are reported separately and never masked.

    A detection this wide sitting on a comb position is having its width set by the
    broadband activity underneath it rather than by a line. It is still reported as a
    member, because arithmetic coincidence at millihertz precision is not plausible, but
    it is kept out of band-power masking, where counting a real rhythm as artifact would
    overstate contamination.
    """
    line_mask_half_width_hz: float = 0.15
    """How much spectrum around a line counts as belonging to it, when charging bands."""
    max_subharmonic_divisor: int = 6
    min_subharmonic_gain: float = 0.2
    """Governs :func:`decomb.spectral.refine_comb_fundamental`: how far below the
    commonest gap the true fundamental may lie, and how much more of the comb a divisor
    must explain before it is accepted."""
    spacing_search_fraction: float = 0.02
    """How far either side of a candidate spacing the fundamental is refined. A refinement
    of a measured gap, not a search for a period -- keep it small."""
    bootstrap_resamples: int = 10_000
    bootstrap_alpha: float = 0.05
    """Interval level of the reported bootstrap: 0.05 gives a 95% interval."""
    bootstrap_seed: int = 42
    """Percentile bootstrap of each line's prominence across the sampling unit."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_hz < self.high_hz:
            raise ValueError("The detection band must be increasing and non-negative.")
        if not 0.0 < self.fdr_alpha < 1.0:
            raise ValueError("fdr_alpha must lie strictly between zero and one.")
        for name in (
            "background_half_width_hz",
            "null_lower_percentile",
            "comb_chance_sigma",
            "tr_tolerance_bins",
            "comb_tolerance_hz",
            "max_pair_spacing_hz",
            "narrow_linewidth_ratio",
            "wide_member_ratio",
            "line_mask_half_width_hz",
            "min_subharmonic_gain",
            "spacing_search_fraction",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"detection.{name} must be finite and positive.")
        if self.narrow_linewidth_ratio > self.wide_member_ratio:
            raise ValueError("narrow_linewidth_ratio cannot exceed wide_member_ratio.")
        if self.null_min_bins < 2:
            raise ValueError("null_min_bins must be at least two.")
        if self.max_subharmonic_divisor < 1:
            raise ValueError("max_subharmonic_divisor must be at least one.")
        if self.bootstrap_resamples < 2:
            raise ValueError("bootstrap_resamples must be at least two.")
        if not 0.0 < self.bootstrap_alpha < 1.0:
            raise ValueError("bootstrap_alpha must lie strictly between zero and one.")

    @classmethod
    def from_config(cls, config) -> DetectionSettings:
        block = dict(config.get("detection") or {})
        known = {entry.name for entry in fields(cls)}
        unknown = set(block) - known
        if unknown:
            raise ValueError(
                f"Unknown `detection` setting(s): {sorted(unknown)}. Known settings are "
                f"{sorted(known)}."
            )
        integers = {
            "max_subharmonic_divisor",
            "bootstrap_resamples",
            "bootstrap_seed",
            "null_min_bins",
        }
        return cls(
            **{
                name: int(value) if name in integers else float(value)
                for name, value in block.items()
            }
        )


@dataclass
class Grid:
    """One frequency grid with the cohort's spectra and prominences on it."""

    freqs: np.ndarray
    subject_psd: np.ndarray  # (n_subjects, n_freqs)
    subject_prominence: np.ndarray  # (n_subjects, n_freqs)
    half_width_bins: int

    @property
    def bin_width_hz(self) -> float:
        return float(self.freqs[1] - self.freqs[0])


def half_width_bins(freqs: np.ndarray, half_width_hz: float = BACKGROUND_HALF_WIDTH_HZ) -> int:
    return int(round(half_width_hz / float(freqs[1] - freqs[0])))


def prominence_of(
    spectrum: np.ndarray, freqs: np.ndarray, half_width_hz: float = BACKGROUND_HALF_WIDTH_HZ
) -> np.ndarray:
    return spectral.prominence_db(
        spectral.to_db(spectrum), half_width_bins=half_width_bins(freqs, half_width_hz)
    )


def build_grid(
    freqs: np.ndarray,
    subject_psd: np.ndarray,
    half_width_hz: float = BACKGROUND_HALF_WIDTH_HZ,
) -> Grid:
    prominence = np.stack(
        [prominence_of(spectrum, freqs, half_width_hz) for spectrum in subject_psd]
    )
    return Grid(freqs, subject_psd, prominence, half_width_bins(freqs, half_width_hz))


class NoLinesDetected(RuntimeError):
    """No line survived FDR control, which is a measurement rather than a fault.

    Separated from the other RuntimeErrors this module raises so a caller can report a
    genuinely clean dataset without also swallowing "no usable background estimate",
    which means the analysis could not run. Recorded identically, that would read as
    successful cleaning -- the most dangerous direction for a verification step to fail
    in.
    """


def detection_mask(
    freqs: np.ndarray,
    *,
    low_hz: float = 3.0,
    high_hz: float = 95.0,
    exclude_hz: tuple[float, float] | None = None,
) -> np.ndarray:
    """Bins the detector is allowed to look at.

    ``exclude_hz`` blanks a band that is being handled by something else -- typically
    mains, when a wide notch takes it. A band left in that no sinusoid subtraction can
    reach produces detections nothing can act on.
    """
    inside = (freqs >= low_hz) & (freqs <= high_hz)
    if exclude_hz is not None:
        inside &= ~((freqs >= exclude_hz[0]) & (freqs <= exclude_hz[1]))
    return inside


def detect_cohort_lines(
    grid: Grid,
    settings: DetectionSettings | None = None,
    *,
    exclude_hz: tuple[float, float] | None = None,
    tr_seconds: float | None = None,
) -> pd.DataFrame:
    """Find lines in the cohort-mean prominence spectrum under FDR control.

    The sweep has no prior knowledge of where lines should be, which is what makes it
    usable as a verification of data already written: it can find a line the removal
    never targeted.

    ``tr_seconds`` is optional and purely descriptive. Given one, each detection also
    reports where it sits relative to the ``k / TR`` grid, which separates a line locked
    to a periodic acquisition from one that is not.
    """
    settings = settings or DetectionSettings()
    mask = detection_mask(
        grid.freqs, low_hz=settings.low_hz, high_hz=settings.high_hz, exclude_hz=exclude_hz
    )
    usable = mask & np.all(np.isfinite(grid.subject_prominence), axis=0)
    candidates = np.flatnonzero(usable)
    if candidates.size == 0:
        raise RuntimeError("No frequency bin has a usable background estimate.")
    cohort = np.full(grid.freqs.size, np.nan)
    cohort[candidates] = grid.subject_prominence[:, candidates].mean(axis=0)

    pvalues = spectral.upper_tail_pvalues(
        cohort[candidates],
        min_bins=settings.null_min_bins,
        lower_percentile=settings.null_lower_percentile,
    )
    qvalues = spectral.fdr_by(pvalues)
    significant = np.zeros(grid.freqs.size, dtype=bool)
    significant[candidates] = qvalues < settings.fdr_alpha
    q_by_bin = np.ones(grid.freqs.size)
    q_by_bin[candidates] = qvalues

    peaks = spectral.cluster_peaks(significant, np.nan_to_num(cohort, nan=-np.inf))
    if not peaks:
        raise NoLinesDetected("No line survived FDR control; nothing to characterise.")

    # Each participant is scored against their own null so prevalence is not driven by
    # whichever participants happen to have the largest lines.
    subject_significant = np.zeros_like(grid.subject_prominence, dtype=bool)
    for index in range(grid.subject_prominence.shape[0]):
        values = grid.subject_prominence[index, candidates]
        subject_significant[index, candidates] = (
            spectral.fdr_by(
                spectral.upper_tail_pvalues(
                    values,
                    min_bins=settings.null_min_bins,
                    lower_percentile=settings.null_lower_percentile,
                )
            )
            < settings.fdr_alpha
        )

    # A percentile bootstrap resamples the sampling unit, so a single participant has no
    # interval to report -- and a lone continuous acquisition is a shape this workflow
    # accepts on purpose. The line is still detected and its position, width and prominence
    # are all still measured; only the between-participant interval is undefined, and it is
    # reported as undefined rather than aborting a verification of data already written.
    single_unit = grid.subject_prominence.shape[0] < 2

    records = []
    for index in peaks:
        values = grid.subject_prominence[:, index]
        if single_unit:
            point, low, high = float(np.median(values)), float("nan"), float("nan")
        else:
            point, low, high = spectral.bootstrap_ci(
                values,
                n_resamples=settings.bootstrap_resamples,
                alpha=settings.bootstrap_alpha,
                seed=settings.bootstrap_seed,
            )
        refined = spectral.refine_peak_frequency(grid.freqs, cohort, index)
        position = None if tr_seconds is None else spectral.comb_index(refined, tr=tr_seconds)
        neighbourhood = slice(max(index - 1, 0), index + 2)
        linewidth = spectral.spectral_linewidth_hz(grid.freqs, cohort, index)
        records.append(
            {
                "bin": index,
                "frequency_hz": float(grid.freqs[index]),
                "refined_hz": refined,
                "linewidth_hz": linewidth,
                "linewidth_over_resolution": linewidth
                / spectral.hann_resolution_hz(1.0 / grid.bin_width_hz),
                "is_narrow": bool(
                    linewidth / spectral.hann_resolution_hz(1.0 / grid.bin_width_hz)
                    < settings.narrow_linewidth_ratio
                ),
                "cohort_mean_prominence_db": float(cohort[index]),
                "cohort_median_prominence_db": point,
                "ci_low_db": low,
                "ci_high_db": high,
                "q_value": float(q_by_bin[index]),
                "n_subjects_detected": int(
                    np.sum(np.any(subject_significant[:, neighbourhood], axis=1))
                ),
                "n_subjects": int(grid.subject_prominence.shape[0]),
                # Position on the k/TR grid, when an acquisition period was supplied.
                # Distinct from the comb_harmonic that classify_lines derives from the
                # fitted fundamental: this one asks whether the line is locked to a
                # periodic acquisition, that one asks which comb member it is.
                "tr_harmonic": -1 if position is None else position.harmonic_index,
                "tr_offset_hz": np.nan if position is None else position.offset_hz,
                "on_tr_comb": (
                    False
                    if position is None
                    else bool(
                        abs(position.offset_hz)
                        < settings.tr_tolerance_bins * grid.bin_width_hz
                    )
                ),
            }
        )
    return pd.DataFrame(records).sort_values("frequency_hz").reset_index(drop=True)


def comb_structure(
    lines: pd.DataFrame,
    settings: DetectionSettings | None = None,
    *,
    tr_seconds: float | None = None,
) -> pd.DataFrame:
    """Recover the comb the narrow lines belong to, and fit its fundamental.

    Only narrow lines take part. The dominant repeated spacing is found first, then each
    line is assigned the nearest harmonic index of that spacing and the fundamental is
    re-estimated by least squares over the assignments. The residuals are the evidence:
    a source with one fixed period leaves residuals at the millihertz level, while a set
    of unrelated peaks that merely happen to fall near a common spacing does not.

    ``tr_seconds`` adds one descriptive column, ``tr_comb_uniformity_p``: whether the
    family's phases on the ``k / TR`` grid are uniform. A small value means the lines are
    locked to the acquisition period rather than merely near it.
    """
    settings = settings or DetectionSettings()
    tolerance_hz = settings.comb_tolerance_hz
    narrow = lines.loc[lines["is_narrow"], "refined_hz"].to_numpy()
    rows: list[dict] = []
    if narrow.size < 3:
        return pd.DataFrame(rows)

    try:
        gap, support = spectral.dominant_spacing(
            narrow,
            max_difference_hz=settings.max_pair_spacing_hz,
            tolerance_hz=tolerance_hz,
        )
    except ValueError:
        # No repeated spacing to find. That is a result, not a failure, and it must not
        # take the rest of the cohort analysis down with it.
        spacing, support = float("nan"), 0
        members = np.zeros(narrow.size, dtype=bool)
    else:
        spacing, members = spectral.refine_comb_fundamental(
            narrow,
            gap,
            tolerance_hz=tolerance_hz,
            max_divisor=settings.max_subharmonic_divisor,
            min_gain=settings.min_subharmonic_gain,
            search_fraction=settings.spacing_search_fraction,
            chance_sigma=settings.comb_chance_sigma,
        )

    if members.sum() >= 3:
        try:
            fit_intercept_hz = spectral.fit_arithmetic_comb(narrow[members]).intercept_hz
        except ValueError:
            # Two members inside the tolerance of the *same* harmonic, so rounding puts
            # both on one index and the free-intercept fit cannot be posed. That costs the
            # intercept and nothing else: the fundamental below is fitted through the
            # origin, which is the physically right model anyway.
            #
            # Discarding the family instead is wrong: one close pair would throw away a
            # comb of dozens of members, and `verify` would then report zero comb lines on
            # uncleaned data.
            fit_intercept_hz = float("nan")

        # Re-fit through the origin: a comb generated by one periodic source has lines at
        # exact integer multiples, so the fundamental is the only free parameter.
        harmonics = np.rint(narrow[members] / spacing)
        fundamental = float(np.sum(harmonics * narrow[members]) / np.sum(harmonics**2))
        through_origin = narrow[members] - harmonics * fundamental
        rows.append(
            {
                "family": "narrow_comb",
                "fundamental_hz": fundamental,
                "spacing_from_pairs_hz": spacing,
                "supporting_pairs": support,
                "n_lines": int(members.sum()),
                "harmonic_min": int(harmonics.min()),
                "harmonic_max": int(harmonics.max()),
                "rmse_hz": float(np.sqrt(np.mean(through_origin**2))),
                "max_abs_residual_hz": float(np.max(np.abs(through_origin))),
                "free_intercept_hz": fit_intercept_hz,
                "tr_comb_uniformity_p": (
                    np.nan
                    if tr_seconds is None
                    else spectral.comb_uniformity_pvalue(narrow[members], tr=tr_seconds)
                ),
            }
        )
    remainder = narrow[~members]
    if remainder.size:
        rows.append(
            {
                "family": "narrow_off_comb",
                "fundamental_hz": np.nan,
                "spacing_from_pairs_hz": np.nan,
                "supporting_pairs": 0,
                "n_lines": int(remainder.size),
                "harmonic_min": -1,
                "harmonic_max": -1,
                "rmse_hz": np.nan,
                "max_abs_residual_hz": np.nan,
                "free_intercept_hz": np.nan,
                "tr_comb_uniformity_p": (
                    spectral.comb_uniformity_pvalue(remainder, tr=tr_seconds)
                    if tr_seconds is not None and remainder.size >= 2
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def classify_lines(
    lines: pd.DataFrame,
    structure: pd.DataFrame,
    settings: DetectionSettings | None = None,
) -> pd.DataFrame:
    """Label every detection: comb member, isolated narrow line, or other.

    Comb membership comes first and is decided arithmetically, because it is the sharper
    criterion: a detection sitting within a few millihertz of an integer multiple is a
    member whatever its measured width. Width is used only for the detections that are
    not members.

    Deciding membership by width first would be wrong. Half-power width is measured
    against the peak's own height, so a weak line reaches the half-power point further out
    and measures wider than a strong one from the same source; on a real comb the two are
    strongly anticorrelated. Width separates a monochromatic source from a brain rhythm
    only at comparable amplitude.
    """
    settings = settings or DetectionSettings()
    tolerance_hz = settings.comb_tolerance_hz
    comb = (
        structure.loc[structure["family"] == "narrow_comb"]
        if "family" in structure.columns
        else structure
    )
    labelled = lines.copy()
    if not len(comb):
        labelled["comb_harmonic"] = -1
        labelled["comb_residual_hz"] = np.nan
        labelled["kind"] = np.where(labelled["is_narrow"], "isolated", "other")
        return labelled

    fundamental = float(comb["fundamental_hz"].iloc[0])
    harmonics = np.rint(labelled["refined_hz"] / fundamental)
    residual = labelled["refined_hz"] - harmonics * fundamental
    member = np.abs(residual) <= tolerance_hz

    labelled["comb_harmonic"] = np.where(member, harmonics, -1).astype(int)
    labelled["comb_residual_hz"] = np.where(member, residual, np.nan)
    wide = labelled["linewidth_over_resolution"] >= settings.wide_member_ratio
    labelled["kind"] = np.where(
        member,
        np.where(wide, "comb_wide", "comb"),
        np.where(labelled["is_narrow"], "isolated", "other"),
    )
    return labelled


def band_impact(
    grid: Grid,
    subjects: Sequence[str],
    lines: pd.DataFrame,
    bands: Mapping[str, Sequence[float]],
    settings: DetectionSettings | None = None,
) -> pd.DataFrame:
    """Fraction of each band's power that is artifact, per subject.

    This is the number that decides whether removal is worth doing at all. A comb that
    carries a third of the gamma band is worth a transform; one that carries a percent of
    it is not.

    Measured as excess over the local background at the line bins. Dropping the line bins
    and comparing band powers instead also drops their background, which counts ordinary
    spectrum as contamination and overstates the share.

    Only comb members and isolated narrow lines count. The remaining detections are broad,
    weak, and present in few subjects; charging those to the artifact would be charging it
    for the brain rhythms the band exists to measure.

    A band the share cannot be estimated in -- one lying too close to DC for a symmetric
    background window, say -- is reported as NaN. This is a measurement, and one band it
    cannot answer for must not take the rest of the diagnosis with it.
    """
    settings = settings or DetectionSettings()
    narrow = lines.loc[lines["kind"].isin(("comb", "isolated"))]
    artifact = list(narrow["refined_hz"])
    rows = []
    for position, subject in enumerate(subjects):
        spectrum = grid.subject_psd[position]
        for name, (low, high) in sorted(bands.items(), key=lambda item: item[1][0]):
            low, high = float(low), float(high)
            try:
                fraction = spectral.line_excess_fraction(
                    grid.freqs,
                    spectrum,
                    low_hz=low,
                    high_hz=high,
                    line_freqs=artifact,
                    half_width_bins=grid.half_width_bins,
                    line_half_width_hz=settings.line_mask_half_width_hz,
                )
            except ValueError:
                fraction = float("nan")
            rows.append(
                {
                    "subject": subject,
                    "band": name,
                    "low_hz": low,
                    "high_hz": high,
                    "n_lines_inside": sum(1 for f in artifact if low <= f <= high),
                    "n_artifact_lines_total": int(len(narrow)),
                    "artifact_share": fraction,
                    "artifact_share_percent": 100.0 * fraction,
                    "line_contribution_db": (
                        float("nan")
                        if not np.isfinite(fraction)
                        else float(-10.0 * np.log10(max(1.0 - fraction, 1e-12)))
                    ),
                }
            )
    return pd.DataFrame(rows)
