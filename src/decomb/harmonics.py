"""Threshold-free discovery and localization of a harmonic line comb.

The detector compares two explicit spectral models.  The null says that power at an
integer grid is no different from power halfway between grid points.  The alternative
adds one positive mean contrast at a freely selected fundamental.  The Bayesian
information criterion (BIC), including the description length of the frequency search,
decides between them.  There is no amplitude, prominence, prevalence, or harmonic-number
cutoff to tune.

Once the alternative wins, every integer multiple inside the configured analysis range
belongs to the correction plan. Local spectra refine positions; they never decide
whether an already-authorized harmonic exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

MAXIMUM_FREQUENCY_HZ = 100.0


class NoCombDetected(RuntimeError):
    """The no-comb model describes the spectrum better than a harmonic grid."""


@dataclass(frozen=True)
class CombEstimate:
    """One spectrum's selected fundamental and complete harmonic grid."""

    fundamental_hz: float
    harmonics: tuple[int, ...]
    positions_hz: tuple[float, ...]
    evidence_bic: float

    def __post_init__(self) -> None:
        _validate_harmonic_positions(self.harmonics, self.positions_hz)
        scalars = (
            self.fundamental_hz,
            self.evidence_bic,
        )
        if not np.all(np.isfinite(scalars)):
            raise ValueError("Comb estimates must contain finite values.")
        if self.fundamental_hz <= 0.0:
            raise ValueError("The comb fundamental must be positive.")
        if self.evidence_bic >= 0.0:
            raise ValueError("A selected comb must improve BIC over the no-comb model.")

    @property
    def n_harmonics(self) -> int:
        return len(self.harmonics)


@dataclass(frozen=True)
class HarmonicEvidence:
    """Positions of every authorized harmonic in one spectrum."""

    harmonics: tuple[int, ...]
    positions_hz: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_harmonic_positions(self.harmonics, self.positions_hz)


@dataclass(frozen=True)
class IsolatedLineModel:
    """Whole-recording isolated lines and their position in every window."""

    positions_hz: tuple[float, ...]
    evidence_bic: tuple[float, ...]
    window_positions_hz: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        count = len(self.positions_hz)
        if count != len(self.evidence_bic):
            raise ValueError("Every isolated line requires one BIC value.")
        if any(len(positions) != count for positions in self.window_positions_hz):
            raise ValueError("Every window must localize every isolated line.")
        if not np.all(np.isfinite(self.positions_hz)):
            raise ValueError("Isolated-line positions must be finite.")
        if self.positions_hz != tuple(sorted(self.positions_hz)):
            raise ValueError("Isolated-line positions must be sorted.")
        if self.evidence_bic and max(self.evidence_bic) >= 0.0:
            raise ValueError("Selected isolated lines must improve BIC.")


@dataclass(frozen=True)
class AdaptiveCombModel:
    """Complete comb plus independently supported isolated line components."""

    whole_estimate: CombEstimate
    window_evidence: tuple[HarmonicEvidence, ...]
    isolated_lines: IsolatedLineModel

    def __post_init__(self) -> None:
        if not self.window_evidence:
            raise ValueError("At least one estimation window is required.")
        expected = self.whole_estimate.harmonics
        if any(evidence.harmonics != expected for evidence in self.window_evidence):
            raise ValueError("Every window must localize the complete authorized comb.")
        if len(self.isolated_lines.window_positions_hz) != len(self.window_evidence):
            raise ValueError("Comb and isolated lines must use the same windows.")


def estimate_comb(
    frequencies_hz: Sequence[float],
    spectrum_db: Sequence[float],
    *,
    spectral_resolution_hz: float,
    frequency_range_hz: tuple[float, float] = (0.0, MAXIMUM_FREQUENCY_HZ),
) -> CombEstimate:
    """Select a harmonic model by minimum description length.

    Candidate spacing is exhaustive at the precision needed to keep its highest
    harmonic within half a DFT bin.  Four observable multiples are required because the
    alternative estimates a frequency, a contrast mean, and residual variance; fewer
    observations cannot identify those quantities independently.
    """
    frequencies, spectrum = _validated_spectrum(frequencies_hz, spectrum_db)
    spectral_resolution = _positive(spectral_resolution_hz, "spectral_resolution_hz")
    minimum_frequency, maximum_frequency = _analysis_range(
        frequencies,
        frequency_range_hz,
    )
    bin_width_hz = float(frequencies[1] - frequencies[0])
    candidates = _fundamental_candidates(
        bin_width_hz,
        spectral_resolution,
        maximum_frequency,
    )
    search_description_length = 2.0 * np.log(candidates.size)

    scored = np.array(
        [
            _comb_score(
                frequencies,
                spectrum,
                fundamental_hz=candidate,
                frequency_range_hz=(minimum_frequency, maximum_frequency),
                search_description_length=search_description_length,
            )
            for candidate in candidates
        ]
    )
    supported = np.isfinite(scored[:, 0]) & (scored[:, 0] < 0.0)
    if not np.any(supported):
        best_bic = float(np.min(scored[:, 0]))
        raise NoCombDetected(
            "The spectrum contains no supported harmonic comb through "
            f"{maximum_frequency:g} Hz (best ΔBIC={best_bic:+.3f})."
        )

    # A BIC can decide whether each grid improves on its own null, but BIC values built
    # from grids of different sizes are not comparable likelihoods: a dense grid can
    # accumulate weak curvature at hundreds of locations. Rank supported grids by their
    # matched-contrast score, mean contrast times sqrt(number of positions). The square
    # root puts grids on the common scale of a summed independent signal: subharmonics
    # are diluted by extra empty locations, while multiples lose the evidence in omitted
    # lines. This is a model ranking, not an amplitude cutoff.
    supported_indices = np.flatnonzero(supported)
    supported_scores = scored[supported_indices]
    harmonic_counts = np.array(
        [
            len(
                _harmonic_numbers(
                    float(candidates[index]),
                    (minimum_frequency, maximum_frequency),
                )
            )
            for index in supported_indices
        ]
    )
    matched_contrast = supported_scores[:, 1] * np.sqrt(harmonic_counts)
    strongest_contrast = np.max(matched_contrast)
    strongest = supported_indices[matched_contrast == strongest_contrast]
    selected = int(strongest[np.argmin(scored[strongest, 0])])
    evidence_bic = float(scored[selected, 0])

    fundamental_hz = float(candidates[selected])
    harmonic_numbers = _harmonic_numbers(
        fundamental_hz,
        (minimum_frequency, maximum_frequency),
    )
    evidence = localize_harmonics(
        frequencies,
        spectrum,
        harmonics=harmonic_numbers,
        fundamental_hz=fundamental_hz,
        spectral_resolution_hz=spectral_resolution,
    )
    return CombEstimate(
        fundamental_hz=fundamental_hz,
        harmonics=harmonic_numbers,
        positions_hz=evidence.positions_hz,
        evidence_bic=evidence_bic,
    )


def localize_harmonics(
    frequencies_hz: Sequence[float],
    spectrum_db: Sequence[float],
    *,
    harmonics: Sequence[int],
    fundamental_hz: float,
    spectral_resolution_hz: float,
) -> HarmonicEvidence:
    """Refine every authorized grid position without an amplitude gate."""
    frequencies, spectrum = _validated_spectrum(frequencies_hz, spectrum_db)
    fundamental = _positive(fundamental_hz, "fundamental_hz")
    resolution = _positive(spectral_resolution_hz, "spectral_resolution_hz")
    harmonic_numbers = tuple(int(value) for value in harmonics)
    if harmonic_numbers != tuple(sorted(set(harmonic_numbers))):
        raise ValueError("Harmonics must be sorted and unique.")
    if not harmonic_numbers or harmonic_numbers[0] < 1:
        raise ValueError("Harmonics must contain positive integers.")

    positions = tuple(
        _local_peak_position(
            frequencies,
            spectrum,
            target_hz=harmonic * fundamental,
            spectral_resolution_hz=resolution,
        )
        for harmonic in harmonic_numbers
    )
    return HarmonicEvidence(harmonic_numbers, positions)


def detect_isolated_lines(
    frequencies_hz: Sequence[float],
    whole_spectrum_db: Sequence[float],
    window_spectra_db: Sequence[Sequence[float]],
    *,
    comb: CombEstimate,
    spectral_resolution_hz: float,
    independent_window_indices: Sequence[int],
    frequency_range_hz: tuple[float, float] = (0.0, MAXIMUM_FREQUENCY_HZ),
) -> IsolatedLineModel:
    """Select off-comb narrow lines by BIC across independent Hann windows."""
    frequencies, whole_spectrum = _validated_spectrum(
        frequencies_hz,
        whole_spectrum_db,
    )
    windows = np.asarray(window_spectra_db, dtype=float)
    if windows.ndim != 2 or windows.shape[1] != frequencies.size:
        raise ValueError("Window spectra must form a window-by-frequency array.")
    if windows.shape[0] < 2 or not np.all(np.isfinite(windows)):
        raise ValueError("Isolated-line detection requires two finite spectral windows.")
    resolution = _positive(spectral_resolution_hz, "spectral_resolution_hz")
    independent_indices = np.asarray(independent_window_indices, dtype=int)
    if (
        independent_indices.ndim != 1
        or independent_indices.size < 2
        or np.any(independent_indices < 0)
        or np.any(independent_indices >= windows.shape[0])
        or np.any(np.diff(independent_indices) <= 0)
    ):
        raise ValueError(
            "independent_window_indices must select at least two ordered unique windows."
        )
    minimum_frequency, maximum_frequency = _analysis_range(
        frequencies,
        frequency_range_hz,
    )

    from scipy.signal import find_peaks

    candidate_indices = find_peaks(whole_spectrum)[0]
    inside = (frequencies[candidate_indices] >= max(minimum_frequency, resolution)) & (
        frequencies[candidate_indices] <= maximum_frequency
    )
    candidate_indices = candidate_indices[inside]
    harmonic_grid_hz = comb.fundamental_hz * np.asarray(comb.harmonics)
    off_comb = np.array(
        [
            np.min(np.abs(harmonic_grid_hz - frequencies[index])) > resolution
            for index in candidate_indices
        ],
        dtype=bool,
    )
    candidate_indices = candidate_indices[off_comb]
    if candidate_indices.size == 0:
        return IsolatedLineModel((), (), tuple(() for _ in windows))

    search_description_length = 2.0 * np.log(candidate_indices.size)
    independent_windows = windows[independent_indices]
    selected: list[tuple[float, float]] = []
    for index in candidate_indices:
        position_hz = _local_peak_position(
            frequencies,
            whole_spectrum,
            target_hz=float(frequencies[index]),
            spectral_resolution_hz=resolution,
        )
        contrasts_db = _line_contrasts(
            frequencies,
            independent_windows,
            position_hz=position_hz,
            spectral_resolution_hz=resolution,
        )
        evidence_bic = _positive_mean_bic(
            contrasts_db,
            search_description_length=search_description_length,
        )
        shape_bic = _line_shape_bic(
            frequencies,
            whole_spectrum,
            position_hz=position_hz,
            spectral_resolution_hz=resolution,
            search_description_length=search_description_length,
        )
        least_favourable_bic = max(evidence_bic, shape_bic)
        if least_favourable_bic < 0.0:
            selected.append((position_hz, least_favourable_bic))

    selected.sort()
    positions_hz = tuple(position for position, _ in selected)
    evidence_bic = tuple(value for _, value in selected)
    window_positions = tuple(
        tuple(
            _local_peak_position(
                frequencies,
                window,
                target_hz=position_hz,
                spectral_resolution_hz=resolution,
            )
            for position_hz in positions_hz
        )
        for window in windows
    )
    return IsolatedLineModel(positions_hz, evidence_bic, window_positions)


def _line_shape_bic(
    frequencies_hz: np.ndarray,
    spectrum_db: np.ndarray,
    *,
    position_hz: float,
    spectral_resolution_hz: float,
    search_description_length: float,
) -> float:
    """Compare a resolution-limited Hann line with a smooth local spectrum."""
    radius_hz = 4.0 * spectral_resolution_hz
    inside = np.abs(frequencies_hz - position_hz) <= radius_hz
    local_frequencies_hz = frequencies_hz[inside]
    if local_frequencies_hz.size < 8:
        return float("inf")

    local_db = spectrum_db[inside]
    local_power = 10.0 ** ((local_db - np.max(local_db)) / 10.0)
    scaled_frequency = (local_frequencies_hz - position_hz) / radius_hz
    smooth_design = np.column_stack(
        (
            np.ones(local_frequencies_hz.size),
            scaled_frequency,
            scaled_frequency**2,
        )
    )
    smooth_coefficients, *_ = np.linalg.lstsq(
        smooth_design,
        local_power,
        rcond=None,
    )
    smooth_residual = local_power - smooth_design @ smooth_coefficients
    null_residual_sum = float(smooth_residual @ smooth_residual)
    if null_residual_sum <= 0.0:
        return float("inf")

    bin_width_hz = float(frequencies_hz[1] - frequencies_hz[0])
    bin_offset = (local_frequencies_hz - position_hz) / bin_width_hz
    hann_amplitude = (
        0.5 * np.sinc(bin_offset)
        - 0.25 * np.sinc(bin_offset - 1.0)
        - 0.25 * np.sinc(bin_offset + 1.0)
    )
    hann_power = (hann_amplitude / 0.5) ** 2
    line_design = np.column_stack((smooth_design, hann_power))
    line_coefficients, *_ = np.linalg.lstsq(
        line_design,
        local_power,
        rcond=None,
    )
    if line_coefficients[-1] <= 0.0:
        return float("inf")
    line_residual = local_power - line_design @ line_coefficients
    alternative_residual_sum = float(line_residual @ line_residual)
    alternative_residual_sum = max(
        alternative_residual_sum,
        np.finfo(float).tiny,
    )
    count = local_frequencies_hz.size
    return float(
        count * np.log(alternative_residual_sum / null_residual_sum)
        + np.log(count)
        + search_description_length
    )


def _line_contrasts(
    frequencies_hz: np.ndarray,
    spectra_db: np.ndarray,
    *,
    position_hz: float,
    spectral_resolution_hz: float,
) -> np.ndarray:
    centres_db = np.array(
        [np.interp(position_hz, frequencies_hz, spectrum) for spectrum in spectra_db]
    )
    left_db = np.array(
        [
            np.interp(
                position_hz - 2.0 * spectral_resolution_hz,
                frequencies_hz,
                spectrum,
            )
            for spectrum in spectra_db
        ]
    )
    right_db = np.array(
        [
            np.interp(
                position_hz + 2.0 * spectral_resolution_hz,
                frequencies_hz,
                spectrum,
            )
            for spectrum in spectra_db
        ]
    )
    return centres_db - 0.5 * (left_db + right_db)


def _positive_mean_bic(
    values: np.ndarray,
    *,
    search_description_length: float,
) -> float:
    mean = float(np.mean(values))
    if mean <= 0.0:
        return float("inf")
    null_residual_sum = float(np.sum(values**2))
    alternative_residual_sum = float(np.sum((values - mean) ** 2))
    if null_residual_sum <= 0.0:
        return float("inf")
    if alternative_residual_sum == 0.0:
        alternative_residual_sum = np.finfo(float).tiny
    count = values.size
    return float(
        count * np.log(alternative_residual_sum / null_residual_sum)
        + np.log(count)
        + search_description_length
    )


def _fundamental_candidates(
    bin_width_hz: float,
    spectral_resolution_hz: float,
    maximum_frequency_hz: float,
) -> np.ndarray:
    """All identifiable spacings at sub-bin accuracy for their highest harmonic."""
    minimum_hz = 2.0 * spectral_resolution_hz
    maximum_hz = maximum_frequency_hz / 4.0
    if minimum_hz >= maximum_hz:
        raise ValueError("The spectrum cannot resolve four distinct comb harmonics.")

    candidates = []
    candidate_hz = minimum_hz
    while candidate_hz <= maximum_hz:
        candidates.append(candidate_hz)
        highest_harmonic = int(maximum_frequency_hz / candidate_hz)
        candidate_hz += bin_width_hz / (2.0 * highest_harmonic)
    return np.asarray(candidates, dtype=float)


def _comb_score(
    frequencies_hz: np.ndarray,
    spectrum_db: np.ndarray,
    *,
    fundamental_hz: float,
    frequency_range_hz: tuple[float, float],
    search_description_length: float,
) -> tuple[float, float]:
    harmonic_numbers = np.asarray(
        _harmonic_numbers(fundamental_hz, frequency_range_hz),
        dtype=float,
    )
    if harmonic_numbers.size < 4:
        return float("inf"), float("nan")
    centres_hz = harmonic_numbers * fundamental_hz
    centre_db = np.interp(centres_hz, frequencies_hz, spectrum_db)
    halfway_db = 0.5 * (
        np.interp(centres_hz - fundamental_hz / 2.0, frequencies_hz, spectrum_db)
        + np.interp(centres_hz + fundamental_hz / 2.0, frequencies_hz, spectrum_db)
    )
    contrasts_db = centre_db - halfway_db
    mean_contrast_db = float(np.mean(contrasts_db))
    if mean_contrast_db <= 0.0:
        return float("inf"), mean_contrast_db

    null_residual_sum = float(np.sum(contrasts_db**2))
    alternative_residual_sum = float(
        np.sum((contrasts_db - mean_contrast_db) ** 2)
    )
    if null_residual_sum <= 0.0 or alternative_residual_sum <= 0.0:
        return float("inf"), mean_contrast_db

    observation_count = contrasts_db.size
    fit_gain = observation_count * np.log(
        alternative_residual_sum / null_residual_sum
    )
    mean_parameter_cost = np.log(observation_count)

    evidence_bic = fit_gain + mean_parameter_cost + search_description_length
    return float(evidence_bic), mean_contrast_db


def _local_peak_position(
    frequencies_hz: np.ndarray,
    spectrum_db: np.ndarray,
    *,
    target_hz: float,
    spectral_resolution_hz: float,
) -> float:
    from decomb.spectral import refine_peak_frequency

    low, high = np.searchsorted(
        frequencies_hz,
        [target_hz - spectral_resolution_hz, target_hz + spectral_resolution_hz],
    )
    low = max(1, int(low))
    high = min(frequencies_hz.size - 1, int(high))
    if high <= low:
        raise ValueError(f"Harmonic at {target_hz:g} Hz lies outside the spectrum.")
    index = low + int(np.argmax(spectrum_db[low:high]))
    return refine_peak_frequency(frequencies_hz, spectrum_db, index)


def _validated_spectrum(
    frequencies_hz: Sequence[float],
    spectrum_db: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    frequencies = np.asarray(frequencies_hz, dtype=float)
    spectrum = np.asarray(spectrum_db, dtype=float)
    if frequencies.ndim != 1 or frequencies.size < 5:
        raise ValueError("Comb estimation requires a one-dimensional frequency grid.")
    if frequencies.shape != spectrum.shape:
        raise ValueError("Frequency and spectrum arrays must have equal shapes.")
    if not np.all(np.isfinite(frequencies)) or np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("Comb frequencies must be finite and strictly increasing.")
    if not np.all(np.isfinite(spectrum)):
        raise ValueError("The comb spectrum must contain only finite values.")
    steps = np.diff(frequencies)
    if not np.allclose(steps, steps[0], rtol=1e-8, atol=np.finfo(float).eps):
        raise ValueError("Comb estimation requires an evenly spaced frequency grid.")
    return frequencies, spectrum


def _harmonic_numbers(
    fundamental_hz: float,
    frequency_range_hz: tuple[float, float],
) -> tuple[int, ...]:
    low_hz, high_hz = frequency_range_hz
    first = max(1, int(np.ceil(low_hz / fundamental_hz)))
    last = int(np.floor(high_hz / fundamental_hz))
    return tuple(range(first, last + 1))


def _analysis_range(
    frequencies_hz: np.ndarray,
    frequency_range_hz: tuple[float, float],
) -> tuple[float, float]:
    values = np.asarray(frequency_range_hz, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("frequency_range_hz must contain two finite values.")
    low_hz, requested_high_hz = (float(value) for value in values)
    if not 0.0 <= low_hz < requested_high_hz <= MAXIMUM_FREQUENCY_HZ:
        raise ValueError("frequency_range_hz must increase inside [0, 100] Hz.")
    high_hz = min(requested_high_hz, float(frequencies_hz[-2]))
    if high_hz <= low_hz:
        raise ValueError("The spectrum does not cover the configured frequency range.")
    return low_hz, high_hz


def _positive(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _validate_harmonic_positions(
    harmonic_numbers: tuple[int, ...],
    positions_hz: tuple[float, ...],
) -> None:
    if len(harmonic_numbers) != len(positions_hz) or not harmonic_numbers:
        raise ValueError("Every non-empty harmonic list requires one measured position.")
    if harmonic_numbers != tuple(sorted(set(harmonic_numbers))):
        raise ValueError("Harmonics must be sorted and unique.")
    if harmonic_numbers[0] < 1:
        raise ValueError("Harmonics must be positive.")
    if not np.all(np.isfinite(positions_hz)) or min(positions_hz) <= 0.0:
        raise ValueError("Harmonic positions must be finite and positive.")
