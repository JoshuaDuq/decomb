"""Participant-specific estimation of an arithmetic line comb.

The fitted harmonics establish the fundamental.  Supported harmonics are then localized
against that fitted grid and are the only frequencies a correction may target.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from decomb.spectral import refine_peak_frequency


@dataclass(frozen=True)
class CombEstimate:
    """One spectrum's fitted comb and independently localized supported members."""

    fundamental_hz: float
    fitted_harmonics: tuple[int, ...]
    fitted_positions_hz: tuple[float, ...]
    supported_harmonics: tuple[int, ...]
    supported_positions_hz: tuple[float, ...]
    residual_rms_hz: float
    max_abs_residual_hz: float
    fundamental_jackknife_se_hz: float

    def __post_init__(self) -> None:
        _validate_harmonic_positions(
            self.fitted_harmonics,
            self.fitted_positions_hz,
            name="fitted",
        )
        _validate_harmonic_positions(
            self.supported_harmonics,
            self.supported_positions_hz,
            name="supported",
        )
        scalars = (
            self.fundamental_hz,
            self.residual_rms_hz,
            self.max_abs_residual_hz,
            self.fundamental_jackknife_se_hz,
        )
        if not np.all(np.isfinite(scalars)):
            raise ValueError("Comb estimates must contain finite values.")
        if self.fundamental_hz <= 0.0:
            raise ValueError("The fitted fundamental must be positive.")
        if min(scalars[1:]) < 0.0:
            raise ValueError("Comb residuals and uncertainty must be non-negative.")

    @property
    def n_fitted_harmonics(self) -> int:
        return len(self.fitted_harmonics)


@dataclass(frozen=True)
class AdaptiveCombModel:
    """Whole-recording authorization and optional window-localized positions."""

    whole_estimate: CombEstimate
    window_evidence: tuple[HarmonicEvidence, ...]

    def __post_init__(self) -> None:
        if not self.window_evidence:
            raise ValueError("At least one estimation window is required.")


@dataclass(frozen=True)
class HarmonicEvidence:
    """Harmonics directly visible in one window; an empty window is explicit."""

    harmonics: tuple[int, ...]
    positions_hz: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.harmonics) != len(self.positions_hz):
            raise ValueError("Each window harmonic requires one measured position.")
        if self.harmonics != tuple(sorted(set(self.harmonics))):
            raise ValueError("Window harmonics must be sorted and unique.")
        if self.harmonics and self.harmonics[0] < 1:
            raise ValueError("Window harmonics must be positive.")
        if not np.all(np.isfinite(self.positions_hz)):
            raise ValueError("Window harmonic positions must be finite.")
        if self.positions_hz and min(self.positions_hz) <= 0.0:
            raise ValueError("Window harmonic positions must be positive.")


def estimate_comb(
    frequencies_hz: Sequence[float],
    spectrum_db: Sequence[float],
    prominence_db: Sequence[float],
    *,
    nominal_fundamental_hz: float,
    fit_harmonic_range: tuple[int, int],
    supported_harmonic_range: tuple[int, int],
    search_hz: float,
    min_prominence_db: float,
    min_harmonics: int,
    max_harmonic_residual_hz: float,
    max_residual_rms_hz: float,
) -> CombEstimate:
    """Fit a comb and localize only prominent members consistent with its grid."""
    frequencies = np.asarray(frequencies_hz, dtype=float)
    spectrum = np.asarray(spectrum_db, dtype=float)
    prominence = np.asarray(prominence_db, dtype=float)
    _validate_spectrum(frequencies, spectrum, prominence)
    _validate_estimation_parameters(
        nominal_fundamental_hz=nominal_fundamental_hz,
        fit_harmonic_range=fit_harmonic_range,
        supported_harmonic_range=supported_harmonic_range,
        search_hz=search_hz,
        min_prominence_db=min_prominence_db,
        min_harmonics=min_harmonics,
        max_harmonic_residual_hz=max_harmonic_residual_hz,
        max_residual_rms_hz=max_residual_rms_hz,
    )

    candidate_harmonics, candidate_positions, candidate_weights = _localize_range(
        frequencies,
        spectrum,
        prominence,
        fundamental_hz=nominal_fundamental_hz,
        harmonic_range=fit_harmonic_range,
        search_hz=search_hz,
        min_prominence_db=min_prominence_db,
    )
    if len(candidate_harmonics) < min_harmonics:
        raise ValueError(
            f"Only {len(candidate_harmonics)} comb harmonics exceeded "
            f"{min_prominence_db:g} dB; at least {min_harmonics} are required."
        )

    fitted_harmonics, fitted_positions, fitted_weights, fundamental_hz = _fit_consistent_harmonics(
        candidate_harmonics,
        candidate_positions,
        candidate_weights,
        min_harmonics=min_harmonics,
        max_harmonic_residual_hz=max_harmonic_residual_hz,
    )
    residuals_hz = fitted_positions - fitted_harmonics * fundamental_hz
    residual_rms_hz = float(np.sqrt(np.mean(residuals_hz**2)))
    if residual_rms_hz > max_residual_rms_hz:
        raise ValueError(
            f"Fitted harmonics scatter {residual_rms_hz:.3f} Hz RMS about their grid, "
            f"above the {max_residual_rms_hz:.3f} Hz bound."
        )

    supported_harmonics, supported_positions, _ = _localize_range(
        frequencies,
        spectrum,
        prominence,
        fundamental_hz=fundamental_hz,
        harmonic_range=supported_harmonic_range,
        search_hz=max_harmonic_residual_hz,
        min_prominence_db=min_prominence_db,
    )
    if supported_harmonics.size == 0:
        raise ValueError("The fitted comb has no supported harmonic eligible for removal.")

    return CombEstimate(
        fundamental_hz=fundamental_hz,
        fitted_harmonics=tuple(int(value) for value in fitted_harmonics),
        fitted_positions_hz=tuple(float(value) for value in fitted_positions),
        supported_harmonics=tuple(int(value) for value in supported_harmonics),
        supported_positions_hz=tuple(float(value) for value in supported_positions),
        residual_rms_hz=residual_rms_hz,
        max_abs_residual_hz=float(np.max(np.abs(residuals_hz))),
        fundamental_jackknife_se_hz=_fundamental_jackknife_se(
            fitted_harmonics,
            fitted_positions,
            fitted_weights,
        ),
    )


def localize_supported_harmonics(
    frequencies_hz: Sequence[float],
    spectrum_db: Sequence[float],
    prominence_db: Sequence[float],
    *,
    supported_harmonics: Sequence[int],
    fundamental_hz: float,
    search_hz: float,
    min_prominence_db: float,
) -> HarmonicEvidence:
    """Localize visible whole-recording targets without authorizing a new grid."""
    frequencies = np.asarray(frequencies_hz, dtype=float)
    spectrum = np.asarray(spectrum_db, dtype=float)
    prominence = np.asarray(prominence_db, dtype=float)
    _validate_spectrum(frequencies, spectrum, prominence)
    harmonic_numbers = tuple(int(value) for value in supported_harmonics)
    if harmonic_numbers != tuple(sorted(set(harmonic_numbers))):
        raise ValueError("Supported harmonics must be sorted and unique.")
    if harmonic_numbers and harmonic_numbers[0] < 1:
        raise ValueError("Supported harmonics must be positive.")
    if not np.isfinite(fundamental_hz) or fundamental_hz <= 0.0:
        raise ValueError("fundamental_hz must be finite and positive.")
    if not np.isfinite(search_hz) or search_hz <= 0.0:
        raise ValueError("search_hz must be finite and positive.")
    if not np.isfinite(min_prominence_db) or min_prominence_db <= 0.0:
        raise ValueError("min_prominence_db must be finite and positive.")

    found_harmonics = []
    found_positions = []
    for harmonic in harmonic_numbers:
        target_hz = harmonic * fundamental_hz
        peak = _peak_near(
            frequencies,
            spectrum,
            prominence,
            target_hz=target_hz,
            search_hz=search_hz,
        )
        if peak is None:
            continue
        position_hz, strength_db = peak
        if strength_db < min_prominence_db:
            continue
        if abs(position_hz - target_hz) > search_hz:
            continue
        found_harmonics.append(harmonic)
        found_positions.append(position_hz)
    return HarmonicEvidence(
        tuple(found_harmonics),
        tuple(float(value) for value in found_positions),
    )


def _validate_harmonic_positions(
    harmonic_numbers: tuple[int, ...],
    positions_hz: tuple[float, ...],
    *,
    name: str,
) -> None:
    if len(harmonic_numbers) != len(positions_hz):
        raise ValueError(f"Each {name} harmonic requires one measured position.")
    if not harmonic_numbers:
        raise ValueError(f"At least one {name} harmonic is required.")
    if harmonic_numbers != tuple(sorted(set(harmonic_numbers))):
        raise ValueError(f"{name.capitalize()} harmonics must be sorted and unique.")
    if harmonic_numbers[0] < 1:
        raise ValueError(f"{name.capitalize()} harmonics must be positive.")
    if not np.all(np.isfinite(positions_hz)) or min(positions_hz) <= 0.0:
        raise ValueError(f"{name.capitalize()} positions must be finite and positive.")


def _validate_spectrum(
    frequencies_hz: np.ndarray,
    spectrum_db: np.ndarray,
    prominence_db: np.ndarray,
) -> None:
    if frequencies_hz.ndim != 1 or frequencies_hz.size < 3:
        raise ValueError("Comb estimation requires a one-dimensional frequency grid.")
    if not frequencies_hz.shape == spectrum_db.shape == prominence_db.shape:
        raise ValueError("Frequency, spectrum, and prominence arrays must have equal shapes.")
    if not np.all(np.isfinite(frequencies_hz)) or np.any(np.diff(frequencies_hz) <= 0.0):
        raise ValueError("Comb frequencies must be finite and strictly increasing.")


def _validate_estimation_parameters(**parameters) -> None:
    nominal = parameters["nominal_fundamental_hz"]
    search_hz = parameters["search_hz"]
    if not np.isfinite(nominal) or nominal <= 0.0:
        raise ValueError("nominal_fundamental_hz must be finite and positive.")
    if not np.isfinite(search_hz) or not 0.0 < search_hz < nominal / 2.0:
        raise ValueError("search_hz must lie below half the nominal fundamental.")
    for name in ("fit_harmonic_range", "supported_harmonic_range"):
        first, last = parameters[name]
        if first < 1 or last < first:
            raise ValueError(f"{name} must contain increasing positive integers.")
    if parameters["min_harmonics"] < 3:
        raise ValueError("min_harmonics must be at least three.")
    for name in (
        "min_prominence_db",
        "max_harmonic_residual_hz",
        "max_residual_rms_hz",
    ):
        value = parameters[name]
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")


def _localize_range(
    frequencies_hz: np.ndarray,
    spectrum_db: np.ndarray,
    prominence_db: np.ndarray,
    *,
    fundamental_hz: float,
    harmonic_range: tuple[int, int],
    search_hz: float,
    min_prominence_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    harmonics = []
    positions = []
    weights = []
    first, last = harmonic_range
    for harmonic in range(first, last + 1):
        target_hz = harmonic * fundamental_hz
        peak = _peak_near(
            frequencies_hz,
            spectrum_db,
            prominence_db,
            target_hz=target_hz,
            search_hz=search_hz,
        )
        if peak is None:
            continue
        position_hz, strength_db = peak
        if strength_db < min_prominence_db:
            continue
        if abs(position_hz - target_hz) > search_hz:
            continue
        harmonics.append(harmonic)
        positions.append(position_hz)
        weights.append(strength_db)
    return (
        np.asarray(harmonics, dtype=float),
        np.asarray(positions, dtype=float),
        np.asarray(weights, dtype=float),
    )


def _peak_near(
    frequencies_hz: np.ndarray,
    spectrum_db: np.ndarray,
    prominence_db: np.ndarray,
    *,
    target_hz: float,
    search_hz: float,
) -> tuple[float, float] | None:
    low, high = np.searchsorted(
        frequencies_hz,
        [target_hz - search_hz, target_hz + search_hz],
    )
    if high <= low:
        return None
    window = prominence_db[low:high]
    if not np.any(np.isfinite(window)):
        return None
    index = low + int(np.nanargmax(window))
    if not 0 < index < frequencies_hz.size - 1:
        return None
    return (
        refine_peak_frequency(frequencies_hz, spectrum_db, index),
        float(prominence_db[index]),
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, cumulative[-1] / 2.0, side="left"))
    return float(values[order[index]])


def _fit_consistent_harmonics(
    harmonics: np.ndarray,
    positions_hz: np.ndarray,
    weights: np.ndarray,
    *,
    min_harmonics: int,
    max_harmonic_residual_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    seed_hz = _weighted_median(positions_hz / harmonics, weights)
    retained = np.abs(positions_hz - harmonics * seed_hz) <= max_harmonic_residual_hz
    visited: set[bytes] = set()
    while True:
        membership = retained.tobytes()
        if membership in visited:
            raise RuntimeError("Robust comb membership entered a cycle.")
        visited.add(membership)
        if np.count_nonzero(retained) < min_harmonics:
            raise ValueError(
                f"Only {np.count_nonzero(retained)} mutually consistent comb harmonics remain."
            )
        selected_harmonics = harmonics[retained]
        selected_positions = positions_hz[retained]
        selected_weights = weights[retained]
        fundamental_hz = float(
            np.sum(selected_weights * selected_harmonics * selected_positions)
            / np.sum(selected_weights * selected_harmonics**2)
        )
        updated = np.abs(positions_hz - harmonics * fundamental_hz) <= max_harmonic_residual_hz
        if np.array_equal(updated, retained):
            return (
                selected_harmonics,
                selected_positions,
                selected_weights,
                fundamental_hz,
            )
        retained = updated


def _fundamental_jackknife_se(
    harmonics: np.ndarray,
    positions_hz: np.ndarray,
    weights: np.ndarray,
) -> float:
    count = harmonics.size
    estimates = np.empty(count, dtype=float)
    for omitted in range(count):
        retained = np.arange(count) != omitted
        estimates[omitted] = np.sum(
            weights[retained] * harmonics[retained] * positions_hz[retained]
        ) / np.sum(weights[retained] * harmonics[retained] ** 2)
    centre_hz = float(np.mean(estimates))
    return float(np.sqrt((count - 1) / count * np.sum((estimates - centre_hz) ** 2)))
