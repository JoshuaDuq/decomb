"""Paired neural-like injections through multitaper recovery and residual FIR."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from decomb import (
    injection,
    notch,
    recordings,
    recovery,
    recovery_benchmark,
    recovery_evaluation,
    validation,
)

POSITIONS = ("exact", "near", "between")
COMPONENT_TO_BACKGROUND_DB = -10.0
INJECTION_CHANNEL_NAME = "Cz"
RANDOM_SEED = 20260814
RESIDUAL_PROTOCOLS = ("adaptive", "frozen", "notch_only", "spatial")
#: Which residual FIR stage runs after recovery. "lines" is main's published
#: pipeline -- converged line rounds only. "joint" adds this branch's target-local
#: rounds. It is orthogonal to whether recovery runs, so the two are separate.
RESIDUAL_STAGES = ("joint", "lines")


@dataclass(frozen=True)
class FrequencyPlacement:
    """A neural-like component positioned relative to two authorized lines."""

    band_name: str
    band_low_hz: float
    band_high_hz: float
    position: str
    authorized_line_hz: float
    neighbouring_line_hz: float
    centre_frequency_hz: float

    def __post_init__(self) -> None:
        if not self.band_name.strip():
            raise ValueError("band_name must not be empty")
        if self.position not in POSITIONS:
            raise ValueError(f"position must be one of {POSITIONS}")
        frequencies = (
            self.band_low_hz,
            self.band_high_hz,
            self.authorized_line_hz,
            self.neighbouring_line_hz,
            self.centre_frequency_hz,
        )
        if not np.isfinite(frequencies).all():
            raise ValueError("frequency placement values must be finite")
        if not (
            0.0 < self.band_low_hz
            < self.authorized_line_hz
            < self.neighbouring_line_hz
            < self.band_high_hz
        ):
            raise ValueError("authorized line pair must lie strictly inside its band")
        if not self.band_low_hz < self.centre_frequency_hz < self.band_high_hz:
            raise ValueError("centre frequency must lie strictly inside its band")


@dataclass(frozen=True)
class MultitaperCleaningResult:
    """Both stages and artifact-null evidence for one adaptive cleaning."""

    recovered: object
    cleaned: object
    residual_filter_plans: tuple[notch.HarmonicNotchPlan, ...]
    residual_round_count: int
    terminal_residual_detector_null: bool
    targeted_local_background_excess_null: bool
    recovery_runtime_s: float
    residual_runtime_s: float
    residual_stage: str = "joint"

    @property
    def artifact_gate_passed(self) -> bool:
        """Both terminal nulls, except that `lines` does not contract for the local one.

        The `lines` stage is main's pipeline, which never runs target-local rounds, so
        requiring that null of it would fail an arm for not doing something it never
        claimed to do. Both booleans are written to the output either way.
        """
        if self.residual_stage == "lines":
            return self.terminal_residual_detector_null
        return (
            self.terminal_residual_detector_null
            and self.targeted_local_background_excess_null
        )


def frequency_placements(
    targets: recovery_benchmark.RecoveryTargets,
    band: tuple[str, float, float],
    *,
    frequency_bin_width_hz: float,
) -> tuple[FrequencyPlacement, ...]:
    """Choose a central resolved line pair and exact, near, and midpoint probes."""
    band_name, band_low_hz, band_high_hz = band
    bin_width_hz = _positive(frequency_bin_width_hz, "frequency_bin_width_hz")
    frequencies_hz = tuple(
        frequency
        for frequency in targets.all_frequencies_hz
        if band_low_hz < frequency < band_high_hz
    )
    pairs = tuple(
        (lower_hz, upper_hz)
        for lower_hz, upper_hz in zip(
            frequencies_hz[:-1],
            frequencies_hz[1:],
            strict=True,
        )
        if upper_hz - lower_hz > 4.0 * bin_width_hz
    )
    if not pairs:
        raise ValueError(
            f"band {band_name!r} requires two resolved authorized lines"
        )
    band_centre_hz = (float(band_low_hz) + float(band_high_hz)) / 2.0
    authorized_line_hz, neighbouring_line_hz = min(
        pairs,
        key=lambda pair: abs((pair[0] + pair[1]) / 2.0 - band_centre_hz),
    )
    centres_hz = (
        authorized_line_hz,
        authorized_line_hz + 2.0 * bin_width_hz,
        (authorized_line_hz + neighbouring_line_hz) / 2.0,
    )
    return tuple(
        FrequencyPlacement(
            str(band_name),
            float(band_low_hz),
            float(band_high_hz),
            position,
            authorized_line_hz,
            neighbouring_line_hz,
            centre_hz,
        )
        for position, centre_hz in zip(POSITIONS, centres_hz, strict=True)
    )


def injection_target(
    placement: FrequencyPlacement,
    kind: str,
    *,
    frequency_bin_width_hz: float,
    component_to_background_db: float,
) -> injection.FactorialInjectionTarget:
    """Build one weak neural-like oscillator centred on its placement."""
    bin_width_hz = _positive(frequency_bin_width_hz, "frequency_bin_width_hz")
    if kind == "drifting":
        drift_hz = 4.0 * bin_width_hz
        frequency_hz = placement.centre_frequency_hz - drift_hz / 2.0
        occupancy = 1.0
        phase_modulation_hz = 0.0
        phase_deviation_rad = 0.0
    elif kind == "intermittent":
        drift_hz = 0.0
        frequency_hz = placement.centre_frequency_hz
        occupancy = 0.5
        phase_modulation_hz = 0.0
        phase_deviation_rad = 0.0
    elif kind == "phase_modulated":
        drift_hz = 0.0
        frequency_hz = placement.centre_frequency_hz
        occupancy = 1.0
        phase_modulation_hz = bin_width_hz
        phase_deviation_rad = 1.0
    elif kind == "stationary":
        drift_hz = 0.0
        frequency_hz = placement.centre_frequency_hz
        occupancy = 1.0
        phase_modulation_hz = 0.0
        phase_deviation_rad = 0.0
    else:
        raise ValueError(f"kind must be one of {injection.KINDS}")
    if not (
        placement.band_low_hz < frequency_hz
        and frequency_hz + drift_hz < placement.band_high_hz
    ):
        raise ValueError("injection trajectory must lie strictly inside its band")
    return injection.FactorialInjectionTarget(
        kind=kind,
        frequency_hz=frequency_hz,
        component_to_background_db=component_to_background_db,
        drift_hz=drift_hz,
        occupancy=occupancy,
        phase_rad=np.pi / 7.0,
        phase_modulation_hz=phase_modulation_hz,
        phase_deviation_rad=phase_deviation_rad,
    )


def completed_trial_keys(
    rows: Iterable[dict[str, object]],
) -> set[tuple[str, str, str, str]]:
    """Validate a metrics checkpoint and return its complete paired trials."""
    grouped: dict[tuple[str, str, str, str], set[str]] = {}
    for row in rows:
        key = tuple(
            str(row[field])
            for field in ("recording", "band", "kind", "position")
        )
        grouped.setdefault(key, set()).add(str(row["stage"]))
    for key, stages in grouped.items():
        if stages != {"recovery", "final"}:
            raise ValueError(f"trial {key} requires recovery and final stages")
    return set(grouped)


def validation_design(
    targets: recovery_benchmark.RecoveryTargets,
    bands: Sequence[tuple[str, float, float]],
    *,
    frequency_bin_width_hz: float,
    component_to_background_db: float,
) -> tuple[tuple[FrequencyPlacement, injection.FactorialInjectionTarget], ...]:
    """Full factorial neural-like validation design for one recording."""
    return tuple(
        (placement, injection_target(
            placement,
            kind,
            frequency_bin_width_hz=frequency_bin_width_hz,
            component_to_background_db=component_to_background_db,
        ))
        for band in bands
        for placement in frequency_placements(
            targets,
            band,
            frequency_bin_width_hz=frequency_bin_width_hz,
        )
        for kind in injection.KINDS
    )


def has_resolved_line_pair(
    targets: recovery_benchmark.RecoveryTargets,
    band: tuple[str, float, float],
    *,
    frequency_bin_width_hz: float,
) -> bool:
    """Whether a band can support exact, near, and between-line probes."""
    try:
        frequency_placements(
            targets,
            band,
            frequency_bin_width_hz=frequency_bin_width_hz,
        )
    except ValueError as error:
        if "requires two resolved authorized lines" not in str(error):
            raise
        return False
    return True


def _converged_residual_result(
    recovered,
    recovery_runtime_s: float,
    targets: recovery_benchmark.RecoveryTargets,
    notch_settings: notch.HarmonicNotchSettings,
    *,
    n_jobs: int = -1,
    residual_stage: str = "joint",
) -> MultitaperCleaningResult:
    """Converge the residual FIR on an already-prepared recording.

    Shared by every arm so that the arms differ only in what, if anything, was
    subtracted before this point, and in which residual stage runs after it.
    """
    if residual_stage not in RESIDUAL_STAGES:
        raise ValueError(f"residual stage must be one of {RESIDUAL_STAGES}")
    started = time.perf_counter()
    if residual_stage == "lines":
        result = notch.clean_until_no_supported_lines(
            recovered,
            notch_settings,
            n_jobs=n_jobs,
        )
        cleaned = result.cleaned
        filter_plans = tuple(round_.filter_plan for round_ in result.rounds)
        round_count = len(result.rounds)
        terminal_null = (
            not result.residual_model.channels
            and result.residual_scanner_harmonics is None
        )
    else:
        residual = recovery_benchmark.clean_joint_residuals(
            recovered,
            targets,
            notch_settings,
            n_jobs=n_jobs,
        )
        cleaned = residual.cleaned
        filter_plans = residual.filter_plans
        round_count = residual.round_count
        terminal_null = all(
            not result.residual_model.channels
            and result.residual_scanner_harmonics is None
            for result in residual.harmonic_results
        )
    residual_runtime_s = time.perf_counter() - started
    # Measured under both stages; only `joint` contracts for it.
    local_null = recovery_benchmark.targeted_local_background_is_null(
        cleaned,
        targets,
        window_s=notch_settings.estimation_window_s,
        familywise_error_rate=notch_settings.familywise_error_rate,
        frequency_range_hz=_available_frequency_range(cleaned, notch_settings),
    )
    return MultitaperCleaningResult(
        recovered=recovered,
        cleaned=cleaned,
        residual_filter_plans=filter_plans,
        residual_round_count=round_count,
        terminal_residual_detector_null=terminal_null,
        targeted_local_background_excess_null=local_null,
        recovery_runtime_s=recovery_runtime_s,
        residual_runtime_s=residual_runtime_s,
        residual_stage=residual_stage,
    )


def _frozen_residual_result(
    recovered,
    recovery_runtime_s: float,
    targets: recovery_benchmark.RecoveryTargets,
    notch_settings: notch.HarmonicNotchSettings,
    residual_filter_plans: tuple[notch.HarmonicNotchPlan, ...],
    *,
    n_jobs: int = -1,
    residual_stage: str = "joint",
) -> MultitaperCleaningResult:
    """Replay one immutable residual FIR sequence on an already-prepared recording."""
    started = time.perf_counter()
    cleaned = recovered
    for plan in residual_filter_plans:
        cleaned = notch.apply_harmonic_notches(
            cleaned,
            plan,
            n_jobs=n_jobs,
        )
    terminal_evidence = notch.fit_harmonic_round(cleaned, notch_settings)
    terminal_null = (
        not terminal_evidence.model.channels
        and terminal_evidence.scanner_harmonics is None
    )
    local_null = recovery_benchmark.targeted_local_background_is_null(
        cleaned,
        targets,
        window_s=notch_settings.estimation_window_s,
        familywise_error_rate=notch_settings.familywise_error_rate,
        frequency_range_hz=_available_frequency_range(cleaned, notch_settings),
    )
    return MultitaperCleaningResult(
        recovered=recovered,
        cleaned=cleaned,
        residual_filter_plans=residual_filter_plans,
        residual_round_count=len(residual_filter_plans),
        terminal_residual_detector_null=terminal_null,
        targeted_local_background_excess_null=local_null,
        recovery_runtime_s=recovery_runtime_s,
        residual_runtime_s=time.perf_counter() - started,
        residual_stage=residual_stage,
    )


def clean_with_multitaper(
    raw,
    targets: recovery_benchmark.RecoveryTargets,
    notch_settings: notch.HarmonicNotchSettings,
    *,
    recovery_window_s: float,
    n_jobs: int = -1,
    residual_stage: str = "joint",
) -> MultitaperCleaningResult:
    """Run multitaper recovery followed by the converged residual FIR."""
    started = time.perf_counter()
    recovered = recovery_benchmark.recover_with_multitaper(
        raw,
        targets,
        window_s=recovery_window_s,
        n_jobs=n_jobs,
    )
    return _converged_residual_result(
        recovered,
        time.perf_counter() - started,
        targets,
        notch_settings,
        n_jobs=n_jobs,
        residual_stage=residual_stage,
    )


def clean_with_frozen_residuals(
    raw,
    targets: recovery_benchmark.RecoveryTargets,
    notch_settings: notch.HarmonicNotchSettings,
    residual_filter_plans: tuple[notch.HarmonicNotchPlan, ...],
    *,
    recovery_window_s: float,
    n_jobs: int = -1,
    residual_stage: str = "joint",
) -> MultitaperCleaningResult:
    """Run multitaper recovery, then replay one immutable residual FIR sequence."""
    started = time.perf_counter()
    recovered = recovery_benchmark.recover_with_multitaper(
        raw,
        targets,
        window_s=recovery_window_s,
        n_jobs=n_jobs,
    )
    return _frozen_residual_result(
        recovered,
        time.perf_counter() - started,
        targets,
        notch_settings,
        residual_filter_plans,
        n_jobs=n_jobs,
        residual_stage=residual_stage,
    )


def clean_without_recovery(
    raw,
    targets: recovery_benchmark.RecoveryTargets,
    notch_settings: notch.HarmonicNotchSettings,
    *,
    n_jobs: int = -1,
    residual_stage: str = "joint",
) -> MultitaperCleaningResult:
    """Notch only: converge the residual FIR with nothing subtracted beforehand.

    This is the published pipeline's behaviour, expressed inside the paired-injection
    harness so that notching and pre-notch subtraction can be measured on identical
    injections. `recovered` is the input recording, unmodified.
    """
    return _converged_residual_result(
        raw,
        0.0,
        targets,
        notch_settings,
        n_jobs=n_jobs,
        residual_stage=residual_stage,
    )


def clean_frozen_residuals_without_recovery(
    raw,
    targets: recovery_benchmark.RecoveryTargets,
    notch_settings: notch.HarmonicNotchSettings,
    residual_filter_plans: tuple[notch.HarmonicNotchPlan, ...],
    *,
    n_jobs: int = -1,
    residual_stage: str = "joint",
) -> MultitaperCleaningResult:
    """Notch only, replaying the background's immutable residual FIR sequence."""
    return _frozen_residual_result(
        raw,
        0.0,
        targets,
        notch_settings,
        residual_filter_plans,
        n_jobs=n_jobs,
        residual_stage=residual_stage,
    )


def clean_background_with_spatial_subspace(
    raw,
    targets: recovery_benchmark.RecoveryTargets,
    notch_settings: notch.HarmonicNotchSettings,
    spatial_model: recovery.SpatialLineSubspaceModel,
    *,
    recovery_window_s: float,
    n_jobs: int = -1,
) -> MultitaperCleaningResult:
    """Apply frozen spatial recovery and learn residual FIR from background."""
    started = time.perf_counter()
    recovered = recovery_benchmark.recover_with_spatial_line_subspace(
        raw,
        spatial_model,
        window_s=recovery_window_s,
        n_jobs=n_jobs,
    )
    recovery_runtime_s = time.perf_counter() - started

    started = time.perf_counter()
    residual = recovery_benchmark.clean_joint_residuals(
        recovered,
        targets,
        notch_settings,
        n_jobs=n_jobs,
    )
    residual_runtime_s = time.perf_counter() - started
    terminal_null = all(
        not result.residual_model.channels
        and result.residual_scanner_harmonics is None
        for result in residual.harmonic_results
    )
    local_null = recovery_benchmark.targeted_local_background_is_null(
        residual.cleaned,
        targets,
        window_s=notch_settings.estimation_window_s,
        familywise_error_rate=notch_settings.familywise_error_rate,
        frequency_range_hz=_available_frequency_range(
            residual.cleaned,
            notch_settings,
        ),
    )
    return MultitaperCleaningResult(
        recovered=recovered,
        cleaned=residual.cleaned,
        residual_filter_plans=residual.filter_plans,
        residual_round_count=residual.round_count,
        terminal_residual_detector_null=terminal_null,
        targeted_local_background_excess_null=local_null,
        recovery_runtime_s=recovery_runtime_s,
        residual_runtime_s=residual_runtime_s,
    )


def clean_with_frozen_spatial_residuals(
    raw,
    targets: recovery_benchmark.RecoveryTargets,
    notch_settings: notch.HarmonicNotchSettings,
    spatial_model: recovery.SpatialLineSubspaceModel,
    residual_filter_plans: tuple[notch.HarmonicNotchPlan, ...],
    *,
    recovery_window_s: float,
    n_jobs: int = -1,
) -> MultitaperCleaningResult:
    """Apply frozen spatial recovery and frozen background residual FIR."""
    started = time.perf_counter()
    recovered = recovery_benchmark.recover_with_spatial_line_subspace(
        raw,
        spatial_model,
        window_s=recovery_window_s,
        n_jobs=n_jobs,
    )
    recovery_runtime_s = time.perf_counter() - started

    started = time.perf_counter()
    cleaned = recovered
    for plan in residual_filter_plans:
        cleaned = notch.apply_harmonic_notches(
            cleaned,
            plan,
            n_jobs=n_jobs,
        )
    terminal_evidence = notch.fit_harmonic_round(cleaned, notch_settings)
    terminal_null = (
        not terminal_evidence.model.channels
        and terminal_evidence.scanner_harmonics is None
    )
    local_null = recovery_benchmark.targeted_local_background_is_null(
        cleaned,
        targets,
        window_s=notch_settings.estimation_window_s,
        familywise_error_rate=notch_settings.familywise_error_rate,
        frequency_range_hz=_available_frequency_range(cleaned, notch_settings),
    )
    return MultitaperCleaningResult(
        recovered=recovered,
        cleaned=cleaned,
        residual_filter_plans=residual_filter_plans,
        residual_round_count=len(residual_filter_plans),
        terminal_residual_detector_null=terminal_null,
        targeted_local_background_excess_null=local_null,
        recovery_runtime_s=recovery_runtime_s,
        residual_runtime_s=time.perf_counter() - started,
    )


def paired_trial_rows(
    background,
    background_cleaning: MultitaperCleaningResult,
    targets: recovery_benchmark.RecoveryTargets,
    placement: FrequencyPlacement,
    target: injection.FactorialInjectionTarget,
    rng: np.random.Generator,
    *,
    recording: str,
    participant: str,
    channel_name: str,
    notch_settings: notch.HarmonicNotchSettings,
    recovery_window_s: float,
    n_jobs: int = -1,
    residual_stage: str = "joint",
) -> tuple[dict[str, object], dict[str, object]]:
    """Measure one pair with independently adaptive residual FIR geometry."""
    spec, realization, injected = _realize_paired_injection(
        background,
        channel_name,
        target,
        rng,
    )
    injected_cleaning = clean_with_multitaper(
        injected,
        targets,
        notch_settings,
        recovery_window_s=recovery_window_s,
    )
    return _paired_measurement_rows(
        background,
        injected,
        background_cleaning,
        injected_cleaning,
        placement,
        target,
        spec,
        realization,
        candidate="multitaper",
        spatial_rank="",
        residual_protocol="adaptive",
        recording=recording,
        participant=participant,
        channel_name=channel_name,
        recovery_window_s=recovery_window_s,
    )


def frozen_paired_trial_rows(
    background,
    background_cleaning: MultitaperCleaningResult,
    targets: recovery_benchmark.RecoveryTargets,
    placement: FrequencyPlacement,
    target: injection.FactorialInjectionTarget,
    rng: np.random.Generator,
    *,
    recording: str,
    participant: str,
    channel_name: str,
    notch_settings: notch.HarmonicNotchSettings,
    recovery_window_s: float,
    n_jobs: int = -1,
    residual_stage: str = "joint",
) -> tuple[dict[str, object], dict[str, object]]:
    """Measure one pair using the background's immutable residual FIR sequence."""
    spec, realization, injected = _realize_paired_injection(
        background,
        channel_name,
        target,
        rng,
    )
    injected_cleaning = clean_with_frozen_residuals(
        injected,
        targets,
        notch_settings,
        background_cleaning.residual_filter_plans,
        recovery_window_s=recovery_window_s,
        n_jobs=n_jobs,
        residual_stage=residual_stage,
    )
    return _paired_measurement_rows(
        background,
        injected,
        background_cleaning,
        injected_cleaning,
        placement,
        target,
        spec,
        realization,
        candidate="multitaper",
        spatial_rank="",
        residual_protocol="frozen",
        recording=recording,
        participant=participant,
        channel_name=channel_name,
        recovery_window_s=recovery_window_s,
    )


def notch_only_paired_trial_rows(
    background,
    background_cleaning: MultitaperCleaningResult,
    targets: recovery_benchmark.RecoveryTargets,
    placement: FrequencyPlacement,
    target: injection.FactorialInjectionTarget,
    rng: np.random.Generator,
    *,
    recording: str,
    participant: str,
    channel_name: str,
    notch_settings: notch.HarmonicNotchSettings,
    recovery_window_s: float,
    n_jobs: int = -1,
    residual_stage: str = "joint",
) -> tuple[dict[str, object], dict[str, object]]:
    """Measure one pair under notching alone, on the background's frozen sequence.

    Identical to `frozen_paired_trial_rows` except that nothing is subtracted before
    the FIR, so a `frozen` and a `notch_only` run on the same seed differ only in the
    recovery step and their component retention is directly comparable.
    """
    spec, realization, injected = _realize_paired_injection(
        background,
        channel_name,
        target,
        rng,
    )
    injected_cleaning = clean_frozen_residuals_without_recovery(
        injected,
        targets,
        notch_settings,
        background_cleaning.residual_filter_plans,
        n_jobs=n_jobs,
        residual_stage=residual_stage,
    )
    return _paired_measurement_rows(
        background,
        injected,
        background_cleaning,
        injected_cleaning,
        placement,
        target,
        spec,
        realization,
        candidate="notch_only",
        spatial_rank="",
        residual_protocol="notch_only",
        recording=recording,
        participant=participant,
        channel_name=channel_name,
        recovery_window_s=recovery_window_s,
    )


def spatial_paired_trial_rows(
    background,
    background_cleaning: MultitaperCleaningResult,
    targets: recovery_benchmark.RecoveryTargets,
    placement: FrequencyPlacement,
    target: injection.FactorialInjectionTarget,
    rng: np.random.Generator,
    *,
    spatial_model: recovery.SpatialLineSubspaceModel,
    recording: str,
    participant: str,
    channel_name: str,
    notch_settings: notch.HarmonicNotchSettings,
    recovery_window_s: float,
    n_jobs: int = -1,
    residual_stage: str = "joint",
) -> tuple[dict[str, object], dict[str, object]]:
    """Measure frozen spatial recovery and background residual geometry."""
    spec, realization, injected = _realize_paired_injection(
        background,
        channel_name,
        target,
        rng,
    )
    injected_cleaning = clean_with_frozen_spatial_residuals(
        injected,
        targets,
        notch_settings,
        spatial_model,
        background_cleaning.residual_filter_plans,
        recovery_window_s=recovery_window_s,
        n_jobs=n_jobs,
        residual_stage=residual_stage,
    )
    return _paired_measurement_rows(
        background,
        injected,
        background_cleaning,
        injected_cleaning,
        placement,
        target,
        spec,
        realization,
        candidate="spatial_ssp",
        spatial_rank=spatial_model.rank,
        residual_protocol="spatial",
        recording=recording,
        participant=participant,
        channel_name=channel_name,
        recovery_window_s=recovery_window_s,
    )


def trial_function(residual_protocol: str):
    """Return the paired-trial implementation for one explicit protocol."""
    functions = {
        "adaptive": paired_trial_rows,
        "frozen": frozen_paired_trial_rows,
        "notch_only": notch_only_paired_trial_rows,
        "spatial": spatial_paired_trial_rows,
    }
    if residual_protocol not in functions:
        raise ValueError(
            f"residual protocol must be one of {RESIDUAL_PROTOCOLS}, "
            f"got {residual_protocol!r}"
        )
    return functions[residual_protocol]


def prepare_background(
    raw,
    targets: recovery_benchmark.RecoveryTargets,
    notch_settings: notch.HarmonicNotchSettings,
    *,
    residual_protocol: str,
    spatial_rank: int | None,
    n_jobs: int = -1,
    residual_stage: str = "joint",
):
    """Clean one background and bind its immutable trial geometry."""
    rank = validated_spatial_rank(residual_protocol, spatial_rank)
    recovery_window_s = notch_settings.estimation_window_s
    if rank is not None:
        spatial_model = recovery_benchmark.fit_spatial_line_subspace(
            raw,
            targets,
            window_s=recovery_window_s,
            rank=rank,
            n_jobs=n_jobs,
        )
        cleaning = clean_background_with_spatial_subspace(
            raw,
            targets,
            notch_settings,
            spatial_model,
            recovery_window_s=recovery_window_s,
            n_jobs=n_jobs,
        )
        paired_trial = partial(
            spatial_paired_trial_rows,
            spatial_model=spatial_model,
        )
        return cleaning, paired_trial

    if residual_protocol == "notch_only":
        cleaning = clean_without_recovery(
            raw,
            targets,
            notch_settings,
            n_jobs=n_jobs,
            residual_stage=residual_stage,
        )
        return cleaning, trial_function(residual_protocol)

    cleaning = clean_with_multitaper(
        raw,
        targets,
        notch_settings,
        recovery_window_s=recovery_window_s,
        n_jobs=n_jobs,
        residual_stage=residual_stage,
    )
    return cleaning, trial_function(residual_protocol)


def _realize_paired_injection(
    background,
    channel_name: str,
    target: injection.FactorialInjectionTarget,
    rng: np.random.Generator,
) -> tuple[injection.SinusoidInjection, injection.InjectionRealization, object]:
    spec, realization = validation.realize_factorial_injection(
        background,
        channel_name,
        target,
        rng,
    )
    injected = injection.inject_spatially_balanced(
        background,
        channel_name,
        realization,
    )
    return spec, realization, injected


def _paired_measurement_rows(
    background,
    injected,
    background_cleaning: MultitaperCleaningResult,
    injected_cleaning: MultitaperCleaningResult,
    placement: FrequencyPlacement,
    target: injection.FactorialInjectionTarget,
    spec: injection.SinusoidInjection,
    realization: injection.InjectionRealization,
    *,
    candidate: str,
    spatial_rank: int | str,
    residual_protocol: str,
    recording: str,
    participant: str,
    channel_name: str,
    recovery_window_s: float,
) -> tuple[dict[str, object], dict[str, object]]:
    """Measure recovery and final paired differences for one cleaning protocol."""
    import mne

    picks = mne.pick_types(background.info, eeg=True, exclude="bads")
    if len(picks) < 2:
        raise ValueError("paired neural validation requires two non-bad EEG channels")
    valid_samples = _valid_sample_mask(background)
    temporal_basis = realization.temporal_basis * valid_samples
    background_data = background.get_data(picks=picks)
    injected_data = injected.get_data(picks=picks)
    component = injected_data - background_data

    common = _trial_common_fields(
        spec,
        target,
        placement,
        background_cleaning,
        injected_cleaning,
        candidate=candidate,
        spatial_rank=spatial_rank,
        residual_protocol=residual_protocol,
        recording=recording,
        participant=participant,
        channel_name=channel_name,
    )
    stage_pairs = (
        (
            "recovery",
            background_cleaning.recovered.get_data(picks=picks),
            injected_cleaning.recovered.get_data(picks=picks),
        ),
        (
            "final",
            background_cleaning.cleaned.get_data(picks=picks),
            injected_cleaning.cleaned.get_data(picks=picks),
        ),
    )
    rows = []
    for stage, cleaned_background, cleaned_injected in stage_pairs:
        cleaned_difference = cleaned_injected - cleaned_background
        energy = validation.paired_energy_metrics(
            background_data,
            injected_data,
            cleaned_background,
            cleaned_injected,
            temporal_basis,
            valid_samples,
        )
        band = recovery_evaluation.measure_band_preservation(
            component,
            cleaned_difference,
            float(background.info["sfreq"]),
            (
                placement.band_name,
                placement.band_low_hz,
                placement.band_high_hz,
            ),
            window_s=recovery_window_s,
        )
        final_stage = stage == "final"
        rows.append(
            {
                **common,
                "stage": stage,
                "injected_energy_v2": energy.injected_energy_v2,
                "difference_energy_v2": energy.difference_energy_v2,
                "measured_component_to_background_db": (
                    energy.component_to_background_db
                ),
                "remaining_fraction": energy.remaining_fraction,
                "collateral_fraction": energy.collateral_fraction,
                "amplitude_ratio": energy.amplitude_ratio,
                "component_error_fraction": energy.component_error_fraction,
                "component_correlation": (
                    ""
                    if energy.component_correlation is None
                    else energy.component_correlation
                ),
                "subspace_phase_error_degrees": (
                    ""
                    if energy.phase_error_degrees is None
                    else energy.phase_error_degrees
                ),
                "band_power_ratio": band.power_ratio,
                "band_power_change_db": band.power_change_db,
                "band_phase_error_degrees": (
                    ""
                    if band.phase_error_degrees is None
                    else band.phase_error_degrees
                ),
                "terminal_residual_detector_null": (
                    injected_cleaning.terminal_residual_detector_null
                    if final_stage
                    else ""
                ),
                "targeted_local_background_excess_null": (
                    injected_cleaning.targeted_local_background_excess_null
                    if final_stage
                    else ""
                ),
                "artifact_gate_passed": (
                    background_cleaning.artifact_gate_passed
                    and injected_cleaning.artifact_gate_passed
                    if final_stage
                    else ""
                ),
                "preservation_gate_passed": "",
            }
        )
    return tuple(rows)


def _trial_common_fields(
    spec: injection.SinusoidInjection,
    target: injection.FactorialInjectionTarget,
    placement: FrequencyPlacement,
    background: MultitaperCleaningResult,
    injected: MultitaperCleaningResult,
    *,
    candidate: str,
    spatial_rank: int | str,
    residual_protocol: str,
    recording: str,
    participant: str,
    channel_name: str,
) -> dict[str, object]:
    return {
        "recording": recording,
        "participant": participant,
        "candidate": candidate,
        "residual_stage": background.residual_stage,
        "spatial_rank": spatial_rank,
        "residual_protocol": residual_protocol,
        "channel_name": channel_name,
        "band": placement.band_name,
        "band_low_hz": placement.band_low_hz,
        "band_high_hz": placement.band_high_hz,
        "kind": spec.kind,
        "position": placement.position,
        "authorized_line_hz": placement.authorized_line_hz,
        "neighbouring_line_hz": placement.neighbouring_line_hz,
        "centre_frequency_hz": placement.centre_frequency_hz,
        "injection_start_frequency_hz": spec.frequency_hz,
        "injection_drift_hz": spec.drift_hz,
        "injection_occupancy": spec.occupancy,
        "injection_phase_rad": spec.phase_rad,
        "phase_modulation_hz": spec.phase_modulation_hz,
        "phase_deviation_rad": spec.phase_deviation_rad,
        "injection_amplitude_v": spec.amplitude_v,
        "requested_component_to_background_db": (
            target.component_to_background_db
        ),
        "background_recovery_runtime_s": background.recovery_runtime_s,
        "background_residual_runtime_s": background.residual_runtime_s,
        "injected_recovery_runtime_s": injected.recovery_runtime_s,
        "injected_residual_runtime_s": injected.residual_runtime_s,
        "background_residual_round_count": background.residual_round_count,
        "injected_residual_round_count": injected.residual_round_count,
        "filter_geometry_changed": (
            background.residual_filter_plans != injected.residual_filter_plans
        ),
        "injected_frequency_fir_unavailable": _injection_is_unavailable(
            spec,
            injected.residual_filter_plans,
        ),
        "background_band_retained_share": _band_retained_share(
            background.residual_filter_plans,
            placement,
        ),
        "injected_band_retained_share": _band_retained_share(
            injected.residual_filter_plans,
            placement,
        ),
        "background_terminal_residual_detector_null": (
            background.terminal_residual_detector_null
        ),
        "background_targeted_local_background_excess_null": (
            background.targeted_local_background_excess_null
        ),
    }


def _injection_is_unavailable(
    spec: injection.SinusoidInjection,
    plans: tuple[notch.HarmonicNotchPlan, ...],
) -> bool:
    low_hz, high_hz = injection.injected_frequency_band_hz(
        spec,
        half_width_hz=0.0,
    )
    return any(
        low_hz <= unavailable_high_hz and high_hz >= unavailable_low_hz
        for plan in plans
        for unavailable_low_hz, unavailable_high_hz in plan.unavailable_edges()
    )


def _band_retained_share(
    plans: tuple[notch.HarmonicNotchPlan, ...],
    placement: FrequencyPlacement,
) -> float:
    if not plans:
        return 1.0
    fields = notch._band_availability_fields(
        notch.merge_recording_plans(plans),
        ((placement.band_name, placement.band_low_hz, placement.band_high_hz),),
    )
    return float(fields[f"{placement.band_name}_retained_share"])


def _valid_sample_mask(raw) -> np.ndarray:
    mask = np.zeros(raw.n_times, dtype=bool)
    for start, stop in recordings.acquisition_segments(raw):
        mask[start:stop] = True
    if not np.any(mask):
        raise ValueError("paired neural validation requires acquisition samples")
    return mask


def _available_frequency_range(
    raw,
    settings: notch.HarmonicNotchSettings,
) -> tuple[float, float]:
    maximum_hz = min(
        settings.frequency_range_hz[1],
        float(np.nextafter(float(raw.info["sfreq"]) / 2.0, 0.0)),
    )
    return settings.frequency_range_hz[0], maximum_hz


def _positive(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def validated_spatial_rank(
    residual_protocol: str,
    spatial_rank: int | None,
) -> int | None:
    """Require an explicit experimental rank only for spatial recovery."""
    if residual_protocol == "spatial":
        if spatial_rank not in (1, 2) or isinstance(spatial_rank, bool):
            raise ValueError("spatial protocol requires --spatial-rank 1 or 2")
        return spatial_rank
    if spatial_rank is not None:
        raise ValueError("--spatial-rank is only valid for the spatial protocol")
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure neural-like signal retention through multitaper recovery and "
            "converged residual FIR without writing cleaned EEG."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recordings", nargs="+", required=True)
    parser.add_argument(
        "--residual-protocol",
        choices=RESIDUAL_PROTOCOLS,
        required=True,
    )
    parser.add_argument("--spatial-rank", type=int, choices=(1, 2))
    parser.add_argument(
        "--residual-stage",
        choices=RESIDUAL_STAGES,
        default="joint",
        help=(
            "which residual FIR runs after recovery: 'lines' is the published "
            "pipeline's converged line rounds only; 'joint' adds this branch's "
            "target-local rounds. Orthogonal to --residual-protocol."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "override execution.n_jobs for this run: -1 for every core, or a positive "
            "integer. Channels are independent, so this changes speed, not results."
        ),
    )
    return parser


def _trial_rng(key: tuple[str, str, str, str]) -> np.random.Generator:
    payload = "\x1f".join((str(RANDOM_SEED), *key)).encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def _selected_runs(source_root: Path, recording_names: Sequence[str]) -> list[Path]:
    requested = tuple(str(name) for name in recording_names)
    if len(set(requested)) != len(requested):
        raise ValueError("recording names must be unique")
    indexed = {
        path.stem: path
        for path in recordings.discover_runs(source_root, None, task="*")
    }
    missing = set(requested) - set(indexed)
    if missing:
        raise FileNotFoundError(f"recordings not found: {sorted(missing)}")
    return [indexed[name] for name in requested]


def run(args: argparse.Namespace) -> None:
    """Run a checkpointed paired-injection cohort and write metrics only."""
    from decomb.config import load_config

    config = load_config(args.config)
    source_root = config.path("bids_root")
    notch_settings = notch.HarmonicNotchSettings.from_config(config)
    bands = notch.analysed_bands_from_config(config)
    n_jobs = (
        recordings.n_jobs_from_config(config)
        if args.n_jobs is None
        else recordings.validated_n_jobs(args.n_jobs)
    )
    manifest = pd.read_csv(
        args.manifest,
        sep="\t",
        keep_default_na=False,
        float_precision="round_trip",
    )
    runs = _selected_runs(source_root, args.recordings)
    spatial_rank = validated_spatial_rank(
        args.residual_protocol,
        args.spatial_rank,
    )

    if args.output.exists():
        raise FileExistsError(f"Refusing to replace validation output: {args.output}")
    staging = args.output.with_name(f".{args.output.name}.staging")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        checkpoint = pd.read_csv(
            staging,
            sep="\t",
            keep_default_na=False,
            float_precision="round_trip",
        )
        protocols = set(checkpoint["residual_protocol"])
        if protocols != {args.residual_protocol}:
            raise ValueError(
                f"validation checkpoint protocols {sorted(protocols)} do not match "
                f"{args.residual_protocol!r}"
            )
        stages = set(checkpoint.get("residual_stage", [args.residual_stage]))
        if stages != {args.residual_stage}:
            raise ValueError(
                f"validation checkpoint stages {sorted(stages)} do not match "
                f"{args.residual_stage!r}"
            )
        rows = checkpoint.to_dict(orient="records")
        completed = completed_trial_keys(rows)
        print(f"Resuming {len(completed)} paired trial checkpoints", flush=True)
    else:
        rows: list[dict[str, object]] = []
        completed = set()

    for recording_index, path in enumerate(runs, start=1):
        recording = path.stem
        targets = recovery_benchmark.targets_from_manifest(manifest, recording)
        applicable_bands = tuple(
            band
            for band in bands
            if has_resolved_line_pair(
                targets,
                band,
                frequency_bin_width_hz=notch_settings.frequency_bin_width_hz,
            )
        )
        omitted_bands = [
            name
            for name, _, _ in bands
            if name not in {band[0] for band in applicable_bands}
        ]
        if omitted_bands:
            print(
                f"[{recording_index}/{len(runs)}] {recording}: no resolved "
                f"first-round line pair in {', '.join(omitted_bands)}",
                flush=True,
            )
        design = validation_design(
            targets,
            applicable_bands,
            frequency_bin_width_hz=notch_settings.frequency_bin_width_hz,
            component_to_background_db=COMPONENT_TO_BACKGROUND_DB,
        )
        pending = [
            (placement, target)
            for placement, target in design
            if (
                recording,
                placement.band_name,
                target.kind,
                placement.position,
            )
            not in completed
        ]
        if not pending:
            continue

        raw = recordings.read_bids_raw(path)
        participant = recordings.subject_of(path)
        if INJECTION_CHANNEL_NAME not in raw.ch_names:
            raise ValueError(
                f"{recording} does not contain {INJECTION_CHANNEL_NAME!r}"
            )
        if INJECTION_CHANNEL_NAME in raw.info["bads"]:
            raise ValueError(
                f"{recording} marks {INJECTION_CHANNEL_NAME!r} bad"
            )
        print(
            f"[{recording_index}/{len(runs)}] {recording}: cleaning paired "
            f"background for {len(pending)} {args.residual_protocol} trials",
            flush=True,
        )
        background_cleaning, paired_trial = prepare_background(
            raw,
            targets,
            notch_settings,
            residual_protocol=args.residual_protocol,
            spatial_rank=spatial_rank,
            n_jobs=n_jobs,
            residual_stage=args.residual_stage,
        )
        if not background_cleaning.artifact_gate_passed:
            raise RuntimeError(f"{recording}: background artifact gate failed")

        for trial_index, (placement, target) in enumerate(pending, start=1):
            key = (
                recording,
                placement.band_name,
                target.kind,
                placement.position,
            )
            started = time.perf_counter()
            measured = paired_trial(
                raw,
                background_cleaning,
                targets,
                placement,
                target,
                _trial_rng(key),
                recording=recording,
                participant=participant,
                channel_name=INJECTION_CHANNEL_NAME,
                notch_settings=notch_settings,
                recovery_window_s=notch_settings.estimation_window_s,
                n_jobs=n_jobs,
                residual_stage=args.residual_stage,
            )
            rows.extend(measured)
            recordings.write_tsv_atomic(pd.DataFrame(rows), staging)
            final = measured[-1]
            print(
                f"[{recording_index}/{len(runs)} {trial_index}/{len(pending)}] "
                f"{placement.band_name} {target.kind} {placement.position}: "
                f"artifact gate {'passed' if final['artifact_gate_passed'] else 'failed'}, "
                f"remaining={float(final['remaining_fraction']):.3f}, "
                f"error={float(final['component_error_fraction']):.3f} "
                f"({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
    os.replace(staging, args.output)
    print(f"Wrote {args.output}", flush=True)


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
