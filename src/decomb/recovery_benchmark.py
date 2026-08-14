"""Evaluate recovery candidates before the production residual FIR stage."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from decomb import lines, notch, recordings, recovery, recovery_evaluation


def _frequency_tuple(values) -> tuple[float, ...]:
    frequencies = tuple(sorted({float(value) for value in values}))
    if not np.isfinite(frequencies).all() or any(
        frequency <= 0.0 for frequency in frequencies
    ):
        raise ValueError("recovery target frequencies must be finite and positive")
    return frequencies


@dataclass(frozen=True)
class RecoveryTargets:
    """First-round ordinary lines and trigger-authorized scanner harmonics."""

    ordinary_frequencies_hz: tuple[float, ...]
    scanner_frequencies_hz: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ordinary_frequencies_hz",
            _frequency_tuple(self.ordinary_frequencies_hz),
        )
        object.__setattr__(
            self,
            "scanner_frequencies_hz",
            _frequency_tuple(self.scanner_frequencies_hz),
        )
        if not self.all_frequencies_hz:
            raise ValueError("at least one first-round recovery target is required")

    @property
    def all_frequencies_hz(self) -> tuple[float, ...]:
        return _frequency_tuple(
            (*self.ordinary_frequencies_hz, *self.scanner_frequencies_hz)
        )


def targets_from_manifest(
    manifest: pd.DataFrame,
    recording: str,
) -> RecoveryTargets:
    """Read targets authorized before any removal from an apply manifest."""
    required = {
        "recording",
        "removal_round",
        "outcome",
        "stopband_low_hz",
        "stopband_high_hz",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"recovery manifest is missing columns: {sorted(missing)}")

    recording_rows = manifest.loc[manifest["recording"] == recording]
    if recording_rows.empty:
        raise ValueError(f"recovery manifest has no rows for {recording!r}")
    round_indices = pd.to_numeric(recording_rows["removal_round"], errors="raise")
    first_round = recording_rows.loc[round_indices == 1].copy()
    detected = first_round.loc[
        first_round["outcome"].isin(
            ("line_detected", "scanner_harmonics_detected")
        )
    ].copy()
    if detected.empty:
        raise ValueError(f"recording {recording!r} has no first-round targets")

    low_hz = pd.to_numeric(detected["stopband_low_hz"], errors="raise")
    high_hz = pd.to_numeric(detected["stopband_high_hz"], errors="raise")
    detected["centre_hz"] = (low_hz + high_hz) / 2.0
    return RecoveryTargets(
        _frequency_tuple(
            detected.loc[detected["outcome"] == "line_detected", "centre_hz"]
        ),
        _frequency_tuple(
            detected.loc[
                detected["outcome"] == "scanner_harmonics_detected",
                "centre_hz",
            ]
        ),
    )


def _eeg_data(raw) -> tuple[np.ndarray, np.ndarray]:
    import mne

    segments = recordings.acquisition_segments(raw)
    if segments != ((0, raw.n_times),):
        raise ValueError(
            "recovery benchmarking currently requires one continuous acquisition"
        )
    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    if len(picks) == 0:
        raise ValueError("signal recovery requires at least one EEG channel")
    return picks, raw.get_data(picks=picks)


def _replace_eeg(raw, picks: np.ndarray, data: np.ndarray):
    recovered = raw.copy().load_data()
    recovered._data[picks] = data
    return recovered


def recover_with_multitaper(
    raw,
    targets: RecoveryTargets,
    *,
    window_s: float,
):
    """Apply the multitaper candidate to every first-round target."""
    picks, data = _eeg_data(raw)
    result = recovery.subtract_multitaper_sinusoids(
        data,
        float(raw.info["sfreq"]),
        targets.all_frequencies_hz,
        window_s=window_s,
    )
    return _replace_eeg(raw, picks, result.cleaned_data)


def _subtract_ordinary_lines(
    data: np.ndarray,
    sampling_frequency_hz: float,
    targets: RecoveryTargets,
    window_s: float,
) -> np.ndarray:
    if not targets.ordinary_frequencies_hz:
        return data
    return recovery.subtract_multitaper_sinusoids(
        data,
        sampling_frequency_hz,
        targets.ordinary_frequencies_hz,
        window_s=window_s,
    ).cleaned_data


def _trigger_samples(raw, event_name: str) -> np.ndarray:
    if not isinstance(event_name, str) or not event_name.strip():
        raise ValueError("trigger_event_name must be a non-empty string")
    descriptions = np.asarray(raw.annotations.description, dtype=str)
    onsets_s = np.asarray(raw.annotations.onset, dtype=float)[
        descriptions == event_name
    ]
    if onsets_s.size < 2:
        raise ValueError(f"trigger event {event_name!r} must occur at least twice")
    relative_onsets_s = onsets_s - float(raw.first_time)
    return raw.time_as_index(relative_onsets_s, use_rounding=True)


def recover_with_trigger_basis(
    raw,
    targets: RecoveryTargets,
    *,
    repetition_time_s: float,
    trigger_event_name: str,
    maximum_component_count: int,
    ordinary_window_s: float,
):
    """Apply trigger-locked OBA to scanner targets, then ordinary line fits."""
    if (
        not isinstance(maximum_component_count, int)
        or isinstance(maximum_component_count, bool)
        or maximum_component_count < 1
    ):
        raise ValueError("maximum_component_count must be a positive integer")
    picks, data = _eeg_data(raw)
    sampling_frequency_hz = float(raw.info["sfreq"])
    recovered_data = data
    if targets.scanner_frequencies_hz:
        triggers = _trigger_samples(raw, trigger_event_name)
        component_count = min(
            maximum_component_count,
            2 * len(targets.scanner_frequencies_hz),
            len(triggers) - 1,
        )
        recovered_data = recovery.subtract_trigger_locked_optimal_basis(
            data,
            sampling_frequency_hz,
            targets.scanner_frequencies_hz,
            triggers,
            repetition_time_s=repetition_time_s,
            component_count=component_count,
        ).cleaned_data
    recovered_data = _subtract_ordinary_lines(
        recovered_data,
        sampling_frequency_hz,
        targets,
        ordinary_window_s,
    )
    return _replace_eeg(raw, picks, recovered_data)


def recover_with_trajectory_pca(
    raw,
    targets: RecoveryTargets,
    *,
    recovery_settings: recovery.TrajectoryPCASettings,
    ordinary_window_s: float,
):
    """Apply authorized trajectory PCA, then ordinary multitaper line fits."""
    picks, data = _eeg_data(raw)
    sampling_frequency_hz = float(raw.info["sfreq"])
    recovered_data = data
    if targets.scanner_frequencies_hz:
        recovered_data = recovery.subtract_recursive_trajectory_pca(
            data,
            sampling_frequency_hz,
            targets.scanner_frequencies_hz,
            recovery_settings,
        ).cleaned_data
    recovered_data = _subtract_ordinary_lines(
        recovered_data,
        sampling_frequency_hz,
        targets,
        ordinary_window_s,
    )
    return _replace_eeg(raw, picks, recovered_data)


def targeted_local_background_detection(
    raw,
    targets: RecoveryTargets,
    *,
    window_s: float,
    familywise_error_rate: float,
    frequency_range_hz: tuple[float, float],
):
    """Detect persistent local spectral excess only at authorized targets."""
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(picks) < 2:
        raise ValueError("local-background testing requires at least two EEG channels")
    bounds = recordings.valid_window_bounds(raw, window_s=window_s, overlap=0.5)
    data = raw.get_data(picks=picks)
    windows = np.stack([data[:, start:stop] for start, stop in bounds])
    frequencies_hz, p_values = lines.persistent_peak_p_values(
        windows,
        float(raw.info["sfreq"]),
        frequency_range_hz=frequency_range_hz,
    )
    bin_width_hz = float(np.median(np.diff(frequencies_hz)))
    target_indices = []
    for target_hz in targets.all_frequencies_hz:
        distances_hz = np.abs(frequencies_hz - target_hz)
        nearest = int(np.argmin(distances_hz))
        if distances_hz[nearest] > bin_width_hz / 2.0 + np.finfo(float).eps:
            raise ValueError(f"target {target_hz:g} Hz has no local-background bin")
        target_indices.append(nearest)
    unique_indices = np.unique(target_indices)
    return lines.detect_lines_from_p_values(
        frequencies_hz[unique_indices],
        p_values[..., unique_indices],
        familywise_error_rate=familywise_error_rate,
    )


def targeted_local_background_is_null(
    raw,
    targets: RecoveryTargets,
    *,
    window_s: float,
    familywise_error_rate: float,
    frequency_range_hz: tuple[float, float],
) -> bool:
    result = targeted_local_background_detection(
        raw,
        targets,
        window_s=window_s,
        familywise_error_rate=familywise_error_rate,
        frequency_range_hz=frequency_range_hz,
    )
    return not result.detections


@dataclass(frozen=True)
class RecoveryBenchmarkSettings:
    """Fixed candidate controls used for development and confirmation."""

    ordinary_window_s: float = 10.0
    trigger_maximum_component_count: int = 4
    trajectory_segment_s: float = 2.0

    def trajectory_settings(self) -> recovery.TrajectoryPCASettings:
        return recovery.TrajectoryPCASettings(segment_s=self.trajectory_segment_s)


@dataclass(frozen=True)
class JointResidualCleaningResult:
    """Residual FIR sequence satisfying coherent and local-background nulls."""

    cleaned: object
    harmonic_results: tuple[notch.HarmonicCleaningResult, ...]
    local_background_plans: tuple[notch.HarmonicNotchPlan, ...]

    @property
    def filter_plans(self) -> tuple[notch.HarmonicNotchPlan, ...]:
        harmonic_plans = tuple(
            round_.filter_plan
            for result in self.harmonic_results
            for round_ in result.rounds
        )
        return (*harmonic_plans, *self.local_background_plans)

    @property
    def round_count(self) -> int:
        return sum(len(result.rounds) for result in self.harmonic_results) + len(
            self.local_background_plans
        )


def clean_joint_residuals(
    raw,
    targets: RecoveryTargets,
    settings: notch.HarmonicNotchSettings,
) -> JointResidualCleaningResult:
    """Alternate residual line and target-local tests until both are null."""
    maximum_hz = min(
        settings.frequency_range_hz[1],
        float(np.nextafter(float(raw.info["sfreq"]) / 2.0, 0.0)),
    )
    frequency_range_hz = (settings.frequency_range_hz[0], maximum_hz)
    harmonic_results = [notch.clean_until_no_supported_lines(raw, settings)]
    local_background_plans = []

    while True:
        cleaned = harmonic_results[-1].cleaned
        detection = targeted_local_background_detection(
            cleaned,
            targets,
            window_s=settings.estimation_window_s,
            familywise_error_rate=settings.familywise_error_rate,
            frequency_range_hz=frequency_range_hz,
        )
        if not detection.detections:
            return JointResidualCleaningResult(
                cleaned,
                tuple(harmonic_results),
                tuple(local_background_plans),
            )

        model = lines.build_line_model(
            detection,
            channel_names=notch.eeg_channel_names(cleaned),
        )
        channel_plans = notch.plan_channel_notches(model, settings)
        local_plan = notch.recording_plan_from_channel_plans(channel_plans)
        filtered = notch.apply_harmonic_notches(cleaned, local_plan)
        if np.array_equal(filtered.get_data(), cleaned.get_data()):
            raise RuntimeError(
                "A target-local residual remains, but its FIR changed no samples"
            )
        local_background_plans.append(local_plan)
        harmonic_results.append(
            notch.clean_until_no_supported_lines(filtered, settings)
        )


def _rms_ratio(change: np.ndarray, reference: np.ndarray) -> float:
    reference_rms = float(np.sqrt(np.mean(reference**2)))
    if reference_rms <= 0.0:
        raise ValueError("benchmark reference data must have positive RMS")
    return float(np.sqrt(np.mean(change**2)) / reference_rms)


def _retained_shares(
    result: JointResidualCleaningResult,
    bands: tuple[tuple[str, float, float], ...],
) -> dict[str, float]:
    if not result.filter_plans:
        return {name: 1.0 for name, _, _ in bands}
    availability = notch._band_availability_fields(
        notch.merge_recording_plans(result.filter_plans),
        bands,
    )
    return {
        name: float(availability[f"{name}_retained_share"])
        for name, _, _ in bands
    }


def _measurement_rows(
    common: dict[str, float | int | str | bool],
    stage: str,
    metrics: recovery_evaluation.PreservationMetrics,
    retained_shares: dict[str, float] | None,
) -> list[dict[str, float | int | str | bool]]:
    rows = []
    for band in metrics.bands:
        row = {
            **common,
            "stage": stage,
            "signal_correlation": metrics.signal_correlation,
            "normalized_change_rms": metrics.normalized_change_rms,
            "band": band.name,
            "band_low_hz": band.low_hz,
            "band_high_hz": band.high_hz,
            "band_power_change_db": band.power_change_db,
            "band_phase_error_degrees": (
                "" if band.phase_error_degrees is None else band.phase_error_degrees
            ),
            "residual_fir_retained_share": (
                "" if retained_shares is None else retained_shares[band.name]
            ),
        }
        rows.append(row)
    return rows


def _evaluate_recovered(
    original,
    recovered,
    *,
    targets: RecoveryTargets,
    recording: str,
    participant: str,
    candidate: str,
    recovery_runtime_s: float,
    settings: notch.HarmonicNotchSettings,
    bands: tuple[tuple[str, float, float], ...],
) -> list[dict[str, float | int | str | bool]]:
    import mne

    picks = mne.pick_types(original.info, eeg=True, exclude="bads")
    original_data = original.get_data(picks=picks)
    recovered_data = recovered.get_data(picks=picks)
    recovered_metrics = recovery_evaluation.measure_preservation(
        original_data,
        recovered_data,
        float(original.info["sfreq"]),
        bands,
        window_s=settings.estimation_window_s,
    )

    residual_started = time.perf_counter()
    cleaning = clean_joint_residuals(recovered, targets, settings)
    residual_runtime_s = time.perf_counter() - residual_started
    cleaned_data = cleaning.cleaned.get_data(picks=picks)
    final_metrics = recovery_evaluation.measure_preservation(
        original_data,
        cleaned_data,
        float(original.info["sfreq"]),
        bands,
        window_s=settings.estimation_window_s,
    )

    initial_harmonic_result = cleaning.harmonic_results[0]
    if initial_harmonic_result.rounds:
        first_residual_stopband_count = len(
            initial_harmonic_result.rounds[0].filter_plan.stopbands
        )
    elif cleaning.local_background_plans:
        first_residual_stopband_count = len(
            cleaning.local_background_plans[0].stopbands
        )
    else:
        first_residual_stopband_count = 0
    retained_shares = _retained_shares(
        cleaning,
        bands,
    )
    common: dict[str, float | int | str | bool] = {
        "recording": recording,
        "participant": participant,
        "candidate": candidate,
        "recovery_runtime_s": recovery_runtime_s,
        "residual_fir_runtime_s": residual_runtime_s,
        "residual_fir_round_count": cleaning.round_count,
        "targeted_local_background_fir_round_count": len(
            cleaning.local_background_plans
        ),
        "first_residual_stopband_count": first_residual_stopband_count,
        "terminal_residual_detector_null": True,
        "targeted_local_background_excess_null": True,
        "artifact_gate_passed": True,
        "recovery_change_rms_ratio": _rms_ratio(
            recovered_data - original_data,
            original_data,
        ),
        "residual_fir_change_rms_ratio": _rms_ratio(
            cleaned_data - recovered_data,
            original_data,
        ),
    }
    return [
        *_measurement_rows(common, "recovery", recovered_metrics, None),
        *_measurement_rows(common, "final", final_metrics, retained_shares),
    ]


def _benchmark_multitaper(
    raw,
    targets: RecoveryTargets,
    settings: RecoveryBenchmarkSettings,
):
    return recover_with_multitaper(
        raw,
        targets,
        window_s=settings.ordinary_window_s,
    )


def _benchmark_trigger_basis(
    raw,
    targets: RecoveryTargets,
    settings: RecoveryBenchmarkSettings,
    notch_settings: notch.HarmonicNotchSettings,
):
    return recover_with_trigger_basis(
        raw,
        targets,
        repetition_time_s=notch_settings.scanner_repetition_time_s,
        trigger_event_name=notch_settings.scanner_trigger_event_name,
        maximum_component_count=settings.trigger_maximum_component_count,
        ordinary_window_s=settings.ordinary_window_s,
    )


def _benchmark_trajectory_pca(
    raw,
    targets: RecoveryTargets,
    settings: RecoveryBenchmarkSettings,
):
    return recover_with_trajectory_pca(
        raw,
        targets,
        recovery_settings=settings.trajectory_settings(),
        ordinary_window_s=settings.ordinary_window_s,
    )


def _benchmark_candidate(
    raw,
    targets: RecoveryTargets,
    *,
    candidate: str,
    recover,
    recording: str,
    participant: str,
    notch_settings: notch.HarmonicNotchSettings,
    bands: tuple[tuple[str, float, float], ...],
) -> list[dict[str, float | int | str | bool]]:
    started = time.perf_counter()
    recovered = recover()
    recovery_runtime_s = time.perf_counter() - started
    return _evaluate_recovered(
        raw,
        recovered,
        targets=targets,
        recording=recording,
        participant=participant,
        candidate=candidate,
        recovery_runtime_s=recovery_runtime_s,
        settings=notch_settings,
        bands=bands,
    )


def benchmark_multitaper_recording(
    raw,
    targets: RecoveryTargets,
    *,
    recording: str,
    participant: str,
    notch_settings: notch.HarmonicNotchSettings,
    bands: tuple[tuple[str, float, float], ...],
    benchmark_settings: RecoveryBenchmarkSettings,
) -> list[dict[str, float | int | str | bool]]:
    return _benchmark_candidate(
        raw,
        targets,
        candidate="multitaper",
        recover=lambda: _benchmark_multitaper(raw, targets, benchmark_settings),
        recording=recording,
        participant=participant,
        notch_settings=notch_settings,
        bands=bands,
    )


def benchmark_trigger_recording(
    raw,
    targets: RecoveryTargets,
    *,
    recording: str,
    participant: str,
    notch_settings: notch.HarmonicNotchSettings,
    bands: tuple[tuple[str, float, float], ...],
    benchmark_settings: RecoveryBenchmarkSettings,
) -> list[dict[str, float | int | str | bool]]:
    return _benchmark_candidate(
        raw,
        targets,
        candidate="trigger_optimal_basis",
        recover=lambda: _benchmark_trigger_basis(
            raw,
            targets,
            benchmark_settings,
            notch_settings,
        ),
        recording=recording,
        participant=participant,
        notch_settings=notch_settings,
        bands=bands,
    )


def benchmark_trajectory_recording(
    raw,
    targets: RecoveryTargets,
    *,
    recording: str,
    participant: str,
    notch_settings: notch.HarmonicNotchSettings,
    bands: tuple[tuple[str, float, float], ...],
    benchmark_settings: RecoveryBenchmarkSettings,
) -> list[dict[str, float | int | str | bool]]:
    return _benchmark_candidate(
        raw,
        targets,
        candidate="recursive_trajectory_pca",
        recover=lambda: _benchmark_trajectory_pca(
            raw,
            targets,
            benchmark_settings,
        ),
        recording=recording,
        participant=participant,
        notch_settings=notch_settings,
        bands=bands,
    )


def benchmark_recording(
    raw,
    targets: RecoveryTargets,
    *,
    recording: str,
    participant: str,
    notch_settings: notch.HarmonicNotchSettings,
    bands: tuple[tuple[str, float, float], ...],
    benchmark_settings: RecoveryBenchmarkSettings,
) -> list[dict[str, float | int | str | bool]]:
    """Run every candidate independently before the same residual FIR stage."""
    benchmark_functions = (
        benchmark_multitaper_recording,
        benchmark_trigger_recording,
        benchmark_trajectory_recording,
    )
    return [
        row
        for benchmark in benchmark_functions
        for row in benchmark(
            raw,
            targets,
            recording=recording,
            participant=participant,
            notch_settings=notch_settings,
            bands=bands,
            benchmark_settings=benchmark_settings,
        )
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark signal recovery before residual Decomb FIR notching."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    return parser


def _completed_candidate_keys(
    frame: pd.DataFrame,
    bands: tuple[tuple[str, float, float], ...],
) -> set[tuple[str, str]]:
    required = {"recording", "candidate", "stage", "band"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"benchmark checkpoint is missing columns: {sorted(missing)}")
    expected_bands = {name for name, _, _ in bands}
    completed = set()
    for key, rows in frame.groupby(["recording", "candidate"], sort=False):
        if set(rows["stage"]) != {"recovery", "final"}:
            raise ValueError(f"benchmark checkpoint has incomplete stages for {key}")
        for stage in ("recovery", "final"):
            observed_bands = set(rows.loc[rows["stage"] == stage, "band"])
            if observed_bands != expected_bands:
                raise ValueError(
                    f"benchmark checkpoint has incomplete {stage} bands for {key}"
                )
        completed.add((str(key[0]), str(key[1])))
    return completed


def run(args: argparse.Namespace) -> None:
    """Run all recovery candidates without writing cleaned EEG signals."""
    from decomb.config import load_config

    config = load_config(args.config)
    source_root = config.path("bids_root")
    notch_settings = notch.HarmonicNotchSettings.from_config(config)
    bands = notch.analysed_bands_from_config(config)
    benchmark_settings = RecoveryBenchmarkSettings(
        ordinary_window_s=notch_settings.estimation_window_s
    )
    manifest = pd.read_csv(args.manifest, sep="\t", keep_default_na=False)
    runs = recordings.discover_runs(source_root, args.subjects, task="*")

    if args.output.exists():
        raise FileExistsError(f"Refusing to replace benchmark output: {args.output}")
    staging = args.output.with_name(f".{args.output.name}.staging")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if staging.exists():
        checkpoint = pd.read_csv(staging, sep="\t", keep_default_na=False)
        completed = _completed_candidate_keys(checkpoint, bands)
        rows = checkpoint.to_dict(orient="records")
        print(f"Resuming {len(completed)} completed candidate checkpoints", flush=True)
    else:
        completed = set()
        rows: list[dict[str, float | int | str | bool]] = []

    benchmark_functions = (
        ("multitaper", benchmark_multitaper_recording),
        ("trigger_optimal_basis", benchmark_trigger_recording),
        ("recursive_trajectory_pca", benchmark_trajectory_recording),
    )
    for index, vhdr in enumerate(runs, start=1):
        pending = [
            (candidate, benchmark)
            for candidate, benchmark in benchmark_functions
            if (vhdr.stem, candidate) not in completed
        ]
        if not pending:
            continue
        raw = recordings.read_bids_raw(vhdr)
        targets = targets_from_manifest(manifest, vhdr.stem)
        for candidate, benchmark in pending:
            started = time.perf_counter()
            measured = benchmark(
                raw,
                targets,
                recording=vhdr.stem,
                participant=recordings.subject_of(vhdr),
                notch_settings=notch_settings,
                bands=bands,
                benchmark_settings=benchmark_settings,
            )
            rows.extend(measured)
            recordings.write_tsv_atomic(pd.DataFrame(rows), staging)
            passed = bool(measured[-1]["artifact_gate_passed"])
            print(
                f"[{index}/{len(runs)}] {vhdr.stem} {candidate}: "
                f"artifact gate {'passed' if passed else 'failed'} "
                f"({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
    os.replace(staging, args.output)
    print(f"Wrote {args.output}", flush=True)


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
