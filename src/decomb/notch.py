"""Automatic participant-specific FIR notches for supported comb harmonics.

The transform makes no claim to recover neural activity at a removed harmonic. Its
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

from decomb import __version__, harmonics, recordings, spectral


@dataclass(frozen=True)
class HarmonicNotchSettings:
    """Configuration required to detect and notch supported comb harmonics."""

    task: str
    estimation_window_s: float
    estimation_overlap: float
    filter_jobs: int
    nominal_fundamental_hz: float
    harmonic_range: tuple[int, int]
    removal_harmonic_range: tuple[int, int]
    search_hz: float
    min_prominence_db: float
    uncertainty_confidence_z: float
    low_hz: float
    high_hz: float
    background_half_width_hz: float
    min_harmonics_for_fit: int
    max_harmonic_residual_resolutions: float
    max_fit_residual_rms_resolutions: float
    minimum_stopband_resolutions: float
    transition_bandwidth_resolutions: float
    residual_search_hz: float
    roundtrip_relative_tolerance: float

    def __post_init__(self) -> None:
        positive = (
            "estimation_window_s",
            "nominal_fundamental_hz",
            "search_hz",
            "uncertainty_confidence_z",
            "background_half_width_hz",
            "max_harmonic_residual_resolutions",
            "max_fit_residual_rms_resolutions",
            "minimum_stopband_resolutions",
            "transition_bandwidth_resolutions",
            "residual_search_hz",
            "roundtrip_relative_tolerance",
        )
        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"removal.{name} must be finite and positive.")
        if not self.task.strip():
            raise ValueError("dataset.task must name a BIDS task label.")
        if self.filter_jobs < 1:
            raise ValueError("removal.filter_jobs must be positive.")
        if not 0.0 < self.estimation_overlap < 1.0:
            raise ValueError("removal.estimation_overlap must lie strictly between zero and one.")
        if self.min_harmonics_for_fit < 3:
            raise ValueError("removal.min_harmonics_for_fit must be at least three.")
        if not 0.0 <= self.low_hz < self.high_hz:
            raise ValueError("removal low_hz and high_hz must be increasing.")
        for name in ("harmonic_range", "removal_harmonic_range"):
            first, last = getattr(self, name)
            if first < 1 or last < first:
                raise ValueError(f"removal.{name} must contain increasing positive integers.")
        if self.search_hz >= self.nominal_fundamental_hz / 2.0:
            raise ValueError("removal.search_hz must be below half the nominal fundamental.")
        if self.max_fit_residual_rms_resolutions > self.max_harmonic_residual_resolutions:
            raise ValueError(
                "removal.max_fit_residual_rms_resolutions cannot exceed "
                "max_harmonic_residual_resolutions."
            )

    @property
    def spectral_resolution_hz(self) -> float:
        return spectral.hann_resolution_hz(self.estimation_window_s)

    @property
    def transition_bandwidth_hz(self) -> float:
        return self.transition_bandwidth_resolutions * self.spectral_resolution_hz

    @property
    def minimum_stopband_width_hz(self) -> float:
        return self.minimum_stopband_resolutions * self.spectral_resolution_hz

    @property
    def max_harmonic_residual_hz(self) -> float:
        return self.max_harmonic_residual_resolutions * self.spectral_resolution_hz

    @property
    def max_fit_residual_rms_hz(self) -> float:
        return self.max_fit_residual_rms_resolutions * self.spectral_resolution_hz

    @classmethod
    def from_config(cls, config) -> HarmonicNotchSettings:
        block = dict(config.get("removal") or {})
        known = {entry.name for entry in fields(cls)} - {"task"}
        unknown = set(block) - known
        if unknown:
            raise ValueError(
                f"Unknown `removal` setting(s): {sorted(unknown)}. "
                f"Known settings are {sorted(known)}."
            )
        missing = known - set(block)
        if missing:
            raise ValueError(f"Missing `removal` setting(s): {sorted(missing)}.")
        values: dict[str, object] = {"task": str((config.get("dataset") or {}).get("task", ""))}
        for entry in fields(cls):
            if entry.name not in block:
                continue
            value = block[entry.name]
            if entry.name in {"harmonic_range", "removal_harmonic_range"}:
                values[entry.name] = tuple(int(item) for item in value)
            elif entry.name in {"filter_jobs", "min_harmonics_for_fit"}:
                values[entry.name] = int(value)
            else:
                values[entry.name] = float(value)
        return cls(**values)


@dataclass(frozen=True)
class HarmonicStopband:
    """One measured harmonic interval that is unavailable after filtering."""

    harmonics: tuple[int, ...]
    low_hz: float
    high_hz: float

    def __post_init__(self) -> None:
        if not self.harmonics or tuple(sorted(set(self.harmonics))) != self.harmonics:
            raise ValueError("Stopband harmonics must be sorted unique positive integers.")
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
            raise ValueError("A harmonic notch plan requires at least one supported harmonic.")
        if not np.isfinite(self.transition_bandwidth_hz) or self.transition_bandwidth_hz <= 0.0:
            raise ValueError("The transition bandwidth must be finite and positive.")
        if any(
            later.low_hz < earlier.high_hz
            for earlier, later in zip(self.stopbands, self.stopbands[1:])
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


def _observed_harmonic_intervals(model, settings) -> list[HarmonicStopband]:
    """Intervals supported by the whole run and localized in its adaptive windows."""
    first, last = settings.removal_harmonic_range
    supported = {
        harmonic
        for harmonic in model.whole_estimate.supported_harmonics
        if first <= harmonic <= last
    }
    whole_positions = dict(
        zip(
            model.whole_estimate.supported_harmonics,
            model.whole_estimate.supported_positions_hz,
        )
    )
    intervals = []
    for harmonic in sorted(supported):
        uncertainty_hz = (
            settings.uncertainty_confidence_z
            * harmonic
            * model.whole_estimate.fundamental_jackknife_se_hz
        )
        lower_edges = [whole_positions[harmonic] - uncertainty_hz]
        upper_edges = [whole_positions[harmonic] + uncertainty_hz]
        for evidence in model.window_evidence:
            positions = dict(zip(evidence.harmonics, evidence.positions_hz))
            if harmonic not in positions:
                continue
            lower_edges.append(positions[harmonic])
            upper_edges.append(positions[harmonic])

        low_hz = min(lower_edges)
        high_hz = max(upper_edges)
        centre_hz = (low_hz + high_hz) / 2.0
        if not settings.low_hz <= centre_hz <= settings.high_hz:
            continue
        minimum_width_hz = settings.minimum_stopband_width_hz
        if high_hz - low_hz < minimum_width_hz:
            low_hz = centre_hz - minimum_width_hz / 2.0
            high_hz = centre_hz + minimum_width_hz / 2.0
        intervals.append(HarmonicStopband((harmonic,), low_hz, high_hz))
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
        )
    return tuple(merged)


def plan_harmonic_stopbands(model, settings) -> HarmonicNotchPlan:
    """Build the narrowest plan justified by measured harmonic positions."""
    transition_bandwidth_hz = settings.transition_bandwidth_hz
    stopbands = _merge_stopbands(
        _observed_harmonic_intervals(model, settings),
        minimum_gap_hz=transition_bandwidth_hz,
    )
    return HarmonicNotchPlan(stopbands, transition_bandwidth_hz)


def _estimate_comb_spectrum(spectrum, settings):
    frequencies_hz, spectrum_db, prominence_db = spectrum
    return harmonics.estimate_comb(
        frequencies_hz,
        spectrum_db,
        prominence_db,
        nominal_fundamental_hz=settings.nominal_fundamental_hz,
        fit_harmonic_range=settings.harmonic_range,
        supported_harmonic_range=settings.removal_harmonic_range,
        search_hz=settings.search_hz,
        min_prominence_db=settings.min_prominence_db,
        min_harmonics=settings.min_harmonics_for_fit,
        max_harmonic_residual_hz=settings.max_harmonic_residual_hz,
        max_residual_rms_hz=settings.max_fit_residual_rms_hz,
    )


def _localize_window_evidence(
    spectrum,
    settings,
    supported_harmonics,
    fundamental_hz,
):
    frequencies_hz, spectrum_db, prominence_db = spectrum
    return harmonics.localize_supported_harmonics(
        frequencies_hz,
        spectrum_db,
        prominence_db,
        supported_harmonics=supported_harmonics,
        fundamental_hz=fundamental_hz,
        search_hz=settings.max_harmonic_residual_hz,
        min_prominence_db=settings.min_prominence_db,
    )


def fit_harmonic_model(raw, settings):
    """Authorize the comb once, then localize only those targets in each window."""
    spectra = recordings.session_run_spectra(raw, settings)
    whole_estimate = _estimate_comb_spectrum(spectra.whole, settings)
    window_evidence = tuple(
        _localize_window_evidence(
            spectrum,
            settings,
            whole_estimate.supported_harmonics,
            whole_estimate.fundamental_hz,
        )
        for spectrum in spectra.windows
    )
    return harmonics.AdaptiveCombModel(
        whole_estimate=whole_estimate,
        window_evidence=window_evidence,
    )


def apply_harmonic_notches(raw, plan: HarmonicNotchPlan, *, filter_jobs: int):
    """Return a copy with the evidence-bounded harmonic intervals removed from EEG."""
    import mne

    if filter_jobs < 1:
        raise ValueError("filter_jobs must be positive.")
    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    if len(picks) == 0:
        raise ValueError("Harmonic notching requires at least one EEG channel.")

    nyquist_hz = float(raw.info["sfreq"]) / 2.0
    unavailable_edges = plan.unavailable_edges()
    if unavailable_edges[0][0] <= 0.0:
        raise ValueError("The first harmonic notch transition reaches 0 Hz.")
    if unavailable_edges[-1][1] >= nyquist_hz:
        raise ValueError(
            f"The last harmonic notch transition reaches the {nyquist_hz:g} Hz Nyquist limit."
        )

    filtered = raw.copy()
    filtered.notch_filter(
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
        phase="zero",
        n_jobs=filter_jobs,
        verbose="ERROR",
    )
    return filtered


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
    for stopband, unavailable in zip(plan.stopbands, unavailable_edges):
        row: dict[str, float | str] = {
            "recording": recording,
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
                    harmonics=tuple(int(value) for value in str(row["harmonics"]).split(";")),
                    low_hz=float(row["stopband_low_hz"]),
                    high_hz=float(row["stopband_high_hz"]),
                )
                for row in rows
            ),
            key=lambda stopband: stopband.low_hz,
        )
    )
    return HarmonicNotchPlan(stopbands, transition_bandwidths_hz.pop())


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
    plan = plan_harmonic_stopbands(model, settings)
    filtered = apply_harmonic_notches(raw, plan, filter_jobs=settings.filter_jobs)

    relative_header = vhdr.relative_to(source_root)
    destination = output_root / relative_header.with_suffix(".eeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    recordings.write_eeg_binary(vhdr, destination, filtered.get_data())

    written = recordings.read_bids_raw(output_root / relative_header)
    expected = filtered.get_data()
    deviation_v = float(np.max(np.abs(written.get_data() - expected)))
    tolerance_v = settings.roundtrip_relative_tolerance * float(np.max(np.abs(expected)))
    if deviation_v > tolerance_v:
        raise RuntimeError(
            f"{vhdr.name}: written data differs by {deviation_v:.3e} V, "
            f"above the {tolerance_v:.3e} V float32 round-trip tolerance."
        )

    rows = harmonic_exclusion_rows(vhdr.stem, plan, analysed_bands)
    changes_db = _measure_stopband_changes(raw, filtered, plan, settings)
    for row, stopband, change_db in zip(rows, plan.stopbands, changes_db):
        supporting_window_count = sum(
            any(harmonic in evidence.harmonics for harmonic in stopband.harmonics)
            for evidence in model.window_evidence
        )
        row["fundamental_hz"] = model.whole_estimate.fundamental_hz
        row["estimation_window_count"] = len(model.window_evidence)
        row["supporting_window_count"] = supporting_window_count
        row["in_stopband_change_db"] = change_db
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

    path = output_root / "dataset_description.json"
    if not path.is_file():
        raise FileNotFoundError(f"Source dataset description was not mirrored to {path}.")
    described = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(described, dict):
        raise ValueError("BIDS dataset_description.json must contain a JSON object.")

    described["DatasetType"] = "derivative"
    if "BIDSVersion" not in described:
        raise ValueError("BIDS dataset_description.json must declare BIDSVersion.")
    described["Name"] = "decomb harmonic-notched EEG"
    existing = described.get("GeneratedBy", [])
    if not isinstance(existing, list) or not all(isinstance(entry, dict) for entry in existing):
        raise ValueError("BIDS GeneratedBy must be a list of objects.")
    generated = [entry for entry in existing if entry.get("Name") != "decomb"]
    generated.append(
        {
            "Name": "decomb",
            "Version": __version__,
            "Description": (
                "The whole recording authorized participant-specific comb harmonics; "
                "overlapping Hann-window spectra localized only those supported targets. "
                "They were removed with zero-phase MNE FIR notches. The stopbands and "
                "their transitions are unavailable for inference and are listed per "
                "recording in harmonic_notch_manifest.tsv."
            ),
            "Parameters": {
                "method": "fir",
                "phase": "zero",
                **{
                    name: list(value) if isinstance(value, tuple) else value
                    for name, value in asdict(settings).items()
                },
                "spectral_resolution_hz": settings.spectral_resolution_hz,
                "minimum_stopband_width_hz": settings.minimum_stopband_width_hz,
                "transition_bandwidth_hz": settings.transition_bandwidth_hz,
            },
        }
    )
    described["GeneratedBy"] = generated
    if not source_dataset_url:
        raise ValueError("The source dataset URL must not be empty.")
    described["SourceDatasets"] = [{"URL": source_dataset_url}]
    path.write_text(json.dumps(described, indent=2) + "\n", encoding="utf-8")
    return path


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
    runs = recordings.discover_runs(source_root, subjects=None, task=settings.task)

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
    print(f"Applying automatic harmonic notches to {len(runs)} recordings")
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
        stopband_width_hz = sum(
            float(row["stopband_high_hz"]) - float(row["stopband_low_hz"]) for row in measured
        )
        median_change_db = float(
            np.median([float(row["in_stopband_change_db"]) for row in measured])
        )
        print(
            f"[{index}/{len(runs)}] {vhdr.stem[:44]:44s} "
            f"{len(measured)} stopbands, {stopband_width_hz:.3f} Hz, "
            f"median {median_change_db:+.1f} dB ({time.time() - started:.0f}s)"
        )

    frame = pd.DataFrame(rows)
    manifest_name = "harmonic_notch_manifest.tsv"
    recordings.write_tsv_atomic(frame, staging / manifest_name)
    source_dataset_url = relative_source_dataset_url(source_root, output_root)
    described = write_harmonic_derivative_description(
        staging,
        source_dataset_url,
        settings,
    )
    os.replace(staging, output_root)

    report_dir.mkdir(parents=True, exist_ok=True)
    recordings.write_tsv_atomic(frame, report_dir / manifest_name)
    effective_path = effective.write(
        config,
        settings,
        report_dir / "effective_config_apply.txt",
        stage="apply",
    )
    print(f"  declared {output_root / described.name} a derivative of {source_root}")
    print(f"  wrote {report_dir / manifest_name}")
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


def _adjacent_residual_peak(
    frequencies_hz: np.ndarray,
    prominence_db: np.ndarray,
    stopband: HarmonicStopband,
    unavailable: tuple[float, float],
    settings,
) -> tuple[float, float]:
    """Frequency and prominence of the largest available peak beside a stopband."""
    search = (frequencies_hz >= stopband.low_hz - settings.residual_search_hz) & (
        frequencies_hz <= stopband.high_hz + settings.residual_search_hz
    )
    available = (frequencies_hz < unavailable[0]) | (frequencies_hz > unavailable[1])
    candidate_indices = np.flatnonzero(search & available)
    candidates = np.asarray(prominence_db, dtype=float)[candidate_indices]
    if candidates.size == 0:
        return float("nan"), float("nan")
    selected = int(candidate_indices[int(np.argmax(candidates))])
    return float(frequencies_hz[selected]), float(prominence_db[selected])


def verify_harmonic_run(
    source_vhdr: Path,
    cleaned_vhdr: Path,
    manifest_rows: Sequence[Mapping[str, object]],
    settings,
) -> list[dict[str, float | str]]:
    """Re-measure a written recording using only its declared filter geometry."""
    original = recordings.read_bids_raw(source_vhdr)
    cleaned = recordings.read_bids_raw(cleaned_vhdr)
    _validate_matching_recordings(original, cleaned)
    plan = harmonic_plan_from_rows(manifest_rows)
    changes_db = _measure_stopband_changes(original, cleaned, plan, settings)
    original_frequencies_hz, _, original_prominence_db = recordings.run_spectrum(
        original,
        settings,
    )
    cleaned_frequencies_hz, _, cleaned_prominence_db = recordings.run_spectrum(
        cleaned,
        settings,
    )
    if not np.array_equal(original_frequencies_hz, cleaned_frequencies_hz):
        raise ValueError("Source and cleaned prominence spectra use different grids.")

    rows = []
    for stopband, unavailable, change_db in zip(
        plan.stopbands,
        plan.unavailable_edges(),
        changes_db,
    ):
        original_peak_hz, original_peak_db = _adjacent_residual_peak(
            original_frequencies_hz,
            original_prominence_db,
            stopband,
            unavailable,
            settings,
        )
        cleaned_peak_hz, cleaned_peak_db = _adjacent_residual_peak(
            cleaned_frequencies_hz,
            cleaned_prominence_db,
            stopband,
            unavailable,
            settings,
        )
        rows.append(
            {
                "recording": source_vhdr.stem,
                "harmonics": ";".join(str(value) for value in stopband.harmonics),
                "stopband_low_hz": stopband.low_hz,
                "stopband_high_hz": stopband.high_hz,
                "unavailable_low_hz": unavailable[0],
                "unavailable_high_hz": unavailable[1],
                "verified_stopband_change_db": change_db,
                "original_adjacent_peak_hz": original_peak_hz,
                "original_adjacent_prominence_db": original_peak_db,
                "cleaned_adjacent_peak_hz": cleaned_peak_hz,
                "cleaned_adjacent_prominence_db": cleaned_peak_db,
                "adjacent_max_prominence_change_db": cleaned_peak_db - original_peak_db,
            }
        )
    return rows


def run_verify(args: argparse.Namespace) -> None:
    """Audit the written harmonic-notch derivative without refitting its targets."""
    from decomb import effective
    from decomb.config import load_config

    config = load_config(getattr(args, "config", None))
    source_root = config.path("bids_root", override=getattr(args, "bids_root", None))
    cleaned_root = config.path("output_root", override=getattr(args, "output_root", None))
    report_dir = config.path("removal_dir", override=getattr(args, "report_dir", None))
    settings = HarmonicNotchSettings.from_config(config)
    runs = recordings.discover_runs(source_root, subjects=None, task=settings.task)
    manifest_path = cleaned_root / "harmonic_notch_manifest.tsv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No harmonic notch manifest at {manifest_path}. Run `decomb apply` first."
        )
    manifest = pd.read_csv(manifest_path, sep="\t", float_precision="round_trip")
    required = {
        "recording",
        "harmonics",
        "stopband_low_hz",
        "stopband_high_hz",
        "transition_bandwidth_hz",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Harmonic notch manifest is missing columns: {sorted(missing)}")
    recording_names = {vhdr.stem for vhdr in runs}
    if set(manifest["recording"]) != recording_names:
        raise ValueError("Harmonic notch manifest does not cover exactly the source recordings.")

    rows: list[dict[str, float | str]] = []
    for vhdr in runs:
        block = manifest.loc[manifest["recording"] == vhdr.stem]
        rows.extend(
            verify_harmonic_run(
                vhdr,
                cleaned_root / vhdr.relative_to(source_root),
                block.to_dict("records"),
                settings,
            )
        )
    frame = pd.DataFrame(rows)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / "harmonic_notch_verification.tsv"
    recordings.write_tsv_atomic(frame, output_path)
    effective_path = effective.write(
        config,
        settings,
        report_dir / "effective_config_verify.txt",
        stage="verify",
    )
    print(
        f"Verified {len(runs)} recordings: median stopband change "
        f"{frame['verified_stopband_change_db'].median():+.1f} dB"
    )
    print(f"  wrote {output_path}")
    print(f"  wrote {effective_path}")
