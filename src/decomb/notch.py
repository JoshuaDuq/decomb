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
    """The stationarity horizon and study frequency range supplied by the user."""

    estimation_window_s: float
    frequency_range_hz: tuple[float, float]

    def __post_init__(self) -> None:
        if not np.isfinite(self.estimation_window_s) or self.estimation_window_s <= 0.0:
            raise ValueError("removal.estimation_window_s must be finite and positive.")
        values = np.asarray(self.frequency_range_hz, dtype=float)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError(
                "removal.frequency_range_hz must contain two finite values."
            )
        low_hz, high_hz = (float(value) for value in values)
        if not 0.0 <= low_hz < high_hz <= harmonics.MAXIMUM_FREQUENCY_HZ:
            raise ValueError("removal.frequency_range_hz must increase inside [0, 100] Hz.")

    @property
    def estimation_overlap(self) -> float:
        """Hann's constant-overlap-add hop, derived from the window itself."""
        return 0.5

    @property
    def spectral_resolution_hz(self) -> float:
        return spectral.hann_resolution_hz(self.estimation_window_s)

    @property
    def transition_bandwidth_hz(self) -> float:
        # MNE firwin's automatic Hamming length is 3.3 / transition bandwidth.
        return 3.3 / self.estimation_window_s

    @property
    def minimum_stopband_width_hz(self) -> float:
        return self.spectral_resolution_hz

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


def observed_line_intervals(model, settings) -> list[HarmonicStopband]:
    """Intervals supported by the whole run and localized in its adaptive windows."""
    whole_positions = dict(
        zip(
            model.whole_estimate.harmonics,
            model.whole_estimate.positions_hz,
        )
    )
    intervals = []
    location_uncertainty_hz = settings.frequency_bin_width_hz / 2.0
    for harmonic in model.whole_estimate.harmonics:
        lower_edges = [whole_positions[harmonic] - location_uncertainty_hz]
        upper_edges = [whole_positions[harmonic] + location_uncertainty_hz]
        for evidence in model.window_evidence:
            positions = dict(zip(evidence.harmonics, evidence.positions_hz))
            lower_edges.append(positions[harmonic] - location_uncertainty_hz)
            upper_edges.append(positions[harmonic] + location_uncertainty_hz)

        low_hz = min(lower_edges)
        high_hz = max(upper_edges)
        centre_hz = (low_hz + high_hz) / 2.0
        minimum_width_hz = settings.minimum_stopband_width_hz
        if high_hz - low_hz < minimum_width_hz:
            low_hz = centre_hz - minimum_width_hz / 2.0
            high_hz = centre_hz + minimum_width_hz / 2.0
        intervals.append(HarmonicStopband((harmonic,), low_hz, high_hz))

    for line_index, whole_position_hz in enumerate(model.isolated_lines.positions_hz):
        positions_hz = [whole_position_hz]
        positions_hz.extend(
            window[line_index]
            for window in model.isolated_lines.window_positions_hz
        )
        low_hz = min(positions_hz) - location_uncertainty_hz
        high_hz = max(positions_hz) + location_uncertainty_hz
        centre_hz = (low_hz + high_hz) / 2.0
        if high_hz - low_hz < settings.minimum_stopband_width_hz:
            half_width_hz = settings.minimum_stopband_width_hz / 2.0
            low_hz = centre_hz - half_width_hz
            high_hz = centre_hz + half_width_hz
        intervals.append(
            HarmonicStopband((), low_hz, high_hz, kind="isolated")
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


def _estimate_comb_spectrum(spectrum, settings):
    frequencies_hz, spectrum_db = spectrum
    return harmonics.estimate_comb(
        frequencies_hz,
        spectrum_db,
        spectral_resolution_hz=settings.spectral_resolution_hz,
        frequency_range_hz=settings.frequency_range_hz,
    )


def _localize_window_evidence(
    spectrum,
    settings,
    supported_harmonics,
    fundamental_hz,
):
    frequencies_hz, spectrum_db = spectrum
    return harmonics.localize_harmonics(
        frequencies_hz,
        spectrum_db,
        harmonics=supported_harmonics,
        fundamental_hz=fundamental_hz,
        spectral_resolution_hz=settings.spectral_resolution_hz,
    )


def fit_harmonic_model(raw, settings):
    """Authorize a complete comb and statistically supported isolated lines."""
    spectra = recordings.session_run_spectra(raw, settings)
    whole_estimate = _estimate_comb_spectrum(spectra.whole, settings)
    window_evidence = tuple(
        _localize_window_evidence(
            spectrum,
            settings,
            whole_estimate.harmonics,
            whole_estimate.fundamental_hz,
        )
        for spectrum in spectra.windows
    )
    frequencies_hz, whole_spectrum_db = spectra.whole
    isolated_lines = harmonics.detect_isolated_lines(
        frequencies_hz,
        whole_spectrum_db,
        [spectrum_db for _, spectrum_db in spectra.windows],
        comb=whole_estimate,
        spectral_resolution_hz=settings.spectral_resolution_hz,
        independent_window_indices=recordings.non_overlapping_window_indices(
            spectra.bounds
        ),
        frequency_range_hz=settings.frequency_range_hz,
    )
    return harmonics.AdaptiveCombModel(
        whole_estimate=whole_estimate,
        window_evidence=window_evidence,
        isolated_lines=isolated_lines,
    )


def apply_harmonic_notches(raw, plan: HarmonicNotchPlan):
    """Return a copy with the evidence-bounded harmonic intervals removed from EEG."""
    import mne

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
        n_jobs=-1,
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
    filtered = apply_harmonic_notches(raw, plan)

    relative_header = vhdr.relative_to(source_root)
    destination = output_root / relative_header.with_suffix(".eeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    recordings.write_eeg_binary(vhdr, destination, filtered.get_data())

    written = recordings.read_bids_raw(output_root / relative_header)
    expected = filtered.get_data()
    deviation_v = float(np.max(np.abs(written.get_data() - expected)))
    representable = recordings.quantized_eeg_data(vhdr, expected)
    quantization_deviation_v = float(np.max(np.abs(representable - expected)))
    tolerance_v = float(np.nextafter(quantization_deviation_v, np.inf))
    if deviation_v > tolerance_v:
        raise RuntimeError(
            f"{vhdr.name}: written data differs by {deviation_v:.3e} V, "
            f"above its {tolerance_v:.3e} V float32 quantization bound."
        )

    rows = harmonic_exclusion_rows(vhdr.stem, plan, analysed_bands)
    changes_db = _measure_stopband_changes(raw, filtered, plan, settings)
    for row, stopband, change_db in zip(rows, plan.stopbands, changes_db):
        isolated_targets = [
            (position_hz, evidence_bic)
            for position_hz, evidence_bic in zip(
                model.isolated_lines.positions_hz,
                model.isolated_lines.evidence_bic,
            )
            if stopband.low_hz <= position_hz <= stopband.high_hz
        ]
        row["fundamental_hz"] = model.whole_estimate.fundamental_hz
        row["comb_evidence_bic"] = model.whole_estimate.evidence_bic
        row["isolated_line_frequencies_hz"] = ";".join(
            f"{position_hz:.17g}" for position_hz, _ in isolated_targets
        )
        row["isolated_line_evidence_bic"] = ";".join(
            f"{evidence_bic:.17g}" for _, evidence_bic in isolated_targets
        )
        row["estimation_window_count"] = len(model.window_evidence)
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
                "BIC model comparison identified a participant-specific comb and "
                "resolution-limited isolated lines. Every comb multiple in the configured "
                "range was localized across overlapping Hann windows, then all targets "
                "were removed with zero-phase MNE FIR notches. Stopbands and transitions "
                "are unavailable for inference and are listed per recording in "
                "harmonic_notch_manifest.tsv."
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
    spectrum_db: np.ndarray,
    stopband: HarmonicStopband,
    unavailable_edges: Sequence[tuple[float, float]],
    search_hz: float,
    spectral_resolution_hz: float,
) -> tuple[float, float]:
    """Frequency and narrow-line contrast of the largest available adjacent peak."""
    search = (frequencies_hz >= stopband.low_hz - search_hz) & (
        frequencies_hz <= stopband.high_hz + search_hz
    )
    unavailable = np.zeros(frequencies_hz.shape, dtype=bool)
    for low_hz, high_hz in unavailable_edges:
        if not np.all(np.isfinite((low_hz, high_hz))) or high_hz <= low_hz:
            raise ValueError("Unavailable intervals must be finite and increasing.")
        unavailable |= (frequencies_hz >= low_hz) & (frequencies_hz <= high_hz)
    available = ~unavailable
    candidate_indices = np.flatnonzero(search & available)
    offset_bins = int(
        np.ceil(spectral_resolution_hz / (frequencies_hz[1] - frequencies_hz[0]))
    )
    valid = (candidate_indices >= offset_bins) & (
        candidate_indices < frequencies_hz.size - offset_bins
    )
    candidate_indices = candidate_indices[valid]
    shoulders_available = (
        ~unavailable[candidate_indices - offset_bins]
        & ~unavailable[candidate_indices + offset_bins]
    )
    candidate_indices = candidate_indices[shoulders_available]
    candidates = (
        spectrum_db[candidate_indices]
        - 0.5
        * (
            spectrum_db[candidate_indices - offset_bins]
            + spectrum_db[candidate_indices + offset_bins]
        )
    )
    if candidates.size == 0:
        return float("nan"), float("nan")
    selected = int(candidate_indices[int(np.argmax(candidates))])
    return float(frequencies_hz[selected]), float(candidates[int(np.argmax(candidates))])


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
    original_frequencies_hz, original_spectrum_db = recordings.run_spectrum(
        original,
        settings,
    )
    cleaned_frequencies_hz, cleaned_spectrum_db = recordings.run_spectrum(
        cleaned,
        settings,
    )
    if not np.array_equal(original_frequencies_hz, cleaned_frequencies_hz):
        raise ValueError("Source and cleaned prominence spectra use different grids.")

    fundamental_hz = float(manifest_rows[0]["fundamental_hz"])
    unavailable_edges = plan.unavailable_edges()
    rows = []
    for stopband, unavailable, change_db in zip(
        plan.stopbands,
        unavailable_edges,
        changes_db,
    ):
        original_peak_hz, original_peak_db = _adjacent_residual_peak(
            original_frequencies_hz,
            original_spectrum_db,
            stopband,
            unavailable_edges,
            fundamental_hz / 2.0,
            settings.spectral_resolution_hz,
        )
        cleaned_peak_hz, cleaned_peak_db = _adjacent_residual_peak(
            cleaned_frequencies_hz,
            cleaned_spectrum_db,
            stopband,
            unavailable_edges,
            fundamental_hz / 2.0,
            settings.spectral_resolution_hz,
        )
        rows.append(
            {
                "recording": source_vhdr.stem,
                "kind": stopband.kind,
                "harmonics": ";".join(str(value) for value in stopband.harmonics),
                "stopband_low_hz": stopband.low_hz,
                "stopband_high_hz": stopband.high_hz,
                "unavailable_low_hz": unavailable[0],
                "unavailable_high_hz": unavailable[1],
                "verified_stopband_change_db": change_db,
                "original_adjacent_peak_hz": original_peak_hz,
                "original_adjacent_line_contrast_db": original_peak_db,
                "cleaned_adjacent_peak_hz": cleaned_peak_hz,
                "cleaned_adjacent_line_contrast_db": cleaned_peak_db,
                "adjacent_max_line_contrast_change_db": cleaned_peak_db - original_peak_db,
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
    runs = recordings.discover_runs(source_root, subjects=None, task="*")
    manifest_path = cleaned_root / "harmonic_notch_manifest.tsv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No harmonic notch manifest at {manifest_path}. Run `decomb apply` first."
        )
    manifest = pd.read_csv(manifest_path, sep="\t", float_precision="round_trip")
    required = {
        "recording",
        "kind",
        "harmonics",
        "fundamental_hz",
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
