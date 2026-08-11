"""Strict BrainVision BIDS I/O and spectra shared by the public stages."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from decomb import spectral


@dataclass(frozen=True)
class SessionRunSpectra:
    """Whole-recording and overlapping-window evidence on one frequency grid."""

    whole: tuple[np.ndarray, np.ndarray]
    windows: tuple[tuple[np.ndarray, np.ndarray], ...]
    bounds: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.windows or len(self.windows) != len(self.bounds):
            raise ValueError("Every non-empty spectrum window requires one sample bound.")


def read_bids_raw(vhdr_path: Path):
    """Read one BIDS recording with strict sidecar-derived metadata."""
    from mne_bids import get_bids_path_from_fname, read_raw_bids

    bids_path = get_bids_path_from_fname(vhdr_path)
    return read_raw_bids(
        bids_path,
        extra_params={"preload": True},
        on_ch_mismatch="raise",
        verbose="ERROR",
    )


def discover_runs(
    bids_root: Path,
    subjects: list[str] | None,
    task: str,
) -> list[Path]:
    """Find BrainVision EEG recordings with optional BIDS run and session entities."""
    patterns = (
        f"sub-*/eeg/sub-*_task-{task}_*eeg.vhdr",
        f"sub-*/ses-*/eeg/sub-*_task-{task}_*eeg.vhdr",
    )
    paths = sorted({path for pattern in patterns for path in bids_root.glob(pattern)})
    if subjects:
        wanted = set(subjects)
        paths = [path for path in paths if subject_of(path) in wanted]
    if not paths:
        raise FileNotFoundError(
            f"No recordings of task {task!r} found under {bids_root}. decomb reads "
            "BrainVision recordings at sub-*/[ses-*/]eeg/*_eeg.vhdr."
        )
    return paths


def subject_of(path: Path) -> str:
    """Return the BIDS subject directory owning a recording."""
    for parent in path.parents:
        if parent.name.startswith("sub-"):
            return parent.name
    raise ValueError(f"{path} does not lie under a BIDS subject directory.")


def estimation_window_samples(sampling_frequency_hz: float, window_s: float) -> int:
    """Convert a duration to a valid whole-sample estimation window."""
    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0.0:
        raise ValueError("The sampling frequency must be finite and positive.")
    if not np.isfinite(window_s) or window_s <= 0.0:
        raise ValueError("The estimation window must be finite and positive.")
    samples = int(round(window_s * sampling_frequency_hz))
    if samples < 2:
        raise ValueError(
            f"estimation_window_s={window_s:g} s is under two samples at "
            f"{sampling_frequency_hz:g} Hz."
        )
    return samples


def overlapping_window_bounds(
    *,
    n_times: int,
    window_samples: int,
    overlap: float,
) -> tuple[tuple[int, int], ...]:
    """Fixed windows with configured overlap that include the recording tail."""
    if n_times < window_samples:
        raise ValueError(
            f"The recording holds {n_times} samples, fewer than one "
            f"{window_samples}-sample estimation window."
        )
    if not np.isfinite(overlap) or not 0.0 < overlap < 1.0:
        raise ValueError("Window overlap must lie strictly between zero and one.")
    hop_samples = int(round(window_samples * (1.0 - overlap)))
    if not 0 < hop_samples < window_samples:
        raise ValueError("Window overlap produces an invalid sample hop.")

    tail_start = n_times - window_samples
    starts = list(range(0, tail_start + 1, hop_samples))
    if starts[-1] != tail_start:
        if tail_start - starts[-1] < hop_samples / 2.0:
            starts[-1] = tail_start
        else:
            starts.append(tail_start)
    return tuple((start, start + window_samples) for start in starts)


def non_overlapping_window_indices(
    bounds: tuple[tuple[int, int], ...],
) -> tuple[int, ...]:
    """Greedily select independent windows, including no shifted overlapping tail."""
    selected = []
    previous_stop = -1
    for index, (start, stop) in enumerate(bounds):
        if stop <= start:
            raise ValueError("Estimation-window bounds must be increasing.")
        if start >= previous_stop:
            selected.append(index)
            previous_stop = stop
    if len(selected) < 2:
        raise ValueError("Line detection requires at least two non-overlapping windows.")
    return tuple(selected)


def session_run_spectra(raw, settings) -> SessionRunSpectra:
    """Measure channel-median Hann spectra for one run and its overlapping windows."""
    import mne

    picks = mne.pick_types(raw.info, eeg=True, exclude=())
    if len(picks) == 0:
        raise ValueError("Spectral estimation requires at least one EEG channel.")
    sampling_frequency_hz = float(raw.info["sfreq"])
    window_samples = estimation_window_samples(
        sampling_frequency_hz,
        settings.estimation_window_s,
    )
    data = raw.get_data(picks=picks)
    if not np.all(np.isfinite(data)):
        raise ValueError("EEG data must contain only finite values.")
    bounds = overlapping_window_bounds(
        n_times=data.shape[-1],
        window_samples=window_samples,
        overlap=settings.estimation_overlap,
    )
    windows = np.stack([data[:, start:stop] for start, stop in bounds], axis=1)
    frequencies_hz, power = spectral.hann_periodogram(
        windows,
        sampling_frequency_hz,
    )
    whole_db = spectral.to_db(np.median(power.mean(axis=1), axis=0))
    whole = (frequencies_hz, whole_db)
    window_spectra = tuple(
        (frequencies_hz, spectral.to_db(np.median(window_power, axis=0)))
        for window_power in np.moveaxis(power, 1, 0)
    )
    return SessionRunSpectra(whole, window_spectra, bounds)


def run_spectrum(raw, settings) -> tuple[np.ndarray, np.ndarray]:
    """Return the whole-recording spectrum used by the harmonic estimator."""
    return session_run_spectra(raw, settings).whole


def psd(raw, picks, settings) -> tuple[np.ndarray, np.ndarray]:
    """Mean Hann power across complete estimation windows."""
    return psd_array(
        raw.get_data(picks=picks),
        float(raw.info["sfreq"]),
        settings.estimation_window_s,
    )


def psd_array(
    data: np.ndarray,
    sampling_frequency_hz: float,
    window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean per-channel Hann power on the correction's estimation grid."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("PSD data must be a non-empty channel-by-time array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("PSD data must contain only finite values.")
    block_samples = estimation_window_samples(sampling_frequency_hz, window_s)
    block_count = values.shape[-1] // block_samples
    if block_count < 1:
        raise ValueError("PSD data do not contain one complete estimation window.")
    blocks = values[..., : block_count * block_samples].reshape(
        values.shape[0],
        block_count,
        block_samples,
    )
    frequencies_hz, power = spectral.hann_periodogram(
        blocks,
        sampling_frequency_hz,
    )
    return frequencies_hz, power.mean(axis=1)


def parse_channel_scaling(vhdr_path: Path) -> tuple[list[str], np.ndarray]:
    """Read channel names and binary resolutions from a BrainVision header."""
    text = vhdr_path.read_text(encoding="utf-8", errors="replace")
    binary_format = re.search(r"BinaryFormat=(\S+)", text)
    orientation = re.search(r"DataOrientation=(\S+)", text)
    if binary_format is None or binary_format.group(1) != "IEEE_FLOAT_32":
        raise ValueError(f"{vhdr_path.name}: expected IEEE_FLOAT_32 binary data.")
    if orientation is None or orientation.group(1) != "MULTIPLEXED":
        raise ValueError(f"{vhdr_path.name}: expected MULTIPLEXED data orientation.")

    names = []
    resolutions = []
    pattern = r"^Ch(\d+)=([^,\n]*),([^,\n]*),([^,\n]*),"
    for match in re.finditer(pattern, text, flags=re.MULTILINE):
        names.append(match.group(2))
        resolutions.append(float(match.group(4)))
    if not names:
        raise ValueError(f"{vhdr_path.name}: no channel definitions found.")
    return names, np.asarray(resolutions, dtype=float)


def write_eeg_binary(
    vhdr_path: Path,
    destination: Path,
    data_volts: np.ndarray,
) -> None:
    """Write multiplexed float32 samples in the layout declared by the source header."""
    values = np.asarray(data_volts, dtype=float)
    if values.ndim != 2:
        raise ValueError("EEG output must be a channel-by-time array.")
    if not np.all(np.isfinite(values)):
        bad_count = int(np.count_nonzero(~np.isfinite(values)))
        raise ValueError(f"Refusing to write {destination}: {bad_count} non-finite sample(s).")
    channel_names, resolutions = parse_channel_scaling(vhdr_path)
    if values.shape[0] != len(channel_names):
        raise ValueError(
            f"{vhdr_path.name}: header describes {len(channel_names)} channels, "
            f"got {values.shape[0]}."
        )
    scaled = (values * 1e6) / resolutions[:, np.newaxis]
    scaled.T.astype("<f4").tofile(destination)


def quantized_eeg_data(vhdr_path: Path, data_volts: np.ndarray) -> np.ndarray:
    """Values a BrainVision float32 round trip can represent, in volts."""
    values = np.asarray(data_volts, dtype=float)
    channel_names, resolutions = parse_channel_scaling(vhdr_path)
    if values.ndim != 2 or values.shape[0] != len(channel_names):
        raise ValueError("Quantization requires data matching the BrainVision channels.")
    scaled = (values * 1e6) / resolutions[:, np.newaxis]
    calibration = resolutions[:, np.newaxis] * 1e-6
    return scaled.astype("<f4").astype(float) * calibration


def mirror_sidecars(source_root: Path, output_root: Path) -> int:
    """Copy every BIDS file except the EEG binaries, which are rewritten."""
    excluded_suffixes = {".bak", ".lock", ".orig", ".tmp"}
    copied = 0
    for path in sorted(source_root.rglob("*")):
        relative_path = path.relative_to(source_root)
        suffixes = set(path.suffixes)
        hidden = any(part.startswith(".") for part in relative_path.parts)
        excluded = ".eeg" in suffixes or bool(suffixes & excluded_suffixes)
        if path.is_dir() or hidden or excluded:
            continue
        target = output_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def write_tsv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Publish a complete TSV or leave the previous table untouched."""
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
        temporary_path = Path(stream.name)
        frame.to_csv(stream, sep="\t", index=False, float_format="%.17g")
    os.replace(temporary_path, path)
