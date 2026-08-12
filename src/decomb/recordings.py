"""Strict BrainVision BIDS I/O and spectra shared by the public stages."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from decomb import spectral

BRAINVISION_UNIT_FACTORS = {
    "V": 1.0,
    "mV": 1e-3,
    "µV": 1e-6,
    "uV": 1e-6,
    "nV": 1e-9,
    "C": 1.0,
    "°C": 1.0,
    "n/a": 1.0,
    "µS": 1e-6,
    "uS": 1e-6,
    "ARU": 1.0,
    "S": 1.0,
    "N": 1.0,
}


@dataclass(frozen=True)
class BrainVisionChannelScaling:
    """Header channel order and conversion from stored values to MNE units."""

    channel_names: tuple[str, ...]
    calibrations: tuple[float, ...]


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


def _read_brainvision_text(path: Path) -> tuple[str, str]:
    """Decode a BrainVision text file using its declared codepage."""
    payload = path.read_bytes()
    codepage_match = re.search(
        rb"^Codepage=([^\r\n]+)",
        payload,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    encoding = (
        "utf-8"
        if codepage_match is None
        else codepage_match.group(1).decode("ascii").strip()
    )
    if encoding.upper() == "ANSI":
        encoding = "cp1252"
    return payload.decode(encoding), encoding


def parse_channel_scaling(vhdr_path: Path) -> BrainVisionChannelScaling:
    """Read indexed channel names and calibrations from a BrainVision header."""
    text, _ = _read_brainvision_text(vhdr_path)
    binary_format = re.search(r"BinaryFormat=(\S+)", text)
    orientation = re.search(r"DataOrientation=(\S+)", text)
    if binary_format is None or binary_format.group(1) != "IEEE_FLOAT_32":
        raise ValueError(f"{vhdr_path.name}: expected IEEE_FLOAT_32 binary data.")
    if orientation is None or orientation.group(1) != "MULTIPLEXED":
        raise ValueError(f"{vhdr_path.name}: expected MULTIPLEXED data orientation.")

    section = re.search(
        r"^\[Channel Infos\]\s*$\s*(.*?)(?=^\[|\Z)",
        text,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if section is None:
        raise ValueError(f"{vhdr_path.name}: no channel definitions found.")

    definitions: dict[int, tuple[str, float]] = {}
    for line in section.group(1).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        match = re.fullmatch(r"Ch(\d+)=(.*)", stripped, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"{vhdr_path.name}: invalid channel definition {line!r}.")
        index = int(match.group(1))
        if index in definitions:
            raise ValueError(f"{vhdr_path.name}: duplicate channel index Ch{index}.")
        properties = match.group(2).split(",")
        if len(properties) not in {3, 4}:
            raise ValueError(f"{vhdr_path.name}: invalid Ch{index} definition.")
        name = properties[0].replace(r"\1", ",")
        resolution = 1.0 if properties[2] == "" else float(properties[2])
        unit = "µV" if len(properties) == 3 or properties[3] == "" else properties[3]
        if unit not in BRAINVISION_UNIT_FACTORS:
            raise ValueError(
                f"{vhdr_path.name}: unsupported BrainVision unit {unit!r} on Ch{index}."
            )
        calibration = resolution * BRAINVISION_UNIT_FACTORS[unit]
        if not np.isfinite(calibration) or calibration <= 0.0:
            raise ValueError(f"{vhdr_path.name}: Ch{index} calibration must be positive.")
        definitions[index] = (name, calibration)

    expected_indices = tuple(range(1, len(definitions) + 1))
    if tuple(sorted(definitions)) != expected_indices:
        raise ValueError(f"{vhdr_path.name}: channel indices must be contiguous from Ch1.")
    ordered = tuple(definitions[index] for index in expected_indices)
    return BrainVisionChannelScaling(
        channel_names=tuple(name for name, _ in ordered),
        calibrations=tuple(calibration for _, calibration in ordered),
    )


def _scaled_brainvision_data(
    vhdr_path: Path,
    data: np.ndarray,
    channel_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate MNE-order data and convert it to stored BrainVision values."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ValueError("EEG output must be a channel-by-time array.")
    if not np.all(np.isfinite(values)):
        bad_count = int(np.count_nonzero(~np.isfinite(values)))
        raise ValueError(f"Refusing to write {bad_count} non-finite sample(s).")
    scaling = parse_channel_scaling(vhdr_path)
    observed_names = tuple(channel_names)
    if observed_names != scaling.channel_names:
        raise ValueError(
            f"{vhdr_path.name}: data channel order does not match the BrainVision header."
        )
    if values.shape[0] != len(scaling.channel_names):
        raise ValueError(
            f"{vhdr_path.name}: header describes {len(scaling.channel_names)} channels, "
            f"got {values.shape[0]}."
        )
    calibrations = np.asarray(scaling.calibrations, dtype=float)[:, np.newaxis]
    return values / calibrations, calibrations


def write_eeg_binary(
    vhdr_path: Path,
    destination: Path,
    data: np.ndarray,
    channel_names: Sequence[str],
) -> None:
    """Write multiplexed float32 samples in the layout declared by the source header."""
    scaled, _ = _scaled_brainvision_data(vhdr_path, data, channel_names)
    scaled.T.astype("<f4").tofile(destination)


def quantized_eeg_data(
    vhdr_path: Path,
    data: np.ndarray,
    channel_names: Sequence[str],
) -> np.ndarray:
    """Values a BrainVision float32 round trip can represent in MNE units."""
    scaled, calibrations = _scaled_brainvision_data(vhdr_path, data, channel_names)
    return scaled.astype("<f4").astype(float) * calibrations


def derivative_vhdr_path(
    source_vhdr: Path,
    source_root: Path,
    derivative_root: Path,
) -> Path:
    """Map one raw BrainVision header to its BIDS derivative name."""
    relative_path = source_vhdr.relative_to(source_root)
    if source_vhdr.suffix != ".vhdr" or not source_vhdr.stem.endswith("_eeg"):
        raise ValueError(f"{source_vhdr.name}: expected a raw BIDS *_eeg.vhdr filename.")
    if "_desc-" in source_vhdr.stem:
        raise ValueError(f"{source_vhdr.name}: raw input must not contain a desc entity.")
    source_entities = source_vhdr.stem.removesuffix("_eeg")
    derivative_name = f"{source_entities}_desc-decomb_eeg.vhdr"
    return derivative_root / relative_path.with_name(derivative_name)


def _brainvision_reference(text: str, key: str, path: Path) -> str:
    pattern = re.compile(rf"^{re.escape(key)}=([^\r\n]+)$", re.IGNORECASE | re.MULTILINE)
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise ValueError(f"{path.name}: expected exactly one {key} reference.")
    reference = matches[0].strip()
    if Path(reference).name != reference:
        raise ValueError(f"{path.name}: {key} must name a file in the same directory.")
    return reference


def _replace_brainvision_reference(text: str, key: str, filename: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}=[^\r\n]+$", re.IGNORECASE | re.MULTILINE)
    return pattern.sub(f"{key}={filename}", text, count=1)


def write_brainvision_sidecars(source_vhdr: Path, destination_vhdr: Path) -> None:
    """Copy a BrainVision header and marker under one internally consistent stem."""
    if destination_vhdr.suffix != ".vhdr" or not destination_vhdr.stem.endswith("_eeg"):
        raise ValueError("The derivative BrainVision header must end in _eeg.vhdr.")

    header, header_encoding = _read_brainvision_text(source_vhdr)
    source_data_name = _brainvision_reference(header, "DataFile", source_vhdr)
    source_marker_name = _brainvision_reference(header, "MarkerFile", source_vhdr)
    source_marker = source_vhdr.parent / source_marker_name
    if not (source_vhdr.parent / source_data_name).is_file():
        raise FileNotFoundError(f"{source_vhdr.name}: referenced data file does not exist.")
    if not source_marker.is_file():
        raise FileNotFoundError(f"{source_vhdr.name}: referenced marker file does not exist.")

    marker, marker_encoding = _read_brainvision_text(source_marker)
    marker_data_name = _brainvision_reference(marker, "DataFile", source_marker)
    if marker_data_name != source_data_name:
        raise ValueError(
            f"{source_marker.name}: DataFile does not match the BrainVision header."
        )

    destination_data_name = destination_vhdr.with_suffix(".eeg").name
    destination_marker = destination_vhdr.with_suffix(".vmrk")
    rewritten_header = _replace_brainvision_reference(
        header,
        "DataFile",
        destination_data_name,
    )
    rewritten_header = _replace_brainvision_reference(
        rewritten_header,
        "MarkerFile",
        destination_marker.name,
    )
    rewritten_marker = _replace_brainvision_reference(
        marker,
        "DataFile",
        destination_data_name,
    )

    destination_vhdr.parent.mkdir(parents=True, exist_ok=True)
    destination_vhdr.write_text(rewritten_header, encoding=header_encoding)
    destination_marker.write_text(rewritten_marker, encoding=marker_encoding)


def mirror_sidecars(source_root: Path, output_root: Path) -> int:
    """Copy BIDS metadata except BrainVision triplets, which are rewritten."""
    excluded_suffixes = {".bak", ".lock", ".orig", ".tmp"}
    brainvision_suffixes = {".vhdr", ".vmrk", ".eeg"}
    copied = 0
    for path in sorted(source_root.rglob("*")):
        relative_path = path.relative_to(source_root)
        suffixes = set(path.suffixes)
        hidden = any(part.startswith(".") for part in relative_path.parts)
        excluded = bool(suffixes & (excluded_suffixes | brainvision_suffixes))
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
