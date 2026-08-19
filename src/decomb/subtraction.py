"""Remove authorized lines by fitting and subtracting them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from decomb import recovery


def authorized_frequencies(evidence, settings) -> tuple[float, ...]:
    """Every frequency this round's evidence authorizes removing."""
    frequencies = [
        line.position_hz
        for channel in evidence.model.channels
        for line in channel.lines
    ]
    scanner = getattr(evidence, "scanner_harmonics", None)
    if scanner is not None:
        frequencies.extend(
            harmonic * scanner.fundamental_hz
            for harmonic in scanner.supporting_harmonics
        )
    return tuple(sorted(set(frequencies)))


def damage_intervals(
    frequencies: Sequence[float],
    settings,
) -> tuple[tuple[float, float], ...]:
    """Merged intervals a multitaper subtraction destroys, two bins each side."""
    half_width_hz = 2.0 * settings.frequency_bin_width_hz
    merged: list[list[float]] = []
    for centre_hz in sorted(float(value) for value in frequencies):
        low_hz, high_hz = centre_hz - half_width_hz, centre_hz + half_width_hz
        if merged and low_hz <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high_hz)
        else:
            merged.append([low_hz, high_hz])
    return tuple((low_hz, high_hz) for low_hz, high_hz in merged)


@dataclass(frozen=True)
class SubtractionRecord:
    """What one recording's subtraction removed, and at what resolution."""

    frequencies_hz: tuple[float, ...]
    window_s: float


def subtract_authorized(raw, evidence, settings, *, n_jobs: int = -1):
    """Fit and remove every authorized frequency, returning the record."""
    import mne

    frequencies = authorized_frequencies(evidence, settings)
    record = SubtractionRecord(frequencies, float(settings.estimation_window_s))
    if not frequencies:
        return raw.copy(), record
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    result = recovery.subtract_multitaper_sinusoids(
        raw.get_data(picks=picks),
        float(raw.info["sfreq"]),
        frequencies,
        window_s=record.window_s,
        n_jobs=n_jobs,
    )
    cleaned = raw.copy()
    cleaned._data[picks] = result.cleaned_data
    return cleaned, record
