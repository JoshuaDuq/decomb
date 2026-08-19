"""Remove authorized lines by fitting and subtracting them."""

from __future__ import annotations

from collections.abc import Sequence


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
