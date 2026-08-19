"""Remove authorized lines by fitting and subtracting them."""

from __future__ import annotations


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
