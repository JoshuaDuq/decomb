"""Remove authorized lines by fitting and subtracting them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from decomb import notch, recovery


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


def fit_window_s(settings) -> float:
    """Subtraction fits on twice the detection window, halving the damage it declares."""
    return 2.0 * float(settings.estimation_window_s)


def damage_intervals(
    frequencies: Sequence[float],
    window_s: float,
) -> tuple[tuple[float, float], ...]:
    """Merged intervals a multitaper fit at this window destroys, two bins each side."""
    half_width_hz = 2.0 / float(window_s)
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

    def manifest_rows(
        self,
        recording: str,
        analysed_bands: tuple[tuple[str, float, float], ...],
        settings,
    ) -> list[dict[str, float | str]]:
        """One row per merged interval, in the same contract as a stopband."""
        intervals = damage_intervals(self.frequencies_hz, self.window_s)
        shares = notch.band_availability_from_intervals(intervals, analysed_bands)
        rows = []
        for low_hz, high_hz in intervals:
            covered = [
                frequency
                for frequency in self.frequencies_hz
                if low_hz <= frequency <= high_hz
            ]
            rows.append(
                {
                    **_inapplicable_manifest_fields(settings),
                    "recording": recording,
                    "kind": "subtracted",
                    "subtracted_frequencies_hz": ";".join(
                        str(frequency) for frequency in covered
                    ),
                    "recovery_window_s": self.window_s,
                    "unavailable_low_hz": low_hz,
                    "unavailable_high_hz": high_hz,
                    **shares,
                }
            )
        return rows


def subtract_authorized(raw, evidence, settings, *, n_jobs: int = -1):
    """Fit and remove every authorized frequency, returning the record."""
    import mne

    from decomb import residual

    frequencies = residual.subtraction_targets(raw, evidence, settings)
    record = SubtractionRecord(frequencies, fit_window_s(settings))
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


def _inapplicable_manifest_fields(settings) -> dict[str, float | str]:
    """Every manifest column a subtraction row must carry but does not populate."""
    fields = dict.fromkeys(notch.MANIFEST_REQUIRED_COLUMNS, "")
    fields["outcome"] = "line_subtracted"
    fields["scanner_repetition_time_s"] = settings.scanner_repetition_time_s
    fields["scanner_trigger_event_name"] = settings.scanner_trigger_event_name
    fields["familywise_error_rate"] = settings.familywise_error_rate
    fields["in_stopband_change_db"] = ""
    return fields


def subtraction_rows(rows: Sequence[dict]) -> list[dict]:
    """The subtraction rows of a manifest, empty for manifests written before it."""
    return [row for row in rows if str(row.get("kind", "")) == "subtracted"]


STAGE_KINDS = frozenset({"subtracted", "threshold_notched"})


def cascade_rows(rows: Sequence[dict]) -> list[dict]:
    """The converged-round rows of a manifest, which alone declare removal rounds."""
    return [row for row in rows if str(row.get("kind", "")) not in STAGE_KINDS]


def recorded_frequencies(rows: Sequence[dict]) -> tuple[float, ...]:
    """Every frequency a manifest's subtraction rows say was removed."""
    frequencies = [
        float(value)
        for row in rows
        for value in str(row["subtracted_frequencies_hz"]).split(";")
        if value
    ]
    return tuple(sorted(set(frequencies)))
