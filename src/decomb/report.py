"""Summarise what the removal achieved, band by band.

    decomb report

Reads only what ``apply`` and ``verify`` wrote -- each recording's own transform
provenance and verification spectra -- and turns them into outcome tables and a figure.
Each participant is scored against the targets that were actually removed from their
data, not against a cohort-wide list, because the target set differs between recordings.

The bands are the configuration's own ``frequency_bands``, so the outcome table speaks in
the units the study's analyses do, and the mains band comes from the removal settings
rather than being assumed.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from decomb import catalogue, estimators, spectral  # noqa: E402
from decomb.remove import (  # noqa: E402
    RemovalSettings,
    _write_tsv_atomic,
    discover_runs,
    settings_fingerprint,
    source_input_digests,
)


def _artifact_frequencies(cell) -> tuple[float, ...]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return ()
    values = tuple(float(piece) for piece in str(cell).split(";") if piece.strip())
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Removal manifest contains a non-finite artifact frequency.")
    return values


def subject_artifact_targets(
    manifest: pd.DataFrame,
    subjects: tuple[str, ...],
    settings: RemovalSettings,
) -> dict[str, tuple[float, ...]]:
    """Artifact frequencies actually authorised for each participant."""
    required = {"recording", "fundamental_hz", "isolated_hz", "adjacent_hz"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Removal manifest is missing columns: {sorted(required - set(manifest))}")

    targets = {}
    for subject in subjects:
        rows = manifest[manifest.recording.astype(str).str.startswith(f"{subject}_")]
        if rows.empty:
            raise ValueError(f"Removal manifest has no manifest rows for {subject}.")
        frequencies = []
        for row in rows.itertuples(index=False):
            fundamental_hz = float(row.fundamental_hz)
            if not np.isfinite(fundamental_hz) or fundamental_hz <= 0.0:
                raise ValueError(f"Removal manifest has an invalid fundamental for {subject}.")
            frequencies.extend(
                harmonic * fundamental_hz
                for harmonic in range(
                    settings.removal_harmonic_range[0],
                    settings.removal_harmonic_range[1] + 1,
                )
            )
            frequencies.extend(_artifact_frequencies(row.isolated_hz))
            frequencies.extend(_artifact_frequencies(row.adjacent_hz))
        targets[subject] = tuple(
            sorted(
                {
                    frequency
                    for frequency in frequencies
                    if settings.low_hz <= frequency <= settings.high_hz
                    and not any(
                        low <= frequency <= high for low, high in settings.protected_bands_hz
                    )
                }
            )
        )
    return targets


def _line_sources(frequencies: tuple[float, ...]) -> tuple[float, ...]:
    """Collapse within-source drift while preserving neighbouring physical lines."""
    clusters: list[list[float]] = []
    for frequency in sorted(frequencies):
        eligible = [
            cluster
            for cluster in clusters
            if abs(frequency - float(np.median(cluster))) <= estimators.LINE_CLAIM_HZ
        ]
        if eligible:
            nearest = min(
                eligible,
                key=lambda cluster: abs(frequency - float(np.median(cluster))),
            )
            nearest.append(frequency)
        else:
            clusters.append([frequency])
    return tuple(float(np.median(cluster)) for cluster in clusters)


def artifact_share_by_subject(
    freqs,
    psd,
    band,
    subject_targets: dict[str, tuple[float, ...]],
    subjects: tuple[str, ...],
    half_width_bins: int,
    line_half_width_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Line-excess share using each participant's own detected target set."""
    if psd.shape[0] != len(subjects):
        raise ValueError("The spectrum rows must align one-to-one with subjects.")
    shares = []
    counts = []
    for subject, spectrum in zip(subjects, psd):
        lines = tuple(
            frequency for frequency in subject_targets[subject] if band[0] <= frequency <= band[1]
        )
        counts.append(len(_line_sources(lines)))
        shares.append(
            spectral.line_excess_fraction(
                freqs,
                spectrum,
                low_hz=band[0],
                high_hz=band[1],
                line_freqs=lines,
                half_width_bins=half_width_bins,
                line_half_width_hz=line_half_width_hz,
            )
            if lines
            else 0.0
        )
    return np.asarray(shares), np.asarray(counts, dtype=int)


def _per_subject_residuals(
    freqs: np.ndarray,
    cleaned: np.ndarray,
    subjects: tuple[str, ...],
    subject_targets: dict[str, tuple[float, ...]],
    half_width_bins: int,
    max_line_width_hz: float,
    residual_search_hz: float,
) -> pd.DataFrame:
    rows = []
    for subject, spectrum in zip(subjects, cleaned):
        prominence = spectral.prominence_db(
            spectral.to_db(spectrum), half_width_bins=half_width_bins
        )
        narrow = estimators._narrow_peak_mask(
            freqs, prominence, max_line_width_hz=max_line_width_hz
        )
        for frequency in _line_sources(subject_targets[subject]):
            inside = np.abs(freqs - frequency) <= residual_search_hz
            candidates = np.flatnonzero(inside & narrow & np.isfinite(prominence))
            index = (
                int(candidates[np.argmax(prominence[candidates])])
                if candidates.size
                else int(np.argmin(np.abs(freqs - frequency)))
            )
            rows.append(
                {
                    "subject": subject,
                    "target_frequency_hz": frequency,
                    "residual_frequency_hz": float(freqs[index]),
                    "residual_prominence_db": float(prominence[index]),
                }
            )
    return pd.DataFrame(rows)


def build_report(
    removal_dir: Path,
    source_root: Path,
    settings: RemovalSettings,
    bands: Mapping[str, Sequence[float]],
) -> dict:
    with np.load(removal_dir / "verification_spectra.npz", allow_pickle=False) as handle:
        freqs = handle["freqs"]
        original = handle["original"]
        cleaned = handle["cleaned"]
        subjects = tuple(str(value) for value in handle["subjects"])
        recorded_fingerprint = str(handle["settings_fingerprint"].item())
        recorded_source_digest = str(handle["source_digest"].item())
    expected_fingerprint = settings_fingerprint(settings)
    if recorded_fingerprint != expected_fingerprint:
        raise ValueError(
            "Verification spectra were produced under different settings; rerun `verify`."
        )
    runs = discover_runs(source_root, subjects=None, task=settings.task)
    input_digests, source_digest = source_input_digests(runs, source_root)
    if recorded_source_digest != source_digest:
        raise ValueError("Verification spectra are stale for the current BIDS source dataset.")
    manifest = pd.read_csv(removal_dir / "removal_manifest.tsv", sep="\t")
    if set(manifest.get("settings_fingerprint", ())) != {expected_fingerprint}:
        raise ValueError("Removal manifest does not match the current settings.")
    if set(manifest.get("input_digest", ())) != set(input_digests.values()):
        raise ValueError("Removal manifest does not match the verified BIDS source dataset.")
    subject_targets = subject_artifact_targets(manifest, subjects, settings)

    half_width = catalogue.half_width_bins(freqs, settings.background_half_width_hz)
    rows = []
    for name, (low, high) in sorted(bands.items(), key=lambda item: item[1][0]):
        band = (float(low), float(high))
        before, counts = artifact_share_by_subject(
            freqs,
            original,
            band,
            subject_targets,
            subjects,
            half_width,
            settings.line_claim_hz,
        )
        after, _ = artifact_share_by_subject(
            freqs,
            cleaned,
            band,
            subject_targets,
            subjects,
            half_width,
            settings.line_claim_hz,
        )
        rows.append(
            {
                "band": name,
                "low_hz": band[0],
                "high_hz": band[1],
                "median_artifact_sources": float(np.median(counts)),
                "min_artifact_sources": int(np.min(counts)),
                "max_artifact_sources": int(np.max(counts)),
                "artifact_share_before": float(np.median(before)),
                "artifact_share_before_max": float(np.max(before)),
                "artifact_share_after": float(np.median(after)),
                "artifact_share_after_max": float(np.max(after)),
            }
        )

    per_subject_lines = _per_subject_residuals(
        freqs,
        cleaned,
        subjects,
        subject_targets,
        half_width,
        settings.max_line_width_hz,
        settings.residual_search_hz,
    )
    return {
        "bands": pd.DataFrame(rows),
        "per_subject_lines": per_subject_lines,
        "manifest": manifest,
        "freqs": freqs,
        "original": original,
        "cleaned": cleaned,
        "subjects": subjects,
        "subject_targets": subject_targets,
    }


def figure(
    report: dict,
    path: Path,
    protected_bands_hz: Sequence[tuple[float, float]],
    background_half_width_hz: float,
) -> None:
    """Cohort spectra before and after, with the bands another stage owns shaded out."""
    freqs, original, cleaned = report["freqs"], report["original"], report["cleaned"]
    inside = (freqs >= 3) & (freqs <= 100)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, height_ratios=[2, 1])

    for data, colour, label in (
        (original, "#C1442E", "before removal"),
        (cleaned, "#111827", "after removal"),
    ):
        axes[0].plot(
            freqs[inside],
            spectral.to_db(np.median(data, axis=0))[inside],
            color=colour,
            lw=0.6,
            label=label,
        )
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("cohort median PSD (dB re 1 V²/Hz)")
    axes[0].set_title(
        "Line removal across participant-median spectra. Grey: bands this pass left to a "
        "wide notch."
    )

    half_width = catalogue.half_width_bins(freqs, background_half_width_hz)
    for data, colour, label in ((original, "#C1442E", "before"), (cleaned, "#111827", "after")):
        prominence = np.median(
            np.stack(
                [
                    spectral.prominence_db(spectral.to_db(row), half_width_bins=half_width)
                    for row in data
                ]
            ),
            axis=0,
        )
        axes[1].plot(freqs[inside], prominence[inside], color=colour, lw=0.6, label=label)
    axes[1].axhline(0, color="#6B7280", lw=0.6, ls="--")
    for axis in axes:
        for low, high in protected_bands_hz:
            axis.axvspan(low, high, color="#E5E7EB", zorder=0)
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("local prominence (dB)")
    axes[1].set_xlim(3, 100)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    """Build the outcome tables and the before-and-after figure."""
    from decomb.config import load_config

    config = load_config(getattr(args, "config", None))
    args.removal_dir = config.path("removal_dir", override=args.removal_dir)
    source_root = config.path("bids_root")

    settings = RemovalSettings.from_config(config)
    bands = config.get("frequency_bands") or {}
    if not bands:
        raise ValueError("`frequency_bands` is empty, so there is nothing to report against.")
    report = build_report(args.removal_dir, source_root, settings, bands)
    _write_tsv_atomic(report["bands"], args.removal_dir / "band_outcomes.tsv")
    _write_tsv_atomic(
        report["per_subject_lines"], args.removal_dir / "per_subject_line_residual.tsv"
    )
    figure(
        report,
        args.removal_dir / "removal_before_after.png",
        settings.protected_bands_hz,
        settings.background_half_width_hz,
    )

    frame = report["bands"]
    print(f"{'band':20s} {'sources':>9s} {'before':>9s} {'after':>9s}")
    for _, row in frame.iterrows():
        print(
            f"{row['band']:20s} {row['median_artifact_sources']:8.1f} "
            f"{100 * row['artifact_share_before']:8.2f}% "
            f"{100 * row['artifact_share_after']:8.2f}%"
        )
    per_line = report["per_subject_lines"]
    print(
        f"\nparticipant-specific line positions still above 1 dB: "
        f"{int((per_line.residual_prominence_db > 1).sum())}/{len(per_line)}"
    )
    print(per_line.nlargest(10, "residual_prominence_db").to_string(index=False))
    print(
        f"\n  wrote {args.removal_dir / 'band_outcomes.tsv'}, "
        "per_subject_line_residual.tsv, removal_before_after.png"
    )
