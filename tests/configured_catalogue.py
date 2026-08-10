"""Catalogue facade using the packaged YAML settings in focused numerical tests."""

from __future__ import annotations

from types import SimpleNamespace

from decomb import catalogue as implementation
from decomb.config import DEFAULTS_PATH, load_config

SETTINGS = implementation.DetectionSettings.from_config(load_config(DEFAULTS_PATH))


def half_width_bins(frequencies_hz):
    return implementation.half_width_bins(
        frequencies_hz,
        SETTINGS.background_half_width_hz,
    )


def detection_mask(frequencies_hz, **kwargs):
    return implementation.detection_mask(
        frequencies_hz,
        low_hz=SETTINGS.low_hz,
        high_hz=SETTINGS.high_hz,
        **kwargs,
    )


def build_grid(frequencies_hz, subject_power):
    return implementation.build_grid(
        frequencies_hz,
        subject_power,
        SETTINGS.background_half_width_hz,
    )


def detect_cohort_lines(grid, **kwargs):
    return implementation.detect_cohort_lines(grid, SETTINGS, **kwargs)


def comb_structure(lines, **kwargs):
    return implementation.comb_structure(lines, SETTINGS, **kwargs)


def classify_lines(lines, structure):
    return implementation.classify_lines(lines, structure, SETTINGS)


def band_impact(grid, subjects, lines, bands):
    return implementation.band_impact(grid, subjects, lines, bands, SETTINGS)


catalogue = SimpleNamespace(
    Grid=implementation.Grid,
    NoLinesDetected=implementation.NoLinesDetected,
    band_impact=band_impact,
    build_grid=build_grid,
    classify_lines=classify_lines,
    comb_structure=comb_structure,
    detect_cohort_lines=detect_cohort_lines,
    detection_mask=detection_mask,
    half_width_bins=half_width_bins,
)
