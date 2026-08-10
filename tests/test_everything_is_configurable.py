"""The automatic correction exposes every user decision and no obsolete controls."""

from __future__ import annotations

from dataclasses import MISSING, fields

import pytest
import yaml

from decomb import catalogue, notch
from decomb.config import load_config

OVERRIDES = {
    "estimation_window_s": 60.0,
    "estimation_overlap": 0.6,
    "filter_jobs": 2,
    "nominal_fundamental_hz": 1.5,
    "harmonic_range": [10, 40],
    "removal_harmonic_range": [9, 41],
    "search_hz": 0.3,
    "min_prominence_db": 1.5,
    "uncertainty_confidence_z": 1.5,
    "low_hz": 2.0,
    "high_hz": 90.0,
    "background_half_width_hz": 5.0,
    "min_harmonics_for_fit": 12,
    "max_harmonic_residual_resolutions": 2.0,
    "max_fit_residual_rms_resolutions": 1.25,
    "minimum_stopband_resolutions": 2.5,
    "transition_bandwidth_resolutions": 5.0,
    "residual_search_hz": 0.2,
    "roundtrip_relative_tolerance": 1.0e-5,
}


def _config(tmp_path, document):
    path = tmp_path / "decomb.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return load_config(path)


def test_every_notch_setting_is_covered_by_this_test():
    declared = {entry.name for entry in fields(notch.HarmonicNotchSettings)} - {"task"}

    assert declared == set(OVERRIDES)


def test_notch_settings_have_no_scientific_defaults_in_python():
    assert all(entry.default is MISSING for entry in fields(notch.HarmonicNotchSettings))


def test_detection_settings_have_no_scientific_defaults_in_python():
    assert all(entry.default is MISSING for entry in fields(catalogue.DetectionSettings))


@pytest.mark.parametrize("name", sorted(OVERRIDES))
def test_every_notch_setting_reaches_the_settings_object(tmp_path, name):
    settings = notch.HarmonicNotchSettings.from_config(
        _config(tmp_path, {"removal": {name: OVERRIDES[name]}})
    )
    expected = OVERRIDES[name]
    actual = getattr(settings, name)

    assert list(actual) == expected if isinstance(actual, tuple) else actual == expected


def test_task_is_read_from_the_dataset_block(tmp_path):
    settings = notch.HarmonicNotchSettings.from_config(
        _config(tmp_path, {"dataset": {"task": "rest"}})
    )

    assert settings.task == "rest"


def test_packaged_defaults_match_the_settings_defaults():
    config = load_config()
    settings = notch.HarmonicNotchSettings.from_config(config)
    block = config.get("removal")

    for name, expected in block.items():
        actual = getattr(settings, name)
        assert list(actual) == expected if isinstance(actual, tuple) else actual == expected


def test_obsolete_or_misspelled_correction_settings_are_errors(tmp_path):
    config = _config(tmp_path, {"removal": {"harmonic_trajectory": {"enabled": True}}})

    with pytest.raises(ValueError, match="Unknown `removal` setting"):
        notch.HarmonicNotchSettings.from_config(config)


def test_filter_geometry_in_hz_is_derived_not_configurable(tmp_path):
    config = _config(tmp_path, {"removal": {"transition_bandwidth_hz": 1.0}})

    with pytest.raises(ValueError, match="Unknown `removal` setting"):
        notch.HarmonicNotchSettings.from_config(config)
