"""Only irreducible user decisions belong in the public YAML."""

from __future__ import annotations

from dataclasses import fields

import pytest
import yaml

from decomb import notch
from decomb.config import load_config


def _config(tmp_path, document):
    path = tmp_path / "decomb.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return load_config(path)


def test_correction_config_contains_only_irreducible_user_choices():
    settings = notch.HarmonicNotchSettings.from_config(load_config())

    assert {field.name for field in fields(settings)} == {
        "estimation_window_s",
        "frequency_range_hz",
    }


@pytest.mark.parametrize(
    "obsolete",
    [
        "nominal_fundamental_hz",
        "harmonic_range",
        "removal_harmonic_range",
        "min_prominence_db",
        "min_harmonics_for_fit",
        "search_hz",
        "filter_jobs",
        "transition_bandwidth_resolutions",
        "roundtrip_relative_tolerance",
    ],
)
def test_obsolete_manual_controls_are_errors(tmp_path, obsolete):
    with pytest.raises(ValueError, match="Unknown `removal` setting"):
        _config(tmp_path, {"removal": {obsolete: 1}})


def test_threshold_blocks_are_absent_from_packaged_defaults():
    config = load_config()

    assert config.get("detection") is None
    assert config.get("psd") is None
    assert config.get("dataset") is None


def test_frequency_bands_remain_configurable_because_they_are_study_definitions(tmp_path):
    config = _config(tmp_path, {"frequency_bands": {"custom": [7.0, 11.0]}})

    assert config.get("frequency_bands.custom") == [7.0, 11.0]


def test_detection_frequency_range_is_a_deliberate_user_choice(tmp_path):
    settings = notch.HarmonicNotchSettings.from_config(
        _config(tmp_path, {"removal": {"frequency_range_hz": [1.0, 80.0]}})
    )

    assert settings.frequency_range_hz == (1.0, 80.0)


@pytest.mark.parametrize("value", [50.0, [0.0], [0.0, 50.0, 100.0], [80.0, 20.0]])
def test_malformed_frequency_ranges_fail_at_config_loading(tmp_path, value):
    with pytest.raises(ValueError, match="frequency_range_hz"):
        notch.HarmonicNotchSettings.from_config(
            _config(tmp_path, {"removal": {"frequency_range_hz": value}})
        )


def test_removed_threshold_sections_fail_during_config_loading(tmp_path):
    path = tmp_path / "decomb.yaml"
    path.write_text("detection:\n  fdr_alpha: 0.05\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config section"):
        load_config(path)


def test_removed_simulation_section_fails_during_config_loading(tmp_path):
    path = tmp_path / "decomb.yaml"
    path.write_text("simulation:\n  duration_s: 120\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config section"):
        load_config(path)
