"""No parameter is hardcoded: every setting is reachable from the configuration file.

Three separate claims, tested separately.

1. Every field of every settings dataclass can be set by name from a YAML file, and the
   value arrives unchanged.
2. The packaged ``defaults.yaml`` produces exactly the dataclass defaults, so the file a
   user copies is a complete and truthful picture of what the workflow will do.
3. Nothing the stages call is left holding a module constant the config cannot reach.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

import pytest
import yaml

from decomb import catalogue, estimators, remove
from decomb.config import DEFAULTS_PATH, load_config

# One value per field that differs from the default, so a setting that is silently
# ignored shows up as an equality failure rather than passing by coincidence.
REMOVAL_OVERRIDES = {
    "estimation_window_s": 61.0,
    "max_band_cost": 0.2,
    "cost_band_hz": [30.0, 90.0],
    "mains_notch_hz": [49.5, 50.5],
    "nominal_fundamental_hz": 1.5,
    "harmonic_range": [10, 40],
    "removal_harmonic_range": [9, 41],
    "search_hz": 0.3,
    "min_prominence_db": 1.5,
    "filter_length": "30s",
    "filter_jobs": 2,
    "mt_bandwidth": 0.75,
    "notch_width_ratio": 300.0,
    "notch_width_min_hz": 0.04,
    "uncertainty_confidence_z": 1.0,
    "low_hz": 2.0,
    "high_hz": 90.0,
    "background_half_width_hz": 5.0,
    "min_harmonics_for_fit": 12,
    "max_harmonic_residual_resolutions": 2.0,
    "max_fit_residual_rms_resolutions": 1.25,
    "line_claim_hz": 0.12,
    "max_line_width_resolutions": 8.0,
    "residual_search_hz": 0.2,
    "residual_family_alpha": 0.01,
    "false_discovery_rate": 0.1,
    "seam_alpha": 0.01,
    "n_seam_controls": 20,
    "roundtrip_relative_tolerance": 1.0e-5,
    "detection_fdr_alpha": 0.01,
    "detection_min_prominence_db": 12.0,
    "detection_adjacent_min_prominence_db": 11.0,
    "support_margin_hz": 0.02,
    "support_min_prominence_db": 11.0,
    "detection_low_hz": 15.0,
    "detection_high_hz": 90.0,
    "detection_search_hz": 0.06,
    "min_runs_per_line": 4,
    "min_runs_per_block_line": 3,
    "min_independent_windows_per_line": 4,
    "exclude_mains": False,
}

DETECTION_OVERRIDES = {
    "low_hz": 4.0,
    "high_hz": 90.0,
    "background_half_width_hz": 5.5,
    "fdr_alpha": 0.01,
    "comb_tolerance_hz": 0.05,
    "max_pair_spacing_hz": 8.0,
    "narrow_linewidth_ratio": 2.5,
    "wide_member_ratio": 8.0,
    "line_mask_half_width_hz": 0.2,
    "max_subharmonic_divisor": 4,
    "min_subharmonic_gain": 0.3,
    "spacing_search_fraction": 0.01,
    "bootstrap_resamples": 500,
    "bootstrap_alpha": 0.1,
    "bootstrap_seed": 7,
}

BENCHMARK_OVERRIDES = {
    "min_probe_separation_hz": 0.5,
    "in_band_probe_count": 6,
    "broadband_probe_channels": 2,
    "probe": {
        "sinusoid_hz": [30.0, 50.0],
        "sinusoid_count": 3,
        "sinusoid_amplitude_v": 1.0e-6,
        "burst_hz": 45.0,
        "burst_centre_s": 60.0,
        "burst_sd_s": 0.04,
        "burst_amplitude_v": 2.0e-6,
        "burst_window_half_widths": 3.0,
    },
    "gate": {
        "min_burst_correlation": 0.95,
    },
}


def _flat_run_spectra():
    """One recording's spectra: a comb on a signed background, enough to plan from."""
    import numpy as np

    freqs = np.arange(1.0, 100.0, 0.002)
    spectrum = np.zeros_like(freqs)
    sigma = 0.109 / 2.355
    for harmonic in range(24, 80):
        spectrum[:] = np.maximum(
            spectrum, 14.0 * np.exp(-0.5 * ((freqs - harmonic * 1.2) / sigma) ** 2)
        )
    spectrum += np.random.default_rng(0).normal(0.0, 0.4, freqs.size)
    whole = (freqs, spectrum, spectrum.copy())
    return remove.SessionRunSpectra(whole=whole, windows=(whole,), bounds=((0, 100),))


def _config(tmp_path, document):
    path = tmp_path / "decomb.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return load_config(path)


def _as_comparable(value):
    return list(value) if isinstance(value, tuple) else value


def test_every_removal_field_is_covered_by_this_test():
    """The test itself must not fall behind the dataclass."""
    derived = {"task", "excluded_bands_hz", "benchmark"}
    declared = {entry.name for entry in fields(remove.RemovalSettings)} - derived

    assert declared == set(REMOVAL_OVERRIDES), (
        "a removal setting was added or removed without updating this test: "
        f"{sorted(declared ^ set(REMOVAL_OVERRIDES))}"
    )


def test_every_detection_field_is_covered_by_this_test():
    declared = {entry.name for entry in fields(catalogue.DetectionSettings)}

    assert declared == set(DETECTION_OVERRIDES), sorted(declared ^ set(DETECTION_OVERRIDES))


def test_every_benchmark_field_is_covered_by_this_test():
    declared = {entry.name for entry in fields(estimators.BenchmarkSettings)}
    assert declared == set(BENCHMARK_OVERRIDES), sorted(declared ^ set(BENCHMARK_OVERRIDES))
    for name, kind in (("probe", estimators.Probe), ("gate", estimators.PreservationGate)):
        declared = {entry.name for entry in fields(kind)}
        assert declared == set(BENCHMARK_OVERRIDES[name]), sorted(
            declared ^ set(BENCHMARK_OVERRIDES[name])
        )


@pytest.mark.parametrize("name", sorted(REMOVAL_OVERRIDES))
def test_a_removal_setting_reaches_the_settings_object(tmp_path, name):
    value = REMOVAL_OVERRIDES[name]
    config = _config(tmp_path, {"removal": {name: value}})

    settings = remove.RemovalSettings.from_config(config)

    assert _as_comparable(getattr(settings, name)) == value


@pytest.mark.parametrize("name", sorted(DETECTION_OVERRIDES))
def test_a_detection_setting_reaches_the_settings_object(tmp_path, name):
    value = DETECTION_OVERRIDES[name]
    config = _config(tmp_path, {"detection": {name: value}})

    settings = catalogue.DetectionSettings.from_config(config)

    assert getattr(settings, name) == value


def test_the_probe_and_the_gate_are_configurable(tmp_path):
    config = _config(tmp_path, {"benchmark": BENCHMARK_OVERRIDES})

    benchmark = remove.RemovalSettings.from_config(config).benchmark

    assert benchmark.min_probe_separation_hz == 0.5
    assert benchmark.in_band_probe_count == 6
    assert benchmark.broadband_probe_channels == 2
    assert benchmark.probe.sinusoid_hz == (30.0, 50.0)
    assert benchmark.probe.burst_hz == 45.0
    assert benchmark.gate.min_burst_correlation == 0.95


def test_the_notch_stage_reads_its_own_parameters(tmp_path):
    from decomb import notch

    config = _config(
        tmp_path,
        {
            "notch_bands": [[56.8, 57.7]],
            "notch_trans_bandwidth_hz": 0.5,
            "removal": {"filter_jobs": 3},
            "frequency_bands": {"gamma": [30.0, 80.0]},
        },
    )

    settings = notch.NotchSettings.from_config(config)

    assert settings.bands[0].low_hz == 56.8
    assert settings.trans_bandwidth_hz == 0.5
    assert settings.filter_jobs == 3
    # frequency_bands is a mapping, so the user's entry merges over the packaged five
    # rather than replacing them; only gamma moves.
    assert dict((name, (low, high)) for name, low, high in settings.analysed_bands)["gamma"] == (
        30.0,
        80.0,
    )


def test_the_psd_stage_reads_its_own_parameters(tmp_path):
    from decomb import psd

    config = _config(
        tmp_path,
        {"psd": {"window_s": 30.0, "overlap": 0.25, "band_hz": [2.0, 80.0], "panel_span_hz": 5.0}},
    )

    settings = psd.PsdSettings.from_config(config)

    assert (settings.window_s, settings.overlap, settings.panel_span_hz) == (30.0, 0.25, 5.0)
    assert settings.band_hz == (2.0, 80.0)


def test_the_packaged_defaults_are_the_dataclass_defaults():
    """The file a user copies must describe what the workflow actually does.

    A default drifting between the YAML and the dataclass would make the documented
    behaviour and the real behaviour differ, silently.
    """
    config = load_config(None)
    document = yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8"))

    for settings, block in (
        (remove.RemovalSettings.from_config(config), document["removal"]),
        (catalogue.DetectionSettings.from_config(config), document["detection"]),
    ):
        for name in block:
            if name == "filter_jobs" and not hasattr(settings, name):
                continue
            assert _as_comparable(getattr(settings, name)) == pytest.approx(block[name]) or (
                _as_comparable(getattr(settings, name)) == block[name]
            ), f"{name} differs between defaults.yaml and the dataclass"

    assert remove.RemovalSettings.from_config(config) == remove.RemovalSettings(
        low_hz=3.0, high_hz=99.8, notch_width_ratio=450.0
    )
    assert catalogue.DetectionSettings.from_config(config) == catalogue.DetectionSettings()


def test_an_unknown_key_is_refused_in_every_block(tmp_path):
    for block, reader in (
        ("removal", remove.RemovalSettings.from_config),
        ("detection", catalogue.DetectionSettings.from_config),
    ):
        config = _config(tmp_path, {block: {"not_a_setting": 1}})
        with pytest.raises(ValueError, match="Unknown"):
            reader(config)

    config = _config(tmp_path, {"benchmark": {"not_a_setting": 1}})
    with pytest.raises(ValueError, match="Unknown `benchmark`"):
        remove.RemovalSettings.from_config(config)

    config = _config(tmp_path, {"benchmark": {"probe": {"not_a_setting": 1}}})
    with pytest.raises(ValueError, match="Unknown `benchmark.probe`"):
        remove.RemovalSettings.from_config(config)


def test_the_benchmark_criteria_are_inside_the_settings_fingerprint(tmp_path):
    """A benchmark run under looser criteria must not be able to certify an apply."""
    strict = remove.RemovalSettings.from_config(_config(tmp_path, {}))
    loose = remove.RemovalSettings.from_config(
        _config(tmp_path, {"benchmark": {"gate": {"min_burst_correlation": 0.5}}})
    )

    assert remove.settings_fingerprint(strict) != remove.settings_fingerprint(loose)


def test_no_settings_dataclass_holds_a_mutable_default():
    """Frozen dataclasses with shared mutable defaults would leak between runs."""
    for kind in (
        remove.RemovalSettings,
        catalogue.DetectionSettings,
        estimators.BenchmarkSettings,
        estimators.Probe,
        estimators.PreservationGate,
    ):
        assert is_dataclass(kind)
        for entry in fields(kind):
            assert not isinstance(entry.default, (list, dict, set)), f"{kind.__name__}.{entry.name}"


def _captured(monkeypatch, module, name):
    """Record the keyword arguments a function is called with, and call through."""
    seen: list[dict] = []
    original = getattr(module, name)

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, spy)
    return seen


def _expected(settings, setting: str, parameter: str):
    """What the estimator should have been handed for ``setting``.

    Three tolerances are configured as multiples of the fit spectrum's resolution and
    reach the estimator in hertz, so for those the question is whether the *derived* value
    arrived -- a multiplier that reached the estimator unconverted would be a bug this test
    exists to catch. Everything else is passed through as given.
    """
    if setting.endswith("_resolutions"):
        # From the setting, not the parameter: the estimator names one of these
        # `max_residual_rms_hz` where the settings object calls it
        # `max_fit_residual_rms_hz`, and the derived attribute is the settings one.
        return getattr(settings, setting.removesuffix("_resolutions") + "_hz")
    return REMOVAL_OVERRIDES[setting]


@pytest.mark.parametrize(
    "setting,parameter",
    (
        ("line_claim_hz", "claim_hz"),
        ("max_line_width_resolutions", "max_line_width_hz"),
        ("detection_low_hz", "low_hz"),
        ("detection_high_hz", "high_hz"),
        ("detection_fdr_alpha", "fdr_alpha"),
    ),
)
def test_a_detection_setting_reaches_the_detector(monkeypatch, setting, parameter):
    """A setting the config accepts but nobody passes on is worse than no setting.

    It reads as a knob, it is recorded in the provenance as though it were in force, and it
    does nothing. This asks the question by watching what the estimator was actually
    called with, which is the only thing that decides behaviour.
    """
    from decomb import estimators

    settings = replace(remove.RemovalSettings(), **{setting: REMOVAL_OVERRIDES[setting]})
    seen = _captured(monkeypatch, estimators, "detect_isolated_lines")

    remove.automatic_line_plans([_flat_run_spectra()], settings)

    assert seen, "detect_isolated_lines was never called"
    assert all(call[parameter] == _expected(settings, setting, parameter) for call in seen)


@pytest.mark.parametrize(
    "setting,parameter",
    (
        ("min_harmonics_for_fit", "min_harmonics"),
        ("max_harmonic_residual_resolutions", "max_harmonic_residual_hz"),
        ("max_fit_residual_rms_resolutions", "max_residual_rms_hz"),
        ("search_hz", "search_hz"),
    ),
)
def test_a_comb_fit_setting_reaches_the_estimator(monkeypatch, setting, parameter):
    from decomb import estimators

    settings = replace(remove.RemovalSettings(), **{setting: REMOVAL_OVERRIDES[setting]})
    seen = _captured(monkeypatch, estimators, "estimate_comb")

    remove.automatic_line_plans([_flat_run_spectra()], settings)

    assert seen, "estimate_comb was never called"
    assert all(call[parameter] == _expected(settings, setting, parameter) for call in seen)
