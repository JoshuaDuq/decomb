"""Detection-procedure ablations sharing decomb's tests, windows, and FIR geometry."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from decomb import ablation, lines, notch


def _narrow_settings(frequency_range_hz=(1.0, 10.0)) -> notch.HarmonicNotchSettings:
    return notch.HarmonicNotchSettings(
        estimation_window_s=54.0,
        familywise_error_rate=0.05,
        frequency_range_hz=frequency_range_hz,
    )


def _raw_with_line(*, line_hz: float, n_channels: int = 2, seed: int = 0):
    sampling_frequency_hz = 250.0
    times_s = np.arange(int(120.0 * sampling_frequency_hz)) / sampling_frequency_hz
    data = np.random.default_rng(seed).normal(scale=1e-6, size=(n_channels, times_s.size))
    data[0] += 20e-6 * np.sin(2.0 * np.pi * line_hz * times_s)
    names = [f"C{index}" for index in range(n_channels)]
    return mne.io.RawArray(
        data,
        mne.create_info(names, sampling_frequency_hz, "eeg"),
        verbose="ERROR",
    )


def test_isolated_model_never_assigns_a_harmonic_label():
    frequencies_hz = np.array([5.0, 10.0])
    p_values = np.array([[[0.01, 0.02], [0.9, 0.9]]])
    result = lines.LineDetectionResult(
        detections=(
            lines.LineDetection(5.0, 0.01, 0.01, window_index=0, channel_index=0),
            lines.LineDetection(10.0, 0.02, 0.02, window_index=0, channel_index=0),
        ),
        tested_frequencies_hz=tuple(frequencies_hz),
        window_count=1,
        channel_count=2,
    )
    del p_values

    model = ablation.isolated_model(result, channel_names=("C0", "C1"))

    assert [channel.channel_name for channel in model.channels] == ["C0"]
    channel = model.channels[0]
    assert channel.fundamental_hz is None
    assert channel.comb_corrected_p_value is None
    assert all(line.harmonic is None for line in channel.lines)
    assert {line.position_hz for line in channel.lines} == {5.0, 10.0}


def test_isolated_model_keeps_the_best_p_value_and_every_supporting_window():
    result = lines.LineDetectionResult(
        detections=(
            lines.LineDetection(5.0, 0.03, 0.03, window_index=0, channel_index=0),
            lines.LineDetection(5.0, 0.01, 0.01, window_index=1, channel_index=0),
        ),
        tested_frequencies_hz=(5.0,),
        window_count=2,
        channel_count=1,
    )

    model = ablation.isolated_model(result, channel_names=("C0",))

    line = model.channels[0].lines[0]
    assert line.corrected_p_value == 0.01
    assert line.window_indices == (0, 1)


def test_isolated_model_requires_channel_names_for_every_tested_channel():
    result = lines.LineDetectionResult(
        detections=(lines.LineDetection(5.0, 0.01, 0.01, window_index=0, channel_index=0),),
        tested_frequencies_hz=(5.0,),
        window_count=1,
        channel_count=1,
    )

    with pytest.raises(ValueError, match="channel_names"):
        ablation.isolated_model(result, channel_names=("C0", "C1"))


def test_fit_bonferroni_model_detects_a_strong_line():
    raw = _raw_with_line(line_hz=5.0)

    model = ablation.fit_bonferroni_model(raw, _narrow_settings())

    c0 = next(channel for channel in model.channels if channel.channel_name == "C0")
    assert any(abs(line.position_hz - 5.0) < 0.02 for line in c0.lines)


def test_standalone_first_round_models_use_the_first_alpha_spend(monkeypatch):
    raw = _raw_with_line(line_hz=5.0)
    p_values = np.array([[[0.02], [0.9]]])
    monkeypatch.setattr(
        notch,
        "_thomson_f_p_values",
        lambda raw, settings: (np.array([5.0]), p_values),
    )

    bonferroni = ablation.fit_bonferroni_model(raw, _narrow_settings())
    shared = ablation.fit_holm_and_bonferroni_models(raw, _narrow_settings())

    assert not bonferroni.channels
    assert not shared["holm"].channels
    assert not shared["bonferroni"].channels


def test_bonferroni_arm_never_labels_a_harmonic():
    raw = _raw_with_line(line_hz=5.0)

    model = ablation.fit_bonferroni_model(raw, _narrow_settings())

    assert all(channel.fundamental_hz is None for channel in model.channels)


def test_shared_models_match_calling_each_procedure_separately():
    raw = _raw_with_line(line_hz=5.0)
    settings = _narrow_settings()

    shared = ablation.fit_holm_and_bonferroni_models(raw, settings)
    separate = {
        "holm": notch.fit_harmonic_model(raw, settings),
        "bonferroni": ablation.fit_bonferroni_model(raw, settings),
    }

    for correction in ("holm", "bonferroni"):
        shared_lines = {
            (channel.channel_name, line.position_hz, line.corrected_p_value)
            for channel in shared[correction].channels
            for line in channel.lines
        }
        separate_lines = {
            (channel.channel_name, line.position_hz, line.corrected_p_value)
            for channel in separate[correction].channels
            for line in channel.lines
        }
        assert shared_lines == separate_lines


def test_bonferroni_cleaning_reaches_a_null_model():
    raw = _raw_with_line(line_hz=5.0)

    result = ablation.clean_until_no_bonferroni_lines(raw, _narrow_settings())

    assert result.rounds
    assert not result.residual_model.channels
