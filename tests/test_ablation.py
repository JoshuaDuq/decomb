"""Detection-procedure ablations sharing decomb's tests, windows, and FIR geometry."""

from __future__ import annotations

import time

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


@pytest.mark.parametrize("correction", ["holm", "bonferroni", "none"])
def test_fit_model_detects_a_strong_line_under_every_correction(correction):
    raw = _raw_with_line(line_hz=5.0)

    model = ablation.fit_model(raw, _narrow_settings(), correction=correction)

    c0 = next(channel for channel in model.channels if channel.channel_name == "C0")
    assert any(abs(line.position_hz - 5.0) < 0.02 for line in c0.lines)


def test_bonferroni_and_none_arms_never_label_a_harmonic():
    raw = _raw_with_line(line_hz=5.0)

    for correction in ("bonferroni", "none"):
        model = ablation.fit_model(raw, _narrow_settings(), correction=correction)
        assert all(channel.fundamental_hz is None for channel in model.channels)


def test_fit_models_every_correction_matches_calling_each_correction_separately():
    raw = _raw_with_line(line_hz=5.0)
    settings = _narrow_settings()

    shared = ablation.fit_models_every_correction(raw, settings)
    separate = {
        correction: ablation.fit_model(raw, settings, correction=correction)
        for correction in ("holm", "bonferroni", "none")
    }

    for correction in ("holm", "bonferroni", "none"):
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


def test_uncorrected_ablation_stays_fast_on_a_wide_default_band():
    # This is the regression this module exists to avoid: uncorrected detection on the
    # packaged 0-100 Hz default produces thousands of nominal false detections on pure
    # noise, and routing those through harmonic classification (as decomb's real
    # pipeline does for Holm) took minutes. The isolated-only model must stay fast.
    settings = notch.HarmonicNotchSettings(
        estimation_window_s=54.0,
        familywise_error_rate=0.05,
        frequency_range_hz=(0.0, 100.0),
    )
    sampling_frequency_hz = 250.0
    times_s = np.arange(int(120.0 * sampling_frequency_hz)) / sampling_frequency_hz
    data = np.random.default_rng(2).normal(scale=1e-6, size=(1, times_s.size))
    raw = mne.io.RawArray(
        data,
        mne.create_info(["C0"], sampling_frequency_hz, "eeg"),
        verbose="ERROR",
    )

    started = time.time()
    model = ablation.fit_ablation_model(raw, settings, correction="none")
    elapsed_s = time.time() - started

    assert elapsed_s < 15.0
    assert len(model.channels) <= 1
