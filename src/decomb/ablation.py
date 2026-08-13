"""Complete-family Bonferroni ablation of decomb's Holm procedure.

Both procedures share the Thomson tests, windows, channel families, stopband geometry,
recording-wide filtering, and terminal-null convergence rule. Harmonic classification is
omitted from the Bonferroni arm because it is descriptive and cannot change the filtered
frequencies.
"""

from __future__ import annotations

from decomb import lines, notch


def isolated_model(
    result: lines.LineDetectionResult,
    *,
    channel_names: tuple[str, ...],
) -> lines.ArtifactModel:
    """Build a channel model treating every significant frequency as an isolated line.

    Produces the same stopband geometry harmonic classification would: every filtered
    frequency still gets its own resolution-bounded interval. Only the descriptive
    harmonic label is skipped.
    """
    if len(channel_names) != result.channel_count or len(channel_names) != len(
        set(channel_names)
    ):
        raise ValueError("channel_names must identify every tested EEG channel once.")

    channel_models = []
    for channel_index, channel_name in enumerate(channel_names):
        channel_detections = [
            detection
            for detection in result.detections
            if detection.channel_index == channel_index
        ]
        if not channel_detections:
            continue
        best_by_frequency: dict[float, lines.LineDetection] = {}
        windows_by_frequency: dict[float, set[int]] = {}
        for detection in channel_detections:
            windows_by_frequency.setdefault(detection.frequency_hz, set()).add(
                detection.window_index
            )
            current = best_by_frequency.get(detection.frequency_hz)
            if current is None or detection.corrected_p_value < current.corrected_p_value:
                best_by_frequency[detection.frequency_hz] = detection
        artifact_lines = tuple(
            lines.ArtifactLine(
                position_hz=frequency_hz,
                raw_p_value=best_by_frequency[frequency_hz].raw_p_value,
                corrected_p_value=best_by_frequency[frequency_hz].corrected_p_value,
                window_indices=tuple(sorted(windows_by_frequency[frequency_hz])),
                harmonic=None,
            )
            for frequency_hz in sorted(best_by_frequency)
        )
        channel_models.append(
            lines.ChannelArtifactModel(
                channel_index=channel_index,
                channel_name=channel_name,
                lines=artifact_lines,
                fundamental_hz=None,
                comb_corrected_p_value=None,
            )
        )
    return lines.ArtifactModel(
        channels=tuple(channel_models),
        window_count=result.window_count,
        channel_count=result.channel_count,
        test_count_per_channel=result.test_count_per_channel,
    )


def fit_bonferroni_model(raw, settings) -> lines.ArtifactModel:
    """Fit the first-round recording-family Bonferroni ablation."""
    round_settings = settings.for_round(1)
    frequencies_hz, p_values = notch._thomson_f_p_values(raw, round_settings)
    result = lines.detect_lines_with_bonferroni_from_p_values(
        frequencies_hz,
        p_values,
        familywise_error_rate=round_settings.familywise_error_rate,
    )
    return isolated_model(result, channel_names=notch.eeg_channel_names(raw))


def fit_holm_and_bonferroni_models(raw, settings) -> dict[str, lines.ArtifactModel]:
    """Holm and Bonferroni models from one shared Thomson F-test pass."""
    round_settings = settings.for_round(1)
    channel_names = notch.eeg_channel_names(raw)
    results = notch.detect_channel_lines_holm_and_bonferroni(raw, round_settings)
    holm_model = lines.build_artifact_model(
        results["holm"],
        channel_names=channel_names,
        frequency_bin_width_hz=round_settings.frequency_bin_width_hz,
        spectral_resolution_hz=round_settings.frequency_bin_width_hz,
        familywise_error_rate=round_settings.familywise_error_rate,
    )
    bonferroni_model = isolated_model(
        results["bonferroni"],
        channel_names=channel_names,
    )
    return {"holm": holm_model, "bonferroni": bonferroni_model}


def clean_until_no_bonferroni_lines(raw, settings) -> notch.HarmonicCleaningResult:
    """Apply recording-wide FIR rounds until a fresh Bonferroni fit is null."""
    return notch._clean_until_model_null(
        raw,
        settings,
        lines.detect_lines_with_bonferroni_from_p_values,
        _isolated_model_from_detection,
    )


def _isolated_model_from_detection(raw, result, settings) -> lines.ArtifactModel:
    del settings
    return isolated_model(result, channel_names=notch.eeg_channel_names(raw))
