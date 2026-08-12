"""Detection-procedure ablations sharing decomb's own tests, windows, and FIR geometry.

Isolates the effect of the per-channel multiplicity procedure -- Holm (decomb's real
pipeline), plain Bonferroni, or no correction at all (matching MNE's uncorrected
``spectrum_fit`` threshold) -- by holding every other stage fixed: the same Thomson
F-test, the same windows, and the same stopband-width and merge rule.

Harmonic classification is intentionally skipped here. It is a descriptive label that
never changes which frequencies get filtered (decomb's own README states this: "either
way, exactly the same significant frequencies are filtered"), and its candidate-
fundamental search scales with detection count -- fine under Holm's real-world sparsity,
but combinatorially expensive once an ablation deliberately removes multiplicity control
and produces thousands of nominal detections on ordinary data.
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


def fit_ablation_model(raw, settings, *, correction: str) -> lines.ArtifactModel:
    """Fit an isolated-line-only model with a chosen per-channel correction procedure."""
    result = notch.detect_channel_lines(raw, settings, correction=correction)
    return isolated_model(result, channel_names=notch.eeg_channel_names(raw))


def fit_model(raw, settings, *, correction: str) -> lines.ArtifactModel:
    """Fit the model decomb would filter from, for any of the three ablation arms.

    Holm keeps decomb's real pipeline, including harmonic classification, since Holm's
    false-positive control keeps its detection counts small. Bonferroni and uncorrected
    detection use the cheaper isolated-only model for the reason above.
    """
    if correction == "holm":
        return notch.fit_harmonic_model(raw, settings, correction="holm")
    return fit_ablation_model(raw, settings, correction=correction)


def fit_models_every_correction(raw, settings) -> dict[str, lines.ArtifactModel]:
    """One model per correction procedure, from one shared Thomson F-test pass."""
    channel_names = notch.eeg_channel_names(raw)
    results = notch.detect_channel_lines_every_correction(raw, settings)
    models = {
        correction: isolated_model(result, channel_names=channel_names)
        for correction, result in results.items()
        if correction != "holm"
    }
    models["holm"] = lines.build_artifact_model(
        results["holm"],
        channel_names=channel_names,
        frequency_bin_width_hz=settings.frequency_bin_width_hz,
        spectral_resolution_hz=settings.spectral_resolution_hz,
        familywise_error_rate=settings.familywise_error_rate,
    )
    return models
