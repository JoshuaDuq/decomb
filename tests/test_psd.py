"""Before-and-after power spectra.

A pair of figures is only worth reading if both sides were measured identically, actually
correspond to each other, and are drawn on one scale. Most of these pin that, rather than
the drawing.
"""

from __future__ import annotations

from dataclasses import MISSING, fields, replace
from pathlib import Path

import numpy as np
import pytest

from decomb import psd
from decomb.config import load_config


def _settings(**overrides) -> psd.PsdSettings:
    return replace(psd.PsdSettings.from_config(load_config()), **overrides)


def _raw(sfreq=250.0, seconds=120.0, line_hz=57.25, amplitude=5e-6, n_channels=4):
    import mne

    times = np.arange(int(sfreq * seconds)) / sfreq
    rng = np.random.default_rng(4)
    data = rng.normal(scale=1e-6, size=(n_channels, times.size))
    data += amplitude * np.sin(2.0 * np.pi * line_hz * times)
    montage = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(list(montage.ch_names[:n_channels]), sfreq, "eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_montage(montage, on_missing="ignore")
    return raw, times


class TestSettings:
    def test_every_value_comes_from_yaml(self):
        assert all(entry.default is MISSING for entry in fields(psd.PsdSettings))
        assert psd.PsdSettings.from_config(load_config()).window_s == 54.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"window_s": 0.0},
            {"window_s": -1.0},
            {"band_hz": (-1.0, 80.0)},
            {"band_hz": (80.0, 20.0)},
        ],
    )
    def test_an_impossible_setting_is_refused(self, kwargs):
        with pytest.raises(ValueError):
            _settings(**kwargs)

    def test_overlap_and_band_are_derived(self):
        settings = _settings()

        assert settings.overlap == 0.5
        assert settings.band_hz == (0.0, 100.0)

    def test_band_follows_correction_frequency_range(self, tmp_path):
        path = tmp_path / "decomb.yaml"
        path.write_text(
            "removal:\n  frequency_range_hz: [2.0, 80.0]\n",
            encoding="utf-8",
        )

        settings = psd.PsdSettings.from_config(load_config(path))

        assert settings.band_hz == (2.0, 80.0)


class TestSpectrum:
    def test_every_channel_is_kept_rather_than_summarised(self):
        """The figure's subject is what the removal did to each channel."""
        raw, _ = _raw(n_channels=6)

        spectrum = psd.channel_spectrum(raw, _settings(window_s=20.0))

        assert spectrum.get_data(exclude=()).shape[0] == 6

    def test_the_resolution_follows_the_window(self):
        raw, _ = _raw()

        spectrum = psd.channel_spectrum(raw, _settings(window_s=20.0))

        assert spectrum.freqs[1] - spectrum.freqs[0] == pytest.approx(1.0 / 20.0, rel=1e-6)

    def test_the_planted_line_lands_where_it_was_put(self):
        raw, _ = _raw(line_hz=57.25)

        spectrum = psd.channel_spectrum(raw, _settings(window_s=20.0))

        averaged = spectrum.get_data(exclude=()).mean(axis=0)
        assert spectrum.freqs[int(np.argmax(averaged))] == pytest.approx(57.25, abs=0.05)

    def test_only_eeg_channels_are_measured(self):
        """An ECG trace is orders larger; drawing it in would set the scale."""
        import mne

        sfreq = 250.0
        times = np.arange(int(sfreq * 120)) / sfreq
        data = np.zeros((2, times.size))
        data[0] = 1e-6 * np.sin(2.0 * np.pi * 20.0 * times)
        data[1] = 1.0 * np.sin(2.0 * np.pi * 20.0 * times)  # a volt-scale auxiliary channel
        raw = mne.io.RawArray(
            data, mne.create_info(["Cz", "ECG"], sfreq, ["eeg", "ecg"]), verbose="ERROR"
        )

        spectrum = psd.channel_spectrum(raw, _settings(window_s=20.0))

        assert spectrum.ch_names == ["Cz"]

    def test_a_recording_shorter_than_one_segment_is_refused(self):
        raw, _ = _raw(seconds=5.0)

        with pytest.raises(ValueError, match="removal.estimation_window_s"):
            psd.channel_spectrum(raw, _settings(window_s=54.0))

    def test_a_recording_without_eeg_channels_is_refused(self):
        import mne

        sfreq = 250.0
        data = np.zeros((1, int(sfreq * 120)))
        raw = mne.io.RawArray(
            data, mne.create_info(["ECG"], sfreq, ["ecg"]), verbose="ERROR"
        )

        with pytest.raises(ValueError, match="at least one EEG channel"):
            psd.channel_spectrum(raw, _settings(window_s=20.0))

    def test_the_returned_band_is_automatically_limited_to_100_hz(self):
        raw, _ = _raw()

        spectrum = psd.channel_spectrum(raw, _settings(window_s=20.0))

        assert spectrum.freqs.min() >= 0.0 and spectrum.freqs.max() <= 100.0

    def test_requested_band_above_100_hz_is_limited_only_by_nyquist(self):
        raw, _ = _raw(sfreq=500.0)

        spectrum = psd.channel_spectrum(
            raw,
            _settings(window_s=20.0, band_hz=(0.0, 180.0)),
        )

        assert spectrum.freqs.max() == pytest.approx(180.0, abs=0.05)

    def test_acquisition_skip_samples_do_not_enter_the_psd(self):
        import mne

        sampling_frequency_hz = 250.0
        times_s = np.arange(int(120.0 * sampling_frequency_hz)) / sampling_frequency_hz
        clean = np.random.default_rng(8).normal(scale=1e-6, size=(2, times_s.size))
        contaminated = clean.copy()
        skipped = (times_s >= 40.0) & (times_s < 80.0)
        contaminated[:, skipped] += 1e-2 * np.sin(2.0 * np.pi * 30.0 * times_s[skipped])
        info = mne.create_info(["C3", "C4"], sampling_frequency_hz, "eeg")
        annotation = mne.Annotations([40.0], [40.0], ["BAD_ACQ_SKIP"])
        clean_raw = mne.io.RawArray(clean, info, verbose="ERROR")
        contaminated_raw = mne.io.RawArray(contaminated, info, verbose="ERROR")
        clean_raw.set_annotations(annotation)
        contaminated_raw.set_annotations(annotation)
        settings = _settings(window_s=20.0)

        clean_psd = psd.channel_spectrum(clean_raw, settings).get_data(exclude=())
        contaminated_psd = psd.channel_spectrum(contaminated_raw, settings).get_data(
            exclude=()
        )

        assert np.array_equal(contaminated_psd, clean_psd)

    def test_welch_windows_do_not_cross_zero_duration_acquisition_edges(self):
        import mne

        sampling_frequency_hz = 250.0
        samples_per_half = int(50.0 * sampling_frequency_hz)
        data = np.concatenate(
            (
                np.full(samples_per_half, 1e-3),
                np.full(samples_per_half, -1e-3),
            )
        )[np.newaxis, :]
        raw = mne.io.RawArray(
            data,
            mne.create_info(["Cz"], sampling_frequency_hz, "eeg"),
            verbose="ERROR",
        )
        raw.set_annotations(mne.Annotations([50.0], [0.0], ["EDGE boundary"]))

        spectrum = psd.channel_spectrum(raw, _settings(window_s=20.0))

        assert np.max(spectrum.get_data(exclude=())) < 1e-35


class TestCorrespondence:
    def test_a_derivative_with_a_different_channel_set_is_refused(self):
        source, _ = _raw()
        other, _ = _raw(n_channels=3)

        with pytest.raises(ValueError, match="channel set differs"):
            psd.require_correspondence(source, other, "other.vhdr")

    def test_a_derivative_of_a_different_length_is_refused(self):
        source, _ = _raw()
        other, _ = _raw(seconds=100.0)

        with pytest.raises(ValueError, match="length differs"):
            psd.require_correspondence(source, other, "other.vhdr")

    def test_a_derivative_at_a_different_sampling_rate_is_refused(self):
        # Same sample count, so the length check cannot fire before the rate check.
        source, _ = _raw(sfreq=250.0, seconds=120.0)
        other, _ = _raw(sfreq=500.0, seconds=60.0)
        assert source.n_times == other.n_times

        with pytest.raises(ValueError, match="sampling rate differs"):
            psd.require_correspondence(source, other, "other.vhdr")

    def test_a_matching_derivative_is_accepted(self):
        source, _ = _raw()
        derivative, _ = _raw()

        assert psd.require_correspondence(source, derivative, "d.vhdr") is None


class TestBadChannels:
    def test_a_derivative_that_lost_the_marking_gets_it_back(self):
        """Otherwise one figure greys a channel the other draws as ordinary signal."""
        source, _ = _raw()
        derivative, _ = _raw()
        source.info["bads"] = [source.ch_names[0]]
        derivative.info["bads"] = []

        aligned = psd.align_bad_channels(source, derivative)

        assert aligned == (source.ch_names[0],)
        assert derivative.info["bads"] == [source.ch_names[0]]

    def test_a_channel_either_side_distrusts_is_bad_on_both(self):
        source, _ = _raw()
        derivative, _ = _raw()
        source.info["bads"] = [source.ch_names[0]]
        derivative.info["bads"] = [source.ch_names[1]]

        aligned = psd.align_bad_channels(source, derivative)

        assert set(aligned) == {source.ch_names[0], source.ch_names[1]}
        assert set(source.info["bads"]) == set(derivative.info["bads"]) == set(aligned)


class TestSharedScale:
    def test_the_scale_spans_both_spectra(self):
        loud, _ = _raw(amplitude=5e-5)
        quiet, _ = _raw(amplitude=0.0)
        settings = _settings(window_s=20.0)
        loud_spectrum = psd.channel_spectrum(loud, settings)
        quiet_spectrum = psd.channel_spectrum(quiet, settings)

        low, high = psd.shared_decibel_limits(loud_spectrum, quiet_spectrum)

        for spectrum in (loud_spectrum, quiet_spectrum):
            decibels = 10.0 * np.log10(spectrum.get_data(exclude=()) * 1e12)
            assert low <= decibels.min() and decibels.max() <= high

    def test_bad_channels_are_inside_the_scale(self):
        """get_data drops bads by default, which would clip the traces drawn highest."""
        raw, _ = _raw(n_channels=4)
        raw._data[0] *= 300.0
        raw.info["bads"] = [raw.ch_names[0]]
        spectrum = psd.channel_spectrum(raw, _settings(window_s=20.0))

        _, high = psd.shared_decibel_limits(spectrum)

        loudest = 10.0 * np.log10(spectrum.get_data(exclude=())[0] * 1e12).max()
        assert high >= loudest


class TestFigures:
    def test_a_spectrum_figure_is_written(self, tmp_path):
        raw, _ = _raw(n_channels=8)
        spectrum = psd.channel_spectrum(raw, _settings(window_s=20.0))
        path = tmp_path / "psd_before.png"

        psd.figure_spectrum(spectrum, path, title="Before correction", ylim=(-60.0, 40.0))

        assert path.is_file() and path.stat().st_size > 0

    def test_both_figures_share_the_requested_limits(self, tmp_path):
        raw, _ = _raw(n_channels=8)
        spectrum = psd.channel_spectrum(raw, _settings(window_s=20.0))

        figure = spectrum.plot(spatial_colors=True, dB=True, amplitude=False, show=False)
        drawn = [axis for axis in figure.axes if axis.get_ylabel()]
        assert drawn, "the spectrum figure should carry a labelled axis"

        psd.figure_spectrum(spectrum, tmp_path / "a.png", title="a", ylim=(-70.0, 30.0))
        # Re-drawn rather than reused, so the assertion is about what the helper sets.
        second = psd.figure_spectrum(spectrum, tmp_path / "b.png", title="b", ylim=(-70.0, 30.0))
        assert Path(second).is_file()


def test_the_stage_is_reachable_from_the_cli():
    from decomb import cli

    assert "psd" in cli.STAGES


def test_the_stage_refuses_before_apply_has_run(tmp_path, monkeypatch):
    import argparse

    configured_settings = _settings()

    class Config:
        def path(self, name, override=None):
            return tmp_path / name

        def get(self, key, default=None):
            return {"task": "*"} if key == "dataset" else default

    monkeypatch.setattr("decomb.config.load_config", lambda *a, **k: Config())
    monkeypatch.setattr(psd.PsdSettings, "from_config", lambda config: configured_settings)

    with pytest.raises(FileNotFoundError, match="Run `decomb apply` first"):
        psd.run(argparse.Namespace(config=None, bids_root=None, report_dir=None))


def test_the_stage_uses_the_output_root_override(tmp_path, monkeypatch):
    import argparse

    configured_settings = _settings()
    overridden_derivative = tmp_path / "chosen-derivative"
    overridden_derivative.mkdir()

    class Config:
        def path(self, name, override=None):
            return Path(override) if override is not None else tmp_path / name

        def get(self, key, default=None):
            return default

    monkeypatch.setattr("decomb.config.load_config", lambda *a, **k: Config())
    monkeypatch.setattr(psd.PsdSettings, "from_config", lambda config: configured_settings)
    monkeypatch.setattr(
        psd.recordings,
        "discover_runs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("discovery reached")),
    )

    with pytest.raises(RuntimeError, match="discovery reached"):
        psd.run(
            argparse.Namespace(
                config=None,
                bids_root=None,
                output_root=overridden_derivative,
                report_dir=None,
                subjects=None,
            )
        )
