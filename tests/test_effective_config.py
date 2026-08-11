"""A run has to be able to say what settings it used, and where each came from.

A config file records what someone changed, which is the small part. It says nothing about
the eighty values inherited from the packaged defaults, and nothing at all about the ones
the workflow computes -- those appear in no file, so a reader cannot tell whether a width
was chosen or derived, or from what. Reproducing a run a year later meant reconstructing
the merge order by hand and knowing which attributes were properties.
"""

from __future__ import annotations

import pytest
import yaml

from decomb import effective, notch
from decomb.config import load_config


def _config(tmp_path, document):
    path = tmp_path / "decomb.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return load_config(path)


class TestProvenance:
    def test_an_inherited_value_says_so(self, tmp_path):
        config = _config(tmp_path, {})
        assert config.provenance("removal.estimation_window_s") == "packaged defaults"

    def test_a_value_the_user_set_names_their_file(self, tmp_path):
        config = _config(tmp_path, {"removal": {"estimation_window_s": 60.0}})
        assert config.provenance("removal.estimation_window_s") == str(config.source)

    def test_a_value_matching_the_default_is_still_the_users(self, tmp_path):
        """Writing a value down is a decision, even when it agrees with the default.

        Someone who set it deliberately should not be told they inherited it -- the two
        mean different things when the default later changes.
        """
        default = load_config(None).get("removal.estimation_window_s")
        config = _config(tmp_path, {"removal": {"estimation_window_s": default}})
        assert config.provenance("removal.estimation_window_s") == str(config.source)

    def test_with_no_config_file_everything_is_inherited(self):
        config = load_config(None)
        assert all(origin == "packaged defaults" for _, _, origin in config.effective())


class TestTheReport:
    def test_it_covers_every_setting_in_force(self, tmp_path):
        config = _config(tmp_path, {})
        settings = notch.HarmonicNotchSettings.from_config(config)

        table = effective.rows(config, settings)
        names = {name for name, _, _ in table}

        assert "removal.estimation_window_s" in names
        assert "removal.transition_bandwidth_hz" in names
        assert "frequency_bands.gamma" in names

    def test_derived_values_appear_with_the_expression_that_made_them(self, tmp_path):
        """They are in no config file, so without this they are invisible to a reader."""
        config = _config(tmp_path, {})
        settings = notch.HarmonicNotchSettings.from_config(config)

        table = {name: (value, origin) for name, value, origin in effective.rows(config, settings)}

        value, origin = table["removal.transition_bandwidth_hz"]
        assert origin.startswith("derived:")
        assert "3.3" in origin
        assert float(value) == pytest.approx(settings.transition_bandwidth_hz)

    def test_a_derived_value_tracks_the_setting_it_derives_from(self, tmp_path):
        config = _config(tmp_path, {"removal": {"estimation_window_s": 108.0}})
        settings = notch.HarmonicNotchSettings.from_config(config)

        table = {name: value for name, value, _ in effective.rows(config, settings)}

        assert float(table["removal.spectral_resolution_hz"]) == pytest.approx(1.4382 / 108.0)

    def test_it_is_written_where_the_outputs_go(self, tmp_path):
        config = _config(tmp_path, {})
        settings = notch.HarmonicNotchSettings.from_config(config)

        written = effective.write(
            config, settings, tmp_path / "out" / "effective.txt", stage="apply"
        )

        text = written.read_text(encoding="utf-8")
        assert "removal.estimation_window_s" in text
        assert "packaged defaults" in text
        assert "derived:" in text


class TestAnEmptyBlockIsRefused:
    def test_a_key_with_nothing_under_it_does_not_silently_wipe_the_defaults(self, tmp_path):
        """A mapping with only comments beneath it reads as null.

        Merging that over the defaults would replace the whole block while the file looks
        like a section left deliberately alone.
        """
        path = tmp_path / "decomb.yaml"
        path.write_text("removal:\n  # only a comment\n", encoding="utf-8")

        with pytest.raises(ValueError, match="empty block"):
            load_config(path)

    def test_the_refusal_names_the_key_and_what_would_be_lost(self, tmp_path):
        path = tmp_path / "decomb.yaml"
        path.write_text("removal:\n", encoding="utf-8")

        with pytest.raises(ValueError) as excinfo:
            load_config(path)

        assert "`removal`" in str(excinfo.value)
        assert "Delete the key" in str(excinfo.value)

    def test_a_block_with_settings_in_it_still_merges(self, tmp_path):
        config = _config(tmp_path, {"removal": {"estimation_window_s": 60.0}})
        settings = notch.HarmonicNotchSettings.from_config(config)

        assert settings.estimation_window_s == 60.0
