"""Probe tones are a property of the dataset, so the dataset should choose them.

Asking a user for them fails in a particular way. The safe positions depend on a fitted
fundamental and on isolated lines detected per recording, neither of which is known when a
config is written. On a 15-participant cohort the catalogue named 9 isolated lines while
the 89 fitted plans contained 297 distinct targets; tones chosen against the catalogue --
correctly, by the only evidence available -- sat 0.046 and 0.118 Hz from lines that were
not in it. Nothing detected the collision until the benchmark reached the recording that
carried it, six hours in.

Plans are fitted before probes are injected, so every target is already known at placement
time and the collision is decidable before any measurement starts.
"""

from __future__ import annotations

import numpy as np
import pytest

from decomb import estimators

F0 = 1.199953
COMB = [k * F0 for k in range(22, 80)]


class TestPlacementClearsEveryTarget:
    def test_tones_land_between_harmonics(self):
        placed = estimators.place_probes(COMB, F0, count=4, band_hz=(28.0, 95.0))
        for position in placed:
            offset = (position / F0) % 1.0
            assert offset == pytest.approx(0.5, abs=1e-6)

    def test_a_line_absent_from_the_catalogue_is_still_avoided(self):
        """The collision that ended a 6.3-hour benchmark, as a test that takes no time."""
        detected_per_recording = [35.4444, 43.9167, 43.6296, 44.0185]
        placed = estimators.place_probes(
            COMB + detected_per_recording, F0, count=4, band_hz=(28.0, 95.0)
        )
        for position in placed:
            for target in COMB + detected_per_recording:
                assert abs(position - target) >= 0.3

    def test_clearance_is_reported_as_at_least_the_minimum(self):
        placed = estimators.place_probes(
            COMB, F0, count=4, band_hz=(28.0, 95.0), min_separation_hz=0.3
        )
        targets = np.asarray(COMB)
        for position in placed:
            assert float(np.min(np.abs(targets - position))) >= 0.3

    def test_the_requested_count_is_returned(self):
        for count in (1, 2, 4, 6):
            placed = estimators.place_probes(COMB, F0, count=count, band_hz=(28.0, 95.0))
            assert len(placed) == count

    def test_tones_are_distinct(self):
        placed = estimators.place_probes(COMB, F0, count=4, band_hz=(28.0, 95.0))
        assert len(set(placed)) == len(placed)


class TestPlacementRespectsTheBand:
    def test_tones_stay_inside_the_requested_band(self):
        placed = estimators.place_probes(COMB, F0, count=4, band_hz=(40.0, 70.0))
        assert all(40.0 <= position <= 70.0 for position in placed)

    def test_an_excluded_band_is_avoided_with_its_margin(self):
        """A tone inside a band handed to a wide notch would be removed wholesale."""
        placed = estimators.place_probes(
            COMB, F0, count=4, band_hz=(28.0, 95.0), excluded_hz=((59.5, 60.5),)
        )
        assert not any(59.2 <= position <= 60.8 for position in placed)

    def test_tones_are_spread_rather_than_clustered(self):
        """Four tones in one corner would report on one corner of the spectrum."""
        placed = estimators.place_probes(COMB, F0, count=4, band_hz=(28.0, 95.0))
        assert max(placed) - min(placed) > 30.0


class TestPlacementRefusesRatherThanGuessing:
    def test_a_band_with_no_clear_position_is_refused(self):
        """Better a refusal naming the reason than a probe that measures the removal."""
        dense = [value for value in np.arange(28.0, 95.0, 0.05)]
        with pytest.raises(ValueError, match="clear every removal target"):
            estimators.place_probes(dense, F0, count=4, band_hz=(28.0, 95.0))

    def test_placement_needs_targets_to_avoid(self):
        with pytest.raises(ValueError, match="at least one target"):
            estimators.place_probes([], F0, count=4, band_hz=(28.0, 95.0))

    def test_a_placed_probe_passes_the_clearance_check_it_is_measured_by(self):
        placed = estimators.place_probes(COMB, F0, count=4, band_hz=(28.0, 95.0))
        probe = estimators.Probe(sinusoid_hz=placed)
        estimators.check_probe_clearance(probe, COMB, min_separation_hz=0.3)


class TestAnUnresolvedProbeCannotBeInjected:
    def test_the_default_probe_has_no_tones(self):
        assert estimators.Probe().sinusoid_hz == ()

    def test_injecting_an_unresolved_probe_says_what_is_missing(self):
        with pytest.raises(ValueError, match="place_probes"):
            estimators.Probe().waveform(np.linspace(0.0, 1.0, 100))
