"""The README figures must measure the current correction on known signals."""

from __future__ import annotations

from pathlib import Path

from docs import make_figure

CONFIG_PATH = Path(__file__).parents[1] / "docs" / "simulated_readme.yaml"


def test_simulation_suppresses_harmonics_and_preserves_the_off_grid_tone():
    result = make_figure.build_simulation(CONFIG_PATH)
    representative_hz = (
        result.correction.nominal_fundamental_hz * result.simulation.representative_harmonic
    )

    harmonic_ratio = make_figure.tone_amplitude_ratio(
        result,
        representative_hz,
    )
    test_tone_ratio = make_figure.tone_amplitude_ratio(
        result,
        result.simulation.test_oscillation_hz,
    )

    expected_harmonics = range(
        result.correction.removal_harmonic_range[0],
        result.correction.removal_harmonic_range[1] + 1,
    )
    planned_harmonics = {
        harmonic for stopband in result.plan.stopbands for harmonic in stopband.harmonics
    }
    assert planned_harmonics == set(expected_harmonics)
    assert harmonic_ratio < 0.05
    assert test_tone_ratio > 0.98


def test_readme_figures_are_rendered_from_the_same_simulation(tmp_path):
    result = make_figure.build_simulation(CONFIG_PATH)
    overview = tmp_path / "overview.png"
    detail = tmp_path / "detail.png"

    make_figure.render_figures(result, overview, detail)

    assert overview.stat().st_size > 20_000
    assert detail.stat().st_size > 20_000
