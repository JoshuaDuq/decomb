"""The worker count changes speed, never results."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest
import yaml

from decomb import notch, recordings, recovery
from decomb.config import load_config


def _channels(n_channels: int = 3, n_times: int = 600) -> np.ndarray:
    rng = np.random.default_rng(11)
    times_s = np.arange(n_times) / 100.0
    artifact = np.sin(2.0 * np.pi * 30.0 * times_s)
    return rng.normal(scale=0.5, size=(n_channels, n_times)) + artifact


@pytest.mark.parametrize("n_jobs", [2, -1])
def test_trajectory_pca_is_identical_in_parallel(n_jobs):
    data = _channels()
    settings = recovery.TrajectoryPCASettings(segment_s=0.2)

    serial = recovery.subtract_recursive_trajectory_pca(data, 100.0, settings, n_jobs=1)
    parallel = recovery.subtract_recursive_trajectory_pca(
        data, 100.0, settings, n_jobs=n_jobs
    )

    np.testing.assert_array_equal(serial.cleaned_data, parallel.cleaned_data)
    np.testing.assert_array_equal(serial.artifact_data, parallel.artifact_data)


def test_multitaper_is_identical_in_parallel():
    data = _channels()

    serial = recovery.subtract_multitaper_sinusoids(
        data, 100.0, (30.0,), window_s=2.0, n_jobs=1
    )
    parallel = recovery.subtract_multitaper_sinusoids(
        data, 100.0, (30.0,), window_s=2.0, n_jobs=2
    )

    np.testing.assert_array_equal(serial.cleaned_data, parallel.cleaned_data)


def test_trigger_locked_basis_is_identical_in_parallel():
    data = _channels(n_times=1_000)
    triggers = np.arange(0, 900, 90, dtype=np.int64)

    def subtract(n_jobs):
        return recovery.subtract_trigger_locked_optimal_basis(
            data,
            100.0,
            (11.111111111111111,),
            triggers,
            repetition_time_s=0.9,
            component_count=2,
            n_jobs=n_jobs,
        )

    np.testing.assert_array_equal(
        subtract(1).artifact_data, subtract(2).artifact_data
    )


@pytest.mark.parametrize("n_jobs", [0, -2, 1.0, True, "4"])
def test_invalid_worker_counts_are_refused(n_jobs):
    with pytest.raises(ValueError, match="n_jobs"):
        recordings.validated_n_jobs(n_jobs)


@pytest.mark.parametrize("n_jobs", [-1, 1, 10])
def test_valid_worker_counts_are_accepted(n_jobs):
    assert recordings.validated_n_jobs(n_jobs) == n_jobs


def test_execution_section_supplies_the_worker_count():
    assert recordings.n_jobs_from_config(load_config()) == -1


def test_execution_worker_count_is_configurable(tmp_path):
    path = tmp_path / "decomb.yaml"
    path.write_text(yaml.safe_dump({"execution": {"n_jobs": 4}}))

    assert recordings.n_jobs_from_config(load_config(path)) == 4


def test_invalid_configured_worker_count_is_refused(tmp_path):
    path = tmp_path / "decomb.yaml"
    path.write_text(yaml.safe_dump({"execution": {"n_jobs": 0}}))

    with pytest.raises(ValueError, match="execution.n_jobs"):
        recordings.n_jobs_from_config(load_config(path))


def test_unknown_execution_setting_is_refused(tmp_path):
    path = tmp_path / "decomb.yaml"
    path.write_text(yaml.safe_dump({"execution": {"threads": 4}}))

    with pytest.raises(ValueError, match="Unknown `execution` setting"):
        load_config(path)


def test_worker_count_is_not_a_scientific_setting(tmp_path):
    """`removal` holds the irreducible scientific choices; parallelism is not one."""
    assert "n_jobs" not in {
        field.name for field in fields(notch.HarmonicNotchSettings)
    }

    path = tmp_path / "decomb.yaml"
    path.write_text(yaml.safe_dump({"removal": {"n_jobs": 4}}))
    with pytest.raises(ValueError, match="Unknown `removal` setting"):
        load_config(path)
