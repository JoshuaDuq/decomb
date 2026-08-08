import numpy as np
import pytest

from decomb import config as decomb_config


@pytest.fixture(autouse=True)
def isolated_config(tmp_path_factory, monkeypatch):
    """Keep the suite from reading whatever config the developer happens to be running.

    ``load_config`` resolves ``DECOMB_CONFIG`` and then ``./decomb.yaml``, so a config
    sitting in the working directory is merged over the packaged defaults -- including for
    the tests that assert what the packaged defaults *are*. Running the suite from a
    checkout where someone had configured a real dataset failed four tests that have
    nothing to do with their settings, and the failures pointed at the library rather than
    at the config that caused them.

    Tests that want a config still get one: they pass an explicit path, which takes
    priority over both of these.
    """
    monkeypatch.delenv(decomb_config.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path_factory.mktemp("no-local-config"))


@pytest.fixture
def brainvision_run(tmp_path):
    """A small BrainVision file written the way the BIDS dataset writes them."""
    import mne

    mne.set_log_level("ERROR")
    sfreq, n_times = 1000.0, 4000
    rng = np.random.default_rng(0)
    data = rng.normal(scale=2e-5, size=(4, n_times))
    info = mne.create_info(["Fp1", "Cz", "Oz", "ECG"], sfreq, ["eeg", "eeg", "eeg", "ecg"])
    raw = mne.io.RawArray(data, info)
    path = tmp_path / "sub-0001_task-rest_run-1_eeg.vhdr"
    mne.export.export_raw(path, raw, fmt="brainvision", overwrite=True)
    return path, raw
