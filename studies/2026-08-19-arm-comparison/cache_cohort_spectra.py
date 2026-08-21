"""Cache the cohort-average spectra and the per-frequency exclusion fraction."""
import os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
from decomb import notch, psd, recordings
from decomb.config import load_config

config = load_config("/Users/joduq24/Desktop/decomb/decomb.yaml")
settings = psd.PsdSettings.from_config(config)
src, deriv = config.path("bids_root"), config.path("output_root")
runs = recordings.discover_runs(src, subjects=None, task="*")

# the pipeline's own generator: one pair resident at a time, not all 180 recordings
cohort = psd.cohort_spectrum_pair(psd._read_recording_pairs(runs, src, deriv), settings)
freqs = cohort.before.freqs

m = notch._read_manifest(config.path("removal_dir") / notch.MANIFEST_NAME)
count = np.zeros(freqs.size)
for _, block in m.groupby("recording"):
    zones = notch.merged_intervals(
        (float(x), float(y))
        for x, y in zip(block.unavailable_low_hz, block.unavailable_high_hz)
        if str(x) != "" and str(y) != "")
    mask = np.zeros(freqs.size, bool)
    for lo, hi in zones:
        mask |= (freqs >= lo) & (freqs <= hi)
    count += mask

np.savez_compressed(sys.argv[1], freqs=freqs,
                    before=cohort.before.get_data(), after=cohort.after.get_data(),
                    excluded_fraction=count / m.recording.nunique(),
                    n_recordings=cohort.recording_count, hours=cohort.analysed_hours)
print(f"cached {cohort.recording_count} recordings, {cohort.analysed_hours:.2f} h", flush=True)
