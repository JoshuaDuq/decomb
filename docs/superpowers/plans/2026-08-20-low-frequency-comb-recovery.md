# Target-Blind 1.2 Hz Recovery Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether adjacent high-frequency pump harmonics can predict and remove a
pump-locked 1.2 Hz component without reading or damaging the held-out participant's 1.2 Hz
EEG.

**Architecture:** Twenty-second windows place every 1.2 Hz harmonic exactly on a Fourier
bin. Adjacent-harmonic cross-products carry the fundamental phase while excluding the
fundamental bin itself. A leave-one-participant-out complex reduced-rank ridge model predicts
per-channel 1.2 Hz coefficients from those high-only features; familywise pump-lock tests and
paired exact-frequency injections decide whether each recording qualifies as recovered.

**Tech Stack:** Python 3.11, NumPy, SciPy, pandas, MNE-Python, pytest, Ruff

---

## File structure

- Create `src/decomb/pump_recovery.py`: target-blind feature extraction, reduced-rank model,
  pump-lock inference, and overlap-add artifact reconstruction.
- Create `tests/test_pump_recovery.py`: synthetic unit tests for target blindness,
  prediction, inference, annotation-safe reconstruction, and exact-frequency preservation.
- Create `studies/2026-08-20-low-frequency-recovery/run_experiment.py`: deterministic
  feature cache, leave-one-participant-out cohort run, comparison arms, injections, and TSV
  output.
- Create `studies/2026-08-20-low-frequency-recovery/report.py`: strict aggregation into the
  final Markdown decision report.
- Create `studies/2026-08-20-low-frequency-recovery/README.md`: exact commands, inputs,
  outputs, and interpretation boundary.

No production pipeline caller changes in this plan. A positive experiment requires a
separate production-integration design.

### Task 1: Extract target-blind adjacent-harmonic features

**Files:**
- Create: `src/decomb/pump_recovery.py`
- Create: `tests/test_pump_recovery.py`

- [ ] **Step 1: Write failing tests for harmonic enumeration and target blindness**

```python
import numpy as np
import pytest

from decomb import pump_recovery


def test_high_harmonics_are_consecutive_teeth_between_20_and_95_hz():
    assert pump_recovery.high_harmonic_numbers(1.2, 500.0) == tuple(range(17, 80))


def test_adjacent_features_do_not_read_the_exact_fundamental():
    sfreq = 1000.0
    times = np.arange(40_000) / sfreq
    background = np.stack(
        [
            np.sin(2 * np.pi * 20.4 * times)
            + 0.6 * np.sin(2 * np.pi * 21.6 * times + 0.2),
            np.cos(2 * np.pi * 20.4 * times)
            + 0.8 * np.cos(2 * np.pi * 21.6 * times - 0.3),
        ]
    )
    injection = 7.0 * np.sin(2 * np.pi * 1.2 * times + 0.4)
    bounds = ((0, 20_000), (20_000, 40_000))
    original = pump_recovery.extract_adjacent_features(
        background, sfreq, bounds, fundamental_hz=1.2
    )
    injected = pump_recovery.extract_adjacent_features(
        background + injection, sfreq, bounds, fundamental_hz=1.2
    )
    np.testing.assert_allclose(injected, original, rtol=0.0, atol=1e-12)
```

- [ ] **Step 2: Run tests and verify the module is absent**

```bash
PYTHONPATH=src /Users/joduq24/Desktop/decomb/.venv/bin/python -m pytest \
  tests/test_pump_recovery.py -q
```

Expected: collection fails with `ImportError: cannot import name 'pump_recovery'`.

- [ ] **Step 3: Implement strict harmonic enumeration and Fourier projection**

```python
"""Target-blind prediction of a periodic artifact from its higher harmonics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

LOWEST_REFERENCE_HZ = 20.0
HIGHEST_REFERENCE_HZ = 95.0


def high_harmonic_numbers(fundamental_hz, nyquist_hz):
    if not np.isfinite(fundamental_hz) or fundamental_hz <= 0.0:
        raise ValueError("fundamental_hz must be finite and positive")
    if not np.isfinite(nyquist_hz) or nyquist_hz <= HIGHEST_REFERENCE_HZ:
        raise ValueError("Nyquist must exceed the 95 Hz reference ceiling")
    first = int(np.ceil(LOWEST_REFERENCE_HZ / fundamental_hz))
    last = int(np.floor(HIGHEST_REFERENCE_HZ / fundamental_hz))
    harmonics = tuple(range(first, last + 1))
    if len(harmonics) < 2:
        raise ValueError("at least two adjacent high harmonics are required")
    return harmonics


def _validated_data(data):
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("data must be a finite channels-by-samples array")
    return values


def harmonic_coefficients(data, sampling_frequency_hz, bounds, frequencies_hz):
    values = _validated_data(data)
    windows = tuple((int(start), int(stop)) for start, stop in bounds)
    if not windows or len({stop - start for start, stop in windows}) != 1:
        raise ValueError("bounds must contain equal-length windows")
    window_samples = windows[0][1] - windows[0][0]
    frequencies = np.asarray(frequencies_hz, dtype=float)
    bins = frequencies * window_samples / sampling_frequency_hz
    rounded_bins = np.rint(bins).astype(int)
    if not np.allclose(bins, rounded_bins, rtol=0.0, atol=1e-12):
        raise ValueError("frequencies must lie exactly on the window Fourier grid")
    if rounded_bins[0] <= 0 or rounded_bins[-1] >= window_samples // 2:
        raise ValueError("frequencies must lie strictly between DC and Nyquist")
    coefficients = []
    for start, stop in windows:
        if start < 0 or stop > values.shape[1] or stop <= start:
            raise ValueError("window bounds must lie inside data")
        spectrum = np.fft.rfft(values[:, start:stop], axis=-1)
        coefficients.append(2.0 * spectrum[:, rounded_bins] / window_samples)
    return np.stack(coefficients)


def extract_adjacent_features(
    data, sampling_frequency_hz, bounds, *, fundamental_hz
):
    harmonics = high_harmonic_numbers(
        fundamental_hz, sampling_frequency_hz / 2.0
    )
    frequencies = np.asarray(harmonics) * fundamental_hz
    coefficients = harmonic_coefficients(
        data, sampling_frequency_hz, bounds, frequencies
    )
    products = coefficients[:, :, 1:] * np.conj(coefficients[:, :, :-1])
    return products.mean(axis=1)
```

Rectangular projection is intentional: a 20 s window contains exactly 24 cycles at
1.2 Hz, making the target bin orthogonal to every selected high-harmonic bin. A taper would
leak the target into the predictors.

- [ ] **Step 4: Run Task 1 tests and verify `2 passed`**

- [ ] **Step 5: Commit Task 1**

```bash
git add src/decomb/pump_recovery.py tests/test_pump_recovery.py
git commit -m "Add target-blind pump harmonic features"
```

### Task 2: Fit a participant-independent complex predictor

**Files:**
- Modify: `src/decomb/pump_recovery.py`
- Modify: `tests/test_pump_recovery.py`

- [ ] **Step 1: Write failing prediction and channel-contract tests**

```python
def test_reduced_rank_model_predicts_a_held_out_fundamental():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(120, 6)) + 1j * rng.normal(size=(120, 6))
    mapping = rng.normal(size=(6, 3)) + 1j * rng.normal(size=(6, 3))
    targets = features @ mapping
    model = pump_recovery.fit_complex_model(
        features[:90], targets[:90], ("Fz", "Cz", "Pz"), rank=6, penalty=1e-8
    )
    predicted = model.predict(features[90:], ("Fz", "Cz", "Pz"))
    np.testing.assert_allclose(predicted, targets[90:], rtol=1e-6, atol=1e-8)


def test_model_refuses_reordered_channels():
    features = np.eye(4, dtype=complex)
    targets = np.ones((4, 2), dtype=complex)
    model = pump_recovery.fit_complex_model(
        features, targets, ("Fz", "Cz"), rank=2, penalty=1.0
    )
    with pytest.raises(ValueError, match="channel names"):
        model.predict(features, ("Cz", "Fz"))
```

- [ ] **Step 2: Run tests and verify `fit_complex_model` is absent**

- [ ] **Step 3: Implement reduced-rank complex ridge regression**

```python
@dataclass(frozen=True)
class CrossHarmonicModel:
    channel_names: tuple[str, ...]
    feature_mean: NDArray[np.complex128]
    feature_scale: NDArray[np.float64]
    target_mean: NDArray[np.complex128]
    weights: NDArray[np.complex128]

    def predict(self, features, channel_names):
        if tuple(channel_names) != self.channel_names:
            raise ValueError("channel names do not match the fitted model")
        values = np.asarray(features, dtype=np.complex128)
        if values.ndim != 2 or values.shape[1] != self.weights.shape[0]:
            raise ValueError("features do not match the fitted model")
        standardized = (values - self.feature_mean) / self.feature_scale
        return self.target_mean + standardized @ self.weights


def fit_complex_model(features, targets, channel_names, *, rank, penalty):
    predictors = np.asarray(features, dtype=np.complex128)
    responses = np.asarray(targets, dtype=np.complex128)
    names = tuple(str(name) for name in channel_names)
    if predictors.ndim != 2 or responses.shape != (predictors.shape[0], len(names)):
        raise ValueError("features and targets have incompatible shapes")
    if not np.isfinite(predictors).all() or not np.isfinite(responses).all():
        raise ValueError("training arrays must be finite")
    if not isinstance(rank, int) or not 1 <= rank <= min(predictors.shape):
        raise ValueError("rank must fit inside the feature matrix")
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("penalty must be finite and positive")
    feature_mean = predictors.mean(axis=0)
    feature_scale = np.sqrt(np.mean(np.abs(predictors - feature_mean) ** 2, axis=0))
    if np.any(feature_scale == 0.0):
        raise ValueError("constant cross-harmonic features cannot train a model")
    target_mean = responses.mean(axis=0)
    standardized = (predictors - feature_mean) / feature_scale
    centered_targets = responses - target_mean
    left, singular_values, right = np.linalg.svd(standardized, full_matrices=False)
    left = left[:, :rank]
    singular_values = singular_values[:rank]
    right = right[:rank]
    shrinkage = singular_values / (singular_values**2 + penalty)
    weights = right.conj().T @ (
        shrinkage[:, None] * (left.conj().T @ centered_targets)
    )
    return CrossHarmonicModel(names, feature_mean, feature_scale, target_mean, weights)
```

- [ ] **Step 4: Add grouped inner validation**

Add `select_model()` accepting participant labels. Evaluate
`ranks=(4, 8, 16, 32)` and `penalties=(1e-6, 1e-4, 1e-2, 1.0, 100.0)` using
leave-one-training-participant-out normalized complex squared error. Average errors per
participant before selecting so long recordings cannot dominate. Tests use three synthetic
groups, assert every group is held out once, and assert repeated selection is identical.

- [ ] **Step 5: Run all pump-recovery tests and commit**

```bash
git add src/decomb/pump_recovery.py tests/test_pump_recovery.py
git commit -m "Fit cross-participant pump predictor"
```

### Task 3: Add familywise pump-lock inference

**Files:**
- Modify: `src/decomb/pump_recovery.py`
- Modify: `tests/test_pump_recovery.py`

- [ ] **Step 1: Write failing null and locked-signal tests**

```python
def test_pump_lock_test_separates_shuffled_and_locked_coefficients():
    rng = np.random.default_rng(8)
    prediction = rng.normal(size=(80, 4)) + 1j * rng.normal(size=(80, 4))
    locked = prediction + 0.05 * (
        rng.normal(size=(80, 4)) + 1j * rng.normal(size=(80, 4))
    )
    shuffled = rng.permutation(prediction, axis=0)
    assert pump_recovery.pump_lock_test(
        locked, prediction, surrogate_count=999, seed=10
    ).p_value <= 0.01
    assert pump_recovery.pump_lock_test(
        shuffled, prediction, surrogate_count=999, seed=10
    ).p_value > 0.01
```

- [ ] **Step 2: Run the test and verify `pump_lock_test` is absent**

- [ ] **Step 3: Implement maximum-coherence permutation inference**

```python
@dataclass(frozen=True)
class PumpLockTest:
    maximum_coherence: float
    p_value: float
    surrogate_count: int


def _channel_coherence(observed, predicted):
    numerator = np.abs(np.sum(observed * np.conj(predicted), axis=0)) ** 2
    denominator = np.sum(np.abs(observed) ** 2, axis=0) * np.sum(
        np.abs(predicted) ** 2, axis=0
    )
    if np.any(denominator <= 0.0):
        raise ValueError("pump-lock coherence requires non-zero channel energy")
    return numerator / denominator


def pump_lock_test(observed, predicted, *, surrogate_count, seed):
    values = np.asarray(observed, dtype=np.complex128)
    reference = np.asarray(predicted, dtype=np.complex128)
    if values.shape != reference.shape or values.ndim != 2:
        raise ValueError("observed and predicted coefficients must share a 2D shape")
    if surrogate_count < 99:
        raise ValueError("at least 99 surrogates are required")
    observed_maximum = float(_channel_coherence(values, reference).max())
    rng = np.random.default_rng(seed)
    surrogate_maxima = np.empty(surrogate_count)
    for index in range(surrogate_count):
        permuted = reference[rng.permutation(reference.shape[0])]
        surrogate_maxima[index] = _channel_coherence(values, permuted).max()
    exceedances = int(np.sum(surrogate_maxima >= observed_maximum))
    p_value = (exceedances + 1.0) / (surrogate_count + 1.0)
    return PumpLockTest(observed_maximum, p_value, surrogate_count)
```

The channel maximum controls the within-recording family. The report applies Holm
correction across the 90 recording-level p-values.

- [ ] **Step 4: Run tests and commit**

```bash
git add src/decomb/pump_recovery.py tests/test_pump_recovery.py
git commit -m "Test residual locking to the pump comb"
```

### Task 4: Reconstruct the artifact and prove injection preservation

**Files:**
- Modify: `src/decomb/pump_recovery.py`
- Modify: `tests/test_pump_recovery.py`

- [ ] **Step 1: Write failing reconstruction and paired-injection tests**

```python
def test_overlap_add_reconstructs_one_fundamental_without_boundary_holes():
    sfreq = 1000.0
    bounds = ((0, 20_000), (10_000, 30_000), (20_000, 40_000))
    coefficient = 2.0 * np.exp(1j * 0.3)
    coefficients = np.full((3, 2), coefficient)
    artifact = pump_recovery.reconstruct_artifact(
        40_000, bounds, coefficients, sfreq, 1.2
    )
    times = np.arange(40_000) / sfreq
    expected = np.real(coefficient * np.exp(2j * np.pi * 1.2 * times))
    np.testing.assert_allclose(artifact[0], expected, rtol=0.0, atol=1e-10)


def test_target_blind_cleaner_preserves_an_exact_frequency_injection():
    sfreq = 1000.0
    times = np.arange(40_000) / sfreq
    bounds = ((0, 20_000), (10_000, 30_000), (20_000, 40_000))
    background = np.vstack(
        [np.sin(2 * np.pi * 20.4 * times), np.cos(2 * np.pi * 21.6 * times)]
    )
    injection = np.vstack(
        [0.2 * np.sin(2 * np.pi * 1.2 * times), 0.3 * np.cos(2 * np.pi * 1.2 * times)]
    )
    predicted = np.full((len(bounds), 2), 0.4 + 0.2j)
    clean_background = pump_recovery.subtract_predicted_artifact(
        background, bounds, predicted, sfreq, 1.2
    )
    clean_injected = pump_recovery.subtract_predicted_artifact(
        background + injection, bounds, predicted, sfreq, 1.2
    )
    np.testing.assert_allclose(
        clean_injected - clean_background, injection, rtol=0.0, atol=1e-12
    )
```

- [ ] **Step 2: Run tests and verify reconstruction functions are absent**

- [ ] **Step 3: Implement weighted overlap-add**

```python
def reconstruct_artifact(
    n_times, bounds, coefficients, sampling_frequency_hz, frequency_hz
):
    windows = tuple(bounds)
    values = np.asarray(coefficients, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] != len(windows):
        raise ValueError("one coefficient row is required per window")
    artifact = np.zeros((values.shape[1], n_times), dtype=float)
    weights = np.zeros(n_times, dtype=float)
    for row, (start, stop) in enumerate(windows):
        sample_times = np.arange(start, stop) / sampling_frequency_hz
        carrier = np.exp(2j * np.pi * frequency_hz * sample_times)
        window = np.hamming(stop - start)
        artifact[:, start:stop] += np.real(values[row, :, None] * carrier) * window
        weights[start:stop] += window
    covered = weights > 0.0
    artifact[:, covered] /= weights[covered]
    return artifact


def subtract_predicted_artifact(
    data, bounds, predicted_coefficients, sampling_frequency_hz, frequency_hz
):
    values = _validated_data(data)
    artifact = reconstruct_artifact(
        values.shape[1],
        bounds,
        predicted_coefficients,
        sampling_frequency_hz,
        frequency_hz,
    )
    return values - artifact
```

- [ ] **Step 4: Add strict preservation metrics**

Add `InjectionPreservation` with relative waveform error, amplitude retention, phase error,
and `passes`. Compute recovered injection as `clean(injected) - clean(background)` and the
best complex gain against the known injection. Pass only when waveform error is at most
0.01, amplitude is in `[0.99, 1.01]`, and phase error is at most one degree. Tests cover
exact pass boundaries and reject zero-energy injections.

- [ ] **Step 5: Run tests and commit**

```bash
git add src/decomb/pump_recovery.py tests/test_pump_recovery.py
git commit -m "Reconstruct target-blind pump artifact"
```

### Task 5: Run leave-one-participant-out recovery on the cohort

**Files:**
- Create: `studies/2026-08-20-low-frequency-recovery/run_experiment.py`
- Create: `studies/2026-08-20-low-frequency-recovery/README.md`
- Modify: `tests/test_pump_recovery.py`

- [ ] **Step 1: Write a failing miniature cohort test**

Build three synthetic participants with two recordings each and pass pre-extracted arrays
to `run_leave_one_participant_out`. Assert:

```python
assert set(results["test_participant"]) == {"sub-01", "sub-02", "sub-03"}
assert all(
    row.test_participant not in row.training_participants
    for row in results.itertuples()
)
assert results["predicted_coefficients"].notna().all()
```

- [ ] **Step 2: Implement deterministic extraction and outer folds**

`run_experiment.py` must load the requested config, discover all source runs, obtain 20 s
50%-overlap bounds from `recordings.valid_window_bounds`, use non-bad EEG for predictors and
all named EEG channels for target coefficients, cache arrays with source hashes, select rank
and penalty inside training participants only, freeze the model before reading test targets,
and run source and residual pump-lock tests with 999 surrogates.

The command is:

```bash
PYTHONPATH=src /Users/joduq24/Desktop/decomb/.venv/bin/python \
  studies/2026-08-20-low-frequency-recovery/run_experiment.py \
  --config /Users/joduq24/Desktop/decomb/decomb.yaml \
  --output-dir /private/tmp/decomb-1p2-recovery
```

Write `recording_results.tsv` atomically only after all 90 recordings succeed. No broad
exception handler, skipped row, or partial success is permitted.

- [ ] **Step 3: Add paired real-background injections**

For every held-out recording, generate deterministic stationary, intermittent, and slowly
amplitude-modulated 1.2 Hz waveforms at phases `0` and `pi/2`, two spatially balanced
topographies, and component-to-background ratios `-20`, `-10`, and `0` dB. Re-extract high
predictors after injection, require equality to the uninjected predictors within `1e-12`,
clean both arrays with the frozen model, and write all preservation metrics to
`injection_results.tsv`. Any failed injection aborts instead of disappearing.

- [ ] **Step 4: Add comparison arms without changing production**

Measure the unchanged source, current `spectrum_fit` subtraction at 1.2 Hz, proposed
cross-harmonic prediction, and a narrow FIR notch with declared response. Run MNE
`apply_pca_obs` only when unwrapped predicted phase yields at least two monotonic cycle
events; pass all EEG names explicitly and record OBS only as a suppression benchmark.

Derive OBS events without reading the observed target coefficients:

```python
def predicted_cycle_times(bounds, predicted, sfreq, frequency_hz):
    centers = np.asarray([(start + stop - 1) / 2.0 for start, stop in bounds])
    unit = predicted / np.maximum(np.abs(predicted), np.finfo(float).tiny)
    common = unit.sum(axis=1)
    if np.any(np.abs(common) == 0.0):
        raise ValueError("predicted pump phasor cancels across channels")
    carrier_phase = 2.0 * np.pi * frequency_hz * centers / sfreq
    total_phase = np.unwrap(carrier_phase + np.angle(common))
    if np.any(np.diff(total_phase) <= 0.0):
        raise ValueError("predicted pump phase is not monotonic")
    first_cycle = int(np.ceil(total_phase[0] / (2.0 * np.pi)))
    last_cycle = int(np.floor(total_phase[-1] / (2.0 * np.pi)))
    crossings = 2.0 * np.pi * np.arange(first_cycle, last_cycle + 1)
    return np.interp(crossings, total_phase, centers) / sfreq
```

- [ ] **Step 5: Run the miniature test and targeted suite**

```bash
PYTHONPATH=src /Users/joduq24/Desktop/decomb/.venv/bin/python -m pytest \
  tests/test_pump_recovery.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the cohort harness**

```bash
git add tests/test_pump_recovery.py \
  studies/2026-08-20-low-frequency-recovery/run_experiment.py \
  studies/2026-08-20-low-frequency-recovery/README.md
git commit -m "Evaluate target-blind 1.2 Hz recovery"
```

### Task 6: Produce and verify the decision report

**Files:**
- Create: `studies/2026-08-20-low-frequency-recovery/report.py`
- Create after execution: `studies/2026-08-20-low-frequency-recovery/cohort_result.md`
- Create after execution: `studies/2026-08-20-low-frequency-recovery/recording_results.tsv`
- Create after execution: `studies/2026-08-20-low-frequency-recovery/injection_results.tsv`

- [ ] **Step 1: Write a failing report-completeness test**

Create synthetic TSVs with one missing recording and assert
`ValueError("expected 90 recording results")`. Create a complete fixture and assert the
report includes counts for `artifact_not_detected`, `recovered`, and `not_recoverable`, plus
worst injection waveform, amplitude, and phase metrics.

- [ ] **Step 2: Implement strict aggregation**

Require exactly 90 unique recordings, every planned injection cell, no missing values, and
matching source hashes. Apply Holm correction to source and residual pump-lock p-values.
Assign `artifact_not_detected` when corrected source p is above 0.01, `recovered` when the
source is significant, residual is null, and every injection passes, and `not_recoverable`
otherwise. Report model choices and bandwidth implications without calling target-band
fitting recovery.

- [ ] **Step 3: Run targeted verification before the cohort**

```bash
PYTHONPATH=src /Users/joduq24/Desktop/decomb/.venv/bin/python -m pytest \
  tests/test_pump_recovery.py -q
/Users/joduq24/Desktop/decomb/.venv/bin/ruff check \
  src/decomb/pump_recovery.py tests/test_pump_recovery.py \
  studies/2026-08-20-low-frequency-recovery
git diff --check
```

Expected: zero failures and zero lint or whitespace errors. Run the full repository suite
separately and report the known `tests/test_run_pipeline.py` missing-runner errors rather
than suppressing them.

- [ ] **Step 4: Run the complete cohort experiment**

Run the Task 5 command with a fresh output directory. Require complete TSVs and exit zero;
the scientific outcome is deliberately not predicted.

- [ ] **Step 5: Generate and inspect the report**

```bash
PYTHONPATH=src /Users/joduq24/Desktop/decomb/.venv/bin/python \
  studies/2026-08-20-low-frequency-recovery/report.py \
  --input-dir /private/tmp/decomb-1p2-recovery \
  --output studies/2026-08-20-low-frequency-recovery/cohort_result.md
```

Read every failed-recovery reason and inspect the worst recording before drawing a cohort
conclusion.

- [ ] **Step 6: Copy complete products and commit**

```bash
cp /private/tmp/decomb-1p2-recovery/recording_results.tsv \
  studies/2026-08-20-low-frequency-recovery/recording_results.tsv
cp /private/tmp/decomb-1p2-recovery/injection_results.tsv \
  studies/2026-08-20-low-frequency-recovery/injection_results.tsv
git add studies/2026-08-20-low-frequency-recovery
git commit -m "Report 1.2 Hz recovery experiment"
```

The handoff states one evidence-backed conclusion: artifact not detected at 1.2 Hz,
target-blind recovery qualified for named recordings, or recovery failed and 1.2 Hz must be
notched and declared unavailable.
