# Cycle-Synchronous 1.2 Hz Pump-Recovery Test

## Objective

Test, without changing the production pipeline, whether a run-specific periodic artifact
model can suppress the 1.2 Hz comb substantially while retaining useful neural signal. The
experiment distinguishes practical recovery of non-pump-locked activity from the
unidentifiable case of neural activity exactly phase-locked to the pump.

## Scope

The experiment reads the 90 source BrainVision recordings and writes only to a fresh
temporary directory. It does not alter the BIDS source, the existing derivative, the YAML
configuration, or any production caller. A positive result would require a separate
integration design.

## Pump phase

Estimate instantaneous pump phase independently for each recording using only EEG content
between 20 and 95 Hz. Narrow analytic signals are extracted at consecutive 1.2 Hz
harmonics. For each adjacent harmonic pair, multiplying the upper analytic signal by the
complex conjugate of the lower produces a phasor at the 1.2 Hz fundamental. Robustly
combine normalized pair phasors across non-bad channels, smooth only enough to reject
cycle-scale phase jitter, unwrap the result, and derive monotonically increasing cycle
crossings.

The phase estimator must never read EEG below 20 Hz. It must fail explicitly if the
high-harmonic clock has insufficient amplitude, non-monotonic phase, or implausible cycle
spacing.

## Recovery arms

### Cross-fitted average artifact subtraction

Map each EEG sample to pump phase and divide complete cycles into five contiguous folds.
For each held-out fold, estimate one robust phase-binned artifact template per channel from
the other four folds and subtract the interpolated template from the held-out cycles. Every
sample is therefore cleaned by a template that did not use that sample's cycle. Templates
are run-specific so no topography or amplitude is assumed to transfer between participants.

### MNE PCA-OBS benchmark

Pass the high-only pump-cycle times to `mne.preprocessing.apply_pca_obs` with all EEG names
explicitly selected. Test one through four components on the pilot recordings. PCA-OBS is a
suppression benchmark rather than the preferred recovery method because it builds a basis
from the target EEG and fits that basis to every cycle.

### Controls

Retain the unchanged source and the current exact-frequency sinusoid subtraction as
comparators. Run a phase-shifted negative-control template to detect improvements that do
not depend on correct pump alignment.

## Staged execution

Run a deterministic six-recording pilot first: one run from six distinct participants,
spanning the observed 1.2 Hz line-strength distribution. The pilot selects between robust
mean and median templates, phase-bin count, and PCA-OBS component count using predeclared
metrics. Only a method that passes the pilot proceeds unchanged to all 90 recordings.

The pilot is a computational gate, not cohort evidence. Final claims use all recordings
with the participant as the independent unit.

## Measurements

For every arm and recording, measure:

- 1.2 Hz peak amplitude and power relative to adjacent local bins;
- stationary and time-varying pump-locked coefficient energy;
- residual 1.2 Hz coherence with the high-only pump clock;
- residual 20--95 Hz comb prominence;
- off-target spectral change, broadband correlation, and change RMS;
- amplitude, phase, and waveform retention for paired synthetic injections.

Injection cells include stationary, intermittent, and amplitude-modulated 1.2 Hz signals.
Each is tested both with phase independent of the pump and exactly pump-locked, at multiple
phases, spatial topographies, and amplitudes. Adjacent-frequency injections at 1.1 and
1.3 Hz test spectral selectivity. The report must state directly that pump-locked neural
injections are not identifiable from the artifact if they are attenuated.

## Decision rules

Call the method practically recoverable only if, on the full cohort:

1. the median 1.2 Hz artifact-amplitude reduction is at least 90%;
2. at least 90% of recordings leave the 1.2 Hz peak no more than 1 dB above their local
   background;
3. the participant-level improvement over the phase-shifted control is significant at
   one-sided alpha 0.01;
4. median off-target spectral amplitude changes by at most 5%; and
5. non-pump-locked injections retain 95--105% amplitude with at most five degrees phase
   error and at most 5% relative waveform error.

Failure of any rule means the current recordings do not support reliable retrospective
recovery. Pump-locked injection loss is reported as an identifiability boundary even if the
practical rules pass.

## Outputs

Write the phase-quality table, pilot comparison, full recording metrics, injection metrics,
and a Markdown decision report under one fresh `/private/tmp` experiment directory. Output
tables are atomic and complete: any recording failure aborts the cohort result rather than
creating a partial success.
