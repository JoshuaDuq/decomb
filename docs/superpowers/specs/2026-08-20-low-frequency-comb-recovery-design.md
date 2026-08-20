# Target-blind recovery experiment for the 1.2 Hz pump fundamental

**Date:** 2026-08-20
**Branch:** `feat/apply-subtraction`
**Status:** approved for specification; implementation awaits written-spec review

## Question

Can the artifact component at exactly 1.2 Hz be estimated without using the held-out
recording's 1.2 Hz EEG, so that independent neural activity at the same frequency remains
available?

The experiment answers this question before any production behavior changes. A successful
high-frequency result does not answer it: the current comb stage starts at 20 Hz and the
written derivatives leave the low teeth essentially unchanged.

## Scientific constraint

Artifact and neural activity at the same frequency are not identifiable from that Fourier
coefficient alone. With no pump marker, pump-off recording, accelerometer, microphone, or
carbon-wire loop, recovery is defensible only if the artifact estimate is derived from
independent structure.

This cohort provides two such candidates:

1. the strong 1.2 Hz-spaced harmonics between 20 and 95 Hz, which can provide a time-varying
   pump clock without inspecting 1.2 Hz; and
2. the other participants, which can provide the relationship between that clock and the
   1.2 Hz sensor artifact without fitting the held-out participant's target-frequency EEG.

If these do not predict the target accurately, the experiment must conclude that 1.2 Hz is
not recoverable from the existing recordings. It must then be notched and declared
unavailable.

## Approaches considered

### 1. Leave-one-participant-out cross-harmonic prediction — primary

Estimate the pump phase and amplitude state from high-frequency teeth only. Learn, on the
other participants, how that state predicts the pump-coherent 1.2 Hz waveform at each EEG
sensor. Apply the frozen predictor to the held-out participant without reading that
participant's 1.2 Hz EEG.

This is the only blind approach considered eligible to claim exact-frequency preservation:
an injected 1.2 Hz neural signal in the held-out data cannot become a training target.

### 2. Adaptive cycle averaging / PCA-OBS — suppression benchmark only

Infer pump-cycle times and construct a sliding template from approximately 21 neighboring
cycles, following Rothlübbers et al.'s helium-pump method. MNE's `apply_pca_obs` can provide
an OBS comparison once pump events are known.

This method is likely to suppress the comb well, but it estimates the 1.2 Hz template from
the same recording. A stationary neural oscillation at exactly 1.2 Hz is phase-stationary in
the same cycle coordinates and can enter the template. The arm is therefore a useful
suppression ceiling, but it cannot qualify as recovery unless the exact-frequency injection
gate unexpectedly proves otherwise.

References:

- [MNE `apply_pca_obs`](https://mne.tools/stable/generated/mne.preprocessing.apply_pca_obs.html)
- [Rothlübbers et al., 2015](https://pubmed.ncbi.nlm.nih.gov/25344750/)

### 3. Narrow 1.2 Hz notch — safety control

Remove a predeclared narrow interval around 1.2 Hz and mark the full response unavailable.
This is not recovery. It is the required outcome when the primary method fails and the
reference against which recovered bandwidth is measured.

## Scope

The first experiment targets only the 1.2 Hz fundamental. Harmonics from 2.4 through
19.2 Hz are not promoted into production merely because the fundamental succeeds. Each must
later pass the same experiment independently.

The current 20–95 Hz processing remains unchanged. The experiment reads source recordings
and writes study outputs only; it does not overwrite the shipped derivative.

## Data separation

The outer validation has 15 leave-one-participant-out folds. For each fold:

1. fourteen participants are training data;
2. the remaining participant is the untouched test data;
3. feature selection, tooth weights, model rank, regularization, and every threshold are
   selected inside the training participants only; and
4. the frozen model is evaluated on all six recordings of the held-out participant.

The held-out participant's 1.2 Hz samples must not influence pump-clock estimation,
normalization, parameter selection, artifact prediction, or acceptance thresholds. A test
must demonstrate that adding a 1.2 Hz signal leaves the predicted artifact unchanged within
numerical tolerance.

## Pump-clock estimator

Only valid continuous-acquisition spans are processed. No window crosses an `edge` or
contains a `bad_acq_skip` annotation.

For each span:

1. compute tapered complex coefficients at the declared high-frequency teeth from 20.4 to
   94.8 Hz on non-bad EEG channels;
2. choose reliable teeth and channel weights using training participants only;
3. solve a robust circular model for the common time-varying fundamental phase whose
   multiples explain the selected high-harmonic phases;
4. derive a slowly varying high-harmonic amplitude state; and
5. measure clock quality on high harmonics excluded from the fit.

The target participant's 1.2 Hz band is never an input. If the held-out high harmonics do
not support a coherent clock, that recording receives a failed-recovery decision rather
than an estimated phase.

## Cross-harmonic artifact predictor

Training recordings are expressed in pump-phase coordinates. A deterministic, regularized
low-rank model maps the high-harmonic state to the complex 1.2 Hz artifact coefficient for
each named EEG sensor. Robust fitting limits the influence of participant-specific neural
activity in the training targets.

For a held-out recording, the frozen model receives only its high-frequency pump state. It
predicts a per-channel 1.2 Hz artifact waveform, which is subtracted from every EEG channel.
The model does not refit or rescale itself using the held-out target band.

Channel names and sampling geometry must match the model exactly. Missing channels,
non-finite data, an unsupported annotation layout, a failed clock, or a rank-deficient fit
is an error or an explicit failed-recovery result; there is no silent alternative path.

## Experimental arms

Every held-out recording is evaluated under the same valid spans:

1. unchanged source;
2. current MNE `spectrum_fit` subtraction at 1.2 Hz;
3. adaptive cycle averaging / PCA-OBS;
4. proposed target-blind cross-harmonic prediction; and
5. narrow notch with declared unavailability.

The first arm measures the starting artifact, arms two and three reveal the preservation
cost of same-recording fitting, arm four is the recovery candidate, and arm five is the
safe bandwidth-cost reference.

## Validation

### Artifact presence

Before cleaning, the source's 1.2 Hz component is tested for phase locking to the pump
clock estimated from held-out high harmonics. The test uses phase-randomized surrogates and
a recording-level maximum statistic at familywise alpha 0.01 across channels and windows.

If the source is null, no recovery is attempted and the recording is reported as
`artifact_not_detected`. If the source is significant, the recovery candidate must pass
both gates below. This prevents the model from subtracting a cross-participant prediction
where the held-out recording provides no evidence of the artifact.

### Artifact suppression

The 1.2 Hz residual is tested for phase locking to the independently estimated pump clock.
The same phase-randomized, recording-level maximum statistic controls the familywise error
rate at alpha 0.01. Recovery passes only when no significant pump-locked residual remains
in the written candidate output.

Peak height is secondary because the 1/f slope biases local prominence near 1.2 Hz.

### Exact-frequency preservation

Paired injections are added to the held-out recording only. They cover:

- stationary 1.2 Hz sinusoids at multiple phases and spatial topographies;
- slowly amplitude-modulated 1.2 Hz activity;
- intermittent 1.2 Hz bursts; and
- multiple component-to-background energy ratios derived from the uninjected background.

For cleaner `C`, background `X`, and injection `S`, the recovered injection is
`C(X + S) - C(X)`. Every trial must satisfy:

- relative waveform error no greater than 1%;
- amplitude retention between 0.99 and 1.01;
- absolute phase error no greater than 1 degree; and
- no statistically supported change outside the injected support.

These are per-trial requirements, not cohort averages. A mean cannot hide a destructive
phase or topography.

### Generalization and decision

A recording is eligible to keep 1.2 Hz only when both suppression and every preservation
trial pass. A failed clock, residual pump locking, or failed injection marks 1.2 Hz as not
recoverable for that recording.

Production integration is considered only if at least one recording qualifies and the
implementation is target-blind by construction. Each non-qualifying recording must use the
predeclared narrow notch and manifest the interval as unavailable. If no recording
qualifies, the study conclusion is that the existing cohort cannot support recovery at
1.2 Hz.

## Outputs

The study writes:

- one TSV row per participant fold, recording, arm, and injection;
- pump-clock quality and held-out harmonic residuals;
- artifact suppression, injection amplitude, phase, and waveform error;
- pass/fail reasons without exception suppression; and
- a Markdown cohort report that separates recovered recordings from notched recordings.

All randomness is seeded from recorded identifiers. Model parameters, fold membership,
MNE/SciPy/NumPy versions, and the source file hashes are recorded.

## Tests required before the cohort run

1. Synthetic drifting periodic artifact: high harmonics recover the known phase trajectory.
2. Target blindness: changing only 1.2 Hz does not change the predicted artifact.
3. Known mixed signal: the predicted artifact is removed while an independent exact 1.2 Hz
   injection survives.
4. Leakage sentinel: intentionally exposing held-out 1.2 Hz causes the test to fail.
5. Annotation boundaries: no estimator window crosses excluded spans.
6. Participant folds: every participant appears once as test data and never in its training
   set.
7. Determinism: repeated runs produce identical metrics and decisions.
8. Failure behavior: missing channels, invalid geometry, failed clocks, and incomplete
   outputs surface explicitly.

## Interpretation limit

Passing this experiment would show that the artifact predicted from independent
cross-harmonic and cross-participant structure can be removed while controlled neural-like
signals survive. It would not prove recovery of an unknown real neural oscillator for which
no ground truth exists.

If the pump is too phase-stable, its low-to-high harmonic relationship varies with the
participant, or the 1.2 Hz component is not predictably coupled to the high comb, exact
recovery is not identifiable from these recordings. The scientifically correct result is
then unavailability, not another target-band fit.
