# Arm comparison and subtraction validation, 2026-08-19

Preserved from a scratchpad that was subject to cleanup. Nothing here is imported by
`decomb`; it is the evidence behind the scanner-harmonic authorization fix and the
subtraction-fidelity numbers.

**The scripts import `decomb.recovery_benchmark`, `decomb.recovery_evaluation` and
`decomb.neural_recovery_validation`, which exist only on `feature/signal-recovery`.**
They do not run against `main` alone.

## Cohort arm comparison

- `fixed_config.py` — the six-arm harness.
- `final_config.tsv` — 450 rows, 90 recordings x 5 arms.
- `arm_combined.tsv` — 90 rows, the `combined` arm.
- `*_probe.tsv`, `*.log` — two-recording probes and run logs.

`combined` wins every cohort mean (4.7 residual peaks vs 32.2 for current notching,
gamma 0.914 vs 0.592). It does not win every median: `comb_subtracted` has a better
median gamma and comb flatness, and `lines_declared` a better median correlation and
change-RMS. Against `lines_declared` the mean advantage comes from a rescued tail, not
uniform superiority -- it loses correlation on 75 of 90 recordings and change-RMS on 76,
by negligible margins (median correlation difference -0.00012).

Residuals are run-localized, not participant-localized: 79% of the variance in residual
peaks is within participants. sub-0008 is the one genuine per-participant case, never
clean in any of its six runs.

## Pre-hoc gate search (negative result)

- `comb_evidence.py`, `comb_evidence.tsv` — comb evidence from the original spectrum
  only, all 90 recordings.

No pre-hoc predictor separates the five excavation failures. At a threshold catching all
five, every candidate flags 31-46 of the other 85. sub-0003 run-1 has near-median comb
evidence and still fails. This is why the fix changes the authorization rule rather than
gating on a threshold.

## Subtraction fidelity at 1.2 Hz comb teeth

- `comb_injection.py` + `comb_injection/` — injections anchored on consecutive on-grid
  teeth, 3 recordings, 72 trials.
- `comb_sweep.py`, `comb_sweep*.tsv` — retention vs offset from a tooth, at 10 s and
  20 s subtraction windows.

Activity exactly on a subtracted tooth is destroyed: retention 0.109, band power
-12.8 dB, band phase error 94 degrees. Stationary and intermittent components lose 96%.
Recovery is complete within about two frequency bins of the subtraction window, and the
two windows collapse onto one curve when offset is expressed in bins:

| offset (bins) | 10 s | 20 s |
|---:|---:|---:|
| 0.00 | 0.000 | 0.000 |
| 0.50 | 0.235 | 0.208 |
| 1.00 | 0.967 | 0.963 |
| 1.50 | 1.079 | 1.070 |
| 2.00 | 0.999 | 0.999 |

Practical rule: treat +/- 2/T Hz around every subtracted frequency as unusable, where T
is the subtraction window (+/- 0.1 Hz at 20 s, 17% of 20-95 Hz). The ~7% overshoot at
1.5 bins reproduces in both windows and is why the boundary is 2 bins rather than 1.

A subtracted frequency is **not** marked unavailable in the manifest the way a notched
one is, so this exclusion cannot currently be derived from the derivative. Shipping a
machine-readable list of subtracted frequencies is the outstanding provenance work.

## Caveats

- `peaks_above_2dB` is prominence-based, so the notching arms' counts are inflated by
  their own notched surrounds. Comparisons among subtraction arms are sound.
- Injection trials report `artifact_gate_passed = False`; that is the *injected*
  condition. `background_terminal_residual_detector_null` is True throughout.
- In `comb_sweep_w20.tsv`, sub-0003's offsets >= 0.05 Hz fall inside the mains stopband
  (its neighbour tooth is 60.0 Hz) and measure notching, not subtraction. Filter on
  `injected_frequency_fir_unavailable`. sub-0013 run-3 has no on-grid tooth pair at 20 s.
