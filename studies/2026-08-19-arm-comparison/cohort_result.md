# The shipped derivative: what came out, measured on the written files

`decomb apply` then `decomb verify` over all 90 recordings, 2026-08-20, with
`comb_fundamental_hz: 1.2` declared. Every number below is measured on the **written
BrainVision derivative against its source**, not on an in-memory reconstruction
(`derivative_stats.py`, `derivative_stats.tsv`). Availability is read from the manifest.

sub-0008 has a bad ECG recording and is reported separately; cohort figures are n=84.

## Verification

| | |
|---|---|
| recordings verified | **90 / 90** |
| max sample deviation | **0.000e+00 V** |
| rows with any deviation | **0** |
| median stopband change | -44.7 dB |

`apply` completed on all 90, which it cannot do unless `validate_residual_postcondition`
passes on each written file — it refits the derivative read back from disk and raises on any
surviving Holm-significant line or authorized comb. `verify` then re-derived both pre-cascade
decisions from the source and replayed all three stages plus the FIR cascade to bit-identical
samples.

## Is the 1.2 Hz comb gone?

Essentially yes, on 63 teeth tested per recording between 20 and 95 Hz.

| | before | after |
|---|---:|---:|
| comb_db, mean | +2.81 | **-0.09** |
| comb_db, median | +2.75 | **-0.04** |
| comb_db, worst | +5.12 | +0.44 |
| teeth standing >2 dB proud | 33.6 | **1.8** |

Median removal is 101.5% — the teeth end up level with the local floor, very slightly past it.
19 of 84 recordings have **no** tooth standing more than 2 dB proud. The worst residual comb
in the cohort is +0.44 dB, against +5.12 dB before.

## What residual is left?

Not zero. Measured against each recording's own null:

| | |
|---|---|
| teeth above own null p99, mean | 0.6 |
| median | **0** |
| worst recording | 5 |
| recordings with none | **51 / 84** |

**Caveat on this statistic.** The null here excludes only bins within 0.5 Hz of a tooth, not
bins near removed ordinary lines, so it is contaminated by the notches themselves: p99 comes
out at a mean of 6.28 dB from ~150 bins, against ~3.5 dB when ordinary-line targets are also
excluded. That makes this count **conservative** — it under-reports residual peaks. The
"teeth >2 dB proud" figure above (mean 1.8) is the stricter and more comparable one.

## Bandwidth availability

Declared across all three removal stages — subtraction damage, residual stopbands and the FIR
cascade, merged. n=84.

| band | mean | median | min | max |
|---|---:|---:|---:|---:|
| delta | 0.996 | 1.000 | 0.893 | 1.000 |
| theta | **1.000** | 1.000 | 1.000 | 1.000 |
| alpha | 0.994 | 1.000 | **0.528** | 1.000 |
| beta | 0.928 | 0.929 | 0.840 | 0.985 |
| gamma | **0.782** | 0.782 | 0.698 | 0.882 |

Theta is untouched in every recording. Alpha is untouched in all but one. Gamma pays the
cost, as designed: about a fifth of it is spent, and none of it is undeclared.

These reproduce the arm study's prediction for this configuration exactly (gamma 0.782,
beta 0.928), which is the acceptance criterion the design set.

## Fidelity

| | mean | worst |
|---|---:|---:|
| correlation with source | 0.9928 | 0.9380 |
| change RMS | 0.0624 | 0.2244 |

## Outliers worth knowing before analysis

| recording | what | value |
|---|---|---:|
| sub-0009 run-6 | **alpha availability** | **0.528** |
| sub-0008 run-4 | delta availability | 0.222 |
| sub-0011 runs 4, 6 | lowest gamma | 0.698, 0.700 |
| sub-0001 runs 1, 3, 6 | most signal disturbed (change RMS) | 0.20-0.22 |
| sub-0009 run-1 | most teeth above its own null (strict null, p99 1.42 dB) | 5 |

**sub-0009 run-6 deserves a look before it enters an alpha analysis.** Alpha is untouched in
89 of 90 recordings; this one lost 47% of the band, meaning supported lines were detected and
removed inside 8-12.9 Hz. That is the pipeline behaving as designed on the evidence it found,
but it is not a recording to pool naively into an alpha contrast.

## sub-0008, reported separately

| | before | after |
|---|---:|---:|
| comb_db | +4.84 | **-0.27** |
| teeth >2 dB proud | 44.8 | 2.5 |
| correlation | -- | 0.9704 |
| change RMS | -- | 0.0815 |
| gamma availability | -- | 0.723 |

Its comb is the strongest in the cohort and is removed as completely as anywhere else. The
bad ECG shows up in fidelity and in delta availability (0.222 on run-4), not in comb removal.

## What this does not claim

No statistically supported line remains anywhere — that is a hard postcondition, checked on
disk, and it held 90/90. The spectrum is **not** flat: a median of zero and a mean of 0.6
teeth per recording still stand above that recording's own noise, and by the stricter 2 dB
measure a mean of 1.8 teeth remain. Anything published should say "no statistically supported
line remains", which is the actual guarantee.
