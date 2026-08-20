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

## Confirming the comb is gone, against a control

`comb_db` is a median over teeth, so it could in principle read zero while a third of the
comb stood at 0.5-2 dB. It does not. Binning every tooth's prominence on the derivative
showed 21.7 of 63 teeth (34.5%) in that window -- which looks alarming until the same
measurement is made at the **midpoints between teeth**, the same band and estimator with no
comb there by construction:

| | mean |
|---|---:|
| median prominence at teeth | +0.10 dB |
| median prominence at midpoints | **+0.42 dB** |
| excess at teeth | **-0.32 dB** |
| recordings where teeth are significantly higher (one-sided, p<0.05) | **0 / 84** |

Teeth sit *below* the gaps between them. The 0.5-2 dB population is the ordinary roughness
of the 20-95 Hz band, and the non-tooth frequencies carry more of it. The same test on the
**source** data finds the comb in **84 / 84** recordings at +2.57 dB excess, so this is a
null from a method with demonstrated power, not a weak test.

Below 20 Hz there is no comb to begin with: teeth versus midpoints on the source gives
-0.07 dB excess, significant in 0 of 84. Extending removal into theta and alpha would delete
signal at 9.6 and 10.8 Hz -- the centre of the alpha rhythm -- to chase an artifact that is
measurably absent.

## A narrow basis around each target makes it worse

`subtract_multitaper_sinusoids` fits one exact frequency (`notch_widths=0.0`). Widening that
to a +/- w/2 basis, with the declared damage widened to `w/2 + 2/window_s` to match, was
tested to see whether it would absorb comb drift (7-8 recordings per arm):

| `notch_widths` | comb_db median | \|comb\| mean | teeth >2 dB | gamma |
|---:|---:|---:|---:|---:|
| 0.0 (shipped) | -0.15 | **0.19** | **1.25** | **0.795** |
| 0.1 | -0.30 | 0.30 | 1.13 | 0.737 |
| 0.2 | -1.69 | 1.64 | 1.43 | 0.676 |
| 0.4 | -8.06 | 6.43 | 8.71 | 0.502 |

Strictly worse: `comb_db` goes hard negative, which is excavation below the local floor, and
gamma availability collapses. There was nothing to absorb -- `docs/artifact_survey.md` had
already measured frequency drift at SD 0.00 Hz in 62 of 90 recordings, maximum SD 0.048 Hz.
The lines are amplitude-modulated, not drifting.

## The advisory comb mask

`apply` also publishes `comb_analysis_mask.tsv` and `analysis_availability.tsv` in the report
directory. The mask covers all 63 teeth in 20-95 Hz at +/- 0.1 Hz -- the width a subtracted
tooth already declares -- whether or not a given recording removed that tooth. It removes
nothing and is deliberately **not** part of the manifest, which records only what was
destroyed.

| band | declared | with mask |
|---|---:|---:|
| delta / theta / alpha | 0.996 / 1.000 / 0.994 | unchanged |
| beta | 0.928 | 0.869 |
| gamma | 0.782 | 0.731 |

## Where the constants come from

The four constants in `residual.py` are not tuned here. They are the operating point derived
in `docs/removal_operating_point.md` from measurements on this cohort:

| constant | evidence |
|---|---|
| 20 s subtraction fit | window sweep 10/20/30/54 s; 20 s best on residual, correlation and change RMS |
| 2.0 dB residual floor | floor sweep 1/2/3/4 dB; chosen after subtraction, because pre-subtraction prominence does not transfer between recordings |
| 0.30 Hz cluster gap | one line spans several bins; a worked failure at 57.20 Hz left a neighbour 17 dB louder |
| `comb_fundamental_hz: 1.2` | `docs/artifact_survey.md`: the comb is at 1.200 Hz in all 90 recordings; the 1/TR grid carries nothing (0/90, -0.65 dB) |

That document's recommended configuration is the one shipped here, arrived at independently
by the arm comparison. Its availability figures count FIR stopbands only and do not declare
subtraction damage, so they are not comparable to the declared numbers above.

It also records the alternative this pipeline does **not** take: notching on the corrected
1.2 Hz grid drives the comb to -6.80 dB instead of about -0.3 dB, at alpha 0.744 and gamma
0.556. That is a study-level choice between artifact floor and available bandwidth.

## What this does not claim

No statistically supported line remains anywhere — that is a hard postcondition, checked on
disk, and it held 90/90. The spectrum is **not** flat: a median of zero and a mean of 0.6
teeth per recording still stand above that recording's own noise, and by the stricter 2 dB
measure a mean of 1.8 teeth remain. Anything published should say "no statistically supported
line remains", which is the actual guarantee.
