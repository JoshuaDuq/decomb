# Choosing between subtracting a line and notching it

Notching is model-free: a FIR stopband removes a 0.25 Hz interval plus its transitions,
for all time and all channels, artifact and neural activity alike. Subtracting is
model-based: MNE's `spectrum_fit` estimates one sinusoid's amplitude and phase per
window and removes exactly that, so the frequency stays available and anything the
sinusoid does not describe survives.

That difference sets the whole trade. Subtraction preserves; notching guarantees
removal. Every measurement below is one recording per row unless stated, on the cohort
described in `docs/artifact_survey.md`.

## Subtraction preserves neighbouring activity, not activity at the line

Paired injections of known neural-like components, 60 trials, identical injections in
both arms, scored by the fraction of injected energy retained:

| placement | notching | subtraction |
| --- | ---: | ---: |
| on an authorized line | 0.02 | 0.22 |
| near a line | 0.76 | **0.95** |
| between two lines | 0.98 | **1.00** |

Subtraction wins 20/20 paired trials at `near`. At `exact` both are poor, because the
component sits inside the fitted sinusoid and goes with it. The honest claim is that
subtraction preserves activity *beside* a line, not *at* one.

This only holds under the published residual stage. With the branch's target-local
rounds added, the two arms are indistinguishable (paired medians ~0.000) because those
rounds re-notch what subtraction spared, taking bandwidth back from 0.983 to 0.588.

## A longer estimation window does not help

| window | worst residual | correlation | change RMS | gamma available |
| --- | ---: | ---: | ---: | ---: |
| 10 s | +12.23 | 0.9924 | 0.0759 | 0.902 |
| **20 s** | **+9.01** | **0.9927** | **0.0736** | 0.899 |
| 30 s | +10.09 | 0.9877 | 0.0962 | 0.845 |
| 54 s | +20.57 | 0.9827 | 0.1206 | 0.716 |

Finer frequency resolution should in principle spare more activity at the line. It does
not, because the lines are amplitude-modulated: a long window cannot track a breathing
line, the fit degrades, and the residual it leaves forces more notching. 20 s is the
operating point.

## Which lines still need a notch is decided after subtraction, not before

Choosing by prominence *before* subtraction does not transfer between recordings: a
10 dB threshold selects 1 line in one recording and 31 in another. Choosing by the
residual that survives subtraction is far more stable, because subtraction drives the
median line below its own background (−1.7 to −4.3 dB) and only a handful resist.

| floor | lines notched (median, range) | gamma available | correlation |
| --- | --- | ---: | ---: |
| 1 dB | 12 (4–32) | 0.881 | 0.9938 |
| **2 dB** | **6 (2–23)** | 0.890 | 0.9938 |
| 3 dB | 4 (0–13) | 0.894 | 0.9940 |
| 4 dB | 3 (0–8) | 0.902 | 0.9940 |

## Cluster adjacent bins before applying the floor

One physical line is detected in several adjacent Fourier bins. Applying the floor per
bin can select one bin of a line and leave its neighbour inside the resulting filter's
transition, where attenuation is incomplete. In sub-0005 run 6 the bins 57.20 / 57.25 /
57.30 Hz had residuals of +1.20 / −0.19 / +4.14 dB, so only 57.30 was notched, and
57.20 finished 17 dB louder than its neighbour.

Grouping bins closer than 0.30 Hz into one line, thresholding on the group's strongest
bin and notching the group's whole span fixes it:

| 57.20 Hz | original | per-bin floor | clustered floor |
| --- | ---: | ---: | ---: |
| absolute level | +6.45 dB | −11.00 dB | **−40.56 dB** |
| prominence | — | +25.07 dB | **−0.79 dB** |

## Where it stands against the published pipeline

90 recordings, 20 s window, clustered residual floor at 2 dB:

| | subtraction | notching | paired |
| --- | ---: | ---: | --- |
| gamma available | **0.891** | 0.592 | better 90/90 |
| beta / alpha / theta / delta | **0.935 / 0.969 / 0.959 / 0.949** | 0.772 / 0.850 / 0.813 / 0.802 | — |
| correlation | **0.9935** | 0.9805 | better 89/90 |
| change RMS | **0.0771** | 0.1413 | better 89/90 |
| FIR rounds | **2.09** | 3.38 | — |
| worst residual level | +0.54 dB | **+0.37 dB** | better 8/90 |
| 1.2 Hz comb left | +0.21 dB | **−1.58 dB** | better 2/90 |

Bandwidth and waveform fidelity improve on essentially every recording. The residual is
the cost: the worst surviving peak is comparable to notching, and the comb is clearly
worse, because the configuration subtracts detected lines and never addresses the comb.

## What is not solved

Residual peaks remain, about 7 per recording, and they are not neural — every one sits
at a frequency the pipeline's own tests flagged as a narrowband line. Of 101 surviving
peaks across 15 recordings, 57 are 1.2 Hz comb teeth, 40 are other narrowband lines,
3 are the pump line and 1 is mains. The comb is therefore the dominant remaining gap.

A note on measurement: prominence is taken against a local median, so notching the
neighbourhood inflates it. Comparisons of surviving-peak counts between an arm that
notches heavily and one that does not are not meaningful; absolute level before and
after is the metric to use.

## Declaring the comb's fundamental changes both ends of the trade

Repeating the comparison with `removal.comb_fundamental_hz: 1.2`, so the comb stage
tests the grid the artifact actually occupies. 90 recordings, five arms:

| arm | comb left | worst level | alpha | gamma | correlation | change RMS | rounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| notching, TR grid | −1.58 | +0.37 | 0.850 | 0.592 | 0.9805 | 0.1413 | 3.38 |
| notching, declared 1.2 Hz | **−6.80** | **−6.25** | 0.744 | 0.556 | 0.9736 | 0.1783 | 2.39 |
| subtract lines | +0.21 | +0.54 | 0.969 | 0.891 | 0.9935 | 0.0771 | 2.09 |
| subtract lines and teeth | −0.11 | −0.21 | 0.969 | 0.896 | 0.9934 | 0.0774 | 1.96 |
| **subtract lines, declared comb** | −0.31 | +0.45 | **0.975** | **0.900** | **0.9940** | **0.0736** | **1.87** |

Correcting the grid makes notching much better at removal — the comb falls 5 dB further
and the worst surviving peak 6.6 dB further, on 85 and 86 of 90 recordings — and worse
for availability, because it now spends stopbands on teeth that are really there rather
than on 86 frequencies the artifact never occupied. Alpha availability drops from 0.850
to 0.744, worse on 89 of 90.

On the subtraction side the corrected grid is a strict improvement: `subtract lines,
declared comb` is the best arm in every band, has the highest correlation and the lowest
change RMS, converges in the fewest rounds, and leaves less comb than either arm built on
the uncorrected grid.

## Recommended configuration, and what it still leaves

Subtract the detected ordinary lines with a 20 s multitaper window; group detected bins
closer than 0.30 Hz into one line; notch only groups whose post-subtraction residual
still exceeds 2 dB; declare `removal.comb_fundamental_hz: 1.2` so the residual comb stage
is anchored where the artifact is.

Against the published pipeline that is 0.900 against 0.592 gamma availability, 0.975
against 0.850 alpha, 0.9940 against 0.9805 correlation, and roughly half the signal
disturbance, with a comparable worst surviving peak.

It leaves the comb about 0.3 dB above its background, where correctly anchored notching
drives it 6.8 dB below. That is the remaining trade, and it is a study-level choice
rather than a technical one: analyses limited by spectral availability should prefer
subtraction, analyses that need the artifact floor as low as possible at specific
frequencies should prefer notching on the corrected grid. Both are better on every axis
than notching on the trigger-derived grid, which is what the pipeline does today.
