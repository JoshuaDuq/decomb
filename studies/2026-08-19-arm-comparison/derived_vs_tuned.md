# The shipped subtraction arm against current notching and the tuned arm

Measured on all 90 recordings of the thermalactive cohort with `derived_shipped.py`, which
drives the shipped code path (`subtraction.subtract_authorized`, `subtraction.damage_intervals`,
`notch.band_availability_from_intervals`, `notch.clean_until_no_supported_lines`) rather than
reimplementing it. Raw results: `derived_shipped.tsv`, log in `derived_shipped.log`,
paired analysis in `derived_vs_tuned.py`.

The harness reproduced the six-recording design prototype (`derived_probe.tsv`, arm
`derived_tr`) exactly -- same target counts, residual peaks, `comb_db` and gamma on all six.
What ships computes what was designed.

## Headline

Subtraction beats current notching on every axis, including the one it pays for. It does
**not** beat the tuned arm, and the prototype's claim that it would has not survived the
full cohort.

## Availability: what subtraction costs, declared

| band | derived, declared | derived, FIR only | current notching |
|---|---:|---:|---:|
| gamma | **0.733** | 0.955 | 0.592 |
| beta | **0.915** | 0.983 | 0.766 |
| alpha | **1.000** | 1.000 | 0.846 |

Excluding sub-0008. "Declared" counts the +/- 2-bin damage zone around every subtracted
frequency on top of the FIR stopbands; "FIR only" is what the manifest would have claimed
had subtraction declared nothing.

The honest reading: subtraction destroys real bandwidth, and saying so drops gamma from
0.955 to 0.733. Even after paying that, it retains **0.733 against notching's 0.592** --
subtraction is more available than notching while being more truthful about what it removes.
Alpha is untouched at 1.000 because nothing in this cohort is subtracted below 13 Hz.

## Against current notching (paired, 84 recordings, excluding sub-0008)

| metric | notching | derived | derived better in |
|---|---:|---:|---:|
| comb_db (median) | -0.41 | **-0.06** | 70/84 |
| \|comb_db\| mean | 1.36 | **0.63** | -- |
| \|comb_db\| max | 23.90 | **6.58** | -- |
| correlation | 0.9801 | **0.9930** | 53/84 |
| change RMS | 0.1443 | **0.0627** | 79/84 |
| gamma availability | 0.592 | **0.733** | 53/84 |
| residual peaks, raw 2 dB | 31.8 | **15.2** | -- |

Subtraction wins outright. It halves the residual peak count, more than halves the signal
it disturbs, flattens the comb, and still declares more usable gamma.

## Against the tuned arm (paired, 79 recordings, excluding sub-0008 and the five stale excavation recordings)

| metric | combined (tuned) | derived | derived better in |
|---|---:|---:|---:|
| comb_db (median) | -0.07 | -0.08 | 44/79 |
| \|comb_db\| mean | **0.22** | 0.64 | -- |
| \|comb_db\| max | **1.11** | 6.58 | -- |
| correlation | **0.9962** | 0.9929 | 2/79 |
| change RMS | **0.0625** | 0.0630 | 24/79 |
| gamma availability | **0.928** | 0.728 | 0/79 |

`arm_combined.tsv` predates the scanner-harmonic authorization fix, so sub-0003 run-1,
sub-0006 runs 1-2, sub-0013 run-1 and sub-0014 run-6 carry stale `comb_db` near -6.5.
They are excluded above, which *helps* the tuned arm -- this is the comparison least
favourable to the derived design, and the one worth trusting.

### This overturns the prototype

The six-recording prototype predicted the derived arm would win on comb flatness
(|comb_db| mean 1.32 vs 1.34, max 4.90 vs 6.54) and on change RMS (0.1002 vs 0.1316).
Neither holds on 90 once the stale rows are removed: the tuned arm's |comb_db| mean is
0.22 against 0.64, its max 1.11 against 6.58, and change RMS is a tie (0.0625 vs 0.0630)
rather than a derived win. The prototype's sample was six recordings and included two
where the tuned arm's recorded numbers were stale.

**What survives:** the derived design is a large, unambiguous improvement over what ships
today. **What does not:** the claim that it also beats the hand-tuned arm. It does not,
and it gives up 0.20 of gamma availability to it -- on every single recording.

That gap is the declared damage zone, not lost signal. The tuned arm notches narrowly and
declares only its stopbands; the derived arm subtracts and declares +/- 2 bins per target,
with a mean of 106.5 targets per recording. Whether 0.20 of declared gamma is worth the
simpler, evidence-bound authorization is a design call, not a measurement -- but it should
be made knowing the tuned arm is genuinely ahead here, not behind.

## Residual peaks: the 2 dB threshold is not a measurement

Reported for the derived arm across all 90 recordings:

| statistic | value |
|---|---:|
| raw 2 dB count, mean | 16.1 |
| per-recording null p99 count, mean | **11.2** |
| null p99, mean / min / max | 3.49 / 1.18 / 20.56 dB |
| recordings whose null p99 exceeds 2 dB | **70 of 90** |
| pure-noise bins above 2 dB, mean share | 4.5% |

In 70 of 90 recordings the 2 dB floor sits **below** the noise of its own statistic, so
roughly a third of the raw count is chance. Calibrating each recording against its own p99
drops the mean from 16.1 to 11.2.

Two further reasons not to lean on this metric:

1. **It is circular against the tuned arm**, which explicitly notched every cluster whose
   prominence exceeded 2 dB -- the same threshold the metric counts.
2. **It is not a paired statistic.** Each arm evaluates prominence at its own detected-line
   set union the comb teeth (`fixed_config.py:157`, `derived_shipped.py`), so the arms are
   scored on different check sets. A difference in count is partly a difference in where
   each arm looked.

Weight `comb_db`, `correlation` and `change_rms`, which are computed on the full spectrum
and which no arm targeted directly.

## sub-0008, reported separately

sub-0008 has a bad ECG recording. That is a known data-quality fact, not a pipeline defect,
and it is excluded from every cohort figure above.

| run | targets | peaks 2 dB | peaks null-cal | comb_db | correlation | change RMS | gamma |
|---|---:|---:|---:|---:|---:|---:|---:|
| run-1 | 124 | 17 | 16 | -1.26 | 0.9866 | 0.0881 | 0.668 |
| run-2 | 139 | 19 | 17 | -5.86 | 0.9851 | 0.0791 | 0.651 |
| run-3 | 132 | 41 | 3 | -1.08 | 0.9849 | 0.0662 | 0.620 |
| run-4 | 146 | 33 | 16 | -4.90 | 0.9363 | 0.1342 | 0.626 |
| run-5 | 131 | 35 | 13 | -1.32 | 0.9823 | 0.0476 | 0.656 |
| run-6 | 137 | 30 | 11 | -0.41 | 0.9817 | 0.0457 | 0.641 |

run-3 is the clearest argument for null calibration in the cohort: 41 raw peaks against 3
once calibrated, because its own null p99 is 18.09 dB. The raw count was measuring noise.

## Keep `estimation_window_s` at 10 s

The damage zone is `2 / estimation_window_s`, so widening the window looks like free
bandwidth. It is not. Measured on six recordings (`derived_probe_w20.tsv`):

| | 10 s | 20 s |
|---|---:|---:|
| ordinary lines detected | 108.2 | **15.3** |
| scanner harmonics | 6.2 | 6.2 |
| residual peaks | **18.7** | 28.7 |
| comb_db | **-1.14** | **+1.55** |
| gamma_kept | 0.716 | 0.923 |

Ordinary-line detection collapses sevenfold at 20 s because a longer window demands the
line hold still across twice as long, and the comb drifts. `comb_db` goes positive -- the
teeth are left standing. The apparent gamma gain is bandwidth preserved by failing to
remove the artifact. This was tested; do not re-litigate it without re-running the comparison.

## Known limitation: subtraction leaves shoulders

Single-frequency subtraction removes the stationary component at the fitted bin. The comb
drifts, so what remains sits beside the target. Twelve of 26 real residual peaks lie within
0.2 Hz of a subtracted frequency (`residual_peaks.tsv`). This is the same physics as the
injection sweep, where drifting components retained 0.28 against 0.035 for stationary ones.
Fitting a narrow basis around each target rather than one exact frequency would absorb the
drift; that is a design change, not a bug, and it is out of scope here. It is also the most
likely route to closing the gap against the tuned arm.

Every one of the 84 recordings runs FIR rounds after subtraction (mean 2.18 rounds, none at
zero), which is why `verify` replays subtraction *and* the cascade on top of it rather than
comparing the subtraction stage against the final derivative.
