# Does the adapted rsPCA work on this data?

Tested 2026-08-17 against `step1_scanner_artifact_pulse_marked`, sub-0003 / sub-0000 /
sub-0008 run 1, 20 s excerpts, real artifact plus injected ground-truth signals.

**Short answer: yes for the 57.24 Hz line, no for the 1.2 Hz comb, and `segment_s` must
change from 2.0 to ~0.3.**

## Fidelity to the publication

Every parameter matches `main_rspca.m` exactly. The port is faithful.

| Published (`main_rspca.m`) | Value | `TrajectoryPCASettings` | |
|---|---|---|---|
| `th_nkval` | −0.5 | `maximum_excess_kurtosis = -0.5` | exact |
| `th_var` | 1e-5 | `minimum_variance_fraction = 1e-5` | exact |
| `max_depth` | 2 | `maximum_depth = 2` | exact |
| `max_nkurt` | 100 | hardcoded `<= 100.0` | exact |
| `sigp_dB` | 1–3% | `secondary_peak_ratio = 0.03` | in range |
| `fpt` | `2^(floor(log2(seg))+2)` | `2**(int(log2(dim))+2)` | exact |
| `FOI.gamma` | `[20, fs/2]` | `fc_hz >= 20.0` | exact |

The delay embedding is also faithful: the original builds
`X = flipud(toeplitz(x(dim:-1:1),x(dim:end)))`, which is what `sliding_window_view` gives.

## `segment_s = 2.0` is the one wrong parameter

`dim = round(sf * segment_s)` = 2000 at 1 kHz. Two independent problems.

**It cannot fire.** The eigenvectors are length `dim` at `fs`, so their spectral resolution is
`1/segment_s`. At 2.0 s that is 0.5 Hz — fine enough to *resolve* the 57 Hz feature, which is a
~0.9 Hz-wide cluster of many peaks, into many peaks. `num_peaks <= 1` then never fires. The
rule needs resolution coarser than the cluster, i.e. `segment_s` below ~1 s.

**It cannot run.** Runtime scales ~`dim^2.7` (measured: 0.9 s / 5.3 s / 69 s for dim
50/100/250 on 20 s of one channel), putting dim=2000 at hours per channel.

Measured sweep, 20 s of FC1 with a real artifact plus two injected signals:

| `segment_s` | dim | resolution | runtime | 57.24 Hz artifact | injected 50 Hz sine | injected 40 Hz burst | alpha |
|---|---|---|---|---|---|---|---|
| 0.05 | 50 | 20.0 Hz | 0 s | −6.00 | −27.15 | **−7.46** | −0.15 |
| 0.10 | 100 | 10.0 Hz | 2 s | −0.80 | −33.39 | −0.00 | −0.00 |
| 0.20 | 200 | 5.0 Hz | 7 s | −15.29 | −26.07 | −0.00 | −0.00 |
| **0.30** | **300** | **3.3 Hz** | **14 s** | **−26.58** | −36.23 | −0.00 | −0.00 |
| 0.50 | 500 | 2.0 Hz | 39 s | −24.98 | −40.66 | −0.20 | −0.00 |

`segment_s = 0.30` is the operating point. Note the response is **not monotonic** below 0.2 —
0.10 removes nothing from the artifact while still deleting a stationary sinusoid — so this
parameter cannot be set by intuition and should not be varied without re-measuring.

## What it removes, and what it spares

It is a **stationary-narrowband remover**, not an artifact-specific one. The injected 50 Hz
sinusoid is deleted *harder than the real artifact* at every setting (−36 dB vs −27 dB at
0.30). Anything stationary and narrow in the gamma band will go.

What it spares is the thing that matters: an amplitude-modulated 40 Hz burst train — what real
neural gamma actually looks like — came back at **−0.00 dB in all 12 channel/recording tests**,
as did alpha. The excess-kurtosis gate is doing genuine work here, separating stationary from
bursty rather than artifact from signal.

The honest statement of the limitation: this is safe on this cohort because there is no
stationary narrowband neural gamma to lose, and it would not be safe on a study that claimed
to measure one.

## It is self-limiting across channels

`segment_s=0.30`, artifact SNR before → after:

| recording | ch | line SNR in | out | suppression |
|---|---|---|---|---|
| sub0008 | FC1 | 19.2 | −3.4 | −25.66 |
| sub0008 | T7 | 16.0 | −6.8 | −25.10 |
| sub0008 | Oz | 14.5 | 2.0 | −15.72 |
| sub0008 | Pz | 12.7 | 0.7 | −14.96 |
| sub0003 | FC1 | 12.0 | −7.3 | −25.65 |
| sub0000 | FC1 | 9.5 | −0.0 | −13.98 |
| sub0003 | T7 | 7.5 | 4.6 | −5.02 |
| sub0003 | Oz | 6.4 | 6.4 | 0.00 |
| sub0000 | Oz | −0.6 | −0.6 | 0.00 |

There is a threshold near **8–10 dB input SNR**: above it the line is driven to background,
below it nothing happens. That is safe rather than harmful, and it *reduces* spatial
inhomogeneity overall — the across-channel spread of the line falls from 19.8 dB to 13.7 dB.
It does not eliminate it, which matters for connectivity or source work.

## The 1.2 Hz comb is untouched

Harmonics k=70 (84 Hz) and k=80 (96 Hz) both return −0.00 dB; k=50 is −0.40. Comb teeth are
not isolated single peaks, so the rule cannot fire on them. This method is not a decomb and
should not be scoped as one.

## Performance: one fidelity gap is also the fix

At `segment_s=0.30` a full recording is ~336 s per channel, ~5.9 h per recording, ~22 days for
90 × 63 channels. Still not viable — but the cause is a departure from the original.

The MATLAB ends with `rXbar = sig - sum(reconXbar(idx,:))`: it reconstructs **only the
components it removes**. `_rs_pca_recursive` instead calls `_diagonal_average` for all `dim`
components and sums the keepers. Since the components sum to the original signal, the two are
equivalent in result but not in cost.

Measured on dim=300: only **94 components exceed the 1e-5 variance floor**, and the top 50
carry 99.7% of the variance. Two changes:

1. Subtract the artifact components from the original instead of summing the keepers — the
   inner loop then runs over a handful of components, not 300. **~60×**
2. `_diagonal_average` uses direct `np.convolve`, O(n·dim); FFT-based is **~21×** at these sizes.

Together that is ~22 days → several hours, which is viable.

## Performance fix applied

Two behaviour-preserving changes, verified against a reference implementation of the previous
logic on real EEG: **maximum absolute difference 1.1e-13** on a signal of unit standard
deviation, and the artifact/preservation table above reproduces to the last printed digit.

**Reconstruct only what is removed.** `_rs_pca_recursive` now starts from the signal and
subtracts the removed components, as `rXbar = sig - sum(reconXbar(idx,:))` does, instead of
diagonal-averaging all `dim` components and summing the keepers. A component that survives
untouched needs no reconstruction, and at the leaf level nothing recurses, so only the
single-peak candidates are built. Reconstructions per channel fall from **3,540 to 201**.

**Compute the kurtosis gate directly.** Profiling put `scipy.stats.kurtosis` at 6.6 s of
15.3 s — 13,629 calls per channel, almost all of it `axis_nan_policy_wrapper` dispatch around
two central moments. `_excess_kurtosis` computes the same biased Fisher definition in numpy
and is tested against scipy across normal, skewed, sinusoidal and two-valued samples.

Net **2.0x** on real data, 15.3 s -> 7.7 s for 20 s of one channel at dim=300. (A third
change, `fftconvolve` with closed-form anti-diagonal weights in `_diagonal_average`, was
already in the working tree.)

### What that leaves

About 8 s per 20 s per channel, so roughly 3.4 h per recording and **~12 days for 90 x 63
channels single-threaded** — around a day across ten cores, since channels are independent.

That is close to the algorithm's intrinsic cost rather than remaining overhead. The work is
94 recursive decompositions per channel, and at depth 2 roughly half the components are
single-peak candidates whose kurtosis cannot be known without reconstructing them. Going
materially faster means changing what the method computes, not how.

## Verdict

Use it, scoped as a **57 Hz stationary-line remover** with `segment_s = 0.30`, after fixing
the reconstruction cost. Do not scope it as a comb remover. Re-measure if `segment_s` changes.
Score it on the delivered output, not on the excerpt it was tuned against.
