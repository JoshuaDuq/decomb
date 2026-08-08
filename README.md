# decomb

Audited removal of narrowband line and comb artifacts from continuous EEG, with the
concurrent-fMRI case in mind.

`decomb` measures each contaminating line's frequency in each recording, subtracts a
sinusoid at that frequency, and — before it will write anything — measures what the
subtraction cost and refuses if the answer fails criteria fixed in advance.

## Why simultaneous EEG-fMRI needs this

EEG recorded inside a scanner is corrected in two large steps, and a third class of
artifact is left behind by both.

**Gradient artifact.** Switching gradients induce millivolt-scale EMF in the electrode
leads. It is removed by average artifact subtraction: average over the repeating slice or
volume epoch, subtract the template. This works precisely because the artifact is *locked
to the acquisition* — it repeats at `k/TR` and at the slice rate, so it survives the
average while everything else cancels.

**Pulse artifact.** Cardiac-driven motion of the head and leads in the static field
produces a ballistocardiogram of tens of microvolts, removed by a template locked to the
R peak, an optimal basis set, or ICA.

**What both leave behind.** Any source that is periodic but *not* locked to the imaging
sequence or the heartbeat is untouched by either step. It averages toward zero in every
template and stays in the data at full amplitude. The environment of an MR suite is full of
such sources, and the physics that makes EEG-fMRI hard is what makes them visible: a
conductor moving in a strong static field induces a voltage proportional to `B₀` and to the
rate of change of the loop area. Millimetre vibrations that would be invisible outside the
bore become measurable EEG.

The usual sources are mechanical and run continuously:

- **The cold head and helium compressor.** The cryocooler that reliquefies helium cycles
  continuously and vibrates the magnet and the bore. On many systems it can be switched off
  for the duration of an acquisition, which is the cleanest fix and worth trying first.
- **The bore ventilation fan.** A constant-speed fan puts energy at its rotation rate and
  at the blade-pass frequency. Also often switchable.
- **Room plant.** Chillers, pumps, and HVAC, coupled through the floor and the gantry.
- **Mains, and anything driven by it.** Machinery on synchronous motors is phase-locked to
  the supply, so its repetition rate is a rational fraction of 50 or 60 Hz and stays
  coherent from one session to the next.
- **Stimulus and response hardware.** Projectors, eye trackers, button boxes, and their
  cabling.

Each of these is periodic and none of them is a pure tone. A piston, a blade passing a
strut, a rectified supply: any periodic non-sinusoidal drive puts energy at *every*
harmonic of its repetition rate. What reaches the EEG is therefore not one line but a
**comb** — dozens of narrow lines at integer multiples of a single fundamental, each a few
millihertz wide, each phase-stable over minutes.

This matters most where EEG-fMRI analyses are most fragile. A comb at a low fundamental
puts its harmonics across beta and gamma, where a scanner-EEG study has the least signal to
spare; a phase-stable line shared across electrodes inflates coherence between every pair
carrying it, so it reads as global connectivity at that frequency; and because its power is
concentrated into a few millihertz rather than spread over the band, a line that is small
in microvolts can still be most of that band's power. `decomb diagnose` measures that share
directly, per band and per participant, so the decision to remove anything at all rests on
a number rather than on an impression.

One diagnostic separates the cases. Given `dataset.tr_seconds`, every detected line is
placed on the `k/TR` grid of the acquisition. A line **on** that grid is residual gradient
artifact, and says your gradient correction needs attention rather than this tool. A line
**off** it is the room, and no template locked to the acquisition will ever reach it. That
distinction is what this workflow is built around.

## Why a notch filter is the wrong instrument

A comb is not what a notch filter is for. Fifty FIR notches take the surrounding band with
them; one wide notch takes far more than the lines occupy. Both destroy the spectrum they
are deployed to clean.

The lines are monochromatic and their frequencies are measurable, so the right operation is
a projection onto sinusoids at those measured frequencies. `decomb` costs a few hundredths
of a hertz per line rather than a band.

![Power spectra before and after removal](docs/psd_before_after.png)

Three 300 s synthetic recordings: pink background, a 1.2 Hz comb over harmonics 24-79
standing 12 dB above it, and a rhythm planted on one of those harmonics. The figure is
produced by [`docs/make_figure.py`](docs/make_figure.py), which builds the data, runs
`diagnose`, `benchmark` and `apply` through their ordinary entry points, and prints every
number below as it measures it — so the claims can be checked and regenerated.

Delta, theta and alpha are untouched: not one bin moves by 1 dB. Across the removed span
the 55 targeted harmonics fall from 12.3 dB above background to 1.1 dB, and outside the
lines the spectrum moves by at most 0.06 dB.

Two features survive, and both are the tool declining to act:

- **42 Hz** carries a 2.8 Hz-wide rhythm sitting exactly on comb harmonic 35. The harmonic
  inside it is removed and the rhythm is not, because a rhythm is whole hertz wide and a
  line is a tenth of one. This — not any prominence threshold — is what keeps a real
  oscillation out of the removal's reach.
- **60 Hz** is comb harmonic 50, inside the `mains_notch_hz` band that `exclude_mains`
  hands to `notch`. Two stages must not aim at the same spectrum.

## The method, in equations

Notation: $x[n]$ is one channel of one estimation window, $f_s$ the sampling rate, $N$ the
window length in samples, $T = N/f_s$ its duration, and $\Delta f = 1/T$ the bin width.

### 1. Spectral estimate

Each window is tapered with a Hann window $w[n]$ and transformed. The one-sided power
spectral density, in SciPy's `density` scaling, is

$$
S(f_k) \;=\; \frac{c_k}{f_s \sum_{n} w^2[n]} \left| \sum_{n=0}^{N-1} w[n]\,x[n]\, e^{-2\pi i k n / N} \right|^{2}
$$

on the grid $f_k = k f_s / N$, with $c_k = 2$ everywhere except DC and Nyquist, where
$c_k = 1$, so that the one-sided density integrates to the mean square of the windowed
signal. Channels are combined by median and windows by mean, and the result is expressed in
decibels as $X(f_k) = 10 \log_{10} S(f_k)$.

### 2. Prominence

Every threshold and every test in the workflow is applied to prominence, not to power.
The local background is a running median over a window of half-width
$H = \mathrm{round}(\Delta_{\mathrm{bg}} / \Delta f)$ bins with the centre $2c+1$ bins
excluded, so a line cannot enter its own background:

$$
B(f_k) \;=\; \mathrm{median}\,\left[\, X(f_j) \;\text{ over }\; c < |j - k| \le H \,\right],
\qquad
P(f_k) \;=\; X(f_k) - B(f_k).
$$

$\Delta_{\mathrm{bg}}$ is `background_half_width_hz` and $c$ is one bin, because a
Hann-windowed tone occupies three. Bins within $H$ of either edge have no symmetric window
and are returned as NaN rather than estimated from a lopsided one.

### 3. Detection

Wherever there is no line, $P$ is zero-centred by construction, so the null is fitted from
the prominence spectrum's own lower tail — which the lines, being one-sided contamination,
cannot inflate:

$$
\hat\mu = \mathrm{median}\,P,
\qquad
\hat\sigma = \hat\mu - Q_{0.158655}(P),
$$

where $Q_\alpha$ is the $\alpha$ quantile; for a Gaussian the gap between the median and the
15.87th percentile is exactly $\sigma$. Each searched bin gets a one-sided probability

$$
p_k \;=\; 1 - \Phi\left( \frac{P(f_k) - \hat\mu}{\hat\sigma} \right),
$$

and the family is controlled at `fdr_alpha` by Benjamini–Hochberg over exactly the bins the
search was allowed to reach:

$$
q_{(i)} \;=\; \min_{j \ge i} \min\left(1, \frac{n\, p_{(j)}}{j}\right),
\qquad \text{accept } q_{(i)} < \alpha .
$$

Runs of accepted bins separated by no more than `join_gap_bins` quiet bins are one line,
represented by their largest bin. That bin is then refined below the grid by fitting a
parabola to the three decibel samples around the summit — a Hann-windowed tone has a
near-parabolic log-magnitude peak:

$$
\delta \;=\; \frac{1}{2}\,\frac{X_{k-1} - X_{k+1}}{X_{k-1} - 2X_k + X_{k+1}},
\qquad
\hat f = f_k + \delta\,\Delta f, \qquad |\delta| \le \tfrac{1}{2}.
$$

A detection's half-power width is measured by linear interpolation to $X_{\text{peak}} - 3$
dB on each side and summed. This is the quantity that separates an instrument line from a
brain rhythm: a Hann-windowed pure tone floors at $1.4382/T$, while a biological resonance
is whole hertz wide.

### 4. The comb fit

A comb is an arithmetic series through the origin, so with $\hat f^{(k)}$ the refined
position of harmonic $k$ and $w_k$ its prominence, the fundamental is the weighted
least-squares slope

$$
\hat f_0 \;=\; \frac{\sum_{k \in \mathcal{K}} w_k\, k\, \hat f^{(k)}}{\sum_{k \in \mathcal{K}} w_k\, k^{2}} .
$$

Membership $\mathcal{K}$ is found by iterating to a fixed point: seed $\hat f_0$ with the
weighted median of $\hat f^{(k)}/k$, keep every harmonic with
$|\hat f^{(k)} - k \hat f_0| \le \tau$, refit, repeat. The fit authorises a removal grid
only if $|\mathcal{K}| \ge$ `min_harmonics_for_fit` and the scatter about it is small:

$$
\mathrm{RMS} \;=\; \sqrt{\frac{1}{|\mathcal{K}|} \sum_{k \in \mathcal{K}} \left(\hat f^{(k)} - k \hat f_0\right)^{2}}
$$

stays under `max_fit_residual_rms_hz`. Peaks that do not lie on one arithmetic series are
not a comb, however many of them there are.

The uncertainty in $\hat f_0$ is a delete-one jackknife over the harmonics, which needs no
assumption about how the per-harmonic errors are distributed:

$$
\widehat{\mathrm{SE}}(\hat f_0) \;=\; \sqrt{\frac{n-1}{n} \sum_{i=1}^{n} \left( \hat f_0^{(-i)} - \bar f \right)^{2}},
\qquad
\bar f \;=\; \frac{1}{n} \sum_{i=1}^{n} \hat f_0^{(-i)},
$$

where $\hat f_0^{(-i)}$ is the fundamental refitted with harmonic $i$ left out.

### 5. What is removed

Each target gets a width, not a fixed one: a comb is disciplined by its source, so a wander
$\delta$ in the fundamental moves harmonic $k$ by $k\delta$, and the width has to carry that
propagated uncertainty. For a comb harmonic $k$ at $f_k$, with $\rho$ = `notch_width_ratio`
and $z$ = `uncertainty_confidence_z`,

$$
W_k \;=\; \max\left(\frac{f_k}{\rho},\, W_{\min}\right) \;+\; 2 z\, k\, \widehat{\mathrm{SE}}(\hat f_0),
$$

and for an isolated line, which inherits no such scaling,
$W = \max(f/\rho,\, W_{\min},\, 1/T_{\text{filter}})$.

Inside each width the operation is a projection, not an attenuation. The subtraction is
MNE's `spectrum_fit`, which fits a deterministic sinusoid at each bin and removes it where
Thomson's multitaper *F* test is significant, at MNE's own Bonferroni-corrected default.
`decomb` reimplements the same statistic — in `estimators.thomson_f_statistics` — for the
residual audit, so what checks the result is the test that produced it.

With $L$ DPSS tapers $v_l$ of bandwidth `mt_bandwidth`, $Y_l(f)$ the tapered transforms,
$U_l = \sum_n v_l[n]$, and $\mathcal{S}$ the symmetric tapers (the ones with $U_l \ne 0$),
the least-squares amplitude and its test statistic are

$$
\hat\mu(f) \;=\; \frac{\sum_{l \in \mathcal{S}} Y_l(f)\, U_l}{\sum_{l \in \mathcal{S}} U_l^{2}},
\qquad
F(f) \;=\; \frac{(L-1)\,\left|\hat\mu(f)\right|^{2} \sum_{l \in \mathcal{S}} U_l^{2}}{\sum_{l \in \mathcal{S}} \left| Y_l(f) - \hat\mu(f) U_l \right|^{2} \;+\; \sum_{l \notin \mathcal{S}} \left| Y_l(f) \right|^{2}} .
$$

Under the null of no sinusoid at $f$, $F(f) \sim F(2,\, 2L-2)$, and the family is the whole
transform window, so the critical value is $F^{-1}(1 - \alpha/N;\, 2,\, 2L-2)$ with $N$ the
number of samples in it. Statistics stay channel-specific, so a line present on four
electrodes never authorises subtraction from the rest.

Where the test fires, $\hat\mu(f)$ is subtracted; where it does not, the bin is untouched.
That is the whole reason the cost is a few bins per line rather than a band.

The fundamental is re-fitted in overlapping windows of `estimation_window_s` at a hop of
half that, because a comb drifts over minutes. Each window is cleaned against its own
targets and widths, then the windows are recombined with squared-sine weights normalised to
a partition of unity, so the seams add to one at every sample:

$$
g_m[n] = \sin^{2}\left( \pi \frac{n + \tfrac{1}{2}}{M} \right),
\qquad
\tilde g_m[n] = \frac{g_m[n]}{\sum_{m'} g_{m'}[n]},
\qquad
\sum_m \tilde g_m[n] = 1 .
$$

### 6. What the benchmark measures

Every criterion is an exact test against a matched control, chosen so that no decibel margin
has to be invented.

**Off-target disturbance** is a paired sign test across channels. The real transform's
deviation at frequencies it never targeted is compared, channel by channel, against a
control transform of the same size at frequencies *it* never targeted. Under the null the
pair is exchangeable, so with $s$ channels where the real one is larger out of $m$ decided
pairs, $p = \mathrm{P}[\mathrm{Binom}(m, \tfrac12) \ge s]$.

**Residual lines.** The worst residual inside each target's claimed window is compared
against the same search run where no target is. With $n$ such controls the exact one-sided
probability, counting the observation among the candidates, is

$$
p \;=\; \frac{1 + n_{\ge}}{1 + n},
$$

where $n_{\ge}$ counts the controls that reach or exceed the observation. Counting the
observation among the candidates is what makes this exact rather than optimistic: the
smallest attainable value is $1/(n+1)$.

Because each run's $p$ is uniform under the null, requiring every run to pass would reject a
faultless cohort almost surely; the decision is therefore Benjamini–Hochberg over the runs
at `false_discovery_rate`, and passes when it makes no discovery.

**Seams** use a synchronised-shift test. Each recording contributes one observed
boundary maximum and `n_seam_controls` controls from shift positions where no seam is; each
candidate is scaled by the 95th percentile of the others, and both the largest ratio and the
count of ratios above one are compared against their permutation distributions, each at
$\alpha/2$.

**Band cost** is measured, never asserted. A broadband probe goes through the identical
transform and the loss per bin is $\ell(f) = X_{\text{before}}(f) - X_{\text{after}}(f)$,
averaged over channels; the reported figure is the share of bins in `cost_band_hz` with
$\ell > 1$ dB and with $\ell > 3$ dB.

**Transients.** An injected Gaussian burst is recovered and compared against the same burst
put through the same removal alone, so the unavoidable loss is divided out rather than
charged twice. Inside the burst window,

$$
\text{energy ratio} = \frac{\sum_n r^2[n]}{\sum_n \hat r^2[n]},
\qquad
\text{correlation} = \min_{\text{channels}} \mathrm{corr}(r, \hat r),
\qquad
\text{intrinsic} = \frac{\sum_n \hat r^2[n]}{\sum_n b^2[n]},
$$

with $b$ the injected burst, $\hat r$ the reference, and $r$ the recovered one. The first two
are gated by `PreservationGate`; the third is reported, because a signal exactly at an
artifact frequency is not separable from the artifact.

## What makes it different

**It refuses.** `apply` will not run without a passing `benchmark` for the same data and
the same settings. The benchmark injects known signals, removes the lines, and measures
what came back. Its criteria are stated before the measurement is taken.

**It measures its own cost.** A broadband probe goes through the identical transform, so
the reported band cost is what a signal occupying the band actually loses, not what the
plan asked for. That figure is written into the output dataset's `GeneratedBy` provenance,
so the cost travels with the data.

**It verifies against a detector that doesn't know the answer.** `verify` re-sweeps the
cleaned data under FDR control with no knowledge of where the targets were, so it can find
a line the removal never aimed at.

**It reads no events.** A resting or baseline acquisition, or any continuous recording, is
a valid input. Nothing here requires a task, a trigger channel, or an epoch structure.

## Install

```bash
pip install -e .
```

Python 3.11+. Depends on MNE, MNE-BIDS, NumPy, SciPy, pandas, matplotlib.

## Use

```bash
decomb diagnose     # what lines are there, do they share a fundamental, and do they matter?
decomb benchmark    # does the removal preserve signal? run this before apply
decomb apply        # write the cleaned BIDS copy
decomb verify       # re-measure what was written
decomb report       # band-by-band outcome tables
decomb notch        # optional: wide notch over cluster bands
decomb psd          # before-and-after spectra
```

Start with `diagnose`. It reports the fitted fundamental and the harmonic span that
supports it; put those in your config before benchmarking. It also reports the share of
each analysed band that is line artifact, which is what decides whether removal is worth
doing at all, and counts detections per band, which is how you tell a band `apply` can
clear from one only `notch` can.

## Configuration

One file, and it holds everything. Copy the packaged
[`defaults.yaml`](src/decomb/defaults.yaml) to `decomb.yaml` in your working directory and
change what you need — your file is merged over the defaults, so it only has to contain the
keys you are changing. Use `--config PATH` or `DECOMB_CONFIG` to put it elsewhere.

Every parameter the workflow uses appears in that file: the detector's band and FDR level,
the comb fit's tolerances, the removal geometry, the injected probe, the acceptance
criteria, and the levels each is decided at. Nothing is hardcoded elsewhere. An
unrecognised key is refused rather than ignored, so a misspelling cannot leave you
believing a setting is in force.

Values marked `SITE` describe one room and mean nothing for another. The comb fundamental
is the important one: `1.2` is a seed for the search, not a fact about your data.

## Requirements on your data

- **BIDS**, read at `sub-*/[ses-*/]eeg/*_eeg.vhdr`, with or without `ses-` and `run-`.
- **BrainVision**, `IEEE_FLOAT_32` and `MULTIPLEXED`. `apply` rewrites the `.eeg` binaries
  in place and copies every sidecar byte-for-byte, which is what guarantees sampling rate,
  channel set, length and annotations cannot drift. Other formats are refused rather than
  silently converted.
- **At least one estimation window** per recording — 54 s by default.
- **Gradient and pulse artifact already corrected.** `decomb` is the step after those, not
  a replacement for either. Run it on data that has been through your usual EEG-fMRI
  correction.

Only EEG channels are transformed. `channels.tsv` is authoritative, so ECG and EOG stay
byte-identical and outside the criteria.

## What the criteria actually decide

Not every criterion is the same kind of claim, and it matters which is which.

**Calibrated tests, decided over the recordings.** The residual questions and the seam.
Each measures an observation against controls that repeat the same search where no target
is, so under the null the observation is exchangeable with them and an exact probability
follows by counting. Benjamini-Hochberg over the recordings controls the false discovery
rate; with a single recording that reduces to `p <= false_discovery_rate`, so a lone
acquisition is decided by its own exact test. `apply` refuses on any of them.

**Preservation is reported, not decided.** Two questions ask whether the transform
disturbed spectrum it never targeted — the injected tones, and the band outside the removed
bins. Neither can be given a valid null. Four tones on one channel is four observations,
and the best a sign test can return is 2⁻⁴ = 0.0625. The band question looks answerable and
is not: a control displaced from the real targets subtracts almost nothing and leaks almost
nothing, while leakage from the real transform scales with the power it removed, so the
real transform "fails" against it simply for having removed something.

Both are therefore reported beside their controls and nothing is decided from either. The
numbers are still worth reading — they typically sit orders of magnitude below any
threshold one would be tempted to set — but they are measurements, not passes.

**One derived bound.** `transient_preserved` comes from the instrument and the transform,
not from the data: a Gaussian burst of duration σ spans about `4/(2πσ)` hertz, crossing a
predictable number of comb lines each subtracted over `freq/notch_width_ratio`.

**One invariant.** `transient_undistorted` reads 1.0 on any data with any settings, because
the transform is linear. It is kept because a genuinely non-linear failure would break it,
not as evidence that anything worked.

There is no ceiling on spectral cost by default. The cost is already fixed by the notch
width and the number of targets, so a shipped ceiling could only be one chosen after seeing
the answer. Set `removal.max_band_cost` if your study wants a stated budget; the
declaration is then recorded in the output provenance as the scientific decision it is.

## `apply` and `notch` are counterparts

`apply` subtracts a sinusoid wherever a line is resolvable, at a few hundredths of a hertz
each. `notch` removes a whole band, at its full width whether or not signal was in it, and
exists for contamination that is a *cluster* — many non-stationary peaks packed into a
narrow span, where removing the tallest only promotes its neighbour. Mains itself is often
this shape, which is why `exclude_mains` defaults to true.

They must not both aim at the same spectrum, so the removal excludes every band listed in
`notch_bands`. `notch_bands` ships empty: a band belongs there only on measured evidence
from your own data.

## Tests

```bash
pytest
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
