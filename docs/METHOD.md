# The method, in equations

Every step `decomb` takes, stated so it can be checked against the code. This is the
reference for [README.md](../README.md); start there for what the tool is for and how to
run it.

Notation: $x[n]$ is one channel of one estimation window, $f_s$ the sampling rate, $N$ the
window length in samples, $T = N/f_s$ its duration, and $\Delta f = 1/T$ the bin width.

## 1. Spectral estimate

Each window is tapered with a Hann window $w[n]$ and transformed. The one-sided power
spectral density, in SciPy's `density` scaling, is

$$
S(f_k) \;=\; \frac{c_k}{f_s \sum_{n} w^2[n]} \left| \sum_{n=0}^{N-1} w[n]\,x[n]\, e^{-2\pi i k n / N} \right|^{2}
$$

on the grid $f_k = k f_s / N$, with $c_k = 2$ everywhere except DC and Nyquist, where
$c_k = 1$, so that the one-sided density integrates to the mean square of the windowed
signal. Channels are combined by median and windows by mean, and the result is expressed in
decibels as $X(f_k) = 10 \log_{10} S(f_k)$.

## 2. Prominence

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

## 3. Detection

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

## 4. The comb fit

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

## 5. What is removed

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

## 6. What the benchmark measures

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
