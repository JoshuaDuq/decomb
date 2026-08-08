# decomb

<p align="center">
  <img src="logo.png" alt="decomb logo" width="420">
</p>

Audited removal of narrowband line and harmonic-comb artifacts from continuous EEG.
`decomb` is intended for BIDS BrainVision recordings acquired during or near concurrent
fMRI, after the usual gradient- and pulse-artifact correction.

The tool detects narrow spectral features, tests whether a subset forms an arithmetic comb,
and constructs a per-recording adaptive removal plan. It benchmarks that plan with injected
signals and matched controls before `apply` can write a derivative. It does not identify the
physical source of a line, replace gradient or pulse correction, or guarantee that every
narrow feature is an artifact.

## Abstract

`decomb` is a research software tool for characterising and selectively removing narrowband
spectral artifacts from continuous EEG stored in BIDS-compatible BrainVision recordings. Its
primary use case is residual environmental or electrical contamination in EEG acquired during
or near fMRI, after gradient and ballistocardiographic correction. The software combines
cohort-level spectral diagnosis, per-recording harmonic-comb modelling, adaptive multitaper
sinusoid fitting, matched-control benchmarking, blind remeasurement, and derivative provenance.
It is designed to make the removal decision auditable rather than to infer artifact source or
to replace established EEG--fMRI correction methods. The repository currently demonstrates
implementation behavior on synthetic data; it does not establish general performance across
scanners, sites, or clinical populations.

## Contributions and evidence boundary

The software contribution is methodological and operational rather than a claim of a new
physical artifact mechanism. It provides:

- a catalogue that detects narrow spectral features without a user-supplied frequency list;
- a comb model that estimates a fundamental from multiple harmonics and adapts it across
  overlapping windows;
- target-specific sinusoid fitting alongside an explicitly separate wide-notch stage for
  dense clusters;
- predeclared, matched-control measurements of residuals, seam artifacts, signal disturbance,
  and spectral cost; and
- BIDS derivative writing with effective-configuration records, input and plan digests, and
  read-back checks.

These are properties of the implementation. The literature cited below motivates the artifact
classes, spectral estimators, statistical procedures, and data standards; it does not validate
`decomb` itself. Claims about performance must therefore be supported by study-specific
validation on the recordings to which the software is applied.

## 1. Motivation and related work

Gradient and pulse artifacts are separate preprocessing problems. Gradient artifact is
periodic with the imaging sequence and is commonly reduced with average artifact subtraction
or related methods [[1](#references), [2](#references)]. The ballistocardiogram is linked to
cardiac activity and can be reduced with ECG-locked templates, optimal basis sets, or other
methods [[3](#references), [4](#references)]. Its amplitude and spatial distribution depend
on the static field and recording setup [[5](#references)]. `decomb` does not implement any
of these corrections and should be applied only after the study's established correction
pipeline.

The MR environment can also produce artifacts unrelated to the imaging sequence or cardiac
cycle. Helium-pump artifacts have been reported as repetitive spectral peaks and reduced with
template subtraction, although switching the pump off is preferable [[6](#references),
[7](#references)]. A ventilation-dependent artifact has been reported for Siemens TRIO and
VERIO systems, including peaks in the gamma range that were most prominent on electrodes on
the head [[8](#references)]. These observations motivate the problem but do not establish
that a line in another scanner has the same source.

A periodic non-sinusoidal source can generate harmonics, so a set of narrow peaks may form an
arithmetic frequency comb. `decomb` tests that structure rather than assuming it. A detected
comb is evidence about spectral regularity, not proof of a physical source, phase stability,
or environmental origin. If `dataset.tr_seconds` is supplied, the catalogue also reports each
line's position relative to the acquisition grid `k/TR`; this is a diagnostic comparison, not
an attribution of the line to gradient artifact.

The tool is deliberately narrower than general artifact correction. It targets resolvable
narrow lines and comb members. Broad or non-stationary clusters belong in the optional wide
`notch` stage, and signal exactly coincident with an artifact is not separable from it. Turning
off the source remains the preferred intervention when possible [[9](#references)]. A
systematic review of simultaneous EEG--fMRI artifact reduction emphasizes that artifact
classes, acquisition choices, and correction methods should be reported explicitly, and that
comparative evidence is limited and methods require context-specific validation
[[18](#references)]. This README therefore separates literature-supported motivation from
implementation-specific behavior.

## 2. Narrowband removal and related methods

A wide FIR notch removes its full stop-band, including frequencies where no artifact was
measured. Filtering can introduce attenuation and other distortions outside the intended
stop-band, so filter characteristics and signal-preservation effects should be evaluated for
the analysis at hand [[19](#references)]. The line-removal pass instead gives each target an
explicit, uncertainty-aware width and fits sinusoids only inside those target regions. It is
in the same broad multitaper line-estimation family as Thomson's harmonic analysis
[[10](#references)] and related implementations [[11](#references)]. The CleanLine and
Chronux family is also described in the multitaper methods text by Mitra and Bokil
[[25](#references)]. `decomb` invokes MNE's `spectrum_fit` in explicit-target mode: its own
catalogue supplies the frequencies and widths rather than asking MNE to discover them. The
residual audit is implemented separately, so `decomb` is not a reimplementation of CleanLine.
CleanLine, reference-layer adaptive filtering, and spectrum-interpolation methods make
different assumptions about the source, stationarity, or availability of reference
measurements [[20](#references), [24](#references)]. Reference-layer methods additionally
require dedicated reference recordings from specialized hardware; `decomb` operates on the
standard EEG channels alone. Comparative
studies therefore do not establish that one method dominates in all EEG or scanner settings
[[18](#references), [20](#references)]. `decomb` makes no superiority claim; it reports the
measurements obtained from its own plan and controls.

![Power spectra before and after removal](docs/psd_before_after.png)

The figure is a reproducible synthetic demonstration generated by
[`docs/make_figure.py`](docs/make_figure.py). It contains a 1.2 Hz comb over harmonics 24--79 (inclusive),
a broad 42 Hz rhythm centred on harmonic 35, and a 60 Hz line inside the configured mains
band. The example illustrates that a broad feature can overlap a comb member without being
classified as a narrow line, and that `apply` and `notch` must not target the same band. Its
numbers are properties of this generated dataset, not performance estimates for real EEG.

![What each stage costs, on each artifact structure](docs/notch_comparison.png)

The second figure measures the paragraph above, and is generated by
[`docs/make_notch_figure.py`](docs/make_notch_figure.py). One synthetic recording carries
both structures: the 1.2 Hz comb, and a dense cluster of twelve non-stationary peaks packed
into 1 Hz at 20 Hz. Each column is put through the line-removal pass and through
`mne.filter.notch_filter` at its library defaults.

The lower row is the measurement the prose claims. A broadband recording carrying no
artifact at all is put through each transform, so the attenuation plotted is what a signal
loses rather than what an artifact loses. Clearing the comb takes 55 notches, one per
harmonic, against the same 55 fitted targets: 8.9 Hz of spectrum by fitting them and 50.0 Hz
by notching them, or 13% against 74% of the 28--95 Hz cost band. Each panel states the cost
over the span it draws rather than the whole band, so its numbers match its own picture. The
notch removes the comb further -- to 40.7 dB below background against the fit's 1.1 dB -- and
the difference in cost is what that buys.

The cluster reverses it. Its peaks wander, so detection resolves 18 peaks over 3 dB where 12
were planted and classifies only 8 as lines; the fit leaves the cluster exactly as it found
it, at 20.2 dB, while one 1.16 Hz notch takes it to background. This is the case
[§6.3](#63-apply-and-notch-are-counterparts) describes, and the reason `notch` exists as a
separate stage rather than a fallback. Neither figure supports a general ranking: each stage
is measured on the artifact structure it was written for.

One practical note, since it is easy to miss: `notch_filter` cannot build these filters at
its own `filter_length='auto'` on a 1.2 Hz comb, raising *"the requested filter length 1651
is too short for the requested 0.11 Hz transition band"*. The figure sets a 30 s filter and
leaves every other notch parameter at MNE's default, including the `freq/200` widths and the
1 Hz transitions where most of the cost lives.

## 3. Software contribution and design

**Benchmark before writing.** `apply` requires a benchmark produced from the same input,
settings, and fitted plans. It checks the per-recording gate and then re-evaluates the cohort
residual and seam criteria before writing.

**Measured cost.** A broadband probe passes through the identical transform, so band cost is
measured rather than inferred from planned widths. Probe preservation and band cost are
reported measurements; they are not silently converted into acceptance claims.

**Blind verification.** `verify` re-sweeps the written data with the same detector, without
using the target list. It can therefore reveal a residual or newly exposed line that the
removal plan did not target.

**Continuous recordings.** The workflow reads no events and requires no epoch structure.
It still requires at least one complete estimation window and suitable continuous EEG data.

## 4. Reproducible workflow

### 4.1 Installation

```bash
pip install -e .
```

Python 3.11+. Depends on MNE, MNE-BIDS, pybv, NumPy, SciPy, pandas, matplotlib, joblib and
PyYAML. `pip install -e ".[dev]"` adds pytest and ruff. `decomb --help` lists the stages
and options.

### 4.2 Quickstart

Point `decomb` at a BIDS root and inspect the catalogue. This stage does not write a cleaned
dataset:

```bash
decomb diagnose --bids-root data/bids --output-dir outputs/diagnosis
```

Use the reported comb fundamental and supported harmonic range to set
`removal.nominal_fundamental_hz` and `removal.harmonic_range` for fitting. The separate
`removal.removal_harmonic_range` controls which harmonics are actually removed and may extend
beyond the fitting range. The band-impact table is a measurement of detected lines in the
configured bands; it is not a claim that their physical source has been identified.

Copy the packaged [`defaults.yaml`](src/decomb/defaults.yaml) to `decomb.yaml`, make the
site-specific changes, and run the benchmark:

```bash
decomb benchmark --config decomb.yaml
```

`benchmark` reports both decisions and non-gated measurements. A passing benchmark is
necessary but not sufficient evidence that the method is appropriate for a study; inspect
suppression, residuals, preservation, band cost, and the synthetic or study-specific
validation data before proceeding.

```bash
decomb apply --config decomb.yaml
decomb verify --config decomb.yaml
decomb report --config decomb.yaml
```

`apply` checks the benchmark's settings fingerprint, input digests, fitted-plan digests, and
cohort criteria before writing. It stages the derivative and reads it back before moving it
into place. To reproduce the synthetic demonstration without data of your own:

```bash
python docs/make_figure.py --keep /tmp/decomb-demo
```

### 4.3 Stages

```bash
decomb diagnose     # what lines are there, do they share a fundamental, and do they matter?
decomb benchmark    # does the removal preserve signal? run this before apply
decomb apply        # write the cleaned BIDS copy
decomb verify       # re-measure what was written
decomb report       # band-by-band outcome tables
decomb notch        # optional: wide notch over cluster bands
decomb psd          # before-and-after Welch spectra [[12](#references)]
```

Every stage takes the same options so a run is described by one config and the few
overrides you gave it:

- `--config PATH`: defaults to `./decomb.yaml`, then the packaged defaults. `DECOMB_CONFIG`
  does the same.
- `--bids-root PATH`: the source root without editing the config.
- `--output-root PATH`: where `apply` puts the cleaned copy.
- `--output-dir PATH`, `--report-dir PATH`: where the catalogue and the tables go.
- `--filter-length`, `--mt-bandwidth`: override the removal geometry for one run.

`--subjects sub-01 sub-02` restricts `diagnose` and `psd` to a subset. `benchmark` `apply`
`verify` and `notch` refuse it on purpose. Their criteria are decided over the recordings
jointly. A subset could neither certify a dataset nor leave the output root in a state the
provenance describes.

`diagnose` also counts detections per band. This helps distinguish a band with resolvable
lines from a dense cluster that may require the wide `notch` stage.

### 4.4 Outputs and provenance

The tables are TSV so the numbers a stage decided on can be read without `decomb`.
Locations come from `paths`. `diagnosis_dir` and `removal_dir` default to
`outputs/diagnosis` and `outputs/removal`.

`diagnose` writes the catalogue. `lines.tsv` has a row per detection: refined frequency,
prominence and its bootstrap interval, 3 dB linewidth, the q-value that admitted it, the
number of subjects carrying it, its comb harmonic, and its position on the `k/TR` grid.
`comb.tsv` has the fitted fundamental and spacing, the supporting harmonics, and the
scatter about the grid. `lines_per_band.tsv` and `band_impact.tsv` hold the per-band counts and
artifact shares printed at the end of the run. `spectra.npz` holds the spectra the sweep
saw.

`benchmark` writes `benchmark.tsv`, a row per recording carrying the gate measurements,
matched-control statistics, reported costs, and settings fingerprint.

`apply` writes the cleaned copy to `output_root` with only the EEG `.eeg` binaries rewritten.
Source sidecars are copied byte-for-byte except `dataset_description.json`, which is rewritten
to declare the derivative and record the version, fingerprint, full parameter set, and measured
band cost. Alongside it
`removal_manifest.tsv` gives the fundamental used, the target counts, the suppression and
residual statistics, the read-back check, and the digests tying the write to its benchmark.
The whole derivative is staged in a hidden directory and moved into place only once every
recording has been written and read back within `removal.roundtrip_relative_tolerance`. An
interrupted run cannot leave a half-cleaned dataset. The manifest goes to `removal_dir` and
to the output root so the copy carries its own record of what was done to it.

`verify` writes `verification.tsv`, the blind re-sweep set beside the same sweep of the
original, with `verification_spectra.npz` beside it. `report` writes `band_outcomes.tsv`
(artifact share per band before and after), `per_subject_line_residual.tsv` (what survived
at each target per subject) and `removal_before_after.png`. `psd` writes overall, tiled
and per-recording spectra. `notch` writes `notch_manifest.tsv`.

### 4.5 Configuration

One file holds everything. Copy the packaged
[`defaults.yaml`](src/decomb/defaults.yaml) to `decomb.yaml` and change what you need. Your
file is merged over the defaults so it only has to contain the keys you are changing.

The configurable detector band and FDR level, comb-fit tolerances, removal geometry,
injected probe, and acceptance criteria are declared there. Derived values are written to
`effective_config_*.txt` with their formulas. An unrecognised key is refused instead of
ignored so a misspelling cannot leave you believing a setting is in force.

Values marked `SITE` describe one room and mean nothing for another. The comb fundamental
is the important one. `1.2` is a seed for the search. It is not a fact about your data.

### 4.6 Data contract

The input and derivative conventions follow BIDS and its EEG extension [[21](#references),
[22](#references)]. `decomb` supports the subset described below rather than claiming to be
a general BIDS EEG reader.

- **BIDS**, read at `sub-*/[ses-*/]eeg/*_eeg.vhdr`, with or without `ses-` and `run-`.
- **BrainVision**, `IEEE_FLOAT_32` and `MULTIPLEXED`. These constraints allow `apply` to
  rewrite only the `.eeg` binaries while preserving the source sidecars byte-for-byte, except
  for the derivative `dataset_description.json`. Sampling rate, channel set, length, and
  annotations cannot drift. Other formats are refused instead of silently converted.
- **At least one estimation window** per recording. 54 s by default.
- **Gradient and pulse artifact already corrected.** `decomb` is the step after those. It
  does not replace either. Run it on data that has been through your usual EEG-fMRI
  correction.

Only EEG channels are transformed. `channels.tsv` is authoritative so ECG and EOG stay
byte-identical and outside the criteria.

## 5. Methods

The implementation is intentionally separated into measurement, modelling, transformation, and
verification. The equations below describe the estimators used by the code; they are not
assumptions that every artifact in an EEG recording follows the same model.

Notation: $x[n]$ is one channel of one estimation window, $f_s$ the sampling rate, $N$ the
window length in samples, $T = N/f_s$ its duration, and $\Delta f = 1/T$ the bin width.

### 5.1 Spectral estimate

Each window is tapered with a Hann window $w[n]$ and transformed. The one-sided power
spectral density, in SciPy's `density` scaling, is

$$
S(f_k) \;=\; \frac{c_k}{f_s \sum_{n} w^2[n]} \left| \sum_{n=0}^{N-1} w[n]\,x[n]\, e^{-2\pi i k n / N} \right|^{2}
$$

on the grid $f_k = k f_s / N$, with $c_k = 2$ everywhere except DC and Nyquist, where
$c_k = 1$, so that the one-sided density integrates to the mean square of the windowed
signal. Channels are combined by median and windows by mean. The result is expressed in
decibels as $X(f_k) = 10 \log_{10} S(f_k)$.

### 5.2 Local prominence

Detection thresholds and line-selection tests use local prominence rather than absolute
power. The local background is a running median over a window of half-width
$H = \mathrm{round}(\Delta_{\mathrm{bg}} / \Delta f)$ bins with the centre $2c+1$ bins
excluded so a line cannot enter its own background:

$$
B(f_k) \;=\; \mathrm{median}\,\left[\, X(f_j) \;\text{ over }\; c < |j - k| \le H \,\right],
\qquad
P(f_k) \;=\; X(f_k) - B(f_k).
$$

$\Delta_{\mathrm{bg}}$ is `background_half_width_hz` and $c$ is one bin, because a
Hann-windowed tone occupies three. Bins within $H$ of either edge have no symmetric window
and are returned as NaN instead of estimated from a lopsided one.

### 5.3 Detection and linewidth

The catalogue detector applies the following null model to the cohort-mean prominence
spectrum. The removal planner uses the same local-prominence convention, but its comb and
isolated-line searches are per recording and per adaptive window.

Wherever there is no line $P$ is approximately zero-centred by construction. The null is
fitted from the prominence spectrum's own lower tail, which one-sided line contamination
cannot inflate:

$$
\hat\mu = \mathrm{median}\,P,
\qquad
\hat\sigma = \hat\mu - Q_{0.158655}(P),
$$

where $Q_\alpha$ is the $\alpha$ quantile. The lower-tail Gaussian scale is an empirical-null
assumption: for a Gaussian the gap between the median and the 15.87th percentile is exactly
$\sigma$. Each searched bin gets an empirical-null one-sided probability

$$
p_k \;=\; 1 - \Phi\left( \frac{P(f_k) - \hat\mu}{\hat\sigma} \right),
$$

For the catalogue, the empirical-null screening family is controlled at `fdr_alpha` by the
conservative Benjamini--Yekutieli correction [[27](#references)] over exactly the bins the
search was allowed to reach. The per-recording isolated-line search uses the separately
configured `removal.detection_fdr_alpha`. These are screening probabilities whose calibration
depends on the empirical-null model; they are not unconditional Gaussian p-values.

$$
q_{(i)} \;=\; \min_{j \ge i} \min\left(1, \frac{n\, p_{(j)}}{j}\right),
\qquad \text{accept } q_{(i)} < \alpha .
$$

Adjacent runs of accepted bins are represented by their largest bin. That bin is then
refined below the grid by fitting a parabola to the three decibel samples around the summit.
A Hann-windowed tone has a near-parabolic log-magnitude peak:

$$
\delta \;=\; \frac{1}{2}\,\frac{X_{k-1} - X_{k+1}}{X_{k-1} - 2X_k + X_{k+1}},
\qquad
\hat f = f_k + \delta\,\Delta f, \qquad |\delta| \le \tfrac{1}{2}.
$$

A detection's linewidth is measured by linear interpolation to $X_{\text{peak}} - 3$ dB on
each side and summed. A Hann-windowed pure tone has a minimum measurable width of
$1.4382/T$; this resolution scale is used to distinguish narrow features from broader ones
[[14](#references)]. Linewidth is a classification rule, not proof that a feature is
instrumental or physiological.

### 5.4 Comb fitting and classification

The removal planner searches near the configured nominal fundamental and obtains one
candidate peak for each configured harmonic. With $\hat f^{(k)}$ the refined position of
harmonic $k$ and $w_k$ its prominence, the fundamental is the weighted least-squares slope
through the origin:

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

stays under `max_fit_residual_rms_resolutions` times the Hann spectral resolution
$1.4382 / T$ Hz, where $T$ is the estimation-window duration configured by
`removal.estimation_window_s`. Peaks that do not lie on one arithmetic series are not admitted
to this comb model. The separate `diagnose` catalogue first looks for repeated
pairwise spacings and then refines possible subharmonics; it reports that cohort-level
structure independently from the per-recording removal fit.

The uncertainty in $\hat f_0$ is a delete-one jackknife over the harmonics
[[15](#references)]. It needs no assumption about how the per-harmonic errors are
distributed:

$$
\widehat{\mathrm{SE}}(\hat f_0) \;=\; \sqrt{\frac{n-1}{n} \sum_{i=1}^{n} \left( \hat f_0^{(-i)} - \bar f \right)^{2}},
\qquad
\bar f \;=\; \frac{1}{n} \sum_{i=1}^{n} \hat f_0^{(-i)},
$$

where $\hat f_0^{(-i)}$ is the fundamental refitted with harmonic $i$ left out.

### 5.5 Transformation and reconstruction

The implementation uses MNE-Python for EEG I/O, multitaper sinusoid fitting, and Welch
comparison spectra [[23](#references), [26](#references)]. Each target receives an explicit
width and is authorized independently in each EEG channel and adaptive window by a
channel-specific Thomson statistic. For the comb
model, a frequency shift $\delta$ in the fundamental moves harmonic $k$ by $k\delta$, so the
width carries that propagated uncertainty. For a comb harmonic $k$ at $f_k$, with $\rho$ =
`notch_width_ratio` and $z$ = `uncertainty_confidence_z`,

$$
W_k \;=\; \max\left(\frac{f_k}{\rho},\, W_{\min}\right) \;+\; 2 z\, k\, \widehat{\mathrm{SE}}(\hat f_0),
$$

and for an isolated line which inherits no such scaling,
$W = \max(f/\rho,\, W_{\min},\, 1/T_{\text{filter}})$.

Inside each width, `decomb` calls MNE's `spectrum_fit` with the explicit target frequencies,
widths, filter length, and multitaper bandwidth supplied by its removal plan. It does not use
MNE's automatic target-discovery mode. In explicit-target mode MNE fits and subtracts the
frequency-grid components inside each supplied width; the channel/window authorization occurs
before this call. The package separately reimplements the Thomson statistic in
`estimators.thomson_f_statistics` for residual auditing, with a channel-specific
Bonferroni family over the complete transform grid.

With $L$ DPSS tapers $v_l$ [[17](#references)] of bandwidth `mt_bandwidth`, $Y_l(f)$ the tapered transforms,
$U_l = \sum_n v_l[n]$, and $\mathcal{S}$ the symmetric tapers (the ones with $U_l \ne 0$),
the least-squares amplitude and its test statistic are

$$
\hat\mu(f) \;=\; \frac{\sum_{l \in \mathcal{S}} Y_l(f)\, U_l}{\sum_{l \in \mathcal{S}} U_l^{2}},
\qquad
F(f) \;=\; \frac{(L-1)\,\left|\hat\mu(f)\right|^{2} \sum_{l \in \mathcal{S}} U_l^{2}}{\sum_{l \in \mathcal{S}} \left| Y_l(f) - \hat\mu(f) U_l \right|^{2} \;+\; \sum_{l \notin \mathcal{S}} \left| Y_l(f) \right|^{2}} .
$$

Under the null of no sinusoid at $f$, $F(f) \sim F(2,\, 2L-2)$, and the family is the whole
transform window, so the critical value is $F^{-1}(1 - \alpha/N;\, 2,\, 2L-2)$ with $N$ the
number of samples in it. Statistics stay channel-specific so a line present on four
electrodes never authorises subtraction from the rest.

Where the test fires $\hat\mu(f)$ is subtracted. Where it does not the bin is untouched.
This is why the cost is a few bins per line and not a band.

The fundamental is re-fitted in overlapping windows of `estimation_window_s` at a hop of
half that, allowing the measured comb position to vary over time. Each window is cleaned
against its own targets and widths. The windows are then recombined with squared-sine weights normalised
to a partition of unity, so the seams add to one at every sample:

$$
g_m[n] = \sin^{2}\left( \pi \frac{n + \tfrac{1}{2}}{M} \right),
\qquad
\tilde g_m[n] = \frac{g_m[n]}{\sum_{m'} g_{m'}[n]},
\qquad
\sum_m \tilde g_m[n] = 1 .
$$

## 6. Validation and decision rules

### 6.1 Benchmark measurements

`benchmark` injects four narrow sinusoids away from the targets, a Gaussian-enveloped
sinusoidal transient, a signal at selected target frequencies, and a broadband noise probe.
The default sinusoid locations are derived from the fitted targets and checked for clearance;
they are not a fixed list of protected frequencies.

The off-target probe and broadband probe are compared with a matched control transform of the
same size, shifted to frequencies where no line is expected. These comparisons quantify
leakage and broadband cost. They are reported, not treated as null-hypothesis acceptance
tests: the real transform necessarily removes more than the displaced control when the
recording contains artifact.

For residual lines, the largest residual prominence in each target's responsibility window is
compared with matched no-target searches. With $n$ controls, the one-sided exceedance
probability is

$$
p \;=\; \frac{1 + n_{\ge}}{1 + n},
$$

where $n_{\ge}$ counts controls at least as extreme as the observation. Including the
observation gives a minimum attainable value of $1/(n+1)$ [[16](#references)].
Because each run's probability is uniform under its null, the cohort decision uses
Benjamini--Hochberg over recordings at `false_discovery_rate`, rather than requiring every
recording to pass independently.

Seams use synchronised shifts. Each recording contributes one observed boundary maximum and
`n_seam_controls` no-seam controls. The observed-to-control scale ratio and the count of
ratios above one are compared with their shift distributions at `seam_alpha / 2`.

For the broadband probe, the loss per frequency bin is
$\ell_c(f) = X_{c,\mathrm{before}}(f) - X_{c,\mathrm{after}}(f)$ for every EEG channel
$c$. The reported cost is the worst channel's share of bins in `cost_band_hz` with loss
greater than 1 dB and 3 dB; the median channel cost is retained as a descriptive companion.
This is a measured operator cost, not a prediction from the requested widths.

For the transient, let $b$ be the injection, $\hat r$ the same transient after the removal
alone, and $r$ the recovered transient after it is added to data and cleaned. Inside the
transient window, the reported metrics are

$$
\text{energy ratio} = \frac{\sum_n r^2[n]}{\sum_n \hat r^2[n]},
\qquad
\text{correlation} = \min_{\text{channels}} \mathrm{corr}(r, \hat r),
\qquad
\text{intrinsic} = \frac{\sum_n \hat r^2[n]}{\sum_n b^2[n]}.
$$

Only correlation is a per-recording gate. The energy ratio measures collateral change and
the intrinsic ratio measures the cost of placing a transient across the removal geometry.

### 6.2 What the criteria decide

The per-recording benchmark gate is controlled by `min_burst_correlation`, the configured
minimum correlation between the recovered and reference transient. The resulting
`transient_undistorted` boolean is a linearity and sanity check, not evidence that the artifact
model is correct.

`apply` then independently requires the cohort seam, whole-run residual, and focal-residual
matched-control decisions to pass. If a study sets `removal.max_band_cost`, the measured
broadband cost is also checked against that predeclared budget.

Probe preservation, broadband cost when no budget is declared, in-band probe survival, and
the transient energy ratios remain measurements. They should be interpreted against the
study's scientific bandwidth and signal requirements, not treated as universal pass/fail
thresholds. The package deliberately does not ship a default spectral-cost ceiling.

### 6.3 `apply` and `notch` are counterparts

`apply` performs target-specific sinusoid fitting in the measured line regions. `notch`
removes a whole band at its full width whether or not signal was in it. It exists for
contamination that is a *cluster*: many non-stationary peaks packed into a narrow span, where
removing the tallest only promotes its neighbour. Mains can have this structure, which is why
`exclude_mains` defaults to true.

They must not both aim at the same spectrum. The removal excludes every band listed in
`notch_bands`. `notch_bands` ships empty. A band belongs there only on measured evidence
from your own data.

## 7. Limitations

- Detection and fitting assume that the relevant line remains resolvable within an estimation
  window. Short windows reduce frequency resolution; long windows reduce adaptation to drift.
- A narrow feature is classified from spectral shape and comb structure, not from an independent
  physical reference. A real neural oscillation or an uncharacterised instrument line can be
  misclassified if it satisfies the same spectral criteria.
- Signal at an artifact frequency is not identifiable from the artifact by spectral subtraction.
  The injected probes measure specific aspects of this trade-off; they do not establish
  preservation of all possible neural signals.
- The repository's synthetic demonstration and automated tests are validation of implementation
  behavior, not a claim of general performance on clinical or scanner-specific recordings.
  Report study-specific measurements and retain the generated provenance with any derivative.

## 8. Verification and tests

```bash
pip install -e ".[dev]"
pytest
```

The tests use seeded synthetic recordings and known lines where appropriate, so they test
measurement and invariants rather than relying only on stored fixtures. The test count and
runtime are environment-dependent.

## 9. References

1. Allen PJ, Josephs O, Turner R (2000). A method for removing imaging artifact from
   continuous EEG recorded during functional MRI. *NeuroImage* 12(2):230-239.
   [doi:10.1006/nimg.2000.0599](https://doi.org/10.1006/nimg.2000.0599)
2. Yan WX, Mullinger KJ, Brookes MJ, Bowtell R (2009). Understanding gradient artefacts in
   simultaneous EEG/fMRI. *NeuroImage* 46(2):459-471.
   [doi:10.1016/j.neuroimage.2009.01.029](https://doi.org/10.1016/j.neuroimage.2009.01.029)
3. Allen PJ, Polizzi G, Krakow K, Fish DR, Lemieux L (1998). Identification of EEG events
   in the MR scanner: the problem of pulse artifact and a method for its subtraction.
   *NeuroImage* 8(3):229-239.
   [doi:10.1006/nimg.1998.0361](https://doi.org/10.1006/nimg.1998.0361)
4. Niazy RK, Beckmann CF, Iannetti GD, Brady JM, Smith SM (2005). Removal of FMRI
   environment artifacts from EEG data using optimal basis sets. *NeuroImage*
   28(3):720-737.
   [doi:10.1016/j.neuroimage.2005.06.067](https://doi.org/10.1016/j.neuroimage.2005.06.067)
5. Debener S, Mullinger KJ, Niazy RK, Bowtell RW (2008). Properties of the
   ballistocardiogram artefact as revealed by EEG recordings at 1.5, 3 and 7 T static
   magnetic field strength. *International Journal of Psychophysiology* 67(3):189-199.
   [doi:10.1016/j.ijpsycho.2007.05.015](https://doi.org/10.1016/j.ijpsycho.2007.05.015)
6. Rothlübbers S, Relvas V, Leal A, Murta T, Lemieux L, Figueiredo P (2015).
   Characterisation and reduction of the EEG artefact caused by the helium cooling pump in
   the MR environment: validation in epilepsy patient data. *Brain Topography*
   28(2):208-220.
   [doi:10.1007/s10548-014-0408-0](https://doi.org/10.1007/s10548-014-0408-0)
7. Kim HC, Yoo SS, Lee JH (2015). Recursive approach of EEG-segment-based principal
   component analysis substantially reduces cryogenic pump artifacts in simultaneous
   EEG-fMRI data. *NeuroImage* 104:437-451.
   [doi:10.1016/j.neuroimage.2014.09.049](https://doi.org/10.1016/j.neuroimage.2014.09.049)
8. Nierhaus T, Gundlach C, Goltz D, Thiel SD, Pleger B, Villringer A (2013). Internal
   ventilation system of MR scanners induces specific EEG artifact during simultaneous
   EEG-fMRI. *NeuroImage* 74:70-76.
   [doi:10.1016/j.neuroimage.2013.02.016](https://doi.org/10.1016/j.neuroimage.2013.02.016)
9. Mullinger KJ, Castellone P, Bowtell R (2013). Best current practice for obtaining high
   quality EEG data during simultaneous fMRI. *Journal of Visualized Experiments* (76):50283.
   [doi:10.3791/50283](https://doi.org/10.3791/50283)
10. Thomson DJ (1982). Spectrum estimation and harmonic analysis. *Proceedings of the IEEE*
    70(9):1055-1096.
    [doi:10.1109/PROC.1982.12433](https://doi.org/10.1109/PROC.1982.12433)
11. Mitra PP, Pesaran B (1999). Analysis of dynamic brain imaging data. *Biophysical
    Journal* 76(2):691-708.
    [doi:10.1016/S0006-3495(99)77236-X](https://doi.org/10.1016/S0006-3495(99)77236-X)
12. Welch PD (1967). The use of fast Fourier transform for the estimation of power spectra.
    *IEEE Transactions on Audio and Electroacoustics* 15(2):70-73.
    [doi:10.1109/TAU.1967.1161901](https://doi.org/10.1109/TAU.1967.1161901)
13. Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical and
    powerful approach to multiple testing. *Journal of the Royal Statistical Society B*
    57(1):289-300.
    [doi:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
14. Harris FJ (1978). On the use of windows for harmonic analysis with the discrete Fourier
    transform. *Proceedings of the IEEE* 66(1):51-83.
    [doi:10.1109/PROC.1978.10837](https://doi.org/10.1109/PROC.1978.10837)
15. Efron B, Stein C (1981). The jackknife estimate of variance. *The Annals of Statistics*
    9(3):586-596.
    [doi:10.1214/aos/1176345462](https://doi.org/10.1214/aos/1176345462)
16. Phipson B, Smyth GK (2010). Permutation p-values should never be zero: calculating
    exact p-values when permutations are randomly drawn. *Statistical Applications in
    Genetics and Molecular Biology* 9(1):Article 39.
    [doi:10.2202/1544-6115.1585](https://doi.org/10.2202/1544-6115.1585)
17. Slepian D (1978). Prolate spheroidal wave functions, Fourier analysis, and uncertainty
    V: the discrete case. *Bell System Technical Journal* 57(5):1371-1430.
    [doi:10.1002/j.1538-7305.1978.tb02104.x](https://doi.org/10.1002/j.1538-7305.1978.tb02104.x)
18. Bullock M, Jackson GD, Abbott DF (2021). Artifact reduction in simultaneous EEG-fMRI: a
    systematic review of methods and contemporary usage. *Frontiers in Neurology* 12:622719.
    [doi:10.3389/fneur.2021.622719](https://doi.org/10.3389/fneur.2021.622719)
19. Widmann A, Schröger E, Maess B (2015). Digital filter design for electrophysiological
    data: a practical approach. *Journal of Neuroscience Methods* 250:34-46.
    [doi:10.1016/j.jneumeth.2014.08.002](https://doi.org/10.1016/j.jneumeth.2014.08.002)
20. Leske S, Dalal SS (2019). Reducing power line noise in EEG and MEG data via spectrum
    interpolation. *NeuroImage* 189:763-776.
    [doi:10.1016/j.neuroimage.2019.01.026](https://doi.org/10.1016/j.neuroimage.2019.01.026)
21. Gorgolewski KJ, Auer T, Calhoun VD, et al. (2016). The brain imaging data structure, a
    format for organizing and describing outputs of neuroimaging experiments. *Scientific Data*
    3:160044. [doi:10.1038/sdata.2016.44](https://doi.org/10.1038/sdata.2016.44)
22. Pernet CR, Appelhoff S, Gorgolewski KJ, et al. (2019). EEG-BIDS, an extension to the brain
    imaging data structure for electroencephalography. *Scientific Data* 6:103.
    [doi:10.1038/s41597-019-0104-8](https://doi.org/10.1038/s41597-019-0104-8)
23. Gramfort A, Luessi M, Larson E, et al. (2014). MNE software for processing MEG and EEG
    data. *NeuroImage* 86:446-460.
    [doi:10.1016/j.neuroimage.2013.10.027](https://doi.org/10.1016/j.neuroimage.2013.10.027)
24. Steyrl D, Krausz G, Koschutnig K, Edlinger G, Müller-Putz GR (2018). Online reduction of
    artifacts in EEG of simultaneous EEG-fMRI using reference layer adaptive filtering.
    *Brain Topography* 31(1):129-149.
    [doi:10.1007/s10548-017-0606-7](https://doi.org/10.1007/s10548-017-0606-7)
25. Mitra PP, Bokil H (2007). *Observed Brain Dynamics: Analyzing Brain Activity in Time,
    Frequency, and Space*. Oxford University Press.
26. Gramfort A, Luessi M, Larson E, et al. (2013). MEG and EEG data analysis with MNE-Python.
    *Frontiers in Neuroscience* 7:267.
    [doi:10.3389/fnins.2013.00267](https://doi.org/10.3389/fnins.2013.00267)
27. Benjamini Y, Yekutieli D (2001). The control of the false discovery rate in multiple testing
    under dependency. *The Annals of Statistics* 29(4):1165-1188.
    [doi:10.1214/aos/1013699998](https://doi.org/10.1214/aos/1013699998)

Implementation reference: MNE-Python,
[`mne.filter.notch_filter`](https://mne.tools/stable/generated/mne.filter.notch_filter.html),
for the `spectrum_fit` multitaper fitting mode used by `decomb`.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
