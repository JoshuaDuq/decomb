# decomb

<p align="center">
  <img src="logo.png" alt="decomb logo" width="420">
</p>

`decomb` is Python research software for detecting and suppressing narrowband harmonic
and isolated spectral artifacts in continuous EEG. Detection is performed separately for
each recording. The output is a BrainVision BIDS derivative with a tabular record of all
stopbands and transition bands.

Frequencies affected by a stopband or transition are classified as unavailable for
inference. The method provides no estimate of neural activity within those intervals.

## Scientific scope

The intended input is continuous EEG acquired during or near fMRI after gradient and
pulse artifact correction [1, 2]. Cryogenic pumps and scanner ventilation systems can
produce residual periodic EEG artifacts [3, 4]. A harmonic spectrum identifies regular
frequency structure. Attribution to a physical source requires independent experimental
evidence. Source control should be performed when the source can be identified [5].

Frequency filtering can alter signal amplitude and temporal structure. Filter type,
stopband edges, transition bandwidth, filter length, phase response, and computation
direction should be reported in subsequent analyses [6]. `decomb` records the frequency
geometry and measured spectral change for this purpose. Broad spectral features and
transient artifacts require methods based on their temporal or spatial properties [7].

The implementation reads the EEG signal and its BIDS metadata. Scanner triggers and
scanner clock annotations are not used.

## Requirements and installation

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

Development dependencies are installed with the following command.

```bash
python3 -m pip install -e '.[dev]'
```

## Input data

Input recordings must use BrainVision format within an EEG-BIDS dataset [12, 14]. The
discovery patterns support subject directories with or without session directories and
with optional run entities. MNE-BIDS reads each recording with data preloaded and raises
an error for channel metadata mismatches.

Spectral estimation uses all channels typed as EEG. Each recording must contain finite
values and at least one complete estimation window. Isolated-line detection additionally
requires at least two non-overlapping windows.

## Configuration

The correction exposes two scientific settings. The complete packaged configuration is
available in [`src/decomb/defaults.yaml`](src/decomb/defaults.yaml).

| Setting | Default | Function |
| --- | --- | --- |
| `removal.estimation_window_s` | 54.0 s | Defines the stationarity interval and spectral resolution |
| `removal.frequency_range_hz` | 0.0 to 100.0 Hz | Defines the frequencies eligible for detection and filtering |

`paths.bids_root` identifies the input dataset. The output and report locations have
packaged defaults. Optional `frequency_bands` entries define study bands for reporting
unavailable and retained bandwidth. They do not affect detection or filtering.

Configuration is resolved in the following order.

1. Command-line path overrides
2. The path supplied with `--config`
3. The path in the `DECOMB_CONFIG` environment variable
4. `decomb.yaml` in the working directory
5. Packaged defaults

Unknown settings and obsolete settings raise an error. The configured upper frequency
cannot exceed 100 Hz and is limited by the available spectrum and the recording Nyquist
frequency.

For the default window duration $T=54$ s, the DFT bin spacing is

$$
\Delta f=\frac{1}{T}=0.018519\ \mathrm{Hz}
$$

and the Hann half-power resolution [8] is

$$
r=\frac{1.4382}{T}=0.026633\ \mathrm{Hz}.
$$

## Command-line workflow

```bash
decomb diagnose --config decomb.yaml
decomb apply --config decomb.yaml
decomb verify --config decomb.yaml
decomb psd --config decomb.yaml
```

`diagnose` fits the comb and isolated-line models without changing the recordings. It
writes the fitted fundamental, model evidence, isolated-line evidence, planned
stopbands, and unavailable bandwidth.

`apply` repeats the fit for every discovered recording, filters EEG channels, writes a
complete derivative through a staging directory, and reads every written BrainVision
recording from disk. The stage rejects an existing output directory.

`verify` reconstructs the filter plan from the derivative manifest. It reads the source
and derivative from disk and measures attenuation without fitting new targets.

`psd` computes matched source and derivative spectra with MNE and writes cohort,
frequency-panel, and per-recording figures together with the numerical spectra.

`apply` and `verify` require the complete discovered dataset. `diagnose` and `psd` accept
subject subsets.

## Methods

### Window construction

Let $f_s$ denote the sampling frequency and let $T$ denote the configured estimation
duration. The number of samples per window is

$$
N=\mathrm{round}(T f_s).
$$

Windows overlap by 50 percent. The final window is aligned to the end of the recording
when the regular sequence does not include the final sample. A greedy selection of
non-overlapping windows supplies independent observations for isolated-line model
selection.

### Spectral density

For EEG channel $x[n]$ and Hann window $w[n]$, the implementation computes the one-sided
periodogram

$$
S(f_k)=\frac{c_k}{f_s\sum_n w^2[n]}
\left|\sum_{n=0}^{N-1}w[n]x[n]e^{-2\pi i kn/N}\right|^2
$$

with

$$
f_k=\frac{k f_s}{N}.
$$

The scaling factor $c_k$ is 2 for interior stored bins and 1 for the first and last
stored bins. Window spectra are summarized by the median across EEG channels. The
whole-recording spectrum is obtained by averaging power across windows within each
channel and then taking the channel median. Power is represented in decibels as

$$
X(f)=10\log_{10}S(f).
$$

Local maxima are refined with a three-point parabolic interpolation in decibel space.
The Hann half-power resolution is $r=1.4382/T$ [8].

### Comb model selection

Candidate fundamentals begin at $2r$ and end at one quarter of the upper analysis
frequency. The latter boundary ensures at least four observable multiples. Candidate
spacing is refined according to the highest observable harmonic. For candidate
$f_{0,j}$, the next value is

$$
f_{0,j+1}=f_{0,j}+\frac{\Delta f}
{2\left\lfloor f_{\max}/f_{0,j}\right\rfloor}.
$$

The harmonic numbers evaluated for a candidate are

$$
\mathcal K(f_0)=\left\{k\in\mathbb N\mid
f_{\min}\leq kf_0\leq f_{\max}\right\}.
$$

The contrast at harmonic $k$ compares the candidate grid with the two halfway
frequencies.

$$
d_k=X(kf_0)-\frac{X(kf_0-f_0/2)+X(kf_0+f_0/2)}{2}.
$$

The null model fixes the mean contrast at zero. The alternative model estimates a
positive mean contrast $\bar d$. A nonpositive fitted mean rejects the alternative. Let
$K$ denote the number of evaluated harmonics, let $M$ denote the number of candidate
fundamentals, and define

$$
R_0=\sum_k d_k^2
$$

and

$$
R_1=\sum_k(d_k-\bar d)^2.
$$

The model comparison statistic is

$$
\Delta\mathrm{BIC}=K\log\!\left(\frac{R_1}{R_0}\right)
+\log K+2\log M.
$$

The final term accounts for the search over candidate fundamentals. A candidate is
supported when $\Delta\mathrm{BIC}<0$ [9]. Supported candidates are ranked by

$$
Q(f_0)=\bar d\sqrt{K}.
$$

Ties in $Q$ are resolved by the lower $\Delta\mathrm{BIC}$. After selection, every
integer multiple within the configured analysis range is included. Harmonic number,
amplitude, prominence, and prevalence thresholds are absent from this step.

Each authorized harmonic is localized within $\pm r$ of its predicted grid frequency in
the whole-recording spectrum and in every overlapping window. The local maximum is
refined by the same parabolic interpolation used for the whole spectrum.

### Isolated-line model selection

SciPy `find_peaks` identifies local maxima in the whole-recording decibel spectrum [16].
Candidates outside the configured range and candidates within $r$ of the selected comb
grid are excluded. Each candidate is assigned its contiguous half-power basin, limited
to off-comb bins. Overlapping basins are represented once by their strongest maximum.
This prevents a moving line with a shallow valley in its time-averaged spectrum from
being counted as two sources. Let $M_I$ denote the resulting number of candidate
features.

The strongest local maximum inside a feature's measured basin is refined separately in
every window, producing the trajectory $p_j$. Two model comparisons are evaluated for
each feature. The temporal comparison uses the non-overlapping windows and the contrast

$$
g_j=X_j(p_j)-\frac{X_j(p_j-2r)+X_j(p_j+2r)}{2}.
$$

The zero-mean model is compared with a model containing one positive shared mean. For
$J$ independent windows, the statistic has the same form as the comb comparison.

$$
\Delta\mathrm{BIC}_{\mathrm{temporal}}
=J\log\!\left(\frac{R_1}{R_0}\right)
+\log J+2\log M_I.
$$

The shape comparison uses linear power from $\min_j p_j-4r$ to
$\max_j p_j+4r$. The null model
is a quadratic background. The line model adds a positive coefficient multiplying the
mean Hann response along the measured trajectory. For a frequency offset $u$ in DFT
bins, the Hann power response is

$$
H(u)=\left[
\frac{0.5\,\mathrm{sinc}(u)
+0.25\,\mathrm{sinc}(u-1)
+0.25\,\mathrm{sinc}(u+1)}{0.5}
\right]^2
$$

and for $W$ overlapping windows the fitted response is

$$
\bar H(f)=\frac{1}{W}\sum_{j=1}^{W}
H\!\left(\frac{f-p_j}{\Delta f}\right).
$$

The shape comparison requires at least eight local frequency bins. One trajectory
position parameter is charged for each of the $J$ independent windows. Its statistic is

$$
\Delta\mathrm{BIC}_{\mathrm{shape}}
=N_L\log\!\left(\frac{R_{1,L}}{R_{0,L}}\right)
+(J+1)\log N_L+2\log M_I.
$$

Here $N_L$ is the number of local frequency bins. A line is retained when both
$\Delta\mathrm{BIC}_{\mathrm{temporal}}$ and
$\Delta\mathrm{BIC}_{\mathrm{shape}}$ are negative. The manifest stores their maximum as
the least favorable evidence value. No amplitude, prominence, cohort-prevalence, or
fixed-frequency threshold decides admission. Every retained line is localized in every
overlapping window, so its stopband covers observed movement instead of assuming a
stationary whole-recording peak.

### Stopband construction

For one harmonic or isolated line, let $p_0,p_1,\ldots,p_J$ denote its positions in the
whole spectrum and overlapping windows. The localization uncertainty is

$$
\epsilon=\frac{\Delta f}{2}.
$$

The observed envelope is

$$
a=\min_j p_j-\epsilon
$$

and

$$
b=\max_j p_j+\epsilon.
$$

With centre $m=(a+b)/2$, the stopband half-width is

$$
h=\max\!\left(\frac{b-a}{2},\frac{r}{2}\right)
$$

and the requested stopband is

$$
[L,U]=[m-h,m+h].
$$

The total transition bandwidth is

$$
q=\frac{3.3}{T}.
$$

This value follows the MNE automatic length factor for a Hamming `firwin` design [6,
10, 11]. Stopbands separated by $q$ or less are merged. The transition-inclusive
unavailable interval is

$$
E=[L-q/2,U+q/2].
$$

For analysis band $A=[A_0,A_1]$, unavailable and retained shares are

$$
C_A=\frac{\left|A\cap\bigcup_i E_i\right|}{A_1-A_0}
$$

and

$$
R_A=1-C_A.
$$

### FIR application

All stopbands for a recording are applied to EEG channels in one call to MNE
`Raw.notch_filter` [10, 11]. The call uses the following parameters.

| Parameter | Value |
| --- | --- |
| `freqs` | Measured stopband centres |
| `notch_widths` | Measured stopband widths |
| `trans_bandwidth` | $3.3/T$ |
| `method` | `fir` |
| `filter_length` | `auto` |
| `phase` | `zero` |
| `fir_window` | MNE default `hamming` |
| `fir_design` | MNE default `firwin` |
| `pad` | MNE default `reflect_limited` |
| `n_jobs` | `-1` |

MNE describes this configuration as a one-pass, zero-phase, noncausal FIR filter with
delay compensation. The Hamming `firwin` response has a reported passband ripple of
0.0194 dB and stopband attenuation of 53 dB [6]. The automatic filter length is computed
by MNE from the shortest transition bandwidth and the Hamming factor 3.3. The Nyquist
and zero-frequency boundaries are checked before filtering. A transition reaching either
boundary raises an error.

### Attenuation and verification

Power is summed across the bins inside each stopband for each EEG channel and averaged
across channels. These spectra use complete, non-overlapping Hann blocks of duration $T$.
Samples after the final complete block are excluded. The measured change is

$$
\Delta_i=10\log_{10}\!\left(
\frac{P_{\mathrm{after},i}}{P_{\mathrm{before},i}}
\right)\ \mathrm{dB}.
$$

The derivative is written as multiplexed little-endian BrainVision
`IEEE_FLOAT_32` data using the channel resolutions from the source header. Each file is
read from disk and compared with the expected quantized values. A deviation above the
float32 quantization bound raises an error.

Independent verification reconstructs each stopband from the manifest. It checks channel
names, channel types, sample count, sampling frequency, and spectral frequency grids. It
then recomputes stopband attenuation and measures the largest adjacent available-line
contrast within half the fitted fundamental.

### Power spectral density figures

The quality-control spectra use MNE `Raw.compute_psd` with Welch estimation [18]. Each
segment contains $\mathrm{round}(Tf_s)$ samples and adjacent segments overlap by 50
percent. The frequency range matches the correction range and is limited below Nyquist.
MNE defaults supply a Hamming segment window, mean removal within each segment, mean
aggregation across segments, and omission of spans marked by bad annotations. The median
across EEG channels is used for each recording. Source and derivative spectra use the
same channels, samples, sampling frequency, and frequency grid.

### Conventional-notch reference

The real-data comparison uses one fitted recording and gives both arms the same detected
target identities and the same trajectory-envelope centres. The decomb arm uses the
measured stopband widths and $q=3.3/T$. The reference arm uses the documented MNE
[`notch_filter`](https://mne.tools/stable/generated/mne.filter.notch_filter.html)
defaults: stopband width $f/200$ at centre $f$ and a 1 Hz transition bandwidth [10,
11]. Reference bands are merged wherever those transitions overlap,
because MNE rejects overlapping FIR stopbands. Both arms are applied to copies of the
same real samples and measured on the same Welch grid. The comparison therefore measures
the frequency cost of filter geometry; it does not claim that one algorithm recovered
neural activity inside a removed interval.

## Outputs and provenance

The derivative follows EEG-BIDS and BIDS derivative conventions [12, 13, 19]. Sidecars
are copied from the source dataset. Hidden files, backup files, temporary files, lock
files, and source `.eeg` binaries are excluded. Corrected `.eeg` binaries are written by
the pipeline.

`harmonic_notch_manifest.tsv` contains one row per merged stopband. Each row records the
line class, contributing harmonic numbers, isolated-line frequencies, least favorable
isolated-line evidence, fitted fundamental, comb evidence, stopband edges, unavailable
edges, transition bandwidth, estimation-window count, in-stopband power change,
BrainVision round-trip deviation, and band-specific unavailable and retained shares.
Floating-point geometry is written with 17 significant digits.

`dataset_description.json` declares a BIDS derivative and records `GeneratedBy`, the
`decomb` version, fitted method parameters, and a relative `SourceDatasets` URL.

`effective_config_apply.txt` and `effective_config_verify.txt` record every setting in
force, its value, and whether it came from packaged defaults, the user configuration, or
a derived expression.

`harmonic_notch_verification.tsv` records independently measured stopband attenuation and
adjacent available-line contrast from the files on disk.

The PSD stage writes `psd_before_after.png`, `psd_before_after_panels.png`,
`psd_before_after_per_recording.png`, and `psd_before_after.npz`.

## Real-data demonstration and validation

The displayed example uses six recordings from one participant, 0.827 hours of real EEG
in total. Individual recordings contain 8.18 to 8.48 minutes. Showing one participant
avoids blurring narrow features across participants whose fitted fundamentals differ by
fractions of a spectral bin. Source and derivative spectra use 54 s Welch segments with
50 percent overlap.

![Power spectra before and after harmonic line filtering in six real EEG recordings from one participant](docs/psd_before_after.png)

The same participant is shown below in 10 Hz panels so narrow changes remain visible
rather than collapsing into sub-pixel features on the overview axis.

![Power spectra before and after harmonic line filtering in 10 Hz panels for one real EEG participant](docs/psd_before_after_panels.png)

For these six recordings, the fitted fundamental ranged from 1.199771 to 1.200328 Hz,
comb $\Delta\mathrm{BIC}$ ranged from -44.74 to -24.66, and 5 to 8 isolated lines were
selected per recording. Plans contained 87 to 91 merged stopbands and made 10.34 to
11.03 Hz unavailable. Outside each recording's own transition-inclusive unavailable
intervals, the median absolute PSD change was 0.00784 dB, the 99th percentile was 0.211
dB, and 0.024 percent of bins changed by more than 1 dB. Inside unavailable intervals,
the median bin change was -13.47 dB. The participant-median PSD changed by more than 1 dB
in 11.3 percent of bins, reached -46.97 dB at 51.593 Hz, and had a maximum positive change
of 0.129 dB. These preservation measurements apply only outside the declared intervals.

The comparison below uses the first 8.2-minute recording. Both filter arms receive the
same detected target identities and centres. Decomb's measured widths and transitions
make 10.5 Hz unavailable; MNE's default notch widths and 1 Hz transitions make 96.4 Hz
unavailable after overlapping bands are merged. This is a comparison of geometry for
this dense comb, not a general ranking of MNE or notch filters.

![Measured decomb notch geometry compared with conventional MNE notch defaults on one real EEG recording](docs/notch_comparison_real.png)

The figures can be regenerated from a configured derivative and its source dataset.

```bash
decomb psd --config decomb.yaml --subjects sub-0011 --report-dir docs
python3 docs/make_notch_comparison_real.py \
  --config decomb.yaml --subject sub-0011 --recording-index 1
```

The detector was separately audited on all 90 recordings from 15 participants, totaling
12.1 hours of EEG. Every recording selected 83 authorized harmonics. Estimated
fundamentals ranged from 1.199659 to 1.200551 Hz and comb $\Delta\mathrm{BIC}$ ranged
from -44.74 to -8.88. The trajectory-aware model selected 3 to 13 isolated lines per
recording, with a median of 7. Plans contained 86 to 96 merged stopbands and made 9.92 to
12.39 Hz unavailable within the configured 0 to 100 Hz range.

Independent verification reconstructed all 8,120 stopbands from the written manifest
and files on disk. Median stopband power change was -28.64 dB; per-recording median
attenuation ranged from -30.80 to -25.68 dB. Outside the union of every unavailable
interval, the median change in the largest adjacent line contrast was +0.00004 dB, the
99th percentile was +0.083 dB, and the maximum was +0.608 dB. Some strong available
peaks remain unchanged; automatically relabeling all of them as artifacts admitted about
100 ordinary spectral maxima per recording in a rejected control analysis. The verifier
reports these peaks for inspection instead of silently erasing scientifically ambiguous
frequencies.

## Software implementation

The project version is `0.1.0` and the license is BSD 3-Clause. The declared minimum
runtime versions and their roles are listed below. The table reproduces the constraints
in `pyproject.toml`. Resolved versions for a specific analysis environment should
accompany the derivative or publication.

| Software | Minimum version | Use in this repository |
| --- | --- | --- |
| Python | 3.11 | Runtime and command-line interface |
| NumPy [15] | 1.24 | Array operations, real FFT, least squares, interpolation, and numerical summaries |
| SciPy [16] | 1.11 | Local-maximum detection with `scipy.signal.find_peaks` |
| MNE-Python [10, 11] | 1.6 | EEG channel selection, zero-phase FIR notch filtering, and Welch spectra |
| MNE-BIDS [14] | 0.14 | BIDS path parsing and BrainVision BIDS reading |
| pandas | 2.0 | Manifest and report tables |
| PyYAML | 6.0 | Configuration loading |
| Matplotlib [17] | 3.8 | Noninteractive quality-control figures |
| joblib | 1.3 | Parallel execution backend used by the scientific Python stack |
| pybv | 0.7.5 | BrainVision support for the test and export toolchain |

The development dependencies are pytest 8.0 or newer and Ruff 0.6 or newer. Exact
resolved versions can be recorded with the following command.

```bash
python3 -m pip freeze > software-versions.txt
```

## Tests

```bash
pytest -q
ruff check src tests
```

The test suite covers spectral scaling, off-grid peak refinement, comb recovery, complete
harmonic enumeration, subharmonic ranking, absent-comb errors, stationary and moving
isolated-line selection, irregular trajectories, broad-peak exclusion, window
independence, stopband geometry, MNE filtering, manifest reconstruction, all-interval
residual masking, conventional-notch comparison, BIDS discovery, binary quantization,
sidecar copying, configuration validation, provenance, command routing, and matched PSD
computation.

## Methodological limitations

Neural and artifactual activity at the same frequency are not identifiable from one EEG
recording. Stopbands and their transitions therefore remain unavailable for inference.

The stationarity duration $T$ defines the temporal scale over which frequency movement is
tracked. A change to $T$ changes the DFT spacing, spectral resolution, minimum stopband
width, transition bandwidth, and FIR length.

The comb model establishes periodic spectral structure and does not identify its physical
source. Broad rhythms and transient artifacts fall outside the line model. Prior gradient
and pulse artifact correction remains necessary for simultaneous EEG-fMRI data [1, 2,
7].

## References

1. Allen PJ, Josephs O, Turner R. A method for removing imaging artifact from continuous
   EEG recorded during functional MRI. *NeuroImage*. 2000, 12, 230 to 239.
   [DOI](https://doi.org/10.1006/nimg.2000.0599)
2. Niazy RK, Beckmann CF, Iannetti GD, Brady JM, Smith SM. Removal of FMRI environment
   artifacts from EEG data using optimal basis sets. *NeuroImage*. 2005, 28, 720 to 737.
   [DOI](https://doi.org/10.1016/j.neuroimage.2005.06.067)
3. Rothlübbers S, Relvas V, Leal A, Murta T, Lemieux L, Figueiredo P. Characterisation
   and reduction of the EEG artefact caused by the helium cooling pump in the MR
   environment. *Brain Topography*. 2015, 28, 208 to 220.
   [DOI](https://doi.org/10.1007/s10548-014-0408-0)
4. Nierhaus T, Gundlach C, Goltz D, et al. Internal ventilation system of MR scanners
   induces specific EEG artifact during simultaneous EEG-fMRI. *NeuroImage*. 2013, 74,
   70 to 76. [DOI](https://doi.org/10.1016/j.neuroimage.2013.02.016)
5. Mullinger KJ, Castellone P, Bowtell R. Best current practice for obtaining high quality
   EEG data during simultaneous fMRI. *Journal of Visualized Experiments*. 2013, 76,
   e50283. [DOI](https://doi.org/10.3791/50283)
6. Widmann A, Schröger E, Maess B. Digital filter design for electrophysiological data,
   a practical approach. *Journal of Neuroscience Methods*. 2015, 250, 34 to 46.
   [DOI](https://doi.org/10.1016/j.jneumeth.2014.08.002)
7. Bullock M, Jackson GD, Abbott DF. Artifact reduction in simultaneous EEG-fMRI, a
   systematic review of methods and contemporary usage. *Frontiers in Neurology*. 2021,
   12, 622719. [DOI](https://doi.org/10.3389/fneur.2021.622719)
8. Harris FJ. On the use of windows for harmonic analysis with the discrete Fourier
   transform. *Proceedings of the IEEE*. 1978, 66, 51 to 83.
   [DOI](https://doi.org/10.1109/PROC.1978.10837)
9. Schwarz G. Estimating the dimension of a model. *Annals of Statistics*. 1978, 6,
   461 to 464. [DOI](https://doi.org/10.1214/aos/1176344136)
10. Gramfort A, Luessi M, Larson E, et al. MNE software for processing MEG and EEG data.
    *NeuroImage*. 2014, 86, 446 to 460.
    [DOI](https://doi.org/10.1016/j.neuroimage.2013.10.027)
11. Gramfort A, Luessi M, Larson E, et al. MEG and EEG data analysis with MNE-Python.
    *Frontiers in Neuroscience*. 2013, 7, 267.
    [DOI](https://doi.org/10.3389/fnins.2013.00267)
12. Pernet CR, Appelhoff S, Gorgolewski KJ, et al. EEG-BIDS, an extension to the Brain
    Imaging Data Structure for electroencephalography. *Scientific Data*. 2019, 6, 103.
    [DOI](https://doi.org/10.1038/s41597-019-0104-8)
13. Gorgolewski KJ, Auer T, Calhoun VD, et al. The Brain Imaging Data Structure, a format
    for organizing and describing outputs of neuroimaging experiments. *Scientific
    Data*. 2016, 3, 160044. [DOI](https://doi.org/10.1038/sdata.2016.44)
14. Appelhoff S, Sanderson M, Brooks TL, et al. MNE-BIDS, organizing
    electrophysiological data into the BIDS format and facilitating their analysis.
    *Journal of Open Source Software*. 2019, 4, 1896.
    [DOI](https://doi.org/10.21105/joss.01896)
15. Harris CR, Millman KJ, van der Walt SJ, et al. Array programming with NumPy. *Nature*.
    2020, 585, 357 to 362. [DOI](https://doi.org/10.1038/s41586-020-2649-2)
16. Virtanen P, Gommers R, Oliphant TE, et al. SciPy 1.0, fundamental algorithms for
    scientific computing in Python. *Nature Methods*. 2020, 17, 261 to 272.
    [DOI](https://doi.org/10.1038/s41592-019-0686-2)
17. Hunter JD. Matplotlib, a 2D graphics environment. *Computing in Science and
    Engineering*. 2007, 9, 90 to 95.
    [DOI](https://doi.org/10.1109/MCSE.2007.55)
18. Welch P. The use of fast Fourier transform for the estimation of power spectra, a
    method based on time averaging over short, modified periodograms. *IEEE Transactions
    on Audio and Electroacoustics*. 1967, 15, 70 to 73.
    [DOI](https://doi.org/10.1109/TAU.1967.1161901)
19. BIDS Contributors. BIDS Derivatives. *Brain Imaging Data Structure specification*.
    Version 1.11.1. [Specification](https://bids-specification.readthedocs.io/en/stable/derivatives/introduction.html)

The MNE implementation details are documented in the
[notch-filter API](https://mne.tools/stable/generated/mne.filter.notch_filter.html), the
[filtering methods tutorial](https://mne.tools/stable/auto_tutorials/preprocessing/25_background_filtering.html),
and the
[Welch PSD API](https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.compute_psd).

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).
