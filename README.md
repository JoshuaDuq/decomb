# decomb

<p align="center">
  <img src="logo.png" alt="decomb logo" width="420">
</p>

`decomb` detects and suppresses narrowband harmonic and isolated spectral artifacts in
continuous EEG. Each recording is fitted independently and written as a BrainVision
BIDS derivative with measured stopbands, transition bands, filter response, and
verification results.

## Problem statement

Residual periodic artifacts can remain in EEG acquired during or near fMRI after
gradient and pulse artifact correction [[1](#user-content-ref-1),
[2](#user-content-ref-2)]. Cryogenic pumps and scanner ventilation systems are reported
sources [[3](#user-content-ref-3), [4](#user-content-ref-4)]. Their frequencies can drift
within a recording and can occur as a harmonic series or as isolated narrowband lines.

Filtering removes the measured signal within every stopband and transition. These
frequencies are classified as unavailable for inference. The method does not estimate
the neural activity that occupied them. Physical attribution requires independent
evidence, and source control remains appropriate when the source is known
[[5](#user-content-ref-5)].

## Scope

Input recordings must use BrainVision format in an EEG-BIDS dataset
[[12](#user-content-ref-12), [14](#user-content-ref-14)]. Subject directories may contain
optional session and run entities. MNE-BIDS reads each recording with data preloaded.
Channel metadata mismatches raise an error.

All channels typed as EEG are included in spectral estimation. Recordings must contain
finite values and at least one complete estimation window. Isolated-line detection
requires at least two non-overlapping windows. Scanner triggers and scanner clock
annotations are not used.

The method identifies narrow spectral structure. Broad rhythms and transient artifacts
require temporal or spatial methods [[7](#user-content-ref-7)]. A detected comb does not
identify its physical source. Neural and artifactual activity at the same frequency
cannot be separated from one EEG recording.

## Installation

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

Development dependencies are installed from the source checkout.

```bash
python3 -m pip install -e '.[dev]'
```

## Quick start

```bash
decomb diagnose --config decomb.yaml
decomb apply --config decomb.yaml
decomb verify --config decomb.yaml
decomb psd --config decomb.yaml
```

| Command | Operation |
| --- | --- |
| `diagnose` | Fits comb and isolated-line models and writes the proposed filter plan |
| `apply` | Fits each recording, filters EEG channels, and writes the complete derivative |
| `verify` | Reconstructs manifest stopbands and measures the files written to disk |
| `psd` | Writes matched source and derivative spectra and figures |

`apply` and `verify` require the complete discovered dataset. `diagnose` and `psd`
accept subject subsets. An existing output directory causes `apply` to stop.

## Configuration

The packaged configuration is
[`src/decomb/defaults.yaml`](src/decomb/defaults.yaml). Two settings define the
scientific correction.

| Setting | Default | Function |
| --- | --- | --- |
| `removal.estimation_window_s` | 54.0 s | Stationarity interval and spectral resolution |
| `removal.frequency_range_hz` | 0.0 to 100.0 Hz | Frequencies eligible for detection and filtering |

The default window gives Fourier bins separated by 0.018519 Hz and a Hann half-power
resolution of 0.026633 Hz [[8](#user-content-ref-8)]. Changing the duration changes the
frequency spacing, resolution, stopband width, transition width, and FIR duration.

`paths.bids_root` identifies the input dataset. Output and report locations have
packaged defaults. Optional `frequency_bands` entries report unavailable and retained
bandwidth and do not affect detection.

Configuration is resolved in this order.

1. Command-line path overrides
2. The file supplied with `--config`
3. The `DECOMB_CONFIG` environment variable
4. `decomb.yaml` in the working directory
5. Packaged defaults

Unknown and obsolete settings raise an error. The upper analysis frequency cannot
exceed 100 Hz and is limited by the available spectrum and recording Nyquist frequency.

## Methods

### Spectral estimation

The configured duration is converted to the nearest whole number of samples. Windows
overlap by 50 percent, and the final window is aligned with the end of the recording. A
greedy non-overlapping subset supplies independent observations for model comparison.

Each EEG window is multiplied by a Hann window. A one-sided periodogram is normalized by
sampling frequency and Hann energy. Zero frequency and even-length Nyquist bins retain
their original power. Every other stored positive-frequency bin is doubled, including
the final bin for odd-length windows. Window spectra are summarized by their channel
median. The whole-recording spectrum averages window power within each channel before
the channel median is calculated.

Power is represented in decibels. Local maxima are refined by three-point parabolic
interpolation in decibel space. The Hann half-power resolution follows Harris
[[8](#user-content-ref-8)].

### Comb selection

Candidate fundamentals begin at twice the Hann half-power resolution and end at one
quarter of the upper analysis frequency. This range requires at least four observable
harmonics. Candidate spacing becomes finer as the highest observable harmonic
increases.

For every candidate, power on the harmonic grid is compared with power halfway between
adjacent grid positions. A zero-mean model is compared with a model containing one
positive shared mean. The Bayesian information criterion includes the fitted mean, the
number of evaluated harmonics, and a penalty for the number of candidate fundamentals
searched [[9](#user-content-ref-9)]. Supported candidates have a lower criterion value
than the zero-mean model. They are ranked by mean grid contrast weighted by the square
root of the harmonic count, with the criterion value resolving ties.

Every integer multiple of the selected fundamental within the analysis range is
retained. Each harmonic is localized within one Hann half-power resolution of its
predicted position in the whole-recording spectrum and every overlapping window.
Amplitude, prominence, prevalence, and harmonic-number thresholds do not decide
admission.

### Isolated-line selection

SciPy `find_peaks` finds local maxima outside the selected comb
[[16](#user-content-ref-16)]. Each maximum is assigned its contiguous half-power basin.
Overlapping basins are represented by their strongest maximum. The strongest maximum
inside each basin is localized independently in every window.

Two Bayesian information criterion comparisons are required. The temporal comparison
tests whether power at the tracked peak is consistently greater than power two Hann
resolutions to either side in independent windows. The shape comparison tests a
positive Hann line response against a quadratic background in linear power. The shape
fit uses at least eight local frequency bins and charges one position parameter for each
independent window. Both comparisons must support the line. The least favorable result
is written to the manifest.

### Stopbands and FIR filtering

Each stopband covers the lowest and highest positions observed in the whole recording
and overlapping windows, expanded by half a Fourier bin. Its minimum width is the Hann
half-power resolution. Stopbands are merged when their transitions would overlap.

The total transition bandwidth is 3.3 divided by the estimation-window duration in
seconds. MNE assigns half of this width to each stopband edge. All merged stopbands are
passed to MNE `Raw.notch_filter` in one call, and only EEG channels are modified
[[10](#user-content-ref-10), [11](#user-content-ref-11)].

| Parameter | Value |
| --- | --- |
| `freqs` | Measured stopband centres |
| `notch_widths` | Measured stopband widths |
| `trans_bandwidth` | Total width of 3.3 divided by the estimation-window duration |
| `method` | `fir` |
| `filter_length` | `auto` |
| `phase` | `zero` |
| `fir_window` | `hamming` |
| `fir_design` | `firwin` |
| `pad` | `reflect_limited` |
| `n_jobs` | `-1` |

This is a one-pass, zero-phase, noncausal FIR design with delay compensation. MNE reports
0.0194 dB passband ripple and 53 dB stopband attenuation for the Hamming `firwin` design
[[6](#user-content-ref-6)]. Automatic length uses the per-edge transition width and is
approximately twice the estimation-window duration. The default duration therefore
produces a filter of approximately 108 seconds, with sample rounding to an odd
coefficient count.

Before filtering, `decomb` constructs the same design with MNE `create_filter`. The
manifest records its exact sample count, duration, minimum measured stopband attenuation,
and maximum measured passband deviation. A transition reaching zero frequency or
Nyquist raises an error.

### Attenuation and verification

Stopband power is summed across frequency bins for each EEG channel and averaged across
channels. Source and derivative spectra use complete, non-overlapping Hann blocks with
the configured duration. Samples outside the final complete block are excluded. The
reported change is the decibel ratio of derivative power to source power.

The derivative uses multiplexed little-endian BrainVision `IEEE_FLOAT_32` data and
`_desc-decomb_eeg.vhdr`, `.vmrk`, and `.eeg` filenames. Internal BrainVision references
are rewritten. Every channel retains its declared unit and resolution. Each written
recording is read from disk and compared with the expected quantized values. A deviation
above the float32 quantization bound raises an error.

Independent verification first compares the current scientific settings with the
apply-time settings in `dataset_description.json`. It then checks channel names, channel
types, sample count, sampling frequency, and spectral grids. Stopband attenuation and
the largest adjacent available-line contrast are recomputed from disk without fitting
new targets.

Quality-control spectra use MNE `Raw.compute_psd` with Welch estimation
[[18](#user-content-ref-18)]. Source and derivative files use identical EEG channels,
samples, segment duration, 50 percent overlap, frequency range, and frequency grid. MNE
defaults provide a Hamming segment window, mean removal within each segment, mean
aggregation, and omission of spans marked by bad annotations. The channel median
produces one spectrum per recording.

## Outputs and provenance

The output follows EEG-BIDS and BIDS derivative conventions
[[12](#user-content-ref-12), [13](#user-content-ref-13),
[19](#user-content-ref-19)]. Unchanged metadata files are copied. Hidden, backup,
temporary, lock, and source BrainVision files are excluded. Corrected triplets receive a
`_desc-decomb` entity.

| Output | Contents |
| --- | --- |
| `harmonic_notch_manifest.tsv` | Targets, evidence, intervals, FIR response, attenuation, and quantization |
| `dataset_description.json` | Derivative declaration, version, settings, and source URL |
| `effective_config_apply.txt` | Apply-time values and their packaged, user, or derived origin |
| `effective_config_verify.txt` | Verification-time values and their packaged, user, or derived origin |
| `harmonic_notch_verification.tsv` | Independently measured attenuation and adjacent available-line contrast |
| PSD outputs | Matched figures and numerical source and derivative spectra |

Floating-point geometry is written with 17 significant digits.

## Real-data comparison

The demonstration uses one 8.2-minute EEG recording. Both filter arms receive the same
detected targets, trajectory-envelope centres, real samples, and Welch grid. The decomb
arm uses measured stopband widths and the transition rule described above. The reference
arm uses the documented MNE defaults of centre frequency divided by 200 for notch width
and 1 Hz total transition bandwidth. Reference bands are merged wherever their
transitions overlap because MNE rejects overlapping FIR stopbands
[[10](#user-content-ref-10), [11](#user-content-ref-11)].

The measured decomb geometry makes 10.5 Hz unavailable. MNE default geometry makes
96.4 Hz unavailable, forms one continuous stopband above 40.2 Hz, and retains no
frequency above 39.0 Hz. Retained frequencies reproduce the uncorrected spectrum within
0.07 dB at the 95th percentile in both arms.

![Decomb and MNE default notch geometry on one real EEG recording](docs/notch_comparison_real.png)

The figure can be regenerated from a configured source dataset.

```bash
python3 docs/make_notch_comparison_real.py \
  --config decomb.yaml --subject sub-0011 --recording-index 1
```

The participant audit comprised 90 recordings from 15 participants and 12.1 hours of
EEG. All recordings selected 83 harmonics, with fitted fundamentals from 1.199659 to
1.200551 Hz. Verification reconstructed 8,120 stopbands and measured a median stopband
power change of -28.64 dB.

## Software and testing

The project version is `0.1.0`. Minimum runtime versions reproduce the constraints in
`pyproject.toml`.

The numerical stack uses NumPy [[15](#user-content-ref-15)], SciPy
[[16](#user-content-ref-16)], MNE-Python [[10](#user-content-ref-10),
[11](#user-content-ref-11)], MNE-BIDS [[14](#user-content-ref-14)], and Matplotlib
[[17](#user-content-ref-17)].

| Software | Minimum version | Use |
| --- | --- | --- |
| Python | 3.11 | Runtime and command-line interface |
| NumPy | 1.24 | Arrays, FFT, least squares, interpolation, and summaries |
| SciPy | 1.11 | Local-maximum detection with `find_peaks` |
| MNE-Python | 1.6 | EEG selection, FIR filtering, filter construction, and Welch spectra |
| MNE-BIDS | 0.14 | BIDS path parsing and BrainVision reading |
| pandas | 2.0 | Manifest and report tables |
| PyYAML | 6.0 | Configuration loading |
| Matplotlib | 3.8 | Noninteractive figures |
| joblib | 1.3 | Parallel execution backend |
| pybv | 0.7.5 | BrainVision test and export support |

Resolved versions for an analysis can be recorded with `python3 -m pip freeze`.

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests
```

The tests cover spectral scaling, peak refinement, comb and isolated-line selection,
window independence, stopband geometry, MNE filtering, BIDS discovery and naming,
unit-aware quantization, sidecar rewriting, configuration validation, immutable
verification settings, FIR response provenance, command routing, and matched PSD
computation.

## References

1. <a name="ref-1"></a>Allen PJ, Josephs O, Turner R. A method for removing imaging artifact from continuous
   EEG recorded during functional MRI. *NeuroImage*. 2000, 12, 230 to 239.
   [DOI](https://doi.org/10.1006/nimg.2000.0599)
2. <a name="ref-2"></a>Niazy RK, Beckmann CF, Iannetti GD, Brady JM, Smith SM. Removal of FMRI environment
   artifacts from EEG data using optimal basis sets. *NeuroImage*. 2005, 28, 720 to 737.
   [DOI](https://doi.org/10.1016/j.neuroimage.2005.06.067)
3. <a name="ref-3"></a>Rothlübbers S, Relvas V, Leal A, Murta T, Lemieux L, Figueiredo P. Characterisation
   and reduction of the EEG artefact caused by the helium cooling pump in the MR
   environment. *Brain Topography*. 2015, 28, 208 to 220.
   [DOI](https://doi.org/10.1007/s10548-014-0408-0)
4. <a name="ref-4"></a>Nierhaus T, Gundlach C, Goltz D, et al. Internal ventilation system of MR scanners
   induces specific EEG artifact during simultaneous EEG-fMRI. *NeuroImage*. 2013, 74,
   70 to 76. [DOI](https://doi.org/10.1016/j.neuroimage.2013.02.016)
5. <a name="ref-5"></a>Mullinger KJ, Castellone P, Bowtell R. Best current practice for obtaining high quality
   EEG data during simultaneous fMRI. *Journal of Visualized Experiments*. 2013, 76,
   e50283. [DOI](https://doi.org/10.3791/50283)
6. <a name="ref-6"></a>Widmann A, Schröger E, Maess B. Digital filter design for electrophysiological data,
   a practical approach. *Journal of Neuroscience Methods*. 2015, 250, 34 to 46.
   [DOI](https://doi.org/10.1016/j.jneumeth.2014.08.002)
7. <a name="ref-7"></a>Bullock M, Jackson GD, Abbott DF. Artifact reduction in simultaneous EEG-fMRI, a
   systematic review of methods and contemporary usage. *Frontiers in Neurology*. 2021,
   12, 622719. [DOI](https://doi.org/10.3389/fneur.2021.622719)
8. <a name="ref-8"></a>Harris FJ. On the use of windows for harmonic analysis with the discrete Fourier
   transform. *Proceedings of the IEEE*. 1978, 66, 51 to 83.
   [DOI](https://doi.org/10.1109/PROC.1978.10837)
9. <a name="ref-9"></a>Schwarz G. Estimating the dimension of a model. *Annals of Statistics*. 1978, 6,
   461 to 464. [DOI](https://doi.org/10.1214/aos/1176344136)
10. <a name="ref-10"></a>Gramfort A, Luessi M, Larson E, et al. MNE software for processing MEG and EEG data.
    *NeuroImage*. 2014, 86, 446 to 460.
    [DOI](https://doi.org/10.1016/j.neuroimage.2013.10.027)
11. <a name="ref-11"></a>Gramfort A, Luessi M, Larson E, et al. MEG and EEG data analysis with MNE-Python.
    *Frontiers in Neuroscience*. 2013, 7, 267.
    [DOI](https://doi.org/10.3389/fnins.2013.00267)
12. <a name="ref-12"></a>Pernet CR, Appelhoff S, Gorgolewski KJ, et al. EEG-BIDS, an extension to the Brain
    Imaging Data Structure for electroencephalography. *Scientific Data*. 2019, 6, 103.
    [DOI](https://doi.org/10.1038/s41597-019-0104-8)
13. <a name="ref-13"></a>Gorgolewski KJ, Auer T, Calhoun VD, et al. The Brain Imaging Data Structure, a format
    for organizing and describing outputs of neuroimaging experiments. *Scientific
    Data*. 2016, 3, 160044. [DOI](https://doi.org/10.1038/sdata.2016.44)
14. <a name="ref-14"></a>Appelhoff S, Sanderson M, Brooks TL, et al. MNE-BIDS, organizing
    electrophysiological data into the BIDS format and facilitating their analysis.
    *Journal of Open Source Software*. 2019, 4, 1896.
    [DOI](https://doi.org/10.21105/joss.01896)
15. <a name="ref-15"></a>Harris CR, Millman KJ, van der Walt SJ, et al. Array programming with NumPy. *Nature*.
    2020, 585, 357 to 362. [DOI](https://doi.org/10.1038/s41586-020-2649-2)
16. <a name="ref-16"></a>Virtanen P, Gommers R, Oliphant TE, et al. SciPy 1.0, fundamental algorithms for
    scientific computing in Python. *Nature Methods*. 2020, 17, 261 to 272.
    [DOI](https://doi.org/10.1038/s41592-019-0686-2)
17. <a name="ref-17"></a>Hunter JD. Matplotlib, a 2D graphics environment. *Computing in Science and
    Engineering*. 2007, 9, 90 to 95.
    [DOI](https://doi.org/10.1109/MCSE.2007.55)
18. <a name="ref-18"></a>Welch P. The use of fast Fourier transform for the estimation of power spectra, a
    method based on time averaging over short, modified periodograms. *IEEE Transactions
    on Audio and Electroacoustics*. 1967, 15, 70 to 73.
    [DOI](https://doi.org/10.1109/TAU.1967.1161901)
19. <a name="ref-19"></a>BIDS Contributors. BIDS Derivatives. *Brain Imaging Data Structure specification*.
    Version 1.11.1.
    [Specification](https://bids-specification.readthedocs.io/en/stable/derivatives/introduction.html)

MNE implementation details are documented in the
[notch-filter API](https://mne.tools/stable/generated/mne.filter.notch_filter.html), the
[filtering methods tutorial](https://mne.tools/stable/auto_tutorials/preprocessing/25_background_filtering.html),
and the
[Welch PSD API](https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.compute_psd).

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).
