# decomb

<p align="center">
  <img src="logo.png" alt="decomb logo" width="420">
</p>

`decomb` detects and suppresses statistically supported sinusoidal components in
continuous EEG. Each recording is fitted independently. Each EEG channel is tested and
filtered from its own evidence, then written as a BrainVision BIDS derivative with
measured stopbands, transition bands, filter response, and verification results. A
single recording is a complete valid input; no cohort catalogue is required.

## Problem statement

Residual periodic artifacts can remain in EEG acquired during or near fMRI after
gradient and pulse artifact correction [[1](#user-content-ref-1),
[2](#user-content-ref-2)]. Cryogenic pumps and scanner ventilation systems are reported
sources [[3](#user-content-ref-3), [4](#user-content-ref-4)]. Their frequencies can drift
within a recording and can occur as a harmonic series or as isolated narrowband lines.

The recording alone does not uniquely separate neural and artifactual contributions at
the same frequency. Source-separation methods require additional assumptions and cannot
establish unique recovery when those assumptions are unsupported
[[20](#user-content-ref-20)]. Filtering attenuates neural activity together with the
artifact and does not reconstruct the rejected signal [[6](#user-content-ref-6),
[21](#user-content-ref-21), [22](#user-content-ref-22)].

`decomb` therefore records every stopband and transition as unavailable for inference
and does not impute neural activity. Physical attribution requires independent evidence.
Source control remains preferable when the source is known
[[5](#user-content-ref-5)].

## Scope

Input recordings must use BrainVision format in an EEG-BIDS dataset
[[12](#user-content-ref-12), [14](#user-content-ref-14)]. Recording directories may contain
optional session and run entities. Channel metadata mismatches raise an error.

All channels typed as EEG are included in spectral estimation. Recordings must contain
finite values and at least one complete estimation window inside a continuous
acquisition span. Windows never cross annotations whose descriptions begin with
`edge` or `bad_acq_skip`, matching MNE's FIR boundary policy. Scanner triggers and
scanner clock annotations are otherwise not used.

The method identifies narrow spectral structure. Broad rhythms and transient artifacts
require temporal or spatial methods [[7](#user-content-ref-7)]. A detected comb does not
identify its physical source.

## Installation

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
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
| `diagnose` | Tests sinusoidal components and writes the proposed filter plan |
| `apply` | Fits each recording, filters EEG channels, and writes the complete derivative |
| `verify` | Reconstructs the declared FIR and requires exact quantized derivative samples |
| `psd` | Writes matched source and derivative spectra and figures |

`apply` and `verify` require the complete discovered dataset. `diagnose` and `psd`
accept recording subsets. An existing output directory causes `apply` to stop.

## Configuration

The packaged configuration is
[`src/decomb/defaults.yaml`](src/decomb/defaults.yaml). Three settings define the
scientific correction.

| Setting | Default | Function |
| --- | --- | --- |
| `removal.estimation_window_s` | 54.0 s | Stationarity interval and spectral resolution |
| `removal.familywise_error_rate` | 0.05 | Per-channel probability of any false line detection |
| `removal.frequency_range_hz` | 0.0 to 100.0 Hz | Frequencies eligible for detection and filtering |

The default window gives Fourier bins separated by 0.018519 Hz and a Hann half-power
resolution of 0.026633 Hz [[8](#user-content-ref-8)]. Changing the duration changes the
frequency spacing, resolution, stopband width, transition width, and FIR duration.

`paths.bids_root` identifies the input dataset. Output and report locations have
packaged defaults. Optional `frequency_bands` entries report unavailable and retained
bandwidth and do not affect detection.

Unknown and obsolete settings raise an error. The configured upper frequency has no
fixed software ceiling; each recording clips it to frequencies strictly below Nyquist.

## Methods

### Spectral estimation

The configured duration is converted to the nearest whole number of samples. Windows
overlap by 50 percent, and the final window in each continuous acquisition span is
aligned with that span's end. No estimation window crosses or includes samples marked
by MNE's `edge` or `bad_acq_skip` annotation prefixes.

Each channel and window is evaluated with Thomson's multitaper sinusoid F test, following
MNE's `spectrum_fit` implementation: eight DPSS tapers, time-bandwidth product four,
alternating tapers for the sinusoidal estimate and residual, and an F distribution with
2 and 14 degrees of freedom [[9](#user-content-ref-9)]. DC is not an oscillatory line and
is excluded.

### Line detection and harmonic classification

Within each EEG channel, Holm correction covers every continuous estimation window and
tested Fourier frequency. Holm controls the channel-family false-positive probability
under arbitrary dependence and is uniformly at least as powerful as Bonferroni. A
frequency is eligible for filtering on that channel only when its adjusted Thomson
p-value is below `removal.familywise_error_rate`. Channels are not pooled into a cohort
or recording-wide multiplicity family. There is no prominence, amplitude, prevalence,
minimum-harmonic, or attenuation cutoff.

If no test survives Holm correction, that null result is recorded explicitly and the
recording is copied without filtering. A clean statistical outcome is valid and does not
abort diagnosis, application, or verification.

Harmonic structure is descriptive and cannot create a target. A partial-conjunction test
requires evidence for at least two distinct harmonic components and remains valid under
arbitrary dependence. A second Bonferroni correction covers every candidate fundamental
implied by the complete tested frequency grid [[23](#user-content-ref-23)]. If this test
does not pass, every detected component remains an isolated line; either way, exactly the
same significant frequencies are filtered.

### Isolated-line selection

Every significant frequency not assigned a supported harmonic label remains an isolated
line. Isolated-line detection does not depend on a comb being present.

### Stopbands and FIR filtering

Each stopband covers its statistically supported Fourier-bin positions, expanded by half
a Fourier bin. Its width is at least the Hann half-power resolution derived from the
configured window [[8](#user-content-ref-8)]. Stopbands are merged only when their FIR
transitions would overlap. Missing harmonics and all other unsupported frequencies stay
in the passband.

The total transition bandwidth is 3.3 divided by the estimation-window duration in
seconds. MNE assigns half of this width to each stopband edge. Merged stopbands are
passed to MNE `Raw.notch_filter` for the EEG channel whose statistical evidence supports
them. Channels without a supported line remain sample-for-sample unchanged
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
| `skip_by_annotation` | `edge`, `bad_acq_skip` |
| `n_jobs` | `-1` |

This is a one-pass, zero-phase, noncausal FIR design with delay compensation. The
manifest records its exact sample count and measured response. A transition reaching
zero frequency or Nyquist raises an error [[6](#user-content-ref-6)].

### Attenuation and verification

Stopband power is summed across frequency bins on the EEG channel to which that stopband
was applied. Source and derivative spectra use the same complete Hann windows used by
the boundary policy. The reported change is descriptive: no attenuation threshold
decides whether verification passes.

Verification confirms that scientific settings, library versions, and recording geometry
match the apply stage. It refits the source recording and requires the Holm authorization,
supporting windows, channel counts, harmonic labels, and stopband geometry to reproduce
the manifest. It then reproduces MNE's FIR response, reapplies the filter, and applies the
destination BrainVision calibration and float32 quantization. Every sample must equal the
written derivative exactly. A null manifest must refit as null and reproduce an unchanged
quantized recording.

Quality-control spectra use MNE Welch estimation [[18](#user-content-ref-18)]. Source
and derivative files use identical EEG channels, continuous acquisition windows, segment
duration, 50 percent overlap, frequency range, and frequency grid. The window mean is
computed within channels, followed by the channel median.

## Outputs and provenance

The output follows EEG-BIDS and BIDS derivative conventions
[[12](#user-content-ref-12), [13](#user-content-ref-13),
[19](#user-content-ref-19)]. Corrected BrainVision triplets receive a `_desc-decomb`
entity. The derivative includes a stopband manifest, `dataset_description.json`, apply
and verification configurations, an independent verification table, and matched PSD
products. The manifest records the affected channel, every detected frequency, raw and
Holm-adjusted p-values, supporting windows, harmonic labels where supported, per-channel
and total test counts, interval geometry, FIR response, attenuation, and unavailable
bandwidth. Floating-point geometry is written with 17 significant digits.

`diagnose` writes `model.tsv`, `detected_lines.tsv`, and `stopbands.tsv`. `apply` writes
`line_notch_manifest.tsv`, and `verify` writes `line_notch_verification.tsv`.

## MNE FIR geometry ablation

This conditional, channel-specific ablation uses channel `Fp1` from one 8.2-minute EEG
recording. The plotting command fixes the recording and channel explicitly; it does not
select a channel from the result. Both arms use the same Holm-authorized line centres,
samples, MNE FIR implementation, and Welch grid. Only stopband width and transition
geometry differ.

The decomb arm uses the measured line intervals and the transition rule above. The
reference arm uses MNE's default notch width of centre frequency divided by 200 and its
1 Hz total transition bandwidth. Decomb merges reference intervals only when their
transitions overlap so that MNE receives valid multiband geometry. This is an FIR
geometry ablation, not a comparison of detection methods.

For this channel, decomb declares 0.457 Hz unavailable and requires a 108.001 s FIR. The
reference geometry declares 6.307 Hz unavailable and requires a 6.601 s FIR. Unavailable
bandwidth includes stopbands and transitions; it describes the declared filter geometry,
not neural information loss. The spectral gaps in the figure mark those declared
intervals rather than an attenuation threshold.

```bash
.venv/bin/python docs/make_notch_comparison_real.py \
  --config decomb.yaml \
  --subject BIDS_LABEL \
  --recording-index 1 \
  --channel Fp1
```

![Channel-specific MNE FIR geometry ablation](docs/notch_comparison_real.png)

## Software and testing

Version `0.1.0` requires Python 3.11, NumPy 1.24
[[15](#user-content-ref-15)], SciPy 1.11 [[16](#user-content-ref-16)], MNE-Python 1.6
[[10](#user-content-ref-10), [11](#user-content-ref-11)], MNE-BIDS 0.14
[[14](#user-content-ref-14)], pandas 2.0, PyYAML 6.0, Matplotlib 3.8
[[17](#user-content-ref-17)] and pybv 0.7.5. Package roles and constraints are
declared in [`pyproject.toml`](pyproject.toml).

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests
```

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
9. <a name="ref-9"></a>Thomson DJ. Spectrum estimation and harmonic analysis. *Proceedings of the IEEE*.
   1982, 70, 1055 to 1096. [DOI](https://doi.org/10.1109/PROC.1982.12433)
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
20. <a name="ref-20"></a>Hyvärinen A, Oja E. Independent component analysis, algorithms and applications.
    *Neural Networks*. 2000, 13, 411 to 430.
    [DOI](https://doi.org/10.1016/S0893-6080(00)00026-5)
21. <a name="ref-21"></a>de Cheveigné A, Nelken I. Filters, when, why, and how not to use them. *Neuron*.
    2019, 102, 280 to 293.
    [DOI](https://doi.org/10.1016/j.neuron.2019.02.039)
22. <a name="ref-22"></a>Leske S, Dalal SS. Reducing power line noise in EEG and MEG data via spectrum
    interpolation. *NeuroImage*. 2019, 189, 763 to 776.
    [DOI](https://doi.org/10.1016/j.neuroimage.2019.01.026)
23. <a name="ref-23"></a>Benjamini Y, Heller R. Screening for partial conjunction hypotheses.
    *Biometrics*. 2008, 64, 1215 to 1222.
    [DOI](https://doi.org/10.1111/j.1541-0420.2007.00984.x)

MNE implementation details are documented in the
[notch-filter API](https://mne.tools/stable/generated/mne.filter.notch_filter.html), the
[filtering methods tutorial](https://mne.tools/stable/auto_tutorials/preprocessing/25_background_filtering.html),
and the
[Welch PSD API](https://mne.tools/stable/generated/mne.time_frequency.psd_array_welch.html).

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).
