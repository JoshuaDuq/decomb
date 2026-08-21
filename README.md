# decomb

<p align="center">
  <img src="logo.png" alt="decomb logo" width="420">
</p>

`decomb` detects and removes supported narrow spectral lines in continuous EEG. It
analyzes each recording independently, subtracts authorized sinusoidal components from
non-bad EEG channels, and applies one residual FIR plan to every EEG channel. `apply`
writes a BrainVision BIDS derivative with provenance and verification results. Frequencies
removed by either stage are declared unavailable for downstream inference.

## Use case and limits

Periodic artifacts can remain after gradient and pulse artifact correction in simultaneous
EEG-fMRI. Scanner pumps, ventilation systems, and other periodic hardware sources can
produce drifting lines, isolated lines, or harmonic structure.

The recording alone cannot establish whether activity at a removed frequency was neural
or artifactual. Subtraction and filtering can both remove neural activity at the same
frequency. Neither reconstructs the rejected signal. Broad rhythms and transient or
spatial artifacts require other methods. Source control remains preferable when the
source is known.

## Input requirements

Input recordings must be BrainVision EEG in an EEG-BIDS dataset. Session and run entities
are supported. Channel metadata mismatches and invalid recording geometry fail before
processing.

Each recording needs at least two non-bad EEG channels with finite data and enough
continuous samples for the configured estimation windows. Windows do not cross `edge` or
`bad_acq_skip` annotations. Scanner-trigger annotations must match the configured name
and repetition time. Continuous spans shorter than the designed FIR are rejected.

Each recording is its own inferential family. A single recording is sufficient for
`diagnose`; `apply` and `verify` operate on the complete discovered dataset, while
`diagnose` and `psd` can restrict their analysis with `--subjects`.

## Installation

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

## Workflow

Set `paths.bids_root` in `decomb.yaml`. Other settings inherit the packaged defaults.

```bash
decomb diagnose --config decomb.yaml
decomb apply --config decomb.yaml
decomb verify --config decomb.yaml
decomb psd --config decomb.yaml
```

| Command | Purpose |
| --- | --- |
| `diagnose` | Detect supported lines and write a diagnostic model |
| `apply` | Subtract authorized lines, notch proud residuals, and write the derivative |
| `verify` | Reproduce both removal stages and verify the written samples |
| `psd` | Compare source and derivative spectra |

`apply` refuses to use an existing output directory. Run `diagnose` before `apply`, then
run `verify` and `psd` after the derivative has been written.

## Configuration

The packaged configuration is [`src/decomb/defaults.yaml`](src/decomb/defaults.yaml).
The settings normally relevant to a user are:

| Setting | Default | Purpose |
| --- | --- | --- |
| `paths.bids_root` | `data/bids` | Source BrainVision BIDS dataset |
| `removal.scanner_repetition_time_s` | `0.9` | Scanner repetition time in seconds |
| `removal.scanner_trigger_event_name` | `Volume/V  1` | Exact annotation used to validate scanner timing |
| `removal.comb_fundamental_hz` | `null` | Known source frequency; null derives it from the scanner timing |
| `removal.estimation_window_s` | `10.0` | Stationarity interval for ordinary-line detection |
| `removal.familywise_error_rate` | `0.05` | Initial recording-level error budget |
| `removal.frequency_range_hz` | `[0.0, 100.0]` | Detection and filtering range before Nyquist clipping |
| `execution.n_jobs` | `-1` | Worker count; changes speed, not results |

The scanner repetition time and exact trigger name validate recording timing. Set
`removal.comb_fundamental_hz` when independent hardware information gives a periodic source
frequency different from the scanner volume rate. The frequency is never estimated from
the EEG. Unknown settings raise an error. The configured upper frequency is clipped to
remain below each recording's Nyquist frequency.

## Method

Continuous acquisition spans are analyzed in overlapping windows. MNE's multitaper
sinusoid F test identifies phase-coherent lines. A complementary persistence test detects
narrowband peaks whose phase or frequency changes prevent a sinusoid fit. Initial line and
scanner-harmonic evidence is controlled within each recording before removal targets are
authorized.

Authorized ordinary lines and comb teeth that exceed the predeclared candidate floor are
fit with MNE multitaper sinusoid estimates and subtracted from non-bad EEG channels. The
default subtraction fit uses a 20-second window while detection uses 10 seconds. The
subtracted frequencies remain populated by whatever the sinusoid fit does not explain.

The residual spectrum is then tested once. Residual candidates above the 2 dB local
prominence threshold are clustered when they are within three Fourier bins. Each retained
cluster receives one stopband that includes its measured span and its edge margin. There is
no terminal cascade after this threshold FIR stage.

The same residual FIR plan is applied to every EEG channel, including channels marked bad.
It uses MNE's `Raw.notch_filter` with a zero-phase Hamming `firwin` design, automatic filter
length, reflect-limited padding, and annotation-aware processing. Unsupported frequencies
remain in the passband. A recording with no subtraction target or residual stopband is
copied unchanged.

Verification starts from the source recording, re-runs the authorization and subtraction
decisions, re-derives the residual stopbands, and checks the exact written samples. `psd`
uses matched Welch spectra for source and derivative files with equal recording weight and
a shared decibel scale.

## Outputs

`diagnose` writes `model.tsv`, `detected_lines.tsv`, `stopbands.tsv`, and the effective
configuration used for the run.

`apply` writes corrected BrainVision triplets with the `_desc-decomb` entity, a BIDS
`dataset_description.json`, `line_notch_manifest.tsv`, `comb_analysis_mask.tsv`,
`analysis_availability.tsv`, and the effective configuration. The manifest distinguishes
subtracted frequencies from residual-notched stopbands and records their evidence,
geometry, attenuation, and cumulative unavailable bandwidth. The two analysis tables are
advisory downstream masks and do not describe the derivative.

`verify` writes `line_notch_verification.tsv` with replay results. `psd` writes
`psd_before.png`, `psd_after.png`, `psd_before_declared.png`, and
`psd_after_declared.png`.

## Example result

These figures show the same EEG cohort before and after the threshold-stop correction.
Both use the same recordings, channels, samples, and decibel scale.

![Sensor-level source spectra](docs/psd_before.png)

![Sensor-level threshold-stop derivative spectra](docs/psd_after.png)

The example audit contains 90 recordings and 12.09 hours of continuous acquisition.
Removed frequency intervals are unavailable for inference, so downstream analyses should
account for the cumulative manifest geometry. Cohort-specific validation does not establish
performance on an independent dataset.

## References and further reading

1. Allen PJ, Josephs O, Turner R. A method for removing imaging artifact from continuous
   EEG recorded during functional MRI. *NeuroImage*. 2000. [DOI](https://doi.org/10.1006/nimg.2000.0599)
2. Niazy RK, et al. Removal of fMRI environment artifacts from EEG data using optimal
   basis sets. *NeuroImage*. 2005. [DOI](https://doi.org/10.1016/j.neuroimage.2005.06.067)
3. Mullinger KJ, Castellone P, Bowtell R. Best current practice for obtaining high quality
   EEG data during simultaneous fMRI. *Journal of Visualized Experiments*. 2013.
   [DOI](https://doi.org/10.3791/50283)
4. Widmann A, Schröger E, Maess B. Digital filter design for electrophysiological data.
   *Journal of Neuroscience Methods*. 2015. [DOI](https://doi.org/10.1016/j.jneumeth.2014.08.002)
5. Thomson DJ. Spectrum estimation and harmonic analysis. *Proceedings of the IEEE*.
   1982. [DOI](https://doi.org/10.1109/PROC.1982.12433)
6. Gramfort A, et al. MNE software for processing MEG and EEG data. *NeuroImage*.
   2014. [DOI](https://doi.org/10.1016/j.neuroimage.2013.10.027)
7. Pernet CR, et al. EEG-BIDS, an extension to the Brain Imaging Data Structure for
   electroencephalography. *Scientific Data*. 2019. [DOI](https://doi.org/10.1038/s41597-019-0104-8)
8. Welch P. The use of fast Fourier transform for the estimation of power spectra.
   *IEEE Transactions on Audio and Electroacoustics*. 1967.
   [DOI](https://doi.org/10.1109/TAU.1967.1161901)

Implementation details are available in the MNE
[notch-filter API](https://mne.tools/stable/generated/mne.filter.notch_filter.html) and
[filtering tutorial](https://mne.tools/stable/auto_tutorials/preprocessing/25_background_filtering.html).

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).
