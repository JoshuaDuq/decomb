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

Python 3.11+. Depends on MNE, MNE-BIDS, pybv, NumPy, SciPy, pandas, matplotlib, joblib and
PyYAML. `pip install -e ".[dev]"` adds pytest and ruff.

`decomb --version` confirms the install, and `decomb --help` lists the stages.

## Quickstart

Point `decomb` at a BIDS root and ask what is in it. Nothing is written to your data until
`apply`, and `apply` will not run until `benchmark` has passed.

```bash
decomb diagnose --bids-root data/bids --output-dir outputs/diagnosis
```

```
Measuring 3 recording(s) under data/bids
58 line(s) over 3 subject(s): 53 comb, 0 isolated
  fundamental 1.200000 Hz over harmonics 24-79, residual RMS 0.4 mHz
  set removal.nominal_fundamental_hz to this value and removal.harmonic_range to the span above.

share of each band that is line artifact (median over subjects):
  delta          0.00%  (worst subject 0.00%, 0 line(s) inside)
  theta          0.00%  (worst subject 0.00%, 0 line(s) inside)
  alpha          0.00%  (worst subject 0.00%, 0 line(s) inside)
  beta           1.78%  (worst subject 1.88%, 2 line(s) inside)
  gamma         21.00%  (worst subject 21.62%, 38 line(s) inside)
```

Two numbers decide whether to go on. The **fundamental and its harmonic span** go into your
config, because every later stage measures against the grid they define. The **share of each
band** is what says whether removal is worth doing at all — here a fifth of gamma is line
artifact and delta through alpha are untouched by it, so only the high bands have anything
to gain.

Copy the packaged [`defaults.yaml`](src/decomb/defaults.yaml) to `decomb.yaml`, set what
`diagnose` just reported, then run the rest against that one file:

```bash
decomb benchmark --config decomb.yaml
```

```
passed 3/3 runs
  gate_transient_preserved         3/3
  gate_transient_undistorted       3/3
  seam (cohort criterion)          PASS: 0 exceeded (count p=1.0000, maximum p=0.8780), worst ratio 0.17
  residual (cohort criterion)      PASS: 0 of 3 recordings (smallest p=0.122)
  focal residual (cohort)          PASS: 0 of 3 recordings (smallest p=0.512)
  preservation (measurement)       probes 1.5e-05 dB against a control's 0.00057; off-target band 0.019 dB against 0.014
  band cost (measurement)          median 0.138, worst 0.141 of 28-95 Hz lost by a broadband probe
  in-band probe survival           median 0.002, worst 0.000 (measurement, not a criterion)
```

`benchmark` injects known signals into your own recordings, removes the lines, and measures
what came back. Only then will the write run:

```bash
decomb apply --config decomb.yaml
decomb verify --config decomb.yaml
decomb report --config decomb.yaml
```

```
median suppression 10.1 dB; worst residual line 12.31 dB
  declared data/bids_decombed/dataset_description.json a derivative of data/bids
```

`apply` refuses unless a `benchmark` recorded under the same settings passed on the same
recordings — the fingerprint is checked, so loosening a criterion and re-running invalidates
the certificate rather than inheriting it. `verify` then re-sweeps what was written under
FDR control, knowing nothing about where the targets were.

The transcripts above are real, from three 300 s synthetic recordings carrying a known
1.2 Hz comb, with the paths shortened. [`docs/make_figure.py`](docs/make_figure.py) builds
that dataset and runs these same stages, so the whole sequence is reproducible without any
data of your own:

```bash
python docs/make_figure.py --keep /tmp/decomb-demo
```

## The stages

```bash
decomb diagnose     # what lines are there, do they share a fundamental, and do they matter?
decomb benchmark    # does the removal preserve signal? run this before apply
decomb apply        # write the cleaned BIDS copy
decomb verify       # re-measure what was written
decomb report       # band-by-band outcome tables
decomb notch        # optional: wide notch over cluster bands
decomb psd          # before-and-after spectra
```

Every stage reads the same config file and takes the same options; `decomb --help` lists
them all. The ones you are likely to want:

| Option | Effect |
|---|---|
| `--config PATH` | the config to use (default `./decomb.yaml`, else the packaged defaults). `DECOMB_CONFIG` does the same |
| `--bids-root PATH` | override the source root, without editing the config |
| `--output-root PATH` | `apply`: where the cleaned copy goes |
| `--output-dir PATH`, `--report-dir PATH` | where the catalogue and the tables go |
| `--subjects sub-01 sub-02` | `diagnose`/`psd` only: restrict to a subset |

`--subjects` is refused by `benchmark`, `apply`, `verify` and `notch` on purpose. Their
criteria are decided over the recordings jointly, so a subset could neither certify a
dataset nor leave the output root in a state the provenance describes.

`diagnose` also counts detections per band, which is how you tell a band `apply` can clear
from one only `notch` can.

## What each stage writes

Tables are TSV, so every number a stage decided on can be read without the tool that wrote
it. Locations come from `paths` in the config; `diagnosis_dir` and `removal_dir` default to
`outputs/diagnosis` and `outputs/removal`.

| Stage | Writes | Holds |
|---|---|---|
| `diagnose` | `lines.tsv` | one row per detection: refined frequency, prominence with its bootstrap interval, half-power width, q-value, how many subjects carried it, comb harmonic, and where it sits on the `k/TR` grid |
| | `comb.tsv` | the fitted fundamental and spacing, the harmonics supporting it, and the scatter about the fitted grid |
| | `lines_per_band.tsv`, `band_impact.tsv` | detections per band, and the share of each band that is line artifact |
| | `spectra.npz` | the spectra the sweep ran on |
| `benchmark` | `benchmark.tsv` | one row per recording: every criterion, the control it was measured against, its p-value, and the settings fingerprint |
| `apply` | `<output_root>/` | the cleaned BIDS copy — `.eeg` binaries rewritten, every sidecar byte-identical |
| | `<output_root>/dataset_description.json` | `GeneratedBy` provenance: version, settings fingerprint, the full parameter set, and the measured band cost |
| | `removal_manifest.tsv` | one row per recording: the fundamental used, target counts, suppression and residual statistics, the read-back check, and the digests tying it to its benchmark |
| `verify` | `verification.tsv` | the blind re-sweep of the written data beside the same sweep of the original, and the verdict |
| | `verification_spectra.npz` | the spectra that sweep ran on |
| `report` | `band_outcomes.tsv` | artifact share per band, before and after |
| | `per_subject_line_residual.tsv` | what survived at each target, per subject |
| | `removal_before_after.png` | the summary figure |
| `psd` | `psd_before_after.png`, `_panels.png`, `_per_recording.png` | overall, tiled, and per-recording spectra |
| `notch` | `notch_manifest.tsv` | the bands taken wholesale, if you ran it |

`apply` stages the whole derivative in a hidden directory and moves it into place only after
every recording has been written and read back within
`removal.roundtrip_relative_tolerance`, so an interrupted run cannot leave a half-cleaned
dataset behind. `removal_manifest.tsv` is written into both `removal_dir` and the output
root, so the cleaned copy always carries its own record of what was done to it.

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

## How it works

Six steps, each stated as an equation in **[docs/METHOD.md](docs/METHOD.md)** so it can be
checked against the code:

1. **Spectral estimate.** Hann-tapered periodogram per estimation window, combined by median
   over channels and mean over windows.
2. **Prominence.** Every threshold and test in the workflow applies to a bin's excess over a
   running-median local background that excludes the bin's own neighbourhood — so a line
   cannot enter its own background.
3. **Detection.** The null is fitted from the prominence spectrum's own lower tail, which
   the lines cannot inflate, and the family is controlled at `fdr_alpha` by
   Benjamini-Hochberg. Accepted peaks are refined below the grid by parabolic interpolation,
   and their half-power width is what separates an instrument line from a brain rhythm.
4. **The comb fit.** A comb is an arithmetic series through the origin, so the fundamental is
   a weighted least-squares slope over harmonics found by iterating to a fixed point. It
   authorises a removal grid only if enough mutually consistent harmonics support it and the
   scatter about it is small. Its uncertainty is a delete-one jackknife.
5. **What is removed.** Each target's width carries the fundamental's uncertainty propagated
   to its harmonic number. Inside that width the operation is a projection, not an
   attenuation: a deterministic sinusoid is fitted per bin and subtracted only where
   Thomson's multitaper *F* test is significant. The fundamental is re-fitted in overlapping
   windows recombined with squared-sine weights that sum to one.
6. **What the benchmark measures.** Every criterion is an exact test against a matched
   control that repeats the same search where no target is — a sign test for off-target
   disturbance, permutation for the seams, and counting for residual lines — so no decibel
   margin has to be invented.

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
pip install -e ".[dev]"
pytest
```

535 tests, about a minute. They build synthetic recordings from seeded noise and known
lines, so what they check is the measurement rather than a stored fixture.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
