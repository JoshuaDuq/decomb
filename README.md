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

Synthetic data: three 300 s recordings, pink background, a 1.2 Hz comb at harmonics 24-79
sitting a few dB above it. All 56 harmonics go; the background between them is unchanged,
and the lower panel shows the change is confined to the lines. `verify` finds 56 comb lines
before and none after.

Two peaks deliberately survive, and both are the tool declining to act rather than failing
to. 60 Hz is comb harmonic 50, inside the `mains_notch_hz` band that `exclude_mains` hands
to a wide notch elsewhere. The 57.25 Hz line is an isolated line planted at ~7 dB, below the
10 dB `detection_min_prominence_db` floor — nothing licensed removing it, so nothing did.

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
