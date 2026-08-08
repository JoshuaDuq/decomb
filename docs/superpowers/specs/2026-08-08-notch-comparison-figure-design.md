# A figure showing what each removal stage costs, and on which artifact

## Why

README §2 asserts in prose that "a wide FIR notch removes its full stop-band, including
frequencies where no artifact was measured," and §6.3 states that `apply` and `notch` are
counterparts rather than competitors. Neither is illustrated. This figure measures both
claims on one synthetic dataset.

It does not claim superiority. §1 says `decomb` makes none, and the measurements below
support that: each stage wins on the artifact structure it was built for.

## What it shows

Two columns, two rows, one synthetic dataset, one fitted plan.

|            | column A: a sparse comb        | column B: a dense cluster        |
|------------|--------------------------------|----------------------------------|
| row 1      | spectra before / after `decomb` / after MNE notch, zoomed to ~6 harmonics | the same three traces over the cluster span |
| row 2      | broadband-probe attenuation vs frequency, per method, same x-range | same |

Row 1 shows what each transform did to the artifact. Row 2 shows what it cost a signal that
carries no artifact at all, which is the quantity the prose claims and the one a reader
cannot infer from row 1.

Expected result, already measured on the comb alone:

| | share of 28-95 Hz attenuated >1 dB |
|---|---|
| MNE `notch_filter`, 55 harmonics, library defaults | 0.739 (0.646 at >3 dB) |
| `decomb` targeted removal | 0.140 |

On the cluster the direction reverses: targeted removal takes the narrow peaks it can
resolve and leaves the span standing, and the notch clears it for the cost of its width.

## The dataset

One generator, extending `docs/make_figure.py`'s:

* the existing 1.2 Hz comb over harmonics 24-79, 12 dB over background;
* the existing 42 Hz rhythm, so the figure inherits the case that must not be removed;
* **new**: a dense cluster at 20 Hz -- 12 peaks spread over 1.0 Hz, each wandering
  +-0.08 Hz across the recording. Placed below the comb's harmonic range, clear of the
  rhythm and of the mains band.

Non-stationarity is what makes the cluster a cluster, and it is measured rather than
assumed. Detection on the wandering version finds 18 peaks over 3 dB where 12 were planted,
of which 8 are narrow enough to be classified as lines. The stationary version yields 12
peaks, all 12 targetable -- which `decomb` would simply remove, leaving nothing to show.

## The two arms

**`decomb`**: `build_run_plans` fits the plan, `clean_continuous_raw` applies it. That is the
transform `apply` performs. `benchmark` and `apply` are deliberately not run here: the
cluster is designed to leave a residual, so the cohort residual criterion may refuse the
dataset -- correctly. This figure measures transforms; the acceptance criteria are exercised
by `docs/psd_before_after.png`.

**Notch**: `mne.filter.notch_filter` at library defaults -- `notch_widths=None`
(freq/200), `trans_bandwidth=1`. Column A notches the 55 comb harmonics; column B notches
the cluster span. `filter_length` must be given explicitly: at `'auto'` MNE raises
`"filter length 1651 is too short for the requested 0.11 Hz transition band"` on a 1.2 Hz
comb. The figure notes this rather than hiding it.

## The cost measurement

A separate broadband-noise recording carrying no artifact, put through each transform
unchanged -- the same construction `benchmark` uses for band cost. Attenuation per frequency
is `10*log10(before/after)` of its Welch spectrum. This is why row 2 is a measurement and
not a drawing.

## Output

* `docs/make_notch_figure.py`, run the same way as `make_figure.py`, printing every number
  the caption may state.
* `docs/notch_comparison.png`.
* A README paragraph beside §2 or §6.3 pointing at it, stating both directions.

## Success criteria

1. Column A: the notch's attenuation covers most of the analysed band; `decomb`'s does not.
   Both numbers printed as measured.
2. Column B: the cluster survives targeted removal and does not survive the notch.
3. The rhythm survives both, or the figure says which one ate it.
4. Every number in the caption is printed by the script when it runs.
