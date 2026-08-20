# Four arms on one availability basis

**This supersedes the first version of this file, which was wrong in three ways.** See
"What the earlier version got wrong" at the end.

Measured on all 90 recordings with `arms_declared.py`; raw results `arms_declared.tsv`,
log `arms_declared.log`. Every number here is recomputed with current code. No figure is
carried over from `final_config.tsv` or `arm_combined.tsv`, both of which predate the
scanner-harmonic authorization fix.

## The one basis

Every arm declares **its own** subtraction damage, not just its filter geometry:

- **FIR**: `plan.unavailable_edges()` -- stopband plus transition, exactly what the manifest writes.
- **subtraction**: `+/- 2 bins = +/- 2 / fit_window_s` around every removed frequency.

That second line is the whole point. The `combined` arm recorded in `arm_combined.tsv`
subtracts ~147 frequencies at a 20 s window and then reports availability from its FIR
plans alone, declaring none of that. Comparing it against the shipped arm's fully-declared
number is not a comparison.

## Arms

| arm | what it does |
|---|---|
| `notching` | current shipping behaviour: no subtraction, converged FIR rounds |
| `tuned` | reproduction of `combined`: subtract ordinary lines + strong teeth at 20 s, notch what still stands proud of 2 dB, then converge |
| `derived` | the shipped subtraction pipeline: subtract the authorized set at 10 s, then converge |
| `derived_w20fit` | as `derived`, but the subtraction **fit** uses 20 s while **detection** stays at 10 s |

## Results (n = 84, excluding sub-0008)

| arm | gamma | beta | alpha | unavail Hz | subtracted | FIR blocks |
|---|---:|---:|---:|---:|---:|---:|
| notching | 0.741 | 0.923 | 1.000 | 23.7 | 0 | 33.1 |
| tuned | 0.782 | 0.928 | 0.994 | 18.3 | 146.5 | 7.7 |
| derived | 0.730 | 0.915 | 1.000 | 24.8 | 106.5 | 7.5 |
| **derived_w20fit** | **0.809** | **0.937** | 1.000 | **17.6** | 106.5 | 8.7 |

| arm | signed comb_db median | share positive | \|comb_db\| mean | \|comb_db\| max | correlation | change RMS |
|---|---:|---:|---:|---:|---:|---:|
| notching | -0.07 | 0.37 | 0.91 | 23.90 | 0.9902 | 0.0712 |
| tuned | -0.04 | 0.35 | **0.22** | **1.11** | 0.9928 | 0.0624 |
| derived | -0.06 | 0.46 | 0.63 | 6.58 | 0.9930 | 0.0627 |
| derived_w20fit | **+0.46** | **0.94** | 0.53 | 2.35 | **0.9931** | **0.0583** |

`comb_db` is best at zero. Positive means the teeth are still standing; negative means they
were excavated below the local floor.

## The shipped arm does not beat current notching

Paired, `derived` minus `notching`:

| metric | mean | better in |
|---|---:|---:|
| gamma availability | **-0.0103** | **27/84** |
| \|comb_db\| | -0.2782 | 30/84 |
| correlation | +0.0028 | 77/84 |
| change RMS | -0.0085 | 77/84 |

On honest accounting the shipped subtraction design is **slightly less available than
current notching**, not more. It is also not more reliably flat: its \|comb_db\| mean looks
better only because notching has rare catastrophic failures (\|max\| 23.90 on sub-0009 run-3
against 6.58), and on a per-recording basis notching flattens the comb better in 54 of 84.

What subtraction does buy is fidelity: correlation and change RMS improve in 77 of 84. It
disturbs the retained signal about 12% less and avoids notching's worst-case comb blowups.
That is a real benefit, and it is the honest case for the design -- but it is not the case
the first version of this file made.

## The tuned arm is the best all-rounder

Paired, `tuned` minus `notching`: gamma +0.0413 (better in 71/84), correlation better in
76/84, change RMS better in 76/84, and \|comb_db\| mean 0.22 with a worst case of **1.11 dB
across all 90 recordings**. Nothing else comes close on comb consistency.

Against the shipped arm on the same declared basis, `tuned` still wins gamma by 0.052
(better in 81 of 84) -- not the 0.20 the first version reported, but a real gap. It gets
there by subtracting *more* frequencies (146.5 vs 106.5) through a *narrower* damage zone
(20 s fit, +/- 0.1 Hz, against 10 s and +/- 0.2 Hz). The fit window, not the target count,
is what dominates declared availability.

## The decoupled window: biggest gain, and a catch

`derived_w20fit` fits the subtraction on 20 s windows while leaving detection at 10 s. It
takes the best availability of any arm (gamma 0.809, beta 0.937), the best fidelity
(correlation 0.9931, change RMS 0.0583), and beats `derived` on gamma in **84 of 84** and
`notching` in 82 of 84.

**But it under-removes the comb.** Its signed `comb_db` median is **+0.46** and **94% of
recordings are positive** -- against 46% for `derived`. It beats `derived` on \|comb_db\| in
only 21 of 84. It declares less damage partly because it removes less artifact.

This is a milder version of the failure the 20 s full-window test showed (`comb_db` +1.55,
detection collapsing sevenfold). Keeping detection at 10 s avoids the collapse -- target
counts are identical at 106.5 -- but the wider fit still leaves teeth standing.

So it is a genuine trade, not a free win: **+0.078 gamma for a systematic +0.5 dB of
residual comb.** Whether that is worth taking is a judgement about which failure the study
can better tolerate. It should not be adopted on the availability number alone.

## How much of the comb each arm actually removes

The tables above report residual `comb_db` with no baseline, which makes a residual of
+0.46 dB impossible to judge. Measured on the same recordings before any cleaning
(`raw_comb.py`, `raw_comb.tsv`), the comb stands at a median of **+2.75 dB**.

| arm | residual comb_db | removed | % of comb removed | \|comb_db\| max | share positive |
|---|---:|---:|---:|---:|---:|
| *(uncleaned)* | +2.75 | -- | 0% | -- | -- |
| notching | -0.07 | 2.91 | 102% | 23.90 | 0.37 |
| tuned | -0.04 | 2.79 | 102% | **1.11** | 0.35 |
| derived | -0.06 | 2.93 | 102% | 6.58 | 0.46 |
| derived_w20fit | +0.46 | 2.34 | **82%** | 2.35 | **0.94** |

So `derived_w20fit` removes about 82% of the comb and leaves roughly a sixth of it standing
-- not "the comb survives", but clearly less complete than the other three, which zero it out
(slightly past zero, hence 102%).

Two things make that 0.46 dB worse than its size suggests: it is **systematic** (positive in
94% of recordings, against 35-46% for the others), and a scanner-locked residual is
correlated with the fMRI paradigm, so it is a structured confound rather than added noise.

Against that, its worst case is 2.35 dB where `notching` reaches 23.90 dB and `derived` 6.58
dB. The trade is "a consistent small residual everywhere" against "usually perfect, sometimes
badly wrong."

**`tuned` is the only arm that does both**: complete comb removal *and* a worst case of 1.11
dB across all 90 recordings, with the second-best availability (0.782) and essentially tied
best fidelity. On this evidence it is the strongest configuration measured.

## The strongest configuration is not implemented anywhere

`tuned` exists only as `fixed_config.py`, a benchmark script. No branch contains it as
production code -- `feat/apply-subtraction` is the only branch with `src/decomb/subtraction.py`,
and it implements `derived`.

The reason it was not shipped is a real constraint, not an oversight. `tuned` depends on six
hand-tuned constants: `WINDOW_S=20.0`, `FLOOR_DB=2.0`, `TOOTH_SUBTRACT_DB=1.0`,
`CLUSTER_GAP_HZ=0.30`, `MARGIN_HZ=0.125`, `COMB_HZ=1.200`. The design brief for this feature
required that every constant derive from `estimation_window_s`, `frequency_bin_width_hz`, or
the statistical evidence, and that no new threshold enter the packaged defaults --
`tests/test_everything_is_configurable.py` enforces it.

So the choice is not "which arm performs best" but **"is the measured gain worth six magic
numbers and the thresholds they imply?"** That is a project decision, and this file does not
make it. What the evidence says is that the gain is real: complete comb removal with a 1.11 dB
worst case, against 6.58 dB for what is currently on the branch.

## What is and is not comparable across arms

**Comparable:** availability (same interval arithmetic), `comb_db` (fixed 1.2 Hz tooth grid,
independent of what the arm detected), correlation, change RMS.

**Not comparable:** the residual peak counts, at any threshold. Each arm evaluates prominence
at its own `subtracted ∪ teeth` set, so `notching` is scored at ~62 tooth frequencies while
`derived` is scored at ~106 targets plus teeth. Differences in count are partly differences
in where each arm looked. The columns are in `arms_declared.tsv` for within-arm use only.
The 2 dB threshold is separately unusable: it sits below the noise of its own statistic in
70 of 90 recordings.

## sub-0008, reported separately

sub-0008 has a bad ECG recording -- a known data-quality fact, not a pipeline defect. Mean
over its 6 runs, excluded from everything above:

| arm | gamma | comb_db | correlation | change RMS |
|---|---:|---:|---:|---:|
| notching | 0.669 | -4.65 | 0.9731 | 0.0797 |
| tuned | 0.723 | -0.27 | 0.9704 | 0.0815 |
| derived | 0.640 | -2.47 | 0.9762 | 0.0768 |
| derived_w20fit | 0.727 | +1.06 | 0.9847 | 0.0636 |

It is the one participant no arm cleans well, and the ordering is the same as the cohort.

## What the earlier version got wrong

1. **Mixed accounting.** It compared the tuned arm's FIR-only availability (0.928) against
   the shipped arm's fully-declared 0.728 and reported a 0.20 deficit "on every recording."
   On one basis the gap is 0.052.
2. **A stale baseline.** It used `notching` gamma 0.592 from `final_config.tsv`, generated
   before the scanner-harmonic authorization fix, when every harmonic in range was notched
   rather than only the supported ones. Fresh, current notching is **0.741**. Every claim of
   the form "subtraction beats notching" was measured against a worse-than-current baseline,
   and the availability half of that claim does not survive.
3. **Inverted comb_db direction.** `derived_vs_tuned.py` scored `comb_db` as
   "higher is better," so it reported subtraction flattening the comb better in 70 of 84.
   Scored correctly on \|comb_db\|, it is 30 of 84.

A fourth, smaller error: `derived_shipped.py` measured FIR damage from raw stopbands instead
of `unavailable_edges()`, omitting the transition band and overstating availability by about
0.7 points. `arms_declared.py` uses `unavailable_edges()` throughout. `derived_shipped.tsv`
is superseded by `arms_declared.tsv`.

## Keep detection at 10 s

Widening the whole `estimation_window_s` to 20 s remains rejected: ordinary-line detection
collapses from 108.2 to 15.3 lines, and `comb_db` goes to +1.55 -- the teeth are left
standing and the apparent gamma gain is bandwidth preserved by failing to remove the
artifact (`derived_probe_w20.tsv`). The `derived_w20fit` arm above is a different proposal --
it widens only the fit -- and carries its own milder version of the same cost.
