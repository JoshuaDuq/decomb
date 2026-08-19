# What the artifacts in this cohort actually are

Measured 2026-08-19 on all 90 recordings under `paths.bids_root`, uncorrected source
data. Spectra are 10 s Welch (0.1 Hz bins, matched to the ~0.1 Hz line width), averaged
over EEG channels. Prominence is a peak's height above the median of a 1–4 Hz local
background; comb strength is the median tooth height above the midpoints between teeth.

## Three artifacts, not one

| artifact | prevalence | strength |
| --- | --- | --- |
| 1.200 Hz comb | 88 / 90 recordings | median +2.86 dB (+0.79 to +5.22) |
| isolated line near 57.2 Hz | 14 / 15 participants | median +17.8 dB (+9.8 to +31.3) |
| 60 Hz mains | 6 / 90, all sub-0001 | +28 to +31 dB |
| 1/TR = 1.1111 Hz comb | **0 / 90** | median −0.65 dB, never above +0.64 |

## The comb is not scanner-locked

Scanning the comb fundamental freely rather than assuming it gives **1.200 Hz** in all
90 recordings (range 1.199–1.201), with median tooth prominence +3.15 dB. On the
trigger-anchored 1/TR grid the same measurement gives **−0.60 dB**: nothing.

1.200 Hz is a period of 0.8333 s, or **72.0 cycles per minute**, which is a cold-head
rate rather than anything the pulse sequence produces. An earlier version of this
pipeline recorded `nominal_fundamental_hz=1.199953` in its apply log, so the fundamental
was once estimated from the data and agrees with this measurement to five digits.

The consequence is that `removal.scanner_repetition_time_s` anchors the comb model to a
grid the artifact does not occupy. Of the 86 frequencies that grid authorizes for
notching, only 14 are ever statistically supported anywhere in the cohort, and the most
frequently supported one is harmonic 54 — exactly 60.000 Hz, because TR = 0.9 s places
mains on the grid. Two supported teeth authorize the whole comb, so mains plus one
coincidence is enough to notch 86 frequencies across 1–100 Hz.

## The lines are frequency-stable and amplitude-modulated

Tracking the dominant line across eight consecutive blocks of each recording:

- frequency drift: SD is 0.00 Hz in 62 of 90 recordings; the maximum SD is 0.048 Hz and
  the maximum range 0.10 Hz, at 0.1 Hz measurement resolution
- amplitude: level SD median 1.13 dB, maximum 6.11 dB

This matters for removal. A sinusoid fit refitted per window tolerates slow amplitude
modulation but not modulation inside a window, and it is insensitive to drift this small.
It predicts — correctly, see `docs/removal_operating_point.md` — that lengthening the
estimation window past about 20 s makes removal worse rather than better.

## Participants differ, and some differences are not correctable here

After the best correction measured to date, five participants (sub-0007, sub-0009,
sub-0012, sub-0014, sub-0015) retain one or two peaks below +4 dB. Four (sub-0001,
sub-0006, sub-0011, sub-0013) retain 13 to 23.

sub-0001 is a different problem from the rest: its dominant line is mains at ~30 dB, an
order of magnitude above the pump line elsewhere, and it carries strong narrowband
structure near 81–83 Hz. Cohort statistics that include it are partly measuring mains
removal. It should be analysed separately.

No recording shows a TR-locked harmonic series, so whatever residual gradient artifact
survives the upstream correction does not appear as a comb here and is out of scope for
line removal.
