# MNE FIR Geometry Ablation Design

## Purpose

Replace the broad “MNE notch comparison” framing with a precisely scoped MNE FIR
geometry ablation. The experiment will answer one conditional question: given the line
centres detected by decomb, how do decomb's measured stopband geometry and the MNE API's
default width and transition parameters trade unavailable frequency bandwidth against
FIR duration?

This is not an end-to-end decomb-versus-MNE method benchmark. Both filtered arms use
MNE's FIR implementation.

## Experimental arms

Both arms receive the same in-memory recording, EEG channels, detected line centres,
FIR phase, window, design, padding, and Welch configuration.

1. **decomb measured geometry** uses the measured trajectory envelopes and the existing
   transition-width rule.
2. **MNE default parameters, overlap-merged by decomb** starts from stopband widths of
   centre frequency divided by 200 and a 1 Hz total transition bandwidth. Decomb merges
   bands whose transitions cannot coexist in a valid multiband FIR design.

The second arm is a counterfactual geometry, not a literal invocation of unmodified MNE
defaults. The literal dense-target call will be checked separately and its design error
will surface explicitly.

## Naming and interfaces

The README heading, figure labels, plot metadata, module documentation, tests, and public
comparison data will consistently use “MNE FIR geometry ablation.” Names containing
`traditional` will be replaced with names that identify the overlap-merged MNE-default
parameter geometry. No compatibility aliases will be retained.

The comparison result will carry the source spectrum, both filtered spectra, both notch
plans, recording duration, and exact MNE filter characterizations for both arms. Filter
characterization will reuse `notch.characterize_harmonic_filter` rather than duplicate
MNE filter-design logic.

## Data flow

1. Fit the decomb harmonic model once on the input recording.
2. Build the decomb measured plan.
3. Build the overlap-merged MNE-default-parameter plan from the same detected centres.
4. Ask MNE to design the exact FIR for each plan and record length, attenuation, and
   passband-deviation measurements.
5. Apply each valid plan through the existing common MNE FIR application function.
6. Compute all three spectra with one Welch configuration and require identical
   frequency grids.
7. Calculate unavailable bandwidth from the declared stopbands and full transitions for
   each arm.

## Literal-default feasibility

A focused function will ask MNE to design the unmerged default-parameter notch geometry.
It will not catch and replace MNE's design error. A targeted test will verify that the
literal dense-target geometry raises and will relate that result to the overlapping
default transition intervals. The error text will not be used as a stable programmatic
assertion. The real-data generator will apply only the explicitly named overlap-merged
counterfactual arm and will identify it as such in its output.

## Figure and README

The figure will retain the three spectrum rows and the scaled six-hertz geometry detail.
Each filtered row will report:

- frequency bandwidth left available under the project's conservative policy; and
- exact FIR duration in seconds.

The title will state that this is an MNE FIR geometry ablation with shared detected
centres. The reference labels will always disclose the overlap merge. Figure metadata
will preserve the real-recording provenance.

The README will:

- rename the section to “MNE FIR geometry ablation”;
- state the conditional question and shared MNE implementation;
- distinguish the literal default failure from the merged counterfactual;
- report unavailable bandwidth and exact FIR duration for both arms;
- explain the frequency-selectivity versus temporal-extent trade-off;
- define unavailable bandwidth as the project's conservative stopband-plus-transition
  policy rather than measured neural information loss; and
- retain the existing explanation of upstream nulls in the input recording.

## Error handling

MNE errors will propagate. The implementation will not use exception-message matching,
fallback filter parameters, or compatibility branches. Invalid plans, frequency-grid
mismatches, non-finite spectra, and boundary violations will continue to fail at their
existing entry points.

## Testing

Tests will verify:

- sparse, non-overlapping reference targets remain numerically equivalent to a direct
  MNE call with omitted notch widths and transition bandwidth;
- dense literal MNE defaults raise instead of being silently modified;
- overlap merging is confined to the explicitly named counterfactual plan;
- both result arms contain exact filter characterizations;
- FIR duration differs in the expected direction under the packaged settings;
- unavailable bandwidth differs in the expected direction;
- both spectra use the same real samples and frequency grid; and
- the generated figure contains both geometry labels and both FIR-duration annotations.

The targeted comparison tests, complete test suite, and Ruff checks must pass before the
figure is regenerated. The regenerated PNG will be inspected at full resolution.

## Scope

This change does not alter decomb detection, stopband planning, filtering, derivative
writing, or verification behavior. It does not add alternative artifact-removal methods,
multi-recording benchmarking, impulse-response panels, or compatibility aliases.

## Acceptance criteria

The result is complete when the code, tests, README, generator output, and PNG all call
the experiment an MNE FIR geometry ablation; the merged reference is never labeled as a
literal MNE default method; both frequency and FIR-duration costs are visible; the
literal dense-target error is disclosed; existing provenance edits are preserved; and
all verification commands pass.
