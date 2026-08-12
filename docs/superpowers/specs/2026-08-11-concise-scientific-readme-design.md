# Concise scientific README design

## Objective

Rewrite the project README using the compact organization of the MNE-Denoise README
while preserving the information required to evaluate and reproduce the scientific
method.

## Scope

Only `README.md` will be changed during implementation. Existing modifications to code,
tests, figures, and other files will remain untouched.

## Document structure

The README will contain the following sections.

1. Logo and scientific summary
2. Scope
3. Installation
4. Quick start
5. Configuration
6. Methods
7. Outputs and provenance
8. Real-data comparison
9. Software and testing
10. References
11. License

## Content requirements

The existing logo will remain at the top. The README will contain one scientific plot,
which will be the real-data comparison with MNE default notch geometry.

Equations and mathematical notation will be removed. Each method will be described in
concise natural language. The descriptions will retain the estimation-window procedure,
spectral scaling, comb candidate search, Bayesian information criterion decisions,
isolated-line temporal and shape tests, stopband construction, MNE FIR settings,
attenuation measurement, verification procedure, and Welch spectra settings.

The configuration table will retain the two scientific settings and their defaults. The
MNE filtering table will retain the exact arguments used by the implementation. The
software table will retain the declared packages, minimum versions, and roles.

The output files, provenance records, BrainVision round-trip check, participant audit,
scientific limitations, and all credible references will remain. Reference numbers must
link to their bibliography entries in GitHub.

## Writing requirements

The prose will be factual, concise, and suitable for scientific software review. It will
avoid promotional claims, contrastive sales framing, em dashes, colons, and semicolons.
It will not reproduce wording from MNE-Denoise.

## Verification

The completed README will be checked for one plot, a retained logo, working numbered
references, valid local paths, absence of equations and restricted math commands, and
compliance with the punctuation and style requirements. The README will be submitted to
the GitHub Markdown rendering API. The repository test suite and Ruff checks will also
be run from the project environment.
