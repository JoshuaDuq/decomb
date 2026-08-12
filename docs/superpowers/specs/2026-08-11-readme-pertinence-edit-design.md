# README pertinence edit design

## Objective

Reduce the README to information needed to understand, run, evaluate, and cite the
software.

## Retained content

The README will retain the logo, concise problem statement, input requirements,
installation commands, command workflow, two scientific settings, method descriptions,
exact MNE filtering parameters, principal derivative and provenance outputs, MNE
comparison plot, participant validation summary, complete references, and license.

The methods will continue to identify the windowing procedure, Hann spectral estimate,
comb and isolated-line model criteria, trajectory-based stopbands, FIR design,
attenuation measurement, verification, and matched Welch spectra.

## Removed or compressed content

The mathematical non-identifiability proof will be replaced by a short inferential
limitation. Extended discussion of inversion, interpolation, regression, and source
separation will be removed. Configuration precedence, Fourier edge-bin scaling,
candidate-grid mechanics, FIR coefficient calculations, BrainVision byte-level details,
the output table, package version table, test inventory, and plot-regeneration command
will be removed or compressed.

Repeated limitations will appear once. The participant validation description will be
limited to its sample size and principal measured result.

## Writing constraints

The prose will remain scientific, factual, and concise. Equations, promotional language,
contrastive sales framing, em dashes, colons, and semicolons will be absent. Numbered
citations will link to the complete bibliography. The wording will not reproduce the
MNE-Denoise README.

## Verification

The final document will be checked for the retained sections, one logo, one MNE
comparison plot, complete reference links, valid local paths, absence of equations, and
the writing constraints. GitHub Markdown rendering, the repository test suite, Ruff,
and whitespace checks will be run before completion.
