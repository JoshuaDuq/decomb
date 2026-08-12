# Concise Scientific README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the long formula-based README with a compact scientific README modeled on the organization of MNE-Denoise.

**Architecture:** The work is confined to `README.md`. Existing prose supplies the scientific source material, while `pyproject.toml`, `src/decomb/defaults.yaml`, and the implementation provide verification sources for software versions, settings, and method parameters.

**Tech Stack:** GitHub Flavored Markdown, MNE-Python, MNE-BIDS, NumPy, SciPy, pandas, PyYAML, Matplotlib, joblib, pybv, pytest, and Ruff.

---

### Task 1: Rewrite the README

**Files:**
- Modify: `README.md`
- Reference: `docs/superpowers/specs/2026-08-11-concise-scientific-readme-design.md`
- Reference: `pyproject.toml`
- Reference: `src/decomb/defaults.yaml`

- [ ] **Step 1: Record the required content from the current README**

Confirm that the current document contains the logo, problem statement, scientific
scope, command workflow, configuration defaults, method details, output descriptions,
single comparison plot, participant audit, package table, tests, limitations, nineteen
references, and license.

Run:

```bash
rg '^#{1,3} |<img |^!\[' README.md
```

Expected result includes the logo and `docs/notch_comparison_real.png`.

- [ ] **Step 2: Replace the document structure**

Rewrite `README.md` with these sections in this order.

```text
# decomb
logo
scientific summary
## Problem statement
## Scope
## Installation
## Quick start
## Configuration
## Methods
## Outputs and provenance
## Real-data comparison
## Software and testing
## References
## License
```

Keep the problem statement factual. State that the software detects and suppresses
narrowband harmonic and isolated spectral artifacts in continuous EEG, processes each
recording independently, writes a BrainVision BIDS derivative, and treats stopbands and
transitions as unavailable for inference.

- [ ] **Step 3: Convert the methods to concise prose**

Remove all equations and mathematical notation. Retain the following implementation
facts in short method paragraphs.

```text
Window duration and fifty percent overlap
One-sided Hann periodograms and channel-median summaries
Three-point parabolic peak interpolation in decibel space
Comb candidates with at least four harmonics
Bayesian information criterion support and candidate-search penalty
Independent temporal and local Hann-shape tests for isolated lines
Trajectory-based stopband bounds and Hann-resolution minimum width
Merging of intervals whose MNE transitions overlap
One Raw.notch_filter call for all measured stopbands
Zero-phase Hamming firwin design with automatic length and delay compensation
Complete non-overlapping Hann blocks for attenuation measurement
Independent reconstruction of stopbands during verification
Matched Raw.compute_psd Welch settings for source and derivative files
```

- [ ] **Step 4: Retain exact tables and reviewer information**

Keep the two scientific configuration defaults. Keep the MNE call settings including
measured centers, measured widths, transition bandwidth derived from the estimation
window, FIR method, automatic length, zero phase, Hamming window, firwin design,
reflect-limited padding, and all-job execution.

Keep the package minimum versions from `pyproject.toml`. Keep the manifest fields,
BrainVision round-trip verification, effective configuration records, participant audit,
single MNE comparison plot, methodological limitations, all nineteen numbered
references, and BSD license.

- [ ] **Step 5: Apply the scientific writing constraints**

Use concise declarative prose. Remove promotional claims, contrastive sales framing,
em dashes, colons, and semicolons. Keep the wording independent of MNE-Denoise.

### Task 2: Verify the README

**Files:**
- Verify: `README.md`
- Test: `tests`

- [ ] **Step 1: Run the README structural audit**

Check that the logo remains, exactly one Markdown plot remains, no equation delimiters or
restricted math commands remain, all local paths exist, and all nineteen citation links
and bibliography targets are present.

Expected result is a zero exit status and an explicit audit success message.

- [ ] **Step 2: Render the README with GitHub Markdown**

Submit the README as GitHub Flavored Markdown. Check that all nineteen in-text links point
to their rendered bibliography targets and that the rendered document contains exactly
two images consisting of the logo and one plot.

Expected result is a zero exit status without macro or delimiter errors.

- [ ] **Step 3: Run repository verification**

Run:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests
git diff --check
```

Expected result is a passing test suite, a clean Ruff result, and no whitespace errors.

- [ ] **Step 4: Inspect the final scope**

Run:

```bash
git status --short
git diff -- README.md
```

Confirm that implementation changed only `README.md` and did not alter the user's other
uncommitted files.

- [ ] **Step 5: Commit the README**

Run:

```bash
git add README.md
git commit -m "Rewrite README for concise scientific review"
```

Expected result is one commit containing only `README.md`.
