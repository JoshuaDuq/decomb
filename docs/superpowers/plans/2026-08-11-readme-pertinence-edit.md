# README Pertinence Edit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove implementation detail and repeated argumentation that users do not need while retaining the scientific and operational information required to use and evaluate `decomb`.

**Architecture:** Only `README.md` changes. The document retains its user-facing sequence and complete bibliography while individual sections are shortened according to the approved pertinence specification.

**Tech Stack:** GitHub Flavored Markdown, MNE-Python, pytest, and Ruff.

---

### Task 1: Reduce the README

**Files:**
- Modify: `README.md`
- Reference: `docs/superpowers/specs/2026-08-11-readme-pertinence-edit-design.md`

- [ ] **Step 1: Shorten the problem statement**

Replace the mathematical proof and extended alternative-method discussion with two
short paragraphs. Retain the residual-artifact context, non-identifiability of neural and
artifactual activity at the same frequency, unavailable-frequency policy, and source
control recommendation.

- [ ] **Step 2: Reduce operational detail**

Remove the configuration precedence list and development-installation paragraph. Keep
the runtime installation, four commands, command meanings, input requirements, two
scientific settings, range validation, and failure on an existing output directory.

- [ ] **Step 3: Condense the methods**

Retain the exact window overlap, Hann spectrum, channel aggregation, peak interpolation,
comb criterion, isolated-line temporal and shape criteria, trajectory stopbands,
transition rule, MNE parameter table, attenuation method, independent verification, and
matched Welch settings. Remove Fourier parity details, candidate-grid refinement
mechanics, reported generic MNE response values, automatic coefficient-count discussion,
and BrainVision byte-level implementation details.

- [ ] **Step 4: Condense outputs and implementation information**

Replace the output table with one short paragraph naming the manifest, derivative
description, effective configurations, verification table, and PSD products. Replace the
package version table with a sentence listing the declared scientific packages and a
link to `pyproject.toml`. Remove the test coverage inventory and plot-regeneration
command.

- [ ] **Step 5: Retain validation and references**

Keep the MNE comparison plot, its shared-input design, the unavailable-bandwidth result,
one short participant-validation paragraph, all bibliography entries, and the license.

### Task 2: Verify and commit

**Files:**
- Verify: `README.md`
- Test: `tests`

- [ ] **Step 1: Run the README audit**

Verify the retained headings, logo, one plot, citation links and targets, local paths,
absence of equations, absence of restricted punctuation, and a lower word count.

- [ ] **Step 2: Verify GitHub rendering**

Submit `README.md` to the GitHub Markdown API and confirm that each numbered citation
points to its rendered bibliography target and exactly two images render.

- [ ] **Step 3: Run repository checks**

Run the following commands.

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
git diff --check
```

Expected results are a passing test suite, clean Ruff output, and no whitespace errors.

- [ ] **Step 4: Commit only the README**

```bash
git add README.md
git commit -m "Remove nonessential README detail" -- README.md
```

Confirm that existing unrelated workspace modifications remain untouched.
