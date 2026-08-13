# Sequential Overnight Runner Design

## Goal

Prepare one fail-fast shell command that runs the current decomb workflow over all
15 participants and 90 recordings in the configured BIDS dataset. Recordings run one at
a time. MNE may use multiple CPU cores within the active recording.

## Pipeline

The runner executes the public CLI stages in their declared order:

1. `diagnose`
2. `apply`
3. `verify`
4. `psd`

`diagnose`, `apply`, and `verify` cover every discovered recording. The existing `psd`
stage intentionally draws the first discovered recording only and remains unchanged.

## Preflight

Before changing any output, the runner loads `decomb.yaml` through decomb's configuration
API and discovers recordings through `decomb.recordings.discover_runs`. It requires exactly
15 distinct participants and 90 recordings. A mismatch stops the run with an error.

The runner also requires the configured virtual-environment executable and configuration
file. It does not infer alternate executables or paths.

## Replacement and Recovery

The existing configured derivative and the associated report root are moved to timestamped
archive paths before the first stage. Nothing is deleted. Existing archive targets cause an
error rather than being overwritten.

The archive operation happens only after preflight succeeds. This satisfies `decomb apply`'s
requirement that the output root not exist while preserving the previous results for manual
recovery.

## Execution and Failure Handling

The runner uses strict shell execution with pipeline failure propagation. It starts no
background jobs and invokes one CLI stage at a time. Each stage's combined output is shown
in the terminal and written to a timestamped log.

Any nonzero stage exit stops the pipeline. decomb's existing staging directory remains in
place after an interrupted or failed apply so the incomplete result is explicit and cannot
be mistaken for the final derivative.

## Completion Checks

After `verify`, the runner reads the derivative manifest and verification table and requires
both to cover exactly the 90 source recording names. Missing, additional, or mismatched
recordings fail the run. The `psd` stage runs only after this coverage check succeeds.

The runner prints the final derivative, report, archive, and log locations when every stage
and coverage check succeeds.

## Verification

Automated tests exercise preflight counts, strict sequential stage order, fail-fast behavior,
recoverable archiving, and final recording-coverage validation without processing the real
EEG data. Shell syntax validation and the project's Python tests and linter are run before
handoff.
