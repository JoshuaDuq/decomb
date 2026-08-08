#!/usr/bin/env bash
# Build the BIDS root decomb actually processes: every recording in the source except the
# one excluded for data quality, as symlinks so nothing is duplicated.
#
# decomb's benchmark/apply/verify refuse a --subjects subset on purpose -- the gates are
# decided over the recordings jointly, and a partial write would leave an output root that
# no provenance describes. Defining the cohort as its own root is the supported way to say
# which recordings a derivative was built from, and it keeps that statement on disk next to
# the data rather than in a command line someone has to remember.
#
# Excluded: sub-0010 run-1. A ~6 s high-amplitude burst at t=25-31 s raises the opening
# window's broadband floor ~13 dB, dropping median comb prominence from ~10 dB to 3.8 dB.
# Only 19 of 52 candidate harmonics then agree on one grid, under decomb's floor of 20, so
# the comb cannot be fitted there. See logs/05_margin_w54.log for the cohort-wide margins.
#
# Also skipped: ._* (AppleDouble stubs the exFAT volume leaves behind), *.lock, and *.bak
# (stale upstream backups). None are BIDS files and none belong in a derivative.

set -euo pipefail

SOURCE="/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg"
COHORT="/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg_cohort89"
EXCLUDE="sub-0010_task-thermalactive_run-1"

rm -rf "$COHORT"
mkdir -p "$COHORT"

linked=0
skipped=0
while IFS= read -r path; do
    name="$(basename "$path")"
    case "$name" in
        ._*|*.lock|*.bak) continue ;;
        "$EXCLUDE"*) skipped=$((skipped + 1)); continue ;;
    esac
    relative="${path#"$SOURCE"/}"
    target="$COHORT/$relative"
    mkdir -p "$(dirname "$target")"
    ln -s "$path" "$target"
    linked=$((linked + 1))
done < <(find "$SOURCE" -type f)

cat > "$COHORT/README" <<'NOTE'
decomb processing cohort: 89 of the 90 recordings in ../eeg.

sub-0010_task-thermalactive_run-1 is excluded. A ~6 s high-amplitude burst at t=25-31 s
raises that recording's opening 54 s window broadband floor by ~13 dB, which masks the
1.2 Hz comb: only 19 of 52 candidate harmonics agree on a single grid, below the 20 that
decomb requires before a fitted fundamental may authorise a removal grid. The remaining
16 windows of that recording are unaffected, and every other recording in the cohort
clears the floor with a median worst-window count of 40.

All 15 participants are present. sub-0010 contributes runs 2-6.

Entries are symlinks into ../eeg; no recording is duplicated.
NOTE

echo "cohort root: $COHORT"
echo "  linked  $linked file(s)"
echo "  skipped $skipped file(s) of $EXCLUDE"
echo "  subjects:   $(find "$COHORT" -mindepth 1 -maxdepth 1 -type d -name 'sub-*' | wc -l | tr -d ' ')"
echo "  recordings: $(find "$COHORT" -name '*_eeg.vhdr' | wc -l | tr -d ' ')"
