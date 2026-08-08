#!/usr/bin/env bash
# A small BIDS root of the recordings that fail the preservation gate hardest, so a
# settings change can be measured in minutes instead of the ~7 hours a full benchmark of
# 89 recordings costs.
#
# This is a measurement fixture, not a cohort: nothing it produces may certify an apply.
# It writes to its own report directory for that reason, and `apply` compares the
# benchmark's recording set against the set it is about to write, so a trial benchmark
# cannot be mistaken for the real one even by accident.

set -euo pipefail

SOURCE="/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg"
TRIAL="/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg_trial0011"

RUNS=(
    "sub-0011/eeg/sub-0011_task-thermalactive_run-1"
    "sub-0011/eeg/sub-0011_task-thermalactive_run-2"
    "sub-0011/eeg/sub-0011_task-thermalactive_run-3"
    "sub-0011/eeg/sub-0011_task-thermalactive_run-4"
    "sub-0011/eeg/sub-0011_task-thermalactive_run-5"
    "sub-0011/eeg/sub-0011_task-thermalactive_run-6"
)

rm -rf "$TRIAL"
mkdir -p "$TRIAL"

# Dataset-level sidecars, which mne-bids expects to find beside the subjects.
for name in dataset_description.json participants.tsv participants.json README \
            task-thermalactive_events.json; do
    [ -e "$SOURCE/$name" ] && ln -s "$SOURCE/$name" "$TRIAL/$name"
done

linked=0
for run in "${RUNS[@]}"; do
    subject="${run%%/*}"
    stem="$(basename "$run")"
    mkdir -p "$TRIAL/$subject/eeg"
    for path in "$SOURCE/$subject/eeg/${stem}"*; do
        name="$(basename "$path")"
        case "$name" in ._*|*.lock|*.bak) continue ;; esac
        ln -s "$path" "$TRIAL/$subject/eeg/$name"
        linked=$((linked + 1))
    done
    # Per-subject sidecars the recordings share.
    for path in "$SOURCE/$subject/eeg/${subject}_task-thermalactive_events."* \
                "$SOURCE/$subject/eeg/${subject}_space-"*; do
        name="$(basename "$path")"
        case "$name" in ._*|*.lock|*.bak) continue ;; esac
        [ -e "$TRIAL/$subject/eeg/$name" ] || ln -s "$path" "$TRIAL/$subject/eeg/$name"
    done
done

echo "trial root: $TRIAL"
echo "  linked $linked recording file(s)"
echo "  recordings: $(find "$TRIAL" -name 'sub-*_eeg.vhdr' ! -name '._*' | wc -l | tr -d ' ')"
