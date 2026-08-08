#!/usr/bin/env bash
# Measure how candidate settings move the preservation gate on the six worst recordings.
#
# The gate that refuses the apply is intrinsic_energy_ratio >= 0.85: the share of an
# injected broadband transient that survives the removal. It fails here because the notches
# crossing the transient's band take more of it than the floor allows, so the levers worth
# testing are the ones that set how much spectrum a target costs -- not the floor itself.
#
#   support_min_prominence_db  how prominent a peak must be before its observed extent may
#                              widen a notch that is already removing it. Raising it emties
#                              less band without narrowing the notch that does the work.
#   notch_width_ratio          the notch width itself, as freq/ratio. Raising it narrows
#                              every notch, which costs suppression at the high harmonics.
#
# Suppression is reported beside the ratio for exactly that reason: a setting that passes
# the gate by not removing the artifact has not solved anything.

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
DECOMB="$HERE/.venv/bin/decomb"
TRIAL_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg_trial6"
mkdir -p "$HERE/trials" "$HERE/logs/trials"

# name|support_min_prominence_db|notch_width_ratio
VARIANTS=(
    "A_baseline|10.0|450"
    "B_support20|20.0|450"
    "C_ratio650|10.0|650"
    "D_both|20.0|600"
)

for spec in "${VARIANTS[@]}"; do
    IFS='|' read -r name support ratio <<<"$spec"
    config="$HERE/trials/trial_$name.yaml"

    # Merged, not appended: a second `removal:` block in the same file would replace the
    # measured fundamental and harmonic ranges rather than add to them.
    "$HERE/.venv/bin/python" "$HERE/make_trial_config.py" "$HERE/decomb.yaml" "$config" \
        "paths.bids_root=$TRIAL_ROOT" \
        "paths.output_root=/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg_trial6_out_$name" \
        "paths.removal_dir=$HERE/outputs/trial_$name" \
        "removal.support_min_prominence_db=$support" \
        "removal.notch_width_ratio=$ratio" \
        "removal.filter_jobs=2"

    echo "=== $name  support_min_prominence_db=$support  notch_width_ratio=$ratio ==="
    "$DECOMB" benchmark --config "$config" > "$HERE/logs/trials/$name.log" 2>&1 &
done

wait
echo
echo "all trials finished"
