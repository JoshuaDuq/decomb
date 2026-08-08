#!/usr/bin/env bash
# Run every decomb stage in order against decomb.yaml, and keep going when one fails.
#
# A failing stage is recorded and the next one still starts. Some later stages genuinely
# cannot succeed without an earlier one -- `apply` refuses without a passing benchmark, and
# `verify`, `report` and `psd` read what `apply` wrote -- so a downstream failure after an
# upstream one is the gate doing its job rather than a second fault. The summary at the end
# lists every stage with its exit code, so which is which stays legible.
#
# `notch` is deliberately not here: notch_bands is empty, so it would write a second 11 GB
# copy that is a passthrough.

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
DECOMB="$HERE/.venv/bin/decomb"
CONFIG="$HERE/decomb.yaml"
LOGS="$HERE/logs"
mkdir -p "$LOGS"

STAGES=(diagnose benchmark apply verify report psd)
declare -a RESULTS=()

echo "=== decomb pipeline started $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "config: $CONFIG"
echo

index=1
for stage in "${STAGES[@]}"; do
    log="$LOGS/$(printf '%02d' "$index")_${stage}.log"
    echo "=== ${stage} -> ${log} ($(date '+%H:%M:%S')) ==="
    started=$SECONDS
    "$DECOMB" "$stage" --config "$CONFIG" > "$log" 2>&1
    code=$?
    elapsed=$((SECONDS - started))
    if [ $code -eq 0 ]; then
        echo "    ${stage} ok (${elapsed}s)"
        tail -3 "$log" | sed 's/^/      /'
    else
        echo "    ${stage} FAILED exit ${code} (${elapsed}s); continuing to next stage"
        tail -8 "$log" | sed 's/^/      /'
    fi
    RESULTS+=("$(printf '%-10s exit=%d  %ds' "$stage" "$code" "$elapsed")")
    index=$((index + 1))
    echo
done

echo "=== summary $(date '+%Y-%m-%d %H:%M:%S') ==="
for result in "${RESULTS[@]}"; do
    echo "  ${result}"
done
