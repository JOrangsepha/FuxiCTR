#!/usr/bin/env bash
# Read the Ali-CCP test split once, after the design is frozen.
#
# Do not run this from run_rigor_suite.sh and do not use the numbers to pick
# another architecture. The rigor suite trains with --skip_test and analyzes
# gates on validation. This script only evaluates the checkpoints that already
# exist under $OUT/checkpoints.
#
# Usage:
#   OUT=/root/autodl-tmp/rg_out bash benchmarks/rankmixer/run_final_test.sh 0
#
# SEEDS defaults to 2025 2026 2027. Match the seeds that were trained:
#   SEEDS="2025 2026 2027 2028 2029" bash benchmarks/rankmixer/run_final_test.sh 0

set -u

GPU="${1:-0}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${OUT:-/root/autodl-tmp/rg_out}"
TIMEOUT="${TIMEOUT:-90m}"
SEEDS="${SEEDS:-2025 2026 2027}"
if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    PY=python3
fi

LOGS="$OUT/final_test/logs"
STATUS="$OUT/final_test/STATUS"
mkdir -p "$LOGS"
: > "$STATUS"

stamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

note() {
    echo "[$(stamp)] $*" | tee -a "$STATUS"
}

JOBS=(
    "MTR_rg_g3_mean|model_zoo/multitask/MT_RankMixer"
    "MTR_rg_g6_mean|model_zoo/multitask/MT_RankMixer"
    "MTR_rg_g6_residual|model_zoo/multitask/MT_RankMixer"
    "MTR_rg_g6_gate|model_zoo/multitask/MT_RankMixer"
    "MTR_rg_g6_random|model_zoo/multitask/MT_RankMixer"
    "MTR_rg_g6_sequential|model_zoo/multitask/MT_RankMixer"
    "PLE_rg|model_zoo/multitask/PLE"
)

fail=0
note "START final-test-once gpu=${GPU} seeds=${SEEDS} out=${OUT}"
note "This reads the test split. Run it only after the design is frozen."

for spec in "${JOBS[@]}"; do
    IFS='|' read -r template workdir <<< "$spec"
    for seed in $SEEDS; do
        expid="${template}_s${seed}"
        ckpt="$OUT/checkpoints/AliCCP_x1/${expid}.model"
        cfg="$OUT/configs/${expid}"
        log="$LOGS/${expid}.log"
        note "START test ${expid}"
        if [[ ! -f "$ckpt" ]]; then
            note "EXIT test ${expid} code=missing checkpoint=${ckpt}"
            echo "missing" > "$LOGS/${expid}.exit"
            fail=1
            continue
        fi
        if [[ ! -d "$cfg" ]]; then
            note "EXIT test ${expid} code=missing config=${cfg}"
            echo "missing" > "$LOGS/${expid}.exit"
            fail=1
            continue
        fi
        start=$(date +%s)
        timeout --signal=TERM --kill-after=60s "$TIMEOUT" \
            "$PY" "$ROOT/benchmarks/rankmixer/eval_checkpoint.py" \
            --gpu "$GPU" --stage test \
            --checkpoint "$ckpt" \
            --config "$cfg" \
            --expid "$expid" \
            --workdir "$ROOT/${workdir}" \
            > "$log" 2>&1
        code=$?
        end=$(date +%s)
        echo $((end - start)) > "$LOGS/${expid}.time"
        echo "$code" > "$LOGS/${expid}.exit"
        note "EXIT test ${expid} code=${code} seconds=$((end - start)) log=${log}"
        if [[ "$code" -ne 0 ]]; then
            fail=1
        fi
    done
done

"$PY" "$ROOT/benchmarks/rankmixer/summarize_multiseed.py" \
    --logs "$LOGS" --csv "$OUT/final_test/summary.csv" >> "$STATUS" 2>&1 || fail=1
"$PY" "$ROOT/benchmarks/rankmixer/multiseed_stats.py" \
    --csv "$OUT/final_test/summary.csv" >> "$STATUS" 2>&1 || fail=1

if [[ "$fail" -eq 0 ]]; then
    note "ALL_DONE"
else
    note "ALL_DONE with failures"
fi
date '+%Y-%m-%d %H:%M:%S' > "$OUT/final_test/ALL_DONE"
echo "fail=${fail}" >> "$OUT/final_test/ALL_DONE"
note "marker ${OUT}/final_test/ALL_DONE"
