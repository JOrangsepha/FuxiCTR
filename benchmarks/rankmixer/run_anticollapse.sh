#!/usr/bin/env bash
# Anti-collapse follow-up on the existing Ali-CCP parquet cache.
#
# Two variants x seeds 2025 2026 2027:
#   MTRankMixer_aliccp_semantic_residual   task_pooling: residual
#   MTRankMixer_aliccp_semantic_entropy    gate_entropy_reg: 0.01
#
# Same budget as the previous multi-seed run: epochs <= 6, patience 1,
# batch 8192, embedding 16, Adam 1e-3, semantic groups, monitor = mean AUC.
# Each job is killed after 45 minutes. Logs, checkpoints, gate tables, and
# STATUS go to /root/autodl-tmp/ac_out/. The script does not rebuild
# feature_map.json.
#
# After training, seed 2025 of each variant is scored with analyze_gates.py
# on up to 500000 test rows (--stage test). That reproduces the development
# study. It is not the selection protocol. New runs use run_rigor_suite.sh,
# which stays on validation. The script then writes ALL_DONE.
#
# Usage, repo at /root/autodl-tmp/FuxiCTR:
#   bash /root/autodl-tmp/FuxiCTR/benchmarks/rankmixer/run_anticollapse.sh 0

set -u

GPU="${1:-0}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="/root/autodl-tmp/ac_out"
TIMEOUT="45m"
SEEDS="${SEEDS:-2025 2026 2027}"
if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    PY=python3
fi

SRC="$ROOT/benchmarks/rankmixer/configs/anticollapse"
LOGS="$OUT/logs"
STATUS="$OUT/STATUS"
mkdir -p "$LOGS" "$OUT/configs"
: > "$STATUS"

stamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

note() {
    echo "[$(stamp)] $*" | tee -a "$STATUS"
}

JOBS=(
    "MTRankMixer_aliccp_semantic_residual|gate_residual"
    "MTRankMixer_aliccp_semantic_entropy|gate_entropy"
)

fail=0
note "START gpu=${GPU} timeout=${TIMEOUT} seeds=${SEEDS}"

for spec in "${JOBS[@]}"; do
    IFS='|' read -r template tag <<< "$spec"
    for seed in $SEEDS; do
        expid="${template}_s${seed}"
        cfg="$OUT/configs/${expid}"
        log="$LOGS/${expid}.log"
        note "START ${expid}"
        "$PY" "$ROOT/benchmarks/rankmixer/prepare_multiseed_config.py" \
            --src "$SRC" --dst "$cfg" --template "$template" \
            --expid "$expid" --seed "$seed" >> "$STATUS" 2>&1
        start=$(date +%s)
        timeout --signal=TERM --kill-after=60s "$TIMEOUT" \
            "$PY" "$ROOT/model_zoo/multitask/MT_RankMixer/run_expid.py" \
            --config "$cfg" --expid "$expid" --gpu "$GPU" \
            > "$log" 2>&1
        code=$?
        end=$(date +%s)
        seconds=$((end - start))
        echo "$seconds" > "$LOGS/${expid}.time"
        echo "$code" > "$LOGS/${expid}.exit"
        note "EXIT ${expid} code=${code} seconds=${seconds} log=${log}"
        if [[ "$code" -ne 0 ]]; then
            fail=1
        fi
    done
done

for spec in "${JOBS[@]}"; do
    IFS='|' read -r template tag <<< "$spec"
    expid="${template}_s2025"
    ckpt="$OUT/checkpoints/AliCCP_x1/${expid}.model"
    md="$OUT/${tag}_s2025.md"
    png="$OUT/${tag}_s2025.png"
    log="$LOGS/${tag}_s2025.log"
    note "START analyze ${expid}"
    if [[ ! -f "$ckpt" ]]; then
        note "EXIT analyze ${expid} code=missing checkpoint=${ckpt}"
        fail=1
        continue
    fi
    timeout --signal=TERM --kill-after=60s "$TIMEOUT" \
        "$PY" "$ROOT/benchmarks/rankmixer/analyze_gates.py" --gpu "$GPU" \
        --stage test \
        --checkpoint "$ckpt" \
        --config "$OUT/configs/${expid}" \
        --expid "$expid" \
        --max_samples 500000 \
        --output "$md" \
        --figure "$png" \
        > "$log" 2>&1
    code=$?
    note "EXIT analyze ${expid} code=${code} markdown=${md} figure=${png}"
    if [[ "$code" -ne 0 ]]; then
        fail=1
    fi
done

"$PY" "$ROOT/benchmarks/rankmixer/summarize_multiseed.py" \
    --logs "$LOGS" --csv "$OUT/anticollapse_summary.csv" >> "$STATUS" 2>&1
"$PY" "$ROOT/benchmarks/rankmixer/multiseed_stats.py" \
    --csv "$OUT/anticollapse_summary.csv" >> "$STATUS" 2>&1

if [[ "$fail" -eq 0 ]]; then
    note "ALL_DONE"
else
    note "ALL_DONE with failures"
fi
date '+%Y-%m-%d %H:%M:%S' > "$OUT/ALL_DONE"
echo "fail=${fail}" >> "$OUT/ALL_DONE"
note "marker ${OUT}/ALL_DONE"
