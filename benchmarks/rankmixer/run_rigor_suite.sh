#!/usr/bin/env bash
# Validation-first rigor suite on the existing Ali-CCP parquet cache.
#
# Trains, then analyzes gates on the validation split, then summarizes.
# This script does not read the test split. After the design is frozen:
#   bash benchmarks/rankmixer/run_final_test.sh 0
#
# Variants (expid prefix MTR_rg_ or PLE_rg):
#   g3_mean, g6_mean, g6_residual, g6_gate, g6_random, g6_sequential, PLE
# g6_random uses partition seed 42 (see configs/rigor/model_config.yaml).
# g6_random and g6_sequential are shared-mean controls, not gated models.
#
# Default seeds are 2025 2026 2027. Five seeds:
#   SEEDS="2025 2026 2027 2028 2029" bash benchmarks/rankmixer/run_rigor_suite.sh 0
#
# Budget matches the previous study except the early-stop sensitivity:
#   batch 8192, embedding 16, Adam 1e-3, epochs <= 10, early_stop_patience 3.
# Each job is invoked with --skip_test. Monitor AUC is the mean of click AUC
# and conversion AUC on the validation split.
#
# Parallelism defaults to 2. Three concurrent jobs previously killed the Ray
# raylet (LocalRayletDiedError). Set PARALLEL=1 if that happens again.
# PARALLEL processes share the single GPU index passed as $1.
#
# Layout mirrors the earlier AutoDL runs:
#   $OUT/logs, $OUT/configs, $OUT/checkpoints, $OUT/STATUS, $OUT/ALL_DONE
#   $OUT/analysis/   validation gate tables, json, and across-seed aggregates
# Default OUT is /root/autodl-tmp/rg_out.
#
# Usage, repo at /root/autodl-tmp/FuxiCTR:
#   bash /root/autodl-tmp/FuxiCTR/benchmarks/rankmixer/run_rigor_suite.sh 0
#   bash /root/autodl-tmp/FuxiCTR/benchmarks/rankmixer/run_rigor_suite.sh 0 2
#
# Environment:
#   OUT           output root (default /root/autodl-tmp/rg_out)
#   SEEDS         space-separated training seeds
#   PARALLEL      concurrent jobs (default 2; or pass as the second argument)
#   TIMEOUT       per-job timeout (default 90m)
#   NUM_WORKERS   Ray prefetch workers written into each cloned config (default 4)
#   PYTHON        interpreter

set -u

GPU="${1:-0}"
if [[ -n "${2:-}" ]]; then
    PARALLEL="$2"
else
    PARALLEL="${PARALLEL:-2}"
fi
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${OUT:-/root/autodl-tmp/rg_out}"
TIMEOUT="${TIMEOUT:-90m}"
SEEDS="${SEEDS:-2025 2026 2027}"
NUM_WORKERS="${NUM_WORKERS:-4}"
if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    PY=python3
fi

SRC="$ROOT/benchmarks/rankmixer/configs/rigor"
LOGS="$OUT/logs"
ANALYSIS="$OUT/analysis"
STATUS="$OUT/STATUS"
mkdir -p "$LOGS" "$ANALYSIS/logs" "$OUT/configs" "$OUT/checkpoints/AliCCP_x1"
: > "$STATUS"

stamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

note() {
    echo "[$(stamp)] $*" | tee -a "$STATUS"
}

# template | workdir relative to repo | gate tag (empty skips analysis) | token names
JOBS=(
    "MTR_rg_g3_mean|model_zoo/multitask/MT_RankMixer||"
    "MTR_rg_g6_mean|model_zoo/multitask/MT_RankMixer||"
    "MTR_rg_g6_residual|model_zoo/multitask/MT_RankMixer|g6_residual|user_id,user_profile,item_id,item_attr,cross,scenario"
    "MTR_rg_g6_gate|model_zoo/multitask/MT_RankMixer|g6_gate|user_id,user_profile,item_id,item_attr,cross,scenario"
    "MTR_rg_g6_random|model_zoo/multitask/MT_RankMixer||"
    "MTR_rg_g6_sequential|model_zoo/multitask/MT_RankMixer||"
    "PLE_rg|model_zoo/multitask/PLE||"
)

pids=()

wait_for_slot() {
    while true; do
        if [[ "${#pids[@]}" -eq 0 ]]; then
            return
        fi
        local alive=0
        local newpids=()
        local pid
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive=$((alive + 1))
                newpids+=("$pid")
            else
                wait "$pid" || true
            fi
        done
        if [[ "${#newpids[@]}" -eq 0 ]]; then
            pids=()
        else
            pids=("${newpids[@]}")
        fi
        if [[ "$alive" -lt "$PARALLEL" ]]; then
            return
        fi
        sleep 5
    done
}

wait_all() {
    local pid
    if [[ "${#pids[@]}" -eq 0 ]]; then
        return
    fi
    for pid in "${pids[@]}"; do
        wait "$pid" || true
    done
    pids=()
}

note "START gpu=${GPU} parallel=${PARALLEL} timeout=${TIMEOUT} seeds=${SEEDS} out=${OUT}"
note "protocol=valid-only skip_test=1 gate_stage=valid"

for spec in "${JOBS[@]}"; do
    IFS='|' read -r template workdir _tag _names <<< "$spec"
    for seed in $SEEDS; do
        wait_for_slot
        expid="${template}_s${seed}"
        (
            cfg="$OUT/configs/${expid}"
            log="$LOGS/${expid}.log"
            note "START ${expid}"
            "$PY" "$ROOT/benchmarks/rankmixer/prepare_multiseed_config.py" \
                --src "$SRC" --dst "$cfg" --template "$template" \
                --expid "$expid" --seed "$seed" \
                --model-root "$OUT/checkpoints" \
                --num-workers "$NUM_WORKERS" >> "$STATUS" 2>&1
            start=$(date +%s)
            timeout --signal=TERM --kill-after=60s "$TIMEOUT" \
                "$PY" "$ROOT/${workdir}/run_expid.py" \
                --config "$cfg" --expid "$expid" --gpu "$GPU" --skip_test \
                > "$log" 2>&1
            code=$?
            end=$(date +%s)
            seconds=$((end - start))
            echo "$seconds" > "$LOGS/${expid}.time"
            echo "$code" > "$LOGS/${expid}.exit"
            note "EXIT ${expid} code=${code} seconds=${seconds} log=${log}"
        ) &
        pids+=("$!")
    done
done
wait_all

fail=0
for exit_file in "$LOGS"/*.exit; do
    [[ -e "$exit_file" ]] || continue
    code="$(cat "$exit_file")"
    if [[ "$code" != "0" ]]; then
        fail=1
    fi
done

for spec in "${JOBS[@]}"; do
    IFS='|' read -r template _workdir tag token_names <<< "$spec"
    if [[ -z "$tag" ]]; then
        continue
    fi
    for seed in $SEEDS; do
        wait_for_slot
        expid="${template}_s${seed}"
        (
            ckpt="$OUT/checkpoints/AliCCP_x1/${expid}.model"
            md="$ANALYSIS/${tag}_s${seed}.md"
            png="$ANALYSIS/${tag}_s${seed}.png"
            js="$ANALYSIS/${tag}_s${seed}.json"
            log="$ANALYSIS/logs/${tag}_s${seed}.log"
            note "START analyze ${expid} stage=valid"
            if [[ ! -f "$ckpt" ]]; then
                note "EXIT analyze ${expid} code=missing checkpoint=${ckpt}"
                echo "missing" > "$ANALYSIS/logs/${tag}_s${seed}.exit"
                exit 0
            fi
            timeout --signal=TERM --kill-after=60s "$TIMEOUT" \
                "$PY" "$ROOT/benchmarks/rankmixer/analyze_gates.py" --gpu "$GPU" \
                --stage valid \
                --checkpoint "$ckpt" \
                --config "$OUT/configs/${expid}" \
                --expid "$expid" \
                --seed "$seed" \
                --token_names "$token_names" \
                --max_samples 500000 \
                --output "$md" \
                --figure "$png" \
                --json "$js" \
                > "$log" 2>&1
            code=$?
            echo "$code" > "$ANALYSIS/logs/${tag}_s${seed}.exit"
            note "EXIT analyze ${expid} code=${code} markdown=${md}"
        ) &
        pids+=("$!")
    done
done
wait_all

shopt -s nullglob
for spec in "${JOBS[@]}"; do
    IFS='|' read -r _template _workdir tag _names <<< "$spec"
    if [[ -z "$tag" ]]; then
        continue
    fi
    jsons=("$ANALYSIS/${tag}_s"*.json)
    if [[ "${#jsons[@]}" -eq 0 ]]; then
        note "SKIP aggregate ${tag}: no json"
        fail=1
        continue
    fi
    "$PY" "$ROOT/benchmarks/rankmixer/aggregate_gates.py" \
        --inputs "${jsons[@]}" \
        --output "$ANALYSIS/${tag}_seeds.md" >> "$STATUS" 2>&1 || fail=1
    note "AGGREGATE ${tag} n=${#jsons[@]} markdown=$ANALYSIS/${tag}_seeds.md"
done
shopt -u nullglob

for exit_file in "$ANALYSIS"/logs/*.exit; do
    [[ -e "$exit_file" ]] || continue
    code="$(cat "$exit_file")"
    if [[ "$code" != "0" ]]; then
        fail=1
    fi
done

first_seed="${SEEDS%% *}"
audit_expid="MTR_rg_g6_mean_s${first_seed}"
audit_ckpt="$OUT/checkpoints/AliCCP_x1/${audit_expid}.model"
if [[ -f "$audit_ckpt" ]]; then
    note "START grad audit ${audit_expid} stage=valid"
    "$PY" "$ROOT/benchmarks/rankmixer/audit_trunk_grads.py" --gpu "$GPU" \
        --stage valid --batches 4 \
        --checkpoint "$audit_ckpt" \
        --config "$OUT/configs/${audit_expid}" \
        --expid "$audit_expid" \
        --workdir "$ROOT/model_zoo/multitask/MT_RankMixer" \
        --output "$OUT/grad_audit.md" \
        > "$ANALYSIS/logs/grad_audit.log" 2>&1
    audit_code=$?
    note "EXIT grad audit code=${audit_code} markdown=$OUT/grad_audit.md"
    if [[ "$audit_code" -ne 0 ]]; then
        fail=1
    fi
else
    note "SKIP grad audit, checkpoint missing: ${audit_ckpt}"
    fail=1
fi

"$PY" "$ROOT/benchmarks/rankmixer/summarize_multiseed.py" \
    --logs "$LOGS" --csv "$OUT/rigor_summary.csv" >> "$STATUS" 2>&1 || fail=1
"$PY" "$ROOT/benchmarks/rankmixer/multiseed_stats.py" \
    --csv "$OUT/rigor_summary.csv" >> "$STATUS" 2>&1 || fail=1

if [[ "$fail" -eq 0 ]]; then
    note "ALL_DONE"
else
    note "ALL_DONE with failures"
fi
date '+%Y-%m-%d %H:%M:%S' > "$OUT/ALL_DONE"
echo "fail=${fail}" >> "$OUT/ALL_DONE"
note "marker ${OUT}/ALL_DONE"
note "test split was not read. After freezing the design: bash $ROOT/benchmarks/rankmixer/run_final_test.sh ${GPU}"
