#!/usr/bin/env bash
# Gate-collapse ablation on Ali-CCP. Validation only (--skip_test).
#
# Variants (seeds 2025-2029):
#   Baseline g6_gate EQ — reuse /root/autodl-tmp/rg_out when present
#   Entropy β ∈ {0.001, 0.003, 0.01}
#   Temperature T ∈ {2, 4}
#   Combined β=0.01 × T=2
#   Residual + β=0.01  (optional residual + β=0.003 if GC_EXTRA=1)
#
# Usage:
#   nohup setsid env PATH=/root/miniconda3/bin:$PATH PYTHON=/root/miniconda3/bin/python \
#     OUT=/root/autodl-tmp/gc_out PARALLEL=2 \
#     bash /root/autodl-tmp/FuxiCTR/benchmarks/rankmixer/run_gatecollapse.sh 0 \
#     > /root/autodl-tmp/gc_driver.log 2>&1 &

set -u
GPU="${1:-0}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ ! -d "$ROOT/benchmarks/rankmixer" ]]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
RM="$ROOT/benchmarks/rankmixer"
OUT="${OUT:-/root/autodl-tmp/gc_out}"
export OUT
PARALLEL="${PARALLEL:-2}"
TIMEOUT="${TIMEOUT:-90m}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-30m}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GC_EXTRA="${GC_EXTRA:-0}"
REUSE_BASELINE="${REUSE_BASELINE:-1}"
RG_OUT="${RG_OUT:-/root/autodl-tmp/rg_out}"
PY="${PYTHON:-python}"
SRC="$RM/configs/gatecollapse"
MTR_WD="$ROOT/model_zoo/multitask/MT_RankMixer"
SEEDS5="2025 2026 2027 2028 2029"
TOKENS="user_id,user_profile,item_id,item_attr,cross,scenario"
LOGS="$OUT/logs"
STATUS="$OUT/STATUS"
RUNS="$OUT/runs.tsv"
mkdir -p "$LOGS" "$OUT/configs" "$OUT/checkpoints/AliCCP_x1" \
  "$OUT/eval_valid/logs" "$OUT/analysis/logs" "$OUT/baseline_reuse"
[[ -f "$RUNS" ]] || printf "time\tphase\tkind\texpid\tattempt\texit\tseconds\tlog\n" > "$RUNS"

exec 9> "$OUT/.driver.lock"
if ! flock -n 9; then echo "another run_gatecollapse.sh holds $OUT/.driver.lock"; exit 1; fi

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
note() { echo "[$(stamp)] $*" | tee -a "$STATUS"; }
FAIL=0

pids=()
wait_for_slot() {
    local limit="$1"
    while true; do
        local alive=() pid
        for pid in "${pids[@]+"${pids[@]}"}"; do
            if kill -0 "$pid" 2>/dev/null; then alive+=("$pid"); else wait "$pid" 2>/dev/null || true; fi
        done
        pids=("${alive[@]+"${alive[@]}"}")
        [[ "${#pids[@]}" -lt "$limit" ]] && return
        sleep 5
    done
}
wait_all() { local pid; for pid in "${pids[@]+"${pids[@]}"}"; do wait "$pid" 2>/dev/null || true; done; pids=(); }

ckpt_of() { echo "$OUT/checkpoints/AliCCP_x1/$1.model"; }
train_done() { [[ -f "$LOGS/$1.exit" && "$(cat "$LOGS/$1.exit")" == "0" && -f "$(ckpt_of "$1")" ]]; }

train_one() {
    local phase="$1" template="$2" seed="$3" attempt="$4"
    local expid="${template}_s${seed}" cfg="$OUT/configs/${template}_s${seed}" log="$LOGS/${template}_s${seed}.log"
    "$PY" "$RM/prepare_multiseed_config.py" --src "$SRC" --dst "$cfg" --template "$template" \
        --expid "$expid" --seed "$seed" --model-root "$OUT/checkpoints" --num-workers "$NUM_WORKERS" >> "$STATUS" 2>&1
    note "START train $expid attempt=$attempt"
    local start end code
    start=$(date +%s)
    timeout --signal=TERM --kill-after=60s "$TIMEOUT" \
        "$PY" "$MTR_WD/run_expid.py" --config "$cfg" --expid "$expid" --gpu "$GPU" --skip_test > "$log" 2>&1
    code=$?
    end=$(date +%s)
    echo $((end - start)) > "$LOGS/$expid.time"
    echo "$code" > "$LOGS/$expid.exit"
    printf "%s\t%s\ttrain\t%s\t%s\t%s\t%s\t%s\n" "$(stamp)" "$phase" "$expid" "$attempt" "$code" $((end - start)) "$log" >> "$RUNS"
    note "EXIT train $expid attempt=$attempt code=$code seconds=$((end - start))"
}

run_train_batch() {
    local phase="$1"; shift
    local jobs=("$@")
    local spec template seed expid
    pids=()
    for spec in "${jobs[@]}"; do
        IFS=':' read -r template seed <<< "$spec"
        expid="${template}_s${seed}"
        if train_done "$expid"; then
            note "SKIP train $expid (already done)"
            continue
        fi
        wait_for_slot "$PARALLEL"
        ( train_one "$phase" "$template" "$seed" 1 ) &
        pids+=("$!")
    done
    wait_all
    # Ray retry once at PARALLEL=1
    local missing=()
    for spec in "${jobs[@]}"; do
        IFS=':' read -r template seed <<< "$spec"
        expid="${template}_s${seed}"
        train_done "$expid" && continue
        missing+=("$template:$seed")
    done
    if [[ "${#missing[@]}" -gt 0 ]]; then
        note "RETRY train serially n=${#missing[@]}"
        PARALLEL_SAVE=$PARALLEL
        PARALLEL=1
        for spec in "${missing[@]}"; do
            IFS=':' read -r template seed <<< "$spec"
            expid="${template}_s${seed}"
            mv -f "$LOGS/$expid.log" "$LOGS/$expid.attempt1.log" 2>/dev/null || true
            train_one "${phase}_retry" "$template" "$seed" 2
            train_done "$expid" || { note "FAILED train $expid after retry"; FAIL=1; }
        done
        PARALLEL=$PARALLEL_SAVE
    fi
}

eval_cvr_one() {
    local expid="$1" attempt="$2"
    local cfg="$OUT/configs/$expid" ck="$(ckpt_of "$expid")"
    local outj="$OUT/eval_valid/$expid.json" log="$OUT/eval_valid/logs/$expid.log"
    [[ -f "$outj" ]] && { note "SKIP eval-cvr $expid"; return 0; }
    [[ -f "$ck" ]] || { note "SKIP eval-cvr $expid (no ckpt)"; return 1; }
    note "START eval-cvr $expid attempt=$attempt"
    local start end code
    start=$(date +%s)
    timeout --signal=TERM --kill-after=60s "$EVAL_TIMEOUT" \
        "$PY" "$RM/eval_cvr_clicked.py" --gpu "$GPU" --stage valid \
        --checkpoint "$ck" --config "$cfg" --expid "$expid" \
        --workdir "$MTR_WD" --output "$outj" > "$log" 2>&1
    code=$?
    end=$(date +%s)
    printf "%s\teval\tcvr\t%s\t%s\t%s\t%s\t%s\n" "$(stamp)" "$expid" "$attempt" "$code" $((end - start)) "$log" >> "$RUNS"
    note "EXIT eval-cvr $expid attempt=$attempt code=$code seconds=$((end - start))"
    return "$code"
}

analyze_gates_one() {
    local tag="$1" expid="$2" seed="$3" attempt="$4"
    local ck="$(ckpt_of "$expid")" cfg="$OUT/configs/$expid"
    local md="$OUT/analysis/${tag}_s${seed}.md"
    local js="$OUT/analysis/${tag}_s${seed}.json"
    local png="$OUT/analysis/${tag}_s${seed}.png"
    local log="$OUT/analysis/logs/${tag}_s${seed}.log"
    [[ -f "$js" ]] && { note "SKIP gates $expid"; return 0; }
    [[ -f "$ck" ]] || { note "SKIP gates $expid (no ckpt)"; return 1; }
    note "START gates $expid attempt=$attempt"
    local start end code
    start=$(date +%s)
    timeout --signal=TERM --kill-after=60s "$EVAL_TIMEOUT" \
        "$PY" "$RM/analyze_gates.py" --gpu "$GPU" --stage valid \
        --checkpoint "$ck" --config "$cfg" --expid "$expid" --seed "$seed" \
        --token_names "$TOKENS" --max_samples 500000 \
        --output "$md" --figure "$png" --json "$js" --workdir "$MTR_WD" \
        > "$log" 2>&1
    code=$?
    end=$(date +%s)
    printf "%s\teval\tgates\t%s\t%s\t%s\t%s\t%s\n" "$(stamp)" "$expid" "$attempt" "$code" $((end - start)) "$log" >> "$RUNS"
    note "EXIT gates $expid attempt=$attempt code=$code seconds=$((end - start))"
    return "$code"
}

# -------- templates --------
TEMPLATES=(
  MTR_gc_g6_gate_ent0001
  MTR_gc_g6_gate_ent0003
  MTR_gc_g6_gate_ent001
  MTR_gc_g6_gate_T2
  MTR_gc_g6_gate_T4
  MTR_gc_g6_gate_ent001_T2
  MTR_gc_g6_residual_ent001
)
if [[ "$GC_EXTRA" == "1" ]]; then
  TEMPLATES+=(MTR_gc_g6_residual_ent0003)
fi

# Optionally retrain baseline instead of reuse
if [[ "$REUSE_BASELINE" != "1" ]]; then
  TEMPLATES=(MTR_gc_g6_gate "${TEMPLATES[@]}")
fi

JOBS=()
for t in "${TEMPLATES[@]}"; do
  for s in $SEEDS5; do
    JOBS+=("${t}:${s}")
  done
done

note "DRIVER START gpu=$GPU parallel=$PARALLEL out=$OUT py=$PY REUSE_BASELINE=$REUSE_BASELINE GC_EXTRA=$GC_EXTRA"
note "templates: ${TEMPLATES[*]}"
note "n_train_jobs=${#JOBS[@]}"
note "protocol: --skip_test; patience=3; epochs<=10; validation only"

# Verify knobs exist
if ! grep -q "gate_entropy_reg" "$MTR_WD/src/MTRankMixer.py"; then
  note "FATAL: gate_entropy_reg missing in MTRankMixer.py"; exit 2
fi
if ! grep -q "gate_temperature" "$MTR_WD/src/MTRankMixer.py"; then
  note "FATAL: gate_temperature missing in MTRankMixer.py"; exit 2
fi
if [[ ! -f "$SRC/model_config.yaml" ]]; then
  note "FATAL: missing $SRC/model_config.yaml"; exit 2
fi
note "knobs present; config src=$SRC"

# -------- Reuse rigor baseline --------
if [[ "$REUSE_BASELINE" == "1" ]]; then
  note "START baseline reuse from $RG_OUT"
  for s in $SEEDS5; do
    src_e="MTR_rg_g6_gate_s${s}"
    dst_e="MTR_gc_g6_gate_s${s}"
    src_ck="$RG_OUT/checkpoints/AliCCP_x1/${src_e}.model"
    dst_ck="$(ckpt_of "$dst_e")"
    if [[ -f "$src_ck" ]]; then
      if [[ ! -f "$dst_ck" ]]; then
        cp -f "$src_ck" "$dst_ck"
        cp -f "$RG_OUT/checkpoints/AliCCP_x1/${src_e}.log" "$OUT/checkpoints/AliCCP_x1/${dst_e}.log" 2>/dev/null || true
      fi
      if [[ ! -d "$OUT/configs/$dst_e" ]]; then
        # Build a gatecollapse-style config pointing at reused weights
        "$PY" "$RM/prepare_multiseed_config.py" --src "$SRC" --dst "$OUT/configs/$dst_e" \
          --template MTR_gc_g6_gate --expid "$dst_e" --seed "$s" \
          --model-root "$OUT/checkpoints" --num-workers "$NUM_WORKERS" >> "$STATUS" 2>&1
      fi
      # Reuse train log AUC by copying rigor log under our name if missing
      if [[ ! -f "$LOGS/$dst_e.log" && -f "$RG_OUT/logs/${src_e}.log" ]]; then
        cp -f "$RG_OUT/logs/${src_e}.log" "$LOGS/$dst_e.log"
        echo 0 > "$LOGS/$dst_e.exit"
        echo 0 > "$LOGS/$dst_e.time"
        echo reused_from_rg > "$OUT/baseline_reuse/$dst_e.txt"
      fi
      # Reuse existing gate analysis if present
      if [[ -f "$RG_OUT/analysis/g6_gate_s${s}.json" && ! -f "$OUT/analysis/g6_gate_s${s}.json" ]]; then
        cp -f "$RG_OUT/analysis/g6_gate_s${s}.json" "$OUT/analysis/g6_gate_s${s}.json"
        cp -f "$RG_OUT/analysis/g6_gate_s${s}.md" "$OUT/analysis/g6_gate_s${s}.md" 2>/dev/null || true
        # also tag name matching our template stem
        cp -f "$RG_OUT/analysis/g6_gate_s${s}.json" "$OUT/analysis/g6_gate_baseline_s${s}.json" 2>/dev/null || true
      fi
      note "REUSED baseline $src_e -> $dst_e"
    else
      note "MISSING rigor baseline $src_ck — will train MTR_gc_g6_gate_s${s}"
      JOBS=("MTR_gc_g6_gate:${s}" "${JOBS[@]}")
    fi
  done
fi

# -------- Train --------
run_train_batch A "${JOBS[@]}"

# -------- Gate analysis (all gated variants including baseline) --------
GATE_TAGS=()
# baseline
for s in $SEEDS5; do
  GATE_TAGS+=("g6_gate_baseline:MTR_gc_g6_gate_s${s}:${s}")
done
for t in "${TEMPLATES[@]}"; do
  # skip if we somehow have mean — all gated
  tag="${t#MTR_gc_}"
  for s in $SEEDS5; do
    GATE_TAGS+=("${tag}:${t}_s${s}:${s}")
  done
done

note "START gate analysis n=${#GATE_TAGS[@]}"
pids=()
for spec in "${GATE_TAGS[@]}"; do
  IFS=':' read -r tag expid seed <<< "$spec"
  [[ -f "$(ckpt_of "$expid")" ]] || continue
  [[ -f "$OUT/analysis/${tag}_s${seed}.json" ]] && continue
  wait_for_slot "$PARALLEL"
  ( analyze_gates_one "$tag" "$expid" "$seed" 1 || true ) &
  pids+=("$!")
done
wait_all
# retry missing
for spec in "${GATE_TAGS[@]}"; do
  IFS=':' read -r tag expid seed <<< "$spec"
  [[ -f "$(ckpt_of "$expid")" ]] || continue
  [[ -f "$OUT/analysis/${tag}_s${seed}.json" ]] && continue
  note "RETRY gates $expid"
  mv -f "$OUT/analysis/logs/${tag}_s${seed}.log" "$OUT/analysis/logs/${tag}_s${seed}.attempt1.log" 2>/dev/null || true
  analyze_gates_one "$tag" "$expid" "$seed" 2 || { note "FAILED gates $expid"; FAIL=1; }
done

# Aggregate per variant
note "START aggregate_gates"
for t in MTR_gc_g6_gate "${TEMPLATES[@]}"; do
  if [[ "$t" == "MTR_gc_g6_gate" ]]; then
    tag="g6_gate_baseline"
  else
    tag="${t#MTR_gc_}"
  fi
  inputs=()
  for s in $SEEDS5; do
    [[ -f "$OUT/analysis/${tag}_s${s}.json" ]] && inputs+=("$OUT/analysis/${tag}_s${s}.json")
  done
  if [[ "${#inputs[@]}" -ge 2 ]]; then
    "$PY" "$RM/aggregate_gates.py" --inputs "${inputs[@]}" \
      --output "$OUT/analysis/${tag}_seeds.md" >> "$STATUS" 2>&1 \
      && note "WROTE analysis/${tag}_seeds.md n=${#inputs[@]}" \
      || note "WARN aggregate_gates failed for $tag"
  fi
done

# -------- Clicked CVR on validation --------
EVALS=()
for t in MTR_gc_g6_gate "${TEMPLATES[@]}"; do
  for s in $SEEDS5; do
    EVALS+=("${t}_s${s}")
  done
done
note "START eval-cvr n=${#EVALS[@]}"
pids=()
for expid in "${EVALS[@]}"; do
  [[ -f "$(ckpt_of "$expid")" ]] || continue
  [[ -f "$OUT/eval_valid/$expid.json" ]] && continue
  wait_for_slot "$PARALLEL"
  ( eval_cvr_one "$expid" 1 || true ) &
  pids+=("$!")
done
wait_all
for expid in "${EVALS[@]}"; do
  [[ -f "$(ckpt_of "$expid")" ]] || continue
  [[ -f "$OUT/eval_valid/$expid.json" ]] && continue
  note "RETRY eval-cvr $expid"
  mv -f "$OUT/eval_valid/logs/$expid.log" "$OUT/eval_valid/logs/$expid.attempt1.log" 2>/dev/null || true
  eval_cvr_one "$expid" 2 || { note "FAILED eval-cvr $expid"; FAIL=1; }
done

# -------- Summarize --------
note "START summarize"
export OUT RG_OUT
"$PY" "$RM/gc_report.py" || FAIL=1
note "results.md written (or attempted)"

if [[ "$FAIL" -eq 0 ]]; then
  date '+%Y-%m-%d %H:%M:%S' > "$OUT/ALL_DONE"
  echo "fail=0" >> "$OUT/ALL_DONE"
  note "ALL_DONE"
else
  date '+%Y-%m-%d %H:%M:%S' > "$OUT/FINISHED_WITH_FAILURES"
  echo "fail=$FAIL" >> "$OUT/FINISHED_WITH_FAILURES"
  note "FINISHED_WITH_FAILURES FAIL=$FAIL (no ALL_DONE)"
fi
exit "$FAIL"
