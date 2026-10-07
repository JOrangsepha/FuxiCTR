#!/usr/bin/env bash
# Follow-up suite, one detached driver. Phases run in order:
#   A1 PLE tuning (PLE_fu_wide, PLE_fu_lr5e4, PLE_fu_drop01) x 3 seeds, --skip_test
#   (B below), then A2 NORM g6_mean/g6_gate/g6_residual x 5 seeds (or $OUT/NORM_SET)
#   B  pick the best PLE config by mean validation avg AUC over seeds 2025-2027
#      (PLE_rg baseline is a candidate), then train its seeds 2028-2029
#   C  validation-only post-hoc work: param counts, eval_cvr_clicked.py on valid
#      for every checkpoint (rigor + follow-up), effective-gradient audit,
#      gate analysis for the NORM gate/residual runs
#   D  final test, once: frozen set = 7 rigor variants + 3 NORM variants +
#      selected PLE config, every available seed, saved best-valid checkpoints.
#      Guarded by FINAL_TEST_DONE and a per-expid marker in final_test/guard.
#   S  summary CSV + markdown, then $OUT/ALL_DONE
#
# Resumable: finished training runs (exit 0 + checkpoint) and existing eval
# JSONs are skipped. PHASES selects phases (default "A B C D S").
# A failed training/eval job (not a timeout) is retried once with PARALLEL=1.
#
# Usage (repo at /root/autodl-tmp/FuxiCTR):
#   nohup setsid env PATH=/root/miniconda3/bin:$PATH PYTHON=/root/miniconda3/bin/python \
#     bash benchmarks/rankmixer/run_followup.sh 0 > /root/autodl-tmp/fu_driver.log 2>&1 &

set -u
GPU="${1:-0}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RM="$ROOT/benchmarks/rankmixer"
OUT="${OUT:-/root/autodl-tmp/fu_out}"
RG="${RG_OUT:-/root/autodl-tmp/rg_out}"
PARALLEL="${PARALLEL:-2}"
TIMEOUT="${TIMEOUT:-90m}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-30m}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PHASES="${PHASES:-A B C D S}"
PY="${PYTHON:-python}"
SRC="$RM/configs/followup"
MTR_WD="$ROOT/model_zoo/multitask/MT_RankMixer"
PLE_WD="$ROOT/model_zoo/multitask/PLE"
TOKENS="user_id,user_profile,item_id,item_attr,cross,scenario"
SEEDS5="2025 2026 2027 2028 2029"
SEEDS3="2025 2026 2027"
LOGS="$OUT/logs"
STATUS="$OUT/STATUS"
RUNS="$OUT/runs.tsv"
FT="$OUT/final_test"
mkdir -p "$LOGS" "$OUT/configs" "$OUT/checkpoints/AliCCP_x1" "$OUT/eval_valid/logs" "$OUT/analysis/logs" "$FT/json" "$FT/logs"
[[ -f "$RUNS" ]] || printf "time\tphase\tkind\texpid\tattempt\texit\tseconds\tlog\n" > "$RUNS"

exec 9> "$OUT/.driver.lock"
if ! flock -n 9; then echo "another run_followup.sh holds $OUT/.driver.lock"; exit 1; fi

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
note() { echo "[$(stamp)] $*" | tee -a "$STATUS"; }
has_phase() { [[ " $PHASES " == *" $1 "* ]]; }
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

ckpt_of() { echo "$2/checkpoints/AliCCP_x1/$1.model"; }   # expid outroot
workdir_of() { if [[ "$1" == PLE* ]]; then echo "$PLE_WD"; else echo "$MTR_WD"; fi; }
outroot_of() { if [[ "$1" == *_rg_* || "$1" == PLE_rg_* ]]; then echo "$RG"; else echo "$OUT"; fi; }

train_done() { [[ -f "$LOGS/$1.exit" && "$(cat "$LOGS/$1.exit")" == "0" && -f "$(ckpt_of "$1" "$OUT")" ]]; }

# train_one PHASE TEMPLATE SEED ATTEMPT
train_one() {
    local phase="$1" template="$2" seed="$3" attempt="$4"
    local expid="${template}_s${seed}" cfg="$OUT/configs/${template}_s${seed}" log="$LOGS/${template}_s${seed}.log"
    local wd; wd="$(workdir_of "$template")"
    "$PY" "$RM/prepare_multiseed_config.py" --src "$SRC" --dst "$cfg" --template "$template" \
        --expid "$expid" --seed "$seed" --model-root "$OUT/checkpoints" --num-workers "$NUM_WORKERS" >> "$STATUS" 2>&1
    note "START train $expid attempt=$attempt"
    local start end code
    start=$(date +%s)
    timeout --signal=TERM --kill-after=60s "$TIMEOUT" \
        "$PY" "$wd/run_expid.py" --config "$cfg" --expid "$expid" --gpu "$GPU" --skip_test > "$log" 2>&1
    code=$?
    end=$(date +%s)
    echo $((end - start)) > "$LOGS/$expid.time"
    echo "$code" > "$LOGS/$expid.exit"
    printf "%s\t%s\ttrain\t%s\t%s\t%s\t%s\t%s\n" "$(stamp)" "$phase" "$expid" "$attempt" "$code" $((end - start)) "$log" >> "$RUNS"
    note "EXIT train $expid attempt=$attempt code=$code seconds=$((end - start))"
}

# run_train_batch PHASE "template:seed ..."
run_train_batch() {
    local phase="$1"; shift
    local item template seed
    for item in $1; do
        template="${item%%:*}"; seed="${item##*:}"
        if train_done "${template}_s${seed}"; then note "SKIP train ${template}_s${seed} (already done)"; continue; fi
        wait_for_slot "$PARALLEL"
        train_one "$phase" "$template" "$seed" 1 &
        pids+=("$!")
    done
    wait_all
    # one retry, serial (PARALLEL=1), for failed non-timeout runs
    for item in $1; do
        template="${item%%:*}"; seed="${item##*:}"
        local expid="${template}_s${seed}"
        train_done "$expid" && continue
        local code; code="$(cat "$LOGS/$expid.exit" 2>/dev/null || echo missing)"
        if [[ "$code" == "124" ]]; then note "NO-RETRY $expid timed out"; FAIL=1; continue; fi
        local why="other"
        grep -qE "RayTaskError|raylet|Raylet|ObjectStoreFull|RaySystemError|plasma|GetTimeoutError" "$LOGS/$expid.log" 2>/dev/null && why="ray-like"
        note "RETRY $expid serially (exit=$code, error=$why)"
        mv -f "$LOGS/$expid.log" "$LOGS/$expid.attempt1.log" 2>/dev/null
        train_one "$phase" "$template" "$seed" 2
        train_done "$expid" || { note "FAILED $expid after retry"; FAIL=1; }
    done
}

# eval_one STAGE EXPID ATTEMPT
eval_one() {
    local stage="$1" expid="$2" attempt="$3"
    local root; root="$(outroot_of "$expid")"
    local ckpt; ckpt="$(ckpt_of "$expid" "$root")"
    local cfg="$root/configs/$expid" wd; wd="$(workdir_of "$expid")"
    local json log extra=()
    if [[ "$stage" == "valid" ]]; then
        json="$OUT/eval_valid/$expid.json"; log="$OUT/eval_valid/logs/$expid.log"
    else
        json="$FT/json/$expid.json"; log="$FT/logs/$expid.log"
        extra=(--allow_test --guard_dir "$FT/guard")
    fi
    if [[ ! -f "$ckpt" || ! -d "$cfg" ]]; then
        note "EXIT eval-$stage $expid code=missing ckpt_or_cfg"; echo missing > "${log%.log}.exit"; return
    fi
    local start end code
    start=$(date +%s)
    timeout --signal=TERM --kill-after=60s "$EVAL_TIMEOUT" \
        "$PY" "$RM/eval_cvr_clicked.py" --gpu "$GPU" --stage "$stage" --checkpoint "$ckpt" \
        --config "$cfg" --expid "$expid" --workdir "$wd" --output "$json" "${extra[@]+"${extra[@]}"}" > "$log" 2>&1
    code=$?
    end=$(date +%s)
    echo "$code" > "${log%.log}.exit"
    printf "%s\t%s\teval-%s\t%s\t%s\t%s\t%s\t%s\n" "$(stamp)" "${PHASE_TAG:-?}" "$stage" "$expid" "$attempt" "$code" $((end - start)) "$log" >> "$RUNS"
    note "EXIT eval-$stage $expid attempt=$attempt code=$code seconds=$((end - start))"
}

# run_eval_batch STAGE "expid ..."
run_eval_batch() {
    local stage="$1" list="$2" expid jsondir
    if [[ "$stage" == "valid" ]]; then jsondir="$OUT/eval_valid"; else jsondir="$FT/json"; fi
    for expid in $list; do
        [[ -f "$jsondir/$expid.json" ]] && { note "SKIP eval-$stage $expid (json exists)"; continue; }
        if [[ "$stage" == "test" && -f "$FT/guard/$expid.test_read" ]]; then
            note "SKIP eval-test $expid: guard marker exists without json (needs manual review)"; FAIL=1; continue
        fi
        wait_for_slot "$PARALLEL"
        eval_one "$stage" "$expid" 1 &
        pids+=("$!")
    done
    wait_all
    for expid in $list; do
        [[ -f "$jsondir/$expid.json" ]] && continue
        local logdir="$OUT/eval_valid/logs"; [[ "$stage" == "test" ]] && logdir="$FT/logs"
        [[ "$(cat "$logdir/$expid.exit" 2>/dev/null)" == "missing" ]] && { FAIL=1; continue; }
        if [[ "$stage" == "test" ]]; then
            # The script produced no metrics, so no test number was observed. Allow one
            # retry and keep an audit trail of the cleared marker.
            if grep -q "REFUSED" "$logdir/$expid.log" 2>/dev/null; then FAIL=1; continue; fi
            note "RETRY eval-test $expid: crashed before writing metrics; clearing marker (logged in guard/RETRIED)"
            cat "$FT/guard/$expid.test_read" >> "$FT/guard/RETRIED" 2>/dev/null
            rm -f "$FT/guard/$expid.test_read"
        else
            note "RETRY eval-valid $expid serially"
        fi
        mv -f "$logdir/$expid.log" "$logdir/$expid.attempt1.log" 2>/dev/null
        eval_one "$stage" "$expid" 2
        [[ -f "$jsondir/$expid.json" ]] || { note "FAILED eval-$stage $expid after retry"; FAIL=1; }
    done
}

note "DRIVER START gpu=$GPU parallel=$PARALLEL phases='$PHASES' out=$OUT rg=$RG py=$PY"
note "protocol: training --skip_test; selection on validation only; test read once in phase D"

PLE_T="PLE_fu_wide PLE_fu_lr5e4 PLE_fu_drop01"

# Order: A1 PLE tuning -> B PLE selection + seeds 2028-2029 -> A2 loss-weighting runs.
# A2 reads $OUT/NORM_SET when it starts (space-separated template stems). Without
# that file it runs the per-batch NORM templates as specified. This leaves a window
# of ~80 min after launch to switch, e.g.
#   echo "MTR_fu_g6_mean_w10 MTR_fu_g6_gate_w10 MTR_fu_g6_residual_w10" > $OUT/NORM_SET
NORM_DEFAULT="MTR_fu_g6_mean_norm MTR_fu_g6_gate_norm MTR_fu_g6_residual_norm"

# ---------------- Phase A1 ----------------
if has_phase A; then
    PHASE_TAG=A1
    note "PHASE A1 start: PLE tuning x 3 seeds"
    items=""
    for t in $PLE_T; do for s in $SEEDS3; do items+="$t:$s "; done; done
    run_train_batch A1 "$items"
    note "PHASE A1 done"
fi

# ---------------- Phase B ----------------
BEST=""
if has_phase B; then
    PHASE_TAG=B
    BEST="$("$PY" "$RM/fu_report.py" select-ple --logs "$RG/logs" "$LOGS" --output "$OUT/ple_selection.json" 2>>"$STATUS" | tail -1)"
    note "PHASE B: selected PLE config = '$BEST' (see $OUT/ple_selection.json)"
    if [[ -z "$BEST" || "$BEST" == NO_ELIGIBLE* ]]; then
        note "PHASE B: no eligible PLE config"; FAIL=1; BEST=""
    elif [[ "$BEST" == "PLE_rg" ]]; then
        note "PHASE B: baseline PLE_rg wins; it already has seeds 2025-2029, nothing to train"
    else
        run_train_batch B "$BEST:2028 $BEST:2029"
    fi
    note "PHASE B done"
fi
if [[ -z "$BEST" && -f "$OUT/ple_selection.json" ]]; then
    BEST="$("$PY" -c "import json;print(json.load(open('$OUT/ple_selection.json'))['best'] or '')")"
fi

# ---------------- Phase A2 ----------------
if [[ -f "$OUT/norm_set_used" ]]; then
    NORM_T="$(cat "$OUT/norm_set_used")"
elif [[ -f "$OUT/NORM_SET" ]]; then
    NORM_T="$(tr -s ' \n' ' ' < "$OUT/NORM_SET")"
else
    NORM_T="$NORM_DEFAULT"
fi
if has_phase A; then
    PHASE_TAG=A2
    if [[ -f "$OUT/NORM_SET" && ! -f "$OUT/norm_set_used" ]]; then
        NORM_T="$(tr -s ' \n' ' ' < "$OUT/NORM_SET")"
        note "PHASE A2: control file $OUT/NORM_SET found"
    fi
    echo "$NORM_T" > "$OUT/norm_set_used"
    note "PHASE A2 start: loss-weighting templates [$NORM_T] x 5 seeds"
    items=""
    for t in $NORM_T; do for s in $SEEDS5; do items+="$t:$s "; done; done
    run_train_batch A2 "$items"
    note "PHASE A2 done"
fi
NORM_MEAN="$(for t in $NORM_T; do [[ "$t" == *g6_mean* ]] && echo "$t"; done | head -1)"

# ---------------- Phase C ----------------
if has_phase C; then
    PHASE_TAG=C
    note "PHASE C start: param counts, valid eval (clicked-only CVR), grad audit, gate analysis"
    pc="$OUT/param_counts.csv"; : > "$OUT/param_counts.raw"
    first=1
    for stem in MTR_rg_g3_mean MTR_rg_g6_mean MTR_rg_g6_residual MTR_rg_g6_gate MTR_rg_g6_random MTR_rg_g6_sequential PLE_rg \
                $NORM_T PLE_fu_wide PLE_fu_lr5e4 PLE_fu_drop01; do
        e="${stem}_s2025"; r="$(outroot_of "$e")"
        [[ -d "$r/configs/$e" ]] || continue
        hdr=(); [[ $first -eq 1 ]] && hdr=(--header); first=0
        CUDA_VISIBLE_DEVICES="" "$PY" "$RM/count_params.py" --config "$r/configs/$e" --expid "$e" \
            --workdir "$(workdir_of "$e")" --label "$stem" "${hdr[@]+"${hdr[@]}"}" >> "$OUT/param_counts.raw" 2>>"$OUT/analysis/logs/count_params.err" || FAIL=1
    done
    grep '^PARAMS,' "$OUT/param_counts.raw" | sed 's/^PARAMS,//' > "$pc"
    note "param counts -> $pc ($(($(wc -l < "$pc") - 1)) configs)"

    list=""
    for stem in MTR_rg_g3_mean MTR_rg_g6_mean MTR_rg_g6_residual MTR_rg_g6_gate MTR_rg_g6_random MTR_rg_g6_sequential PLE_rg \
                $NORM_T; do
        for s in $SEEDS5; do list+="${stem}_s${s} "; done
    done
    for t in $PLE_T; do for s in $SEEDS5; do [[ -f "$(ckpt_of "${t}_s${s}" "$OUT")" ]] && list+="${t}_s${s} "; done; done
    run_eval_batch valid "$list"

    note "START grad audit (EQ vs $NORM_MEAN g6_mean s2025, 16 valid batches)"
    "$PY" "$RM/audit_effective_grads.py" --gpu "$GPU" --batches 16 --output "$OUT/grad_audit_norm.md" \
        --item "EQ g6_mean s2025|$RG/configs/MTR_rg_g6_mean_s2025|MTR_rg_g6_mean_s2025|$RG/checkpoints/AliCCP_x1/MTR_rg_g6_mean_s2025.model" \
        --item "$NORM_MEAN s2025|$OUT/configs/${NORM_MEAN}_s2025|${NORM_MEAN}_s2025|$OUT/checkpoints/AliCCP_x1/${NORM_MEAN}_s2025.model" \
        > "$OUT/analysis/logs/grad_audit_norm.log" 2>&1
    code=$?; note "EXIT grad audit code=$code -> $OUT/grad_audit_norm.md"; [[ $code -ne 0 ]] && FAIL=1
    "$PY" "$RM/audit_trunk_grads.py" --gpu "$GPU" --stage valid --batches 4 \
        --checkpoint "$OUT/checkpoints/AliCCP_x1/${NORM_MEAN}_s2025.model" \
        --config "$OUT/configs/${NORM_MEAN}_s2025" --expid "${NORM_MEAN}_s2025" \
        --workdir "$MTR_WD" --output "$OUT/grad_audit_norm_raw.md" > "$OUT/analysis/logs/grad_audit_raw.log" 2>&1
    note "EXIT raw grad audit (original script, ${NORM_MEAN} ckpt) code=$?"

    for tmpl in $NORM_T; do
        [[ "$tmpl" == *g6_gate* || "$tmpl" == *g6_residual* ]] || continue
        tag="${tmpl#MTR_fu_}"
        for s in $SEEDS5; do
            e="${tmpl}_s${s}"; ck="$(ckpt_of "$e" "$OUT")"
            [[ -f "$OUT/analysis/${tag}_s${s}.json" ]] && continue
            [[ -f "$ck" ]] || { note "SKIP gates $e: no checkpoint"; continue; }
            wait_for_slot "$PARALLEL"
            ( timeout --signal=TERM --kill-after=60s "$EVAL_TIMEOUT" "$PY" "$RM/analyze_gates.py" --gpu "$GPU" --stage valid \
                --checkpoint "$ck" --config "$OUT/configs/$e" --expid "$e" --seed "$s" --token_names "$TOKENS" \
                --max_samples 500000 --output "$OUT/analysis/${tag}_s${s}.md" \
                --figure "$OUT/analysis/${tag}_s${s}.png" --json "$OUT/analysis/${tag}_s${s}.json" \
                > "$OUT/analysis/logs/${tag}_s${s}.log" 2>&1
              note "EXIT gates $e code=$?" ) &
            pids+=("$!")
        done
    done
    wait_all
    shopt -s nullglob
    for tmpl in $NORM_T; do
        [[ "$tmpl" == *g6_gate* || "$tmpl" == *g6_residual* ]] || continue
        tag="${tmpl#MTR_fu_}"
        js=("$OUT/analysis/${tag}_s"*.json)
        if [[ ${#js[@]} -gt 0 ]]; then
            "$PY" "$RM/aggregate_gates.py" --inputs "${js[@]}" --output "$OUT/analysis/${tag}_seeds.md" >> "$STATUS" 2>&1 || FAIL=1
            note "AGGREGATE ${tag} n=${#js[@]}"
        else
            note "AGGREGATE ${tag}: no json"; FAIL=1
        fi
    done
    shopt -u nullglob
    note "PHASE C done"
fi

# ---------------- Phase D ----------------
if has_phase D; then
    PHASE_TAG=D
    if [[ -f "$FT/FINAL_TEST_DONE" ]]; then
        note "PHASE D skipped: $FT/FINAL_TEST_DONE exists (test already read once)"
    elif [[ -z "$BEST" ]]; then
        note "PHASE D NOT RUN: no selected PLE config"; echo "no selected PLE config" > "$FT/FINAL_TEST_SKIPPED"; FAIL=1
    else
        frozen=""
        missing=0
        {
            echo "# Pre-registered frozen set, written $(stamp) before any test read."
            echo "# 7 rigor variants + 3 NORM variants + selected PLE config ($BEST), all available seeds."
        } > "$FT/FROZEN_SET.txt"
        for stem in MTR_rg_g3_mean MTR_rg_g6_mean MTR_rg_g6_residual MTR_rg_g6_gate MTR_rg_g6_random MTR_rg_g6_sequential PLE_rg \
                    $NORM_T; do
            for s in $SEEDS5; do frozen+="${stem}_s${s} "; done
        done
        if [[ "$BEST" != "PLE_rg" ]]; then for s in $SEEDS5; do frozen+="${BEST}_s${s} "; done; fi
        for e in $frozen; do
            ck="$(ckpt_of "$e" "$(outroot_of "$e")")"
            if [[ -f "$ck" ]]; then echo "$e|$ck" >> "$FT/FROZEN_SET.txt"; else echo "# MISSING $e" >> "$FT/FROZEN_SET.txt"; missing=$((missing + 1)); fi
        done
        # Do not burn the one-time test read on a broken suite: require >= 90% of the set.
        total=$(echo $frozen | wc -w)
        if [[ $missing -gt $((total / 10)) ]]; then
            note "PHASE D NOT RUN: $missing of $total frozen checkpoints missing"
            echo "missing=$missing total=$total" > "$FT/FINAL_TEST_SKIPPED"; FAIL=1
        else
            note "PHASE D start: final test once on $((total - missing)) checkpoints (missing=$missing)"
            date '+%Y-%m-%d %H:%M:%S' > "$FT/TEST_STARTED"
            run_eval_batch test "$(grep -v '^#' "$FT/FROZEN_SET.txt" | cut -d'|' -f1 | tr '\n' ' ')"
            { date '+%Y-%m-%d %H:%M:%S'; echo "evaluated=$(ls "$FT/json" | wc -l)"; } > "$FT/FINAL_TEST_DONE"
            note "PHASE D done -> $FT/FINAL_TEST_DONE"
        fi
    fi
fi

# ---------------- Summary ----------------
if has_phase S; then
    "$PY" "$RM/fu_report.py" summarize --logs "$RG/logs" "$LOGS" --out "$OUT" >> "$STATUS" 2>&1 || FAIL=1
    note "SUMMARY -> $OUT/followup_summary.csv $OUT/followup_results.md"
fi
{ date '+%Y-%m-%d %H:%M:%S'; echo "fail=$FAIL"; } > "$OUT/ALL_DONE"
note "ALL_DONE fail=$FAIL marker $OUT/ALL_DONE"
