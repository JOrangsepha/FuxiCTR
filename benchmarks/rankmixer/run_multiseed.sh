#!/usr/bin/env bash
# Multi-seed early stopping on AliCCP_x1.
#
# Three models x three seeds (2025 2026 2027):
#   MT-RankMixer semantic, per-task gating     (MTRankMixer_aliccp_semantic_es)
#   MT-RankMixer semantic, shared mean pooling (MTRankMixer_aliccp_semantic_mean_es)
#   PLE                                            (PLE_aliccp_es)
#
# Settings match the 1-epoch run except the training budget:
#   epochs <= 6, early_stop_patience = 1, batch 8192, embedding 16, Adam 1e-3,
#   seed as above, no L2, no dropout.
#
# What "monitor: AUC" means here
#   MultiTaskModel.evaluate stores each task as click_AUC / conversion_AUC and
#   also writes the unweighted mean back under the bare key AUC.
#   Early stopping therefore watches the average of click AUC and conversion AUC.
#   Patience 1 stops on the first validation epoch that does not improve, then
#   fit() reloads the best checkpoint before the test evaluation.
#   FuxiCTR still has reduce_lr_on_plateau on by default. With patience 1 the
#   decayed learning rate is not used for the reported test metrics, because
#   those metrics are computed from the restored best checkpoint.
#
# Data
#   Uses the parquet cache already at data/AliCCP/AliCCP_x1/. This script does
#   not rebuild feature_map.json.
#
# Checkpoint kept for gate analysis (semantic, per-task gate, seed 2025):
#   model_zoo/multitask/MT_RankMixer/checkpoints/AliCCP_x1/MTRankMixer_aliccp_semantic_es_s2025.model
#
# Usage, from anywhere, with the repo at /root/autodl-tmp/FuxiCTR:
#   bash /root/autodl-tmp/FuxiCTR/benchmarks/rankmixer/run_multiseed.sh 0
#
# Optional environment:
#   TIMEOUT_SEC   per-job timeout in seconds (default 14400, four hours)
#   SEEDS         space-separated seeds (default "2025 2026 2027")
#   PYTHON        interpreter (default python, else python3)

set -u

GPU="${1:-0}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TIMEOUT_SEC="${TIMEOUT_SEC:-14400}"
SEEDS="${SEEDS:-2025 2026 2027}"
if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    PY=python3
fi

SRC="$ROOT/benchmarks/rankmixer/configs/multiseed"
RUNS="$ROOT/benchmarks/rankmixer/multiseed_runs"
LOGS="$RUNS/logs"
mkdir -p "$LOGS"

JOBS=(
    "MTRankMixer_aliccp_semantic_es|$ROOT/model_zoo/multitask/MT_RankMixer"
    "MTRankMixer_aliccp_semantic_mean_es|$ROOT/model_zoo/multitask/MT_RankMixer"
    "PLE_aliccp_es|$ROOT/model_zoo/multitask/PLE"
)

for spec in "${JOBS[@]}"; do
    IFS='|' read -r template workdir <<< "$spec"
    for seed in $SEEDS; do
        expid="${template}_s${seed}"
        cfg="$RUNS/configs/${expid}"
        log="$LOGS/${expid}.log"
        echo "========== ${expid} gpu=${GPU} timeout=${TIMEOUT_SEC}s =========="
        "$PY" "$ROOT/benchmarks/rankmixer/prepare_multiseed_config.py" \
            --src "$SRC" --dst "$cfg" --template "$template" \
            --expid "$expid" --seed "$seed"
        start=$(date +%s)
        set +e
        timeout --signal=TERM --kill-after=60s "$TIMEOUT_SEC" \
            "$PY" "$workdir/run_expid.py" --config "$cfg" --expid "$expid" --gpu "$GPU" \
            > "$log" 2>&1
        code=$?
        set -e
        end=$(date +%s)
        echo $((end - start)) > "$LOGS/${expid}.time"
        echo "$code" > "$LOGS/${expid}.exit"
        echo "exit ${code} after $((end - start))s -> ${log}"
    done
done

set +e
"$PY" "$ROOT/benchmarks/rankmixer/summarize_multiseed.py" \
    --logs "$LOGS" --csv "$ROOT/benchmarks/rankmixer/multiseed_summary.csv"
"$PY" "$ROOT/benchmarks/rankmixer/multiseed_stats.py" \
    --csv "$ROOT/benchmarks/rankmixer/multiseed_summary.csv"
