#!/usr/bin/env bash
# 1-epoch GPU comparison.
#   Single-task, Criteo_x1: RankMixer vs DCNv2 vs WuKong vs DNN
#   Multi-task, Ali-CCP:    MT-RankMixer vs MMoE vs PLE vs ShareBottom
#
# Usage:
#   bash benchmarks/rankmixer/run_gpu_benchmark.sh 0          # all eight runs on GPU 0
#   bash benchmarks/rankmixer/run_gpu_benchmark.sh 0 single   # Criteo only
#   bash benchmarks/rankmixer/run_gpu_benchmark.sh 0 multi    # Ali-CCP only
#
# Dataset preparation is documented in benchmarks/rankmixer/README.md.
# Configs are modest on purpose (embedding dim 16, batch 8192, 1 epoch).
# They are not the paper's 100M / 1B production settings.

set -euo pipefail

GPU="${1:-0}"
WHICH="${2:-all}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CFG="$ROOT/benchmarks/rankmixer/configs"

run() {
    local workdir="$1"
    local config="$2"
    local expid="$3"
    echo "========== ${expid} (gpu ${GPU}) =========="
    (
        cd "$workdir"
        python run_expid.py --config "$config" --expid "$expid" --gpu "$GPU"
    )
}

if [[ "$WHICH" == "all" || "$WHICH" == "single" ]]; then
    run "$ROOT/model_zoo/RankMixer" "$ROOT/model_zoo/RankMixer/config" RankMixer_criteo_x1
    run "$ROOT/model_zoo/DCNv2" "$CFG/DCNv2" DCNv2_criteo_x1
    run "$ROOT/model_zoo/WuKong" "$CFG/WuKong" WuKong_criteo_x1
    run "$ROOT/model_zoo/DNN/DNN_torch" "$CFG/DNN" DNN_criteo_x1
fi

if [[ "$WHICH" == "all" || "$WHICH" == "multi" ]]; then
    run "$ROOT/model_zoo/multitask/MT_RankMixer" "$ROOT/model_zoo/multitask/MT_RankMixer/config" MTRankMixer_aliccp
    run "$ROOT/model_zoo/multitask/MMoE" "$CFG/MMoE" MMoE_aliccp
    run "$ROOT/model_zoo/multitask/PLE" "$CFG/PLE" PLE_aliccp
    run "$ROOT/model_zoo/multitask/ShareBottom" "$CFG/ShareBottom" ShareBottom_aliccp
fi

echo "Done. Per-run metrics are appended to model_config.csv in each config directory."
