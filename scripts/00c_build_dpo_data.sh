#!/bin/bash
# Build DPO preference pairs from shipped SFT data + shipped SFT-model
# judgments.
#
# Chosen  (y+): ideal_response from data/sft/sft_train_{dataset}.jsonl
# Rejected (y-): SFT-model failures from
#                data/results/judgments_sft_{model}_{dataset}_b8g2.csv
#                (or eedi/ variant), where judge_category ∉ C*(f).
#
# Outputs (per dataset, per SFT model):
#   data/sft/dpo_train_{dataset}_{model}.jsonl
#
# NOTE: The shipped dpo_train_*.jsonl files were produced by this script;
# running it will regenerate byte-identical outputs given the same inputs.
#
# Positional arg (optional):
#   $1: dataset — one of {malrule, eedi, all}   (default: all)

set -euo pipefail

DATASET=${1:-all}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

build_one() {
    local ds=$1 model=$2
    echo "── Building DPO pairs: dataset=$ds model=$model ───────────"
    python src/generate_dpo_data.py \
        --dataset "$ds" --model "$model" --all-chosen --balance
}

if [ "$DATASET" = "malrule" ] || [ "$DATASET" = "all" ]; then
    build_one malrule qwen
    build_one malrule llama
fi
if [ "$DATASET" = "eedi" ] || [ "$DATASET" = "all" ]; then
    build_one eedi    qwen
    build_one eedi    llama
fi
