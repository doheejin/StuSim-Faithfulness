#!/bin/bash
# Evaluate a trained SFT or GRPO checkpoint (Table 4/5).
#
# Runs simulate + judge, then prints SFS.
#
# Required env vars: HF_TOKEN, CACHE_DIR (see 00_generate_feedback.sh).
#
# Positional args:
#   $1: merged checkpoint path (e.g. checkpoints/sft_qwen3_4b_malrule/merged)
#   $2: model tag              (e.g. sft_qwen3_4b_malrule) — used as output suffix
#   $3: dataset                (malrule|eedi)

set -euo pipefail

MODEL=${1:?"usage: $0 <merged_ckpt> <tag> <dataset>"}
TAG=${2:?}
DATASET=${3:?}

: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${CACHE_DIR:=$HOME/.cache/huggingface}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

JUDGE_MODEL="openai/gpt-oss-120b"
NUM_GPUS=4

run() {
    if [ -n "${ENV_TOML:-}" ]; then
        srun --environment="$ENV_TOML" bash -lc "export HF_TOKEN=$HF_TOKEN; $*"
    else
        HF_TOKEN=$HF_TOKEN bash -lc "$*"
    fi
}

echo "=== [$TAG / $DATASET] simulate ==="
run "python src/run_experiment.py \
    --step simulate --dataset $DATASET \
    --model $MODEL --model-tag $TAG \
    --backend vllm --num-gpus $NUM_GPUS \
    --cache-dir $CACHE_DIR"

echo ""
echo "=== [$TAG / $DATASET] judge ==="
run "python src/run_experiment.py \
    --step judge --dataset $DATASET \
    --model $JUDGE_MODEL --model-tag $TAG \
    --backend vllm --num-gpus $NUM_GPUS \
    --cache-dir $CACHE_DIR"

echo ""
echo "=== [$TAG / $DATASET] SFS ==="
python src/analysis/sfs.py \
    --dataset $DATASET \
    --tag $TAG \
    --baseline-tag "simulate_student_authentic_${TAG%%_*}"
