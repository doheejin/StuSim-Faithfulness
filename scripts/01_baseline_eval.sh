#!/bin/bash
# Reproduce paper Table 3: Base and Reflective prompt SFS for seven
# prompt-only baseline models on Malrule and EEDI.
#
# Runs per model:
#   1. simulate  (with either Base or Reflective/authentic prompt)
#   2. judge     (with gpt-oss-120b as the labeler)
#
# Then prints SFS via src/analysis/sfs.py.
#
# Required env vars:
#   HF_TOKEN, CACHE_DIR (see 00_generate_feedback.sh).
#
# By default this script iterates over ALL seven paper models sequentially.
# To evaluate a single model, set MODEL_ONLY=<tag> (see MODELS array below).

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${CACHE_DIR:=$HOME/.cache/huggingface}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# Model tag -> HF model path (paper's seven prompt-only baselines).
declare -A MODEL_PATH=(
    [qwen4b]="Qwen/Qwen3-4B"
    [llama31_8b]="meta-llama/Llama-3.1-8B-Instruct"
    [oss20b]="openai/gpt-oss-20b"
    [llama33_70b]="meta-llama/Llama-3.3-70B-Instruct"
    [qwen80b]="Qwen/Qwen3-Next-80B-A3B-Instruct"
    [qwen80b_thinking]="Qwen/Qwen3-Next-80B-A3B-Thinking"
    [oss120b]="openai/gpt-oss-120b"
)

JUDGE_MODEL="openai/gpt-oss-120b"
NUM_GPUS=4

run() {
    if [ -n "${ENV_TOML:-}" ]; then
        srun --environment="$ENV_TOML" bash -lc "export HF_TOKEN=$HF_TOKEN; $*"
    else
        HF_TOKEN=$HF_TOKEN bash -lc "$*"
    fi
}

for TAG in "${!MODEL_PATH[@]}"; do
    if [ -n "${MODEL_ONLY:-}" ] && [ "$TAG" != "$MODEL_ONLY" ]; then continue; fi
    MODEL=${MODEL_PATH[$TAG]}

    for DATASET in malrule eedi; do
        for PROMPT in simulate_student simulate_student_authentic; do
            echo ""
            echo "=== [$TAG / $DATASET / $PROMPT] simulate ==="
            run "python src/run_experiment.py \
                --step simulate --dataset $DATASET \
                --simulate-prompt $PROMPT \
                --model $MODEL --model-tag $TAG \
                --backend vllm --num-gpus $NUM_GPUS \
                --cache-dir $CACHE_DIR"

            echo ""
            echo "=== [$TAG / $DATASET / $PROMPT] judge ==="
            run "python src/run_experiment.py \
                --step judge --dataset $DATASET \
                --simulate-prompt $PROMPT \
                --model $JUDGE_MODEL --model-tag $TAG \
                --backend vllm --num-gpus $NUM_GPUS \
                --cache-dir $CACHE_DIR"
        done
    done
done

echo ""
echo "=== Table 3: SFS across all seven baseline models ==="
python src/analysis/sfs.py --dataset malrule --table3
python src/analysis/sfs.py --dataset eedi --table3
