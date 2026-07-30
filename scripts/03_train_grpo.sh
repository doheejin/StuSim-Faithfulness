#!/bin/bash
# GRPO training. Requires a merged SFT checkpoint as the initial policy
# (produced by scripts/02_train_sft.sh + merge_lora.py).
#
# Paper configurations (per (base model, dataset)):
#   Qwen3-4B  × malrule : lr=1e-5, beta=0.1
#   Qwen3-4B  × eedi    : lr=1e-6, beta=0.04
#   Llama-8B  × malrule : lr=1e-6, beta=0.04
#   Llama-8B  × eedi    : lr=1e-5, beta=0.04
#
# Required env vars:
#   HF_TOKEN, CACHE_DIR   (see 00_generate_feedback.sh)
#   OPENAI_API_KEY        (online judge)
#
# Positional args:
#   $1: base model path (SFT merged checkpoint, e.g. checkpoints/sft_qwen3_4b_malrule/merged)
#   $2: sft training data (matching dataset, e.g. data/sft/sft_train_malrule.jsonl)
#   $3: meta-data responses CSV
#         data/results/responses_qwen80b_thinking.csv          (malrule)
#         data/results/eedi/responses_qwen80b_thinking.csv     (eedi)
#   $4: output dir (e.g. checkpoints/grpo_qwen3_4b_malrule)
#   $5: learning rate (e.g. 1e-5)
#   $6: beta (e.g. 0.1)

set -euo pipefail

MODEL=${1:?"usage: $0 <sft_merged> <sft_data> <meta_data> <output_dir> <lr> <beta>"}
SFT_DATA=${2:?}
META_DATA=${3:?}
OUT=${4:?}
LR=${5:?}
BETA=${6:?}

: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set (for online judge)}"
: "${CACHE_DIR:=$HOME/.cache/huggingface}"
: "${JUDGE_MODEL:=gpt-5-nano}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

run() {
    if [ -n "${ENV_TOML:-}" ]; then
        srun --environment="$ENV_TOML" bash -lc \
            "export HF_TOKEN=$HF_TOKEN OPENAI_API_KEY=$OPENAI_API_KEY; $*"
    else
        HF_TOKEN=$HF_TOKEN OPENAI_API_KEY=$OPENAI_API_KEY bash -lc "$*"
    fi
}

run "python src/train_grpo.py \
    --model      $MODEL \
    --sft-data   $SFT_DATA \
    --meta-data  $META_DATA \
    --output     $OUT \
    --cache-dir  $CACHE_DIR \
    --epochs     1 \
    --lr         $LR \
    --beta       $BETA \
    --num-gen    4 \
    --batch-size 4 \
    --grad-accum 2 \
    --max-prompt-length     512 \
    --max-completion-length 256 \
    --judge-model $JUDGE_MODEL \
    --report-to none"

run "python src/merge_lora.py \
    --base-model $MODEL \
    --adapter    $OUT/final \
    --output     $OUT/merged \
    --cache-dir  $CACHE_DIR"
