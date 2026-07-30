#!/bin/bash
# DPO training. Requires a merged SFT checkpoint as the initial policy
# (produced by scripts/02_train_sft.sh + merge_lora.py).
#
# Preferred (chosen)  : ideal student responses (judge-filtered).
# Dispreferred (reject): baseline model responses with wrong judge category.
#
# Required env vars:
#   HF_TOKEN, CACHE_DIR   (see 00_generate_feedback.sh)
#
# Positional args:
#   $1: base model path (SFT merged checkpoint,
#        e.g. checkpoints/sft_qwen3_4b_malrule/merged)
#   $2: DPO training data (per (base, dataset) pair), e.g.
#        data/sft/dpo_train_malrule_qwen.jsonl
#        data/sft/dpo_train_malrule_llama.jsonl
#        data/sft/dpo_train_eedi_qwen.jsonl
#        data/sft/dpo_train_eedi_llama.jsonl
#   $3: output dir (e.g. checkpoints/dpo_qwen3_4b_malrule)
#   $4: learning rate (e.g. 1e-7)
#   $5: beta (e.g. 0.1)

set -euo pipefail

MODEL=${1:?"usage: $0 <sft_merged> <dpo_data> <output_dir> <lr> <beta>"}
DPO_DATA=${2:?}
OUT=${3:?}
LR=${4:?}
BETA=${5:?}

: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${CACHE_DIR:=$HOME/.cache/huggingface}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

run() {
    if [ -n "${ENV_TOML:-}" ]; then
        srun --environment="$ENV_TOML" bash -lc "export HF_TOKEN=$HF_TOKEN; $*"
    else
        HF_TOKEN=$HF_TOKEN bash -lc "$*"
    fi
}

run "python src/train_dpo.py \
    --model      $MODEL \
    --data       $DPO_DATA \
    --output     $OUT \
    --cache-dir  $CACHE_DIR \
    --epochs     1 \
    --lr         $LR \
    --beta       $BETA \
    --batch-size 8 \
    --grad-accum 2 \
    --max-length 1024 \
    --lora-r     16 \
    --lora-alpha 32"

# After training, merge the LoRA adapter into the base model for evaluation:
run "python src/merge_lora.py \
    --base-model $MODEL \
    --adapter    $OUT/final \
    --output     $OUT/merged \
    --cache-dir  $CACHE_DIR"
