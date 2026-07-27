#!/bin/bash
# SFT training with the paper's configuration (LoRA r=16, alpha=32,
# batch=8, grad-accum=2, lr=2e-4, 3 epochs, max_length=768).
#
# Trains a single (base model, dataset) combination. To reproduce Table
# 4/5, run this four times: Qwen3-4B/Llama-3.1-8B × malrule/eedi.
#
# Required env vars: HF_TOKEN, CACHE_DIR (see 00_generate_feedback.sh).
# Optional: WANDB_API_KEY (if using wandb tracking).
#
# Positional args:
#   $1: base model      e.g. Qwen/Qwen3-4B
#   $2: training data   e.g. data/sft/sft_train_malrule.jsonl
#   $3: output dir      e.g. checkpoints/sft_qwen3_4b_malrule

set -euo pipefail

MODEL=${1:?"usage: $0 <base_model> <train_file> <output_dir>"}
TRAIN=${2:?"usage: $0 <base_model> <train_file> <output_dir>"}
OUT=${3:?"usage: $0 <base_model> <train_file> <output_dir>"}

: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${CACHE_DIR:=$HOME/.cache/huggingface}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# Library pins used in the paper (April 2026 snapshot):
#   trl==1.3.0  peft==0.19.1  accelerate==1.13.0  transformers==5.5.4
# See requirements.txt for exact set. Newer trl breaks Qwen3 partial
# forward; older trl lacks --max-length / assistant_only_loss.

run() {
    if [ -n "${ENV_TOML:-}" ]; then
        srun --environment="$ENV_TOML" bash -lc "export HF_TOKEN=$HF_TOKEN; $*"
    else
        HF_TOKEN=$HF_TOKEN bash -lc "$*"
    fi
}

run "python src/train_sft.py \
    --model      $MODEL \
    --train-file $TRAIN \
    --output-dir $OUT \
    --cache-dir  $CACHE_DIR \
    --epochs     3 \
    --lr         2e-4 \
    --batch-size 8 \
    --grad-accum 2 \
    --max-length 768 \
    --lora-r     16 \
    --lora-alpha 32"

# After training, merge the LoRA adapter into the base model for evaluation:
run "python src/merge_lora.py \
    --base-model $MODEL \
    --adapter    $OUT/final \
    --output     $OUT/merged \
    --cache-dir  $CACHE_DIR"
