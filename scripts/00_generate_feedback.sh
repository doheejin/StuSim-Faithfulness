#!/bin/bash
# Generate three feedback types (type1_correct, type2_wrong, type3_generic)
# for the sampled Malrule / EEDI problems, using gpt-oss-120b as the
# feedback generator.
#
# Output: data/results/feedback.csv (malrule)
#         data/results/eedi/feedback.csv (eedi)
#
# NOTE: We include pre-generated feedback CSVs in data/results/ so you can
# skip this step and go directly to scripts/01_baseline_eval.sh.
#
# Required env vars:
#   HF_TOKEN         : HuggingFace token (for gated models)
#   CACHE_DIR        : local HF cache directory
#   ENV_TOML         : (optional) container spec for `srun --environment=...`
#                      Set to empty to run without srun wrapper.

set -euo pipefail

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

# ── Malrule ─────────────────────────────────────────────────────────
run "python src/run_experiment.py \
    --step feedback --dataset malrule \
    --num-problems 1000 \
    --model openai/gpt-oss-120b --backend vllm --num-gpus 4 \
    --cache-dir $CACHE_DIR"

# ── EEDI ────────────────────────────────────────────────────────────
run "python src/run_experiment.py \
    --step feedback --dataset eedi \
    --num-problems 1000 \
    --model openai/gpt-oss-120b --backend vllm --num-gpus 4 \
    --cache-dir $CACHE_DIR"
