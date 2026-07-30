#!/bin/bash
# Build synthetic SFT training data (Data Generation and Filtering stage).
#
# For each (problem, feedback_type) in the train split, generate k=3 candidate
# student responses with gpt-4o-mini, judge each with gpt-4o-mini, and split
# by the ideal-category filter C*(f):
#   type1_correct → {correct_flip}
#   type2_wrong   → {constructive_pushback, passive_maintain, confusion}
#   type3_generic → {constructive_pushback, passive_maintain, confusion}
#
# Outputs (per dataset):
#   data/sft/sft_train_{dataset}.jsonl     (kept — passes C*(f))
#   data/sft/sft_rejected_{dataset}.jsonl  (rejected — fails C*(f); carries
#                                           the judge_category label so the
#                                           filtering step can be audited)
# The union kept ∪ rejected is the complete pre-filter k=3 pool.
#
# NOTE: The shipped data/sft/sft_train_*.jsonl and data/sft/sft_rejected_*.jsonl
# were produced by this pipeline; running this script will regenerate them
# (up to sampling variability at temperature=0.8).
#
# Required env vars:
#   OPENAI_API_KEY   : OpenAI API key (both generator and judge are gpt-4o-mini)
#   ENV_TOML         : (optional) container spec for `srun --environment=...`

set -euo pipefail

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

run() {
    if [ -n "${ENV_TOML:-}" ]; then
        srun --environment="$ENV_TOML" bash -lc \
            "export OPENAI_API_KEY=$OPENAI_API_KEY; $*"
    else
        OPENAI_API_KEY=$OPENAI_API_KEY bash -lc "$*"
    fi
}

# ── Malrule ─────────────────────────────────────────────────────────
run "python src/generate_sft_data.py --dataset malrule --n-samples 3"

# ── EEDI ────────────────────────────────────────────────────────────
run "python src/generate_sft_data.py --dataset eedi    --n-samples 3"
