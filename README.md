# Misconception-Faithfulness

![arXiv](https://img.shields.io/badge/arXiv-2605.12748-b31b1b.svg) [![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b8bbe.svg)](https://neurips.cc/Conferences/2026) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository for our NeurIPS 2026 paper, *"Simulating Students or Sycophantic Problem Solving? On Misconception Faithfulness of LLM Simulators."*

This repository provides:
- **Code** for feedback generation, student simulation, judging, and SFT/DPO/GRPO training;
- **Training and evaluation data**, including per-instance model responses and judge labels;
- **Reproduction scripts** for the main experimental results.

The reported Selective Flip Score (SFS) results can be reproduced **without GPUs or API calls** using the included model responses and judge labels.

## Repository structure

```text
misconception-faithfulness/
├── data/
│   ├── raw_data.csv
│   ├── oss120b_v3.csv
│   ├── eedi/
│   ├── results/              # Model responses and judge labels
│   └── sft/                  # Training data and train/test split
├── src/
│   ├── run_experiment.py     # Feedback, simulation, and judging
│   ├── train_sft.py
│   ├── train_dpo.py
│   ├── train_grpo.py
│   ├── generate_sft_data.py    # k=3 candidate generation + filtering
│   ├── generate_dpo_data.py    # DPO preference pair construction
│   ├── merge_lora.py
│   ├── make_split.py
│   └── analysis/
│       └── sfs.py            # SFS computation
├── scripts/
│   ├── 00_generate_feedback.sh
│   ├── 00b_build_sft_data.sh
│   ├── 00c_build_dpo_data.sh
│   ├── 01_baseline_eval.sh
│   ├── 02_train_sft.sh
│   ├── 03_train_grpo.sh
│   ├── 03b_train_dpo.sh
│   └── 04_eval_trained.sh
├── requirements.txt
└── README.md
```

## Citation

If you find our work useful, please cite our paper:

```bibtex
@inproceedings{do2026simulating,
  title     = {Simulating Students or Sycophantic Problem Solving? On Misconception Faithfulness of {LLM} Simulators},
  author    = {Do, Heejin and Sonkar, Shashank and Sachan, Mrinmaya},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://arxiv.org/abs/2605.12748}
}
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing reported results

Per-instance model responses and judge labels used for the reported results are included in `data/results/`. Therefore, SFS can be recomputed directly without rerunning model inference or training.

### Prompt-only baselines

```bash
python src/analysis/sfs.py --dataset malrule --table3
python src/analysis/sfs.py --dataset eedi --table3
```

### Post-training results

For example, to reproduce the Malrule results for Qwen3-4B:

```bash
# SFT
python src/analysis/sfs.py \
    --dataset malrule \
    --tag sft_qwen3_4b_b8g2 \
    --baseline-tag simulate_student_authentic_qwen4b

# GRPO
python src/analysis/sfs.py \
    --dataset malrule \
    --tag grpo_qwen3_4b_malrule_k3_1e5_b16_b01 \
    --baseline-tag simulate_student_authentic_qwen4b
```

The corresponding response and judgment CSVs can also be inspected directly to audit individual `(problem, feedback, response)` instances.

## Running the full evaluation

To rerun the prompt-only evaluation:

```bash
bash scripts/01_baseline_eval.sh
```

To evaluate a single model:

```bash
MODEL_ONLY=qwen4b bash scripts/01_baseline_eval.sh
```

This runs student simulation followed by response judging under the same configurations used in the paper.

## Synthetic training data

### Pipeline overview

For each `(problem, feedback_type)` in the train split, we generate `k = 3`
candidate student responses with `gpt-4o-mini`, judge each with `gpt-4o-mini`
into the 6-category taxonomy, and split by the ideal-category filter `C*(f)`:

| `feedback_type` | Ideal categories `C*(f)` |
|-----------------|---------------------------|
| `type1_correct` | `{correct_flip}` |
| `type2_wrong`   | `{constructive_pushback, passive_maintain, confusion}` |
| `type3_generic` | `{constructive_pushback, passive_maintain, confusion}` |

Candidates whose judge label falls inside `C*(f)` are kept (used as SFT
targets); the remainder are stored separately with their judge labels so
the filtering step can be audited independently.

### Files

Post-filter (SFT targets):

```text
data/sft/sft_train_malrule.jsonl
data/sft/sft_train_eedi.jsonl
data/sft/sft_train_both.jsonl
```

Pre-filter audit set (candidates that fell outside `C*(f)`, with judge label):

```text
data/sft/sft_rejected_malrule.jsonl
data/sft/sft_rejected_eedi.jsonl
```

The union `sft_train_{dataset}.jsonl ∪ sft_rejected_{dataset}.jsonl` reconstitutes
the complete pre-filter `k = 3` candidate pool. Filter rates: 0.2% (Malrule),
14.6% (EEDI); the higher EEDI rate reflects its finer-grained misconception
categories.

The fixed problem-level train/test split is provided in:

```text
data/sft/split.json
```

### Regenerating

```bash
export OPENAI_API_KEY=<your_key>
bash scripts/00b_build_sft_data.sh
```

This regenerates `sft_train_{dataset}.jsonl` and `sft_rejected_{dataset}.jsonl`
(up to sampling variability at temperature 0.8).

### SFT training

Example training command using the shipped SFT data:

```bash
bash scripts/02_train_sft.sh \
    Qwen/Qwen3-4B \
    data/sft/sft_train_malrule.jsonl \
    checkpoints/sft_qwen3_4b_malrule
```

## DPO

DPO starts from the corresponding merged SFT checkpoint. The preference pairs are provided in:

```text
data/sft/dpo_train_malrule_qwen.jsonl
data/sft/dpo_train_malrule_llama.jsonl
data/sft/dpo_train_eedi_qwen.jsonl
data/sft/dpo_train_eedi_llama.jsonl
```

Each row has the fields `prompt` / `chosen` / `rejected`, where `chosen` is a judge-filtered ideal student response (from `sft_train_{dataset}.jsonl`) and `rejected` is a hard-negative failure from the corresponding SFT model's evaluation whose judge label lies outside `C*(f)`.

To regenerate:

```bash
bash scripts/00c_build_dpo_data.sh          # all four combinations
bash scripts/00c_build_dpo_data.sh malrule  # single dataset
```

Example:

```bash
bash scripts/03b_train_dpo.sh \
    checkpoints/sft_qwen3_4b_malrule/merged \
    data/sft/dpo_train_malrule_qwen.jsonl \
    checkpoints/dpo_qwen3_4b_malrule \
    1e-7 0.1
```

## GRPO

GRPO starts from the corresponding SFT checkpoint.

Example:

```bash
bash scripts/03_train_grpo.sh \
    checkpoints/sft_qwen3_4b_malrule/merged \
    data/sft/sft_train_malrule.jsonl \
    data/results/responses_qwen80b_thinking.csv \
    checkpoints/grpo_qwen3_4b_malrule \
    1e-5 0.1
```

GRPO uses an online LLM judge (`gpt-5-nano`) for reward computation and therefore requires the corresponding API access.

## Evaluating trained models

```bash
bash scripts/04_eval_trained.sh \
    checkpoints/sft_qwen3_4b_malrule/merged \
    sft_qwen3_4b_malrule \
    malrule
```

SFS is computed as:

```text
SFS = F_T - (F_M + F_G) / 2
```

where `F_T`, `F_M`, and `F_G` are flip rates under targeted, misaligned, and generic feedback, respectively.

## Feedback generation

The generated feedback used in the experiments is included in:

```text
data/results/feedback.csv
```

To regenerate it:

```bash
bash scripts/00_generate_feedback.sh
```

## Reproducibility notes

For exact reproduction:
- use the provided `data/sft/split.json`;
- use the model and judge configurations specified in the scripts;
- use `gpt-oss-120b` as the response judge;
- use the included judgment CSVs to reproduce the reported SFS values without rerunning inference.

The included responses are stored separately from their judge labels, allowing individual outputs to be inspected or re-evaluated with an alternative judge.

## License

See `LICENSE`.
