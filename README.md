# Misconception-Faithfulness

Anonymous artifact release for *"Simulating Students or Sycophantic Problem Solving? On Misconception Faithfulness of LLM Simulators."*

This repository provides:
- **Code** for feedback generation, student simulation, judging, and SFT/GRPO training;
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
│   ├── train_grpo.py
│   ├── merge_lora.py
│   ├── make_split.py
│   └── analysis/
│       └── sfs.py            # SFS computation
├── scripts/
│   ├── 00_generate_feedback.sh
│   ├── 01_baseline_eval.sh
│   ├── 02_train_sft.sh
│   ├── 03_train_grpo.sh
│   └── 04_eval_trained.sh
├── requirements.txt
└── README.md
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

## SFT

The synthetic training data are provided in:

```text
data/sft/sft_train_malrule.jsonl
data/sft/sft_train_eedi.jsonl
data/sft/sft_train_both.jsonl
```

The fixed problem-level train/test split is provided in:

```text
data/sft/split.json
```

Example training command:

```bash
bash scripts/02_train_sft.sh \
    Qwen/Qwen3-4B \
    data/sft/sft_train_malrule.jsonl \
    checkpoints/sft_qwen3_4b_malrule
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

GRPO uses an online LLM judge for reward computation and therefore requires the corresponding API/model access.

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