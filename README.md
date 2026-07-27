# Misconception-Faithfulness

Anonymous code and data release for the paper *"Simulating Students or Sycophantic Problem Solving? On Misconception Faithfulness of LLM Simulators."* This repository contains everything needed to reproduce the paper's main results: the seven prompt-only baselines, and the SFT / GRPO post-training experiments on Qwen3-4B and Llama-3.1-8B.

## Contents

```
misconception-faithfulness/
├── data/
│   ├── raw_data.csv                # Malrule instances (from MalruleLib)
│   ├── oss120b_v3.csv              # Malrule taxonomy + descriptions
│   ├── eedi/                       # EEDI raw data (Feng et al.)
│   ├── results/                    # shipped per-model responses + judgments
│   │   ├── feedback.csv            # generated T1/T2/T3 feedback (Malrule)
│   │   ├── responses_simulate_student_authentic_{7 models}.csv  # Table 3 Reflective baseline
│   │   ├── judgments_simulate_student_authentic_{7 models}.csv  # Table 3 Reflective baseline (judged)
│   │   ├── responses_{sft,grpo}_{qwen3_4b,llama3_8b}_*.csv      # Table 4/5 trained-model outputs
│   │   ├── judgments_{sft,grpo}_{qwen3_4b,llama3_8b}_*.csv      # Table 4/5 trained-model judged
│   │   ├── responses_qwen80b_thinking.csv                       # GRPO --meta-data source
│   │   └── eedi/                                                # analogous set for EEDI
│   └── sft/
│       ├── split.json              # 800/200 train/test problem_ids
│       ├── sft_train_malrule.jsonl
│       ├── sft_train_eedi.jsonl
│       └── sft_train_both.jsonl
├── src/
│   ├── run_experiment.py           # feedback + simulate + judge pipeline
│   ├── train_sft.py                # SFT trainer
│   ├── train_grpo.py               # GRPO trainer (online reward)
│   ├── merge_lora.py               # LoRA → merged HF checkpoint
│   ├── make_split.py               # rebuild data/sft/split.json
│   ├── analysis/sfs.py             # SFS metric + Table 3 helper
│   └── llm_inference/              # vLLM / OpenAI wrappers + prompts
├── scripts/
│   ├── 00_generate_feedback.sh     # (optional) regenerate feedback.csv
│   ├── 01_baseline_eval.sh         # reproduce Table 3
│   ├── 02_train_sft.sh             # SFT training (paper cfg)
│   ├── 03_train_grpo.sh            # GRPO training (paper cfg)
│   └── 04_eval_trained.sh          # eval SFT/GRPO checkpoint (Table 4/5)
├── requirements.txt
├── LICENSE
└── README.md
```

## Definitions

**SFS (Sycophancy-Free Sensitivity)** = `F_T − (F_M + F_G) / 2`, where:

- `F_T` = flip rate under **type1 (correct)** feedback,
- `F_M` = flip rate under **type2 (misconception-targeted wrong)** feedback,
- `F_G` = flip rate under **type3 (generic wrong)** feedback,
- a "flip" = judge label ∈ {`correct_flip`, `sycophantic_flip`}.

A high SFS indicates the student flips *selectively* in response to correct feedback and not merely to any signal — i.e., misconception-faithful simulation.

## Installation

```bash
# Python 3.10+ recommended.
pip install -r requirements.txt
```

The exact library pins (`trl==1.3.0`, `peft==0.19.1`, `accelerate==1.13.0`, `transformers==5.5.4`) matter for training: newer `trl` breaks Qwen3 partial forward; older `trl` lacks `--max-length` / `assistant_only_loss`. For inference-only usage (baseline eval), recent versions work.

## Environment variables

All scripts read a small set of environment variables:

| Variable            | Purpose                                          | Required for               |
|---------------------|--------------------------------------------------|----------------------------|
| `HF_TOKEN`          | HuggingFace token (for gated Llama / Qwen)       | all scripts                |
| `CACHE_DIR`         | HuggingFace cache (default `$HOME/.cache/huggingface`) | all scripts        |
| `OPENAI_API_KEY`    | OpenAI key (online judge in GRPO)                | GRPO training              |
| `ENV_TOML`          | (optional) `--environment` toml for `srun`       | if you use containerized SLURM |

Set them once in your shell before invoking scripts:

```bash
export HF_TOKEN=<your_token>
export CACHE_DIR=$HOME/.cache/huggingface
export OPENAI_API_KEY=<your_key>   # GRPO only
```

## Reproducing Table 3 — prompt-only baselines

```bash
bash scripts/01_baseline_eval.sh
```

This iterates over all seven paper baselines (Qwen3-4B, Llama-3.1-8B, gpt-oss-20b, Llama-3.3-70B, Qwen3-Next-80B-Instruct, Qwen3-Next-80B-Thinking, gpt-oss-120b) × both datasets × both prompt variants (`simulate_student` = *Base*, `simulate_student_authentic` = *Reflective*). Each cell runs `simulate` then `judge`. The judge is `gpt-oss-120b` for consistency with the paper. After completion, `src/analysis/sfs.py --table3 --dataset {malrule,eedi}` prints the seven-row SFS table.

To run just one model, set `MODEL_ONLY=qwen4b bash scripts/01_baseline_eval.sh`.

Approximate compute per model on 4×H100:

| Model              | simulate (per prompt) | judge |
|--------------------|-----------------------|-------|
| 4–8B               | 10 min                | 20 min |
| 20B                | 15 min                | 20 min |
| 70–80B             | 30–45 min             | 20 min |
| 120B               | 45–60 min             | 20 min |

## Reproducing Table 4/5 — SFT + GRPO

### Step 1: Train SFT

```bash
bash scripts/02_train_sft.sh \
    Qwen/Qwen3-4B \
    data/sft/sft_train_malrule.jsonl \
    checkpoints/sft_qwen3_4b_malrule
```

Repeat for the four (model, dataset) combinations reported in the paper:

| Base model            | Training data                     | Output dir                       |
|-----------------------|-----------------------------------|----------------------------------|
| Qwen/Qwen3-4B         | `data/sft/sft_train_malrule.jsonl`| `checkpoints/sft_qwen3_4b_malrule` |
| Qwen/Qwen3-4B         | `data/sft/sft_train_eedi.jsonl`   | `checkpoints/sft_qwen3_4b_eedi`    |
| meta-llama/Llama-3.1-8B-Instruct | `data/sft/sft_train_malrule.jsonl` | `checkpoints/sft_llama3_8b_malrule` |
| meta-llama/Llama-3.1-8B-Instruct | `data/sft/sft_train_eedi.jsonl` | `checkpoints/sft_llama3_8b_eedi` |

Each script call trains LoRA (r=16, α=32, batch=8, grad-accum=2,
lr=2e-4, 3 epochs, max_length=768) and then merges the adapter into a
serveable checkpoint at `<output>/merged`.

### Step 2: Train GRPO

GRPO takes a merged SFT checkpoint as its starting policy. Paper
configurations (learning rate, KL β) per (base, dataset):

| Base × Dataset          | lr     | β    |
|-------------------------|--------|------|
| Qwen3-4B × malrule      | 1e-5   | 0.1  |
| Qwen3-4B × eedi         | 1e-6   | 0.04 |
| Llama-3.1-8B × malrule  | 1e-6   | 0.04 |
| Llama-3.1-8B × eedi     | 1e-5   | 0.04 |

Example:

```bash
bash scripts/03_train_grpo.sh \
    checkpoints/sft_qwen3_4b_malrule/merged \
    data/sft/sft_train_malrule.jsonl \
    data/results/responses_qwen80b_thinking.csv \
    checkpoints/grpo_qwen3_4b_malrule \
    1e-5 0.1
```

The reward is provided online by `gpt-4o-mini` (`OPENAI_API_KEY`
required). To swap the reward judge, set `JUDGE_MODEL=<hf-path>` in the
environment.

### Step 3: Evaluate the trained model

```bash
bash scripts/04_eval_trained.sh \
    checkpoints/sft_qwen3_4b_malrule/merged \
    sft_qwen3_4b_malrule \
    malrule
```

Runs simulate + judge, then prints SFS with the paper's formula
(matched-pair comparison against the prompt-only baseline).

## Reproducibility of the paper's tables

For each of the 22 model × dataset cells that back Table 3 (Reflective
baseline, seven models × two datasets) and Table 4/5 (four trained
models × two datasets, SFT + GRPO), we ship BOTH the per-instance
student response (`responses_*.csv`) AND the corresponding judge label
(`judgments_*.csv`). This lets a reviewer, without any GPU:

- re-compute SFS from the shipped judgments (matches the paper values
  within max Δ = 0.001, verified below);
- audit any specific (problem, feedback, student response) triple to
  check whether the judge's classification is reasonable;
- re-judge the shipped responses with an alternative judge model to
  test sensitivity of the SFS metric to the judge choice.

Re-computing SFS from these CSVs matches the paper values within max Δ =
0.001 (rounding), verified on all four (base, dataset) combinations of
Table 4/5:

| Row                                        | Paper SFS | CSV SFS |
|--------------------------------------------|----------:|--------:|
| Malrule Qwen3-4B  baseline (Reflective)    | +0.029    | +0.028  |
| Malrule Qwen3-4B  SFT                      | +0.584    | +0.584  |
| Malrule Qwen3-4B  GRPO (lr=1e-5, β=0.1)    | +0.609    | +0.608  |
| Malrule Llama-3.1-8B baseline              | +0.094    | +0.094  |
| Malrule Llama-3.1-8B SFT                   | +0.541    | +0.542  |
| Malrule Llama-3.1-8B GRPO (lr=1e-6, β=.04) | +0.570    | +0.570  |
| EEDI    Qwen3-4B  baseline                 | −0.005    | −0.006  |
| EEDI    Qwen3-4B  SFT                      | +0.265    | +0.264  |
| EEDI    Qwen3-4B  GRPO (lr=1e-6, β=.04)    | +0.336    | +0.335  |
| EEDI    Llama-3.1-8B baseline              | +0.115    | +0.115  |
| EEDI    Llama-3.1-8B SFT                   | +0.157    | +0.157  |
| EEDI    Llama-3.1-8B GRPO (lr=1e-5, β=.04) | +0.317    | +0.317  |

Reviewers with a Python interpreter but no GPU can reproduce the entire
Table 4/5 without running any inference:

```bash
# Reproduce Table 3 baseline SFS (all seven prompt-only models):
python src/analysis/sfs.py --dataset malrule --table3
python src/analysis/sfs.py --dataset eedi    --table3

# Reproduce Table 4/5 trained-model rows (SFT / GRPO):
python src/analysis/sfs.py --dataset malrule --tag sft_qwen3_4b_b8g2 \
    --baseline-tag simulate_student_authentic_qwen4b
python src/analysis/sfs.py --dataset malrule --tag grpo_qwen3_4b_malrule_k3_1e5_b16_b01 \
    --baseline-tag simulate_student_authentic_qwen4b
# ... same pattern for Llama and for eedi.
```

## Data provenance

- **Malrule**: `data/raw_data.csv` and `data/oss120b_v3.csv` are
  instantiated from MalruleLib (Chen et al., ACL Main 2026,
  <https://arxiv.org/abs/2601.03217>). Sampling is done by
  `src/run_experiment.py --step feedback`, which additionally uses
  `gpt-oss-120b` to generate the three feedback types. To regenerate,
  see `scripts/00_generate_feedback.sh`.
- **EEDI**: `data/eedi/train.csv` and `data/eedi/misconception_mapping.csv`
  are from the EEDI diagnostic questions dataset (Feng et al.).
- The `data/sft/split.json` file defines the 800/200 train/test problem
  split used by all experiments; rebuild via `python src/make_split.py`.
- `data/results/responses_qwen80b_thinking.csv` and its EEDI counterpart
  are the Qwen3-Next-80B-Thinking baseline responses, used as the source
  of prompts/completions for GRPO's reference distribution.

## Important

- The judge model (`gpt-oss-120b`) is required for the paper's reported numbers. Substituting a smaller judge will shift absolute SFS values; use the same judge across baseline and trained conditions.
