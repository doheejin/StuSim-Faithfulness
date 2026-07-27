"""
GRPO fine-tuning starting from SFT checkpoint.
Uses GPT-5 nano judge as online reward signal.

Reward is aligned with the Selective Flip Score (SFS):
  SFS = F_T - 0.5 * (F_M + F_G)
  where F_T/F_M/F_G = P(correct_flip | type1/type2/type3 feedback)

  Flip event = correct_flip OR sycophantic_flip (both: a' = a*)

  T1 correct  → +1.0  if flip (↑F_T),  -1.0 otherwise
  T2 wrong    → -0.5  if flip (↑F_M),  +0.5 otherwise   ← half magnitude so E[r] ∝ SFS
  T3 generic  → -0.5  if flip (↑F_G),  +0.5 otherwise
  unknown     →  0.0  (judge failure, neutral to avoid gradient noise)

Usage:
  export OPENAI_API_KEY=...
  python train_grpo.py \
    --model      checkpoints/sft_llama3_8b_malrule/final \
    --data       data/results/responses_qwen80b_thinking.csv \
    --split-file data/sft/split.json \
    --output     checkpoints/grpo_llama3_8b_malrule
"""

import argparse
import json
import math
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from openai import OpenAI
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from llm_inference.prompts.studentsim.judge_response import build_conversation as build_judge_conv

try:
    import wandb
except ImportError:
    wandb = None


SYSTEM_PROMPT = (
    "You are a student learning math. You worked on the problem below and "
    "believe your answer is correct. Your task is to respond to your teacher "
    "naturally — as a real student would."
)

# SFS-aligned reward values.
# Flip events (a' = a*): correct_flip, sycophantic_flip
# Non-flip: constructive_pushback, passive_maintain, confusion, different_wrong
#
# Per-sample rewards are designed so E[reward] ∝ SFS = F_T - 0.5*(F_M + F_G):
#   type1: flip → +1, no_flip → -1   (E = 2*F_T - 1)
#   type2/3: flip → -0.5, no_flip → +0.5   (E = 0.5 - F_M or F_G, half magnitude)
FLIP_CATEGORIES = {"correct_flip", "sycophantic_flip"}

REWARD = {
    "type1_correct": {"flip": +1.0,  "no_flip": -1.0},
    "type2_wrong":   {"flip": -0.5,  "no_flip": +0.5},
    "type3_generic": {"flip": -0.5,  "no_flip": +0.5},
}
REWARD_UNKNOWN = 0.0
JUDGE_MODEL = "gpt-5-nano"


def _shorten(text: str, limit: int = 600) -> str:
    text = str(text).replace("\n", "\\n")
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ── SFS tracker ────────────────────────────────────────────────────────────────

class SFSTracker:
    """Thread-safe running estimates of F_T, F_M, F_G and SFS."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counts = defaultdict(lambda: {"flip": 0, "total": 0})
        self._unknown = 0
        self._error_counts = defaultdict(int)

    def update(self, feedback_type: str, is_flip: bool):
        with self._lock:
            self._counts[feedback_type]["total"] += 1
            if is_flip:
                self._counts[feedback_type]["flip"] += 1

    def record_unknown(self, reason: str):
        with self._lock:
            self._unknown += 1
            self._error_counts[reason] += 1

    def compute(self) -> dict:
        with self._lock:
            def rate(key):
                c = self._counts[key]
                return c["flip"] / c["total"] if c["total"] > 0 else float("nan")

            f_t = rate("type1_correct")
            f_m = rate("type2_wrong")
            f_g = rate("type3_generic")

            import math
            if any(math.isnan(x) for x in [f_t, f_m, f_g]):
                sfs = float("nan")
                delta_f = float("nan")
            else:
                sfs = f_t - 0.5 * (f_m + f_g)
                delta_f = f_t - max(f_m, f_g)

            return {
                "sfs/F_T": f_t, "sfs/F_M": f_m, "sfs/F_G": f_g,
                "sfs/SFS": sfs, "sfs/delta_F": delta_f,
                "sfs/n_T": self._counts["type1_correct"]["total"],
                "sfs/n_M": self._counts["type2_wrong"]["total"],
                "sfs/n_G": self._counts["type3_generic"]["total"],
                "judge/unknown": self._unknown,
                "judge/error_counts": dict(self._error_counts),
            }

    def reset(self):
        with self._lock:
            self._counts.clear()
            self._unknown = 0
            self._error_counts.clear()


# ── Reward function ────────────────────────────────────────────────────────────

def _supports_reasoning_effort(model_name: str) -> bool:
    model_name = model_name.lower()
    return model_name.startswith(("gpt-5", "o1", "o3", "o4"))


def _judge_one(args):
    client, judge_model, completion, feedback_type, problem_text, student_answer, correct_answer, feedback_text, tracker = args
    conv = build_judge_conv(
        problem_text=problem_text,
        student_answer=student_answer,
        correct_answer=correct_answer,
        feedback_text=feedback_text,
        student_response=completion,
    )
    try:
        request = {
            "model": judge_model,
            "messages": conv,
            "max_completion_tokens": 200,
            "response_format": {"type": "json_object"},
        }
        if _supports_reasoning_effort(judge_model):
            request["reasoning_effort"] = "minimal"
        r = client.chat.completions.create(**request)
        category = json.loads(r.choices[0].message.content).get("category", "unknown")
    except Exception as e:
        tracker.record_unknown(f"{type(e).__name__}: {str(e)[:200]}")
        return REWARD_UNKNOWN

    if category == "unknown":
        tracker.record_unknown("unknown_category")
        return REWARD_UNKNOWN

    is_flip = category in FLIP_CATEGORIES
    tracker.update(feedback_type, is_flip)

    rewards_for_type = REWARD.get(feedback_type)
    if rewards_for_type is None:
        return REWARD_UNKNOWN
    return rewards_for_type["flip"] if is_flip else rewards_for_type["no_flip"]


def make_reward_fn(
    openai_api_key: str,
    judge_model: str = JUDGE_MODEL,
    n_workers: int = 8,
    print_completions: int = 2,
    print_completions_every: int = 10,
):
    client = OpenAI(api_key=openai_api_key)
    tracker = SFSTracker()
    reward_call = 0
    print(f"Using OpenAI judge model: {judge_model}")

    def reward_fn(prompts, completions, completion_ids, **kwargs):
        nonlocal reward_call
        reward_call += 1
        args_list = [
            (client, judge_model, completion, ft, pt, sa, ca, fbt, tracker)
            for completion, ft, pt, sa, ca, fbt in zip(
                completions,
                kwargs["feedback_type"],
                kwargs["problem_text"],
                kwargs["student_answer"],
                kwargs["correct_answer"],
                kwargs["feedback_text"],
            )
        ]
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            rewards = list(pool.map(_judge_one, args_list))

        # Log SFS stats every call (GRPOTrainer will handle logging_steps)
        sfs_stats = tracker.compute()
        print(
            f"  [SFS] F_T={sfs_stats['sfs/F_T']:.3f}  "
            f"F_M={sfs_stats['sfs/F_M']:.3f}  "
            f"F_G={sfs_stats['sfs/F_G']:.3f}  "
            f"SFS={sfs_stats['sfs/SFS']:.3f}  "
            f"ΔF={sfs_stats['sfs/delta_F']:.3f}  "
            f"(n={sfs_stats['sfs/n_T']}/{sfs_stats['sfs/n_M']}/{sfs_stats['sfs/n_G']}, "
            f"unknown={sfs_stats['judge/unknown']})"
        )
        if sfs_stats["judge/error_counts"]:
            top_errors = sorted(
                sfs_stats["judge/error_counts"].items(),
                key=lambda item: item[1],
                reverse=True,
            )[:3]
            print(f"  [Judge errors] {top_errors}")

        sample_rows = []
        should_print_samples = (
            print_completions > 0
            and print_completions_every > 0
            and reward_call % print_completions_every == 0
        )
        if should_print_samples:
            print(f"  [Completion samples] reward_call={reward_call}")
            for i, (completion, reward, feedback_type) in enumerate(
                list(zip(completions, rewards, kwargs["feedback_type"]))[:print_completions],
                start=1,
            ):
                row = {
                    "reward_call": reward_call,
                    "sample": i,
                    "feedback_type": feedback_type,
                    "reward": reward,
                    "completion": _shorten(completion),
                }
                sample_rows.append(row)
                print(
                    f"    #{i} feedback_type={feedback_type} reward={reward:.2f} "
                    f"completion={row['completion']}"
                )

        if wandb is not None and wandb.run is not None:
            reward_mean = sum(rewards) / len(rewards) if rewards else float("nan")
            reward_std = (
                math.sqrt(sum((r - reward_mean) ** 2 for r in rewards) / len(rewards))
                if rewards else float("nan")
            )
            wandb_log = {
                key: value
                for key, value in sfs_stats.items()
                if key != "judge/error_counts"
            }
            wandb_log.update({
                "reward/mean": reward_mean,
                "reward/std": reward_std,
                "reward/unknown_fraction": (
                    rewards.count(REWARD_UNKNOWN) / len(rewards) if rewards else float("nan")
                ),
            })
            if sfs_stats["judge/error_counts"]:
                top_error, top_count = max(
                    sfs_stats["judge/error_counts"].items(),
                    key=lambda item: item[1],
                )
                wandb_log["judge/top_error"] = top_error
                wandb_log["judge/top_error_count"] = top_count
            if sample_rows:
                wandb_log["completion_samples"] = wandb.Table(
                    columns=["reward_call", "sample", "feedback_type", "reward", "completion"],
                    data=[
                        [
                            row["reward_call"],
                            row["sample"],
                            row["feedback_type"],
                            row["reward"],
                            row["completion"],
                        ]
                        for row in sample_rows
                    ],
                )
            wandb.log(wandb_log)

        return rewards

    reward_fn.__name__ = "sfs_judge_reward"
    return reward_fn


# ── Dataset ────────────────────────────────────────────────────────────────────

def build_dataset(sft_path, meta_path, tokenizer, chat_kwargs, dedup: bool = False):
    # Primary source: SFT training data (same split used for SFT/DPO)
    sft = [json.loads(l) for l in open(sft_path)]
    print(f"  Loaded {len(sft)} examples from {sft_path}")

    # Deduplicate to 1 sample per (problem_id, feedback_type) if requested
    if dedup:
        seen, deduped = set(), []
        for ex in sft:
            key = (ex["problem_id"], ex["feedback_type"])
            if key not in seen:
                seen.add(key)
                deduped.append(ex)
        print(f"  Deduped: {len(sft)} → {len(deduped)} (1 per problem×feedback_type)")
        sft = deduped

    # correct_answer is not stored in sft_train.jsonl — join from problem metadata
    meta = pd.read_csv(meta_path, usecols=["problem_id", "feedback_type", "correct_answer"])
    meta_map = {
        (r["problem_id"], r["feedback_type"]): str(r["correct_answer"])
        for _, r in meta.iterrows()
    }

    records, skipped = [], 0
    for ex in sft:
        key = (ex["problem_id"], ex["feedback_type"])
        correct_answer = meta_map.get(key)
        if correct_answer is None:
            skipped += 1
            continue

        # messages already built in SFT format — apply chat template to get prompt string
        prompt = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=True, **chat_kwargs
        )

        # Extract fields from user message for judge calls
        user_content = ex["messages"][1]["content"]
        records.append({
            "prompt":         prompt,
            "feedback_type":  ex["feedback_type"],
            "problem_text":   _parse_field(user_content, "Problem"),
            "student_answer": _parse_field(user_content, "Your answer"),
            "correct_answer": correct_answer,
            "feedback_text":  _parse_feedback(user_content),
        })

    if skipped:
        print(f"  Skipped {skipped} examples (no correct_answer in metadata)")
    print(f"  Final: {len(records)} GRPO prompts")

    dataset = Dataset.from_list(records)
    split_ds = dataset.train_test_split(test_size=0.05, seed=42)
    return split_ds["train"], split_ds["test"]


def _parse_field(content: str, field: str) -> str:
    """Extract **Field**: value from user message."""
    import re
    pattern = rf"\*\*{field}\*\*:\s*(.+?)(?=\n\*\*|\n\nYour teacher)"
    m = re.search(pattern, content, re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_feedback(content: str) -> str:
    """Extract feedback text from 'Your teacher says: "..."'."""
    import re
    m = re.search(r'Your teacher says: "(.+?)"', content, re.DOTALL)
    return m.group(1).strip() if m else ""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="checkpoints/sft_llama3_8b_malrule/merged")
    parser.add_argument("--sft-data",     default="data/sft/sft_train_malrule.jsonl",
                        help="SFT training data (same split as SFT/DPO)")
    parser.add_argument("--meta-data",    default="data/results/responses_qwen80b_thinking.csv",
                        help="Problem metadata CSV (used only for correct_answer lookup)")
    parser.add_argument("--output",       default="checkpoints/grpo_llama3_8b_malrule")
    parser.add_argument("--resume-from-checkpoint", default=None,
                        help="Resume training from this Trainer checkpoint")
    parser.add_argument("--cache-dir",    default=os.path.expanduser("~/.cache/huggingface"))
    parser.add_argument("--epochs",       type=int,   default=2)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--beta",         type=float, default=0.04,
                        help="KL penalty coefficient")
    parser.add_argument("--epsilon",      type=float, default=0.2,
                        help="GRPO clip ratio")
    parser.add_argument("--num-gen",      type=int,   default=4,
                        help="Group size G (completions per prompt)")
    parser.add_argument("--batch-size",   type=int,   default=1)
    parser.add_argument("--grad-accum",   type=int,   default=8)
    parser.add_argument("--max-prompt-length",     type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--lora-r",       type=int,   default=16)
    parser.add_argument("--lora-alpha",   type=int,   default=32)
    parser.add_argument("--judge-workers",type=int,   default=8,
                        help="Parallel threads for OpenAI judge calls")
    parser.add_argument("--judge-model",  default=JUDGE_MODEL,
                        help="OpenAI model to use for online reward judging")
    parser.add_argument("--report-to",    default="none",
                        help="Trainer reporting backend, e.g. 'wandb' or 'none'")
    parser.add_argument("--run-name",     default=None,
                        help="Run name for experiment trackers such as W&B")
    parser.add_argument("--log-completions", action="store_true",
                        help="Save sampled completions through TRL logging")
    parser.add_argument("--print-completions", type=int, default=2,
                        help="Print this many completion samples without saving TRL parquet files")
    parser.add_argument("--print-completions-every", type=int, default=10,
                        help="Print completion samples every N reward calls")
    parser.add_argument("--eval-strategy", choices=["no", "steps", "epoch"], default="epoch",
                        help="Evaluation strategy")
    parser.add_argument("--eval-steps",    type=int,   default=500,
                        help="Evaluate every N update steps when --eval-strategy steps")
    parser.add_argument("--save-strategy", choices=["no", "steps", "epoch"], default="epoch",
                        help="Checkpoint save strategy")
    parser.add_argument("--save-steps",    type=int,   default=500,
                        help="Save every N update steps when --save-strategy steps")
    parser.add_argument("--save-total-limit", type=int, default=2,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--dedup",      action="store_true",
                        help="Use 1 sample per (problem_id, feedback_type) to reduce training time")
    args = parser.parse_args()

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.cache_dir, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    template = getattr(tokenizer, "chat_template", "") or ""
    is_qwen3 = "<|im_start|>" in template and "thinking" in template.lower()
    chat_kwargs = {"enable_thinking": False} if is_qwen3 else {}
    print(f"  chat_kwargs: {chat_kwargs}")

    if chat_kwargs:
        _orig = tokenizer.apply_chat_template
        def _patched(*a, **kw):
            kw.update(chat_kwargs)
            return _orig(*a, **kw)
        tokenizer.apply_chat_template = _patched

    # ── Dataset ────────────────────────────────────────────────────────────────
    print("Building GRPO dataset...")
    train_dataset, eval_dataset = build_dataset(
        args.sft_data, args.meta_data, tokenizer, chat_kwargs, dedup=args.dedup
    )
    print(f"  Train: {len(train_dataset)}  Val: {len(eval_dataset)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── LoRA ───────────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # ── Reward function ────────────────────────────────────────────────────────
    reward_fn = make_reward_fn(
        openai_key,
        judge_model=args.judge_model,
        n_workers=args.judge_workers,
        print_completions=args.print_completions,
        print_completions_every=args.print_completions_every,
    )

    # ── GRPO config ────────────────────────────────────────────────────────────
    grpo_args = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
        report_to=args.report_to,
        run_name=args.run_name,
        # GRPO-specific
        num_generations=args.num_gen,
        beta=args.beta,
        epsilon=args.epsilon,
        max_completion_length=args.max_completion_length,
        log_completions=args.log_completions,
        num_completions_to_print=2 if args.log_completions else 0,
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_cfg,
    )

    if args.resume_from_checkpoint:
        print(f"\nStarting GRPO training (G={args.num_gen}, beta={args.beta}) from {args.resume_from_checkpoint}...")
    else:
        print(f"\nStarting GRPO training (G={args.num_gen}, beta={args.beta})...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_path = Path(args.output) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nModel saved to {final_path}")


if __name__ == "__main__":
    main()
