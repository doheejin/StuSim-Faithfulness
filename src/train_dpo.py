"""
DPO fine-tuning starting from SFT checkpoint.

Preferred (chosen)  : ideal student responses (judge-filtered)
Dispreferred (reject): baseline model responses with wrong judge category

Usage:
  python train_dpo.py \
    --model   checkpoints/sft_qwen3_4b_malrule/merged \
    --data    data/sft/dpo_train_malrule_qwen.jsonl \
    --output  checkpoints/dpo_qwen3_4b_malrule
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def apply_chat_template(tokenizer, messages, chat_kwargs, add_generation_prompt=False):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **chat_kwargs,
    )


def build_dataset(path, tokenizer, chat_kwargs):
    raw = load_jsonl(path)
    prompts  = [apply_chat_template(tokenizer, r["prompt"], chat_kwargs, add_generation_prompt=True) for r in raw]
    chosens  = [r["chosen"]   for r in raw]
    rejecteds = [r["rejected"] for r in raw]
    return Dataset.from_dict({"prompt": prompts, "chosen": chosens, "rejected": rejecteds})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="checkpoints/sft_qwen3_4b_malrule/merged")
    parser.add_argument("--data",        default="data/sft/dpo_train_malrule_qwen.jsonl")
    parser.add_argument("--output",      default="checkpoints/dpo_qwen3_4b_malrule")
    parser.add_argument("--cache-dir",   default=os.path.expanduser("~/.cache/huggingface"))
    parser.add_argument("--epochs",      type=int,   default=2)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--beta",        type=float, default=0.1)
    parser.add_argument("--batch-size",  type=int,   default=2)
    parser.add_argument("--grad-accum",  type=int,   default=8)
    parser.add_argument("--max-length",  type=int,   default=1024)
    parser.add_argument("--lora-r",      type=int,   default=16)
    parser.add_argument("--lora-alpha",  type=int,   default=32)
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.cache_dir, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO needs left-padding

    # Detect Qwen3 and disable thinking mode
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

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Building DPO dataset...")
    dataset = build_dataset(args.data, tokenizer, chat_kwargs)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    val_dataset   = split["test"]
    print(f"  Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── LoRA (fresh adapters on top of SFT-merged weights) ────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── DPO config ────────────────────────────────────────────────────────────
    dpo_args = DPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        beta=args.beta,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        max_length=args.max_length,
        remove_unused_columns=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("\nStarting DPO training...")
    trainer.train()

    final_path = Path(args.output) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nModel saved to {final_path}")


if __name__ == "__main__":
    main()
