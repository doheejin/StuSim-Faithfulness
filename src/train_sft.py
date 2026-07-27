"""
SFT fine-tuning for ideal student simulator behavior.

Trains a model to:
  - T1 (correct feedback)  → correct_flip    (engage with feedback, arrive at correct answer)
  - T2 (wrong feedback)    → constructive_pushback / passive_maintain
  - T3 (generic feedback)  → confusion / constructive_pushback

Usage:
  python train_sft.py --model meta-llama/Llama-3.1-8B-Instruct \
                      --train-file data/sft/sft_train_malrule.jsonl \
                      --val-file   data/sft/sft_val_malrule.jsonl \
                      --output-dir checkpoints/sft_llama8b_malrule
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


GEMMA_CHAT_TEMPLATE = """{%- for message in messages %}
{%- if message['role'] == 'user' %}
{{ '<start_of_turn>user\n' }}{{ message['content'] | trim }}{{ '<end_of_turn>\n' }}
{%- elif message['role'] == 'assistant' %}
{{ '<start_of_turn>model\n' }}{% generation %}{{ message['content'] | trim }}{% endgeneration %}{{ '<end_of_turn>\n' }}
{%- else %}
{{ '<start_of_turn>user\n' }}{{ message['content'] | trim }}{{ '<end_of_turn>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{ '<start_of_turn>model\n' }}
{%- endif %}"""


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def is_gemma_model(model_name: str) -> bool:
    return "gemma" in model_name.lower()


def get_model_cfg(model_name: str, tokenizer) -> dict:
    """Infer chat-format config from tokenizer template + model name."""
    template = getattr(tokenizer, "chat_template", "") or ""
    if is_gemma_model(model_name):
        return {
            "response_template": "<start_of_turn>model\n",
            "chat_kwargs": {},
            "chat_template": GEMMA_CHAT_TEMPLATE,
        }
    elif "<|im_start|>" in template:
        # ChatML family (Qwen2, Qwen3, …)
        is_qwen3 = "qwen3" in model_name.lower()
        return {
            "response_template": "<|im_start|>assistant\n",
            "chat_kwargs": {"enable_thinking": False} if is_qwen3 else {},
        }
    else:
        # Llama-3 / default header format
        import trl as _trl, os as _os
        _tpl = _os.path.join(_os.path.dirname(_trl.__file__), "chat_templates", "llama3_training.jinja")
        return {
            "response_template": "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "chat_kwargs": {},
            "chat_template": open(_tpl).read() if _os.path.exists(_tpl) else None,
        }


def normalize_messages_for_model(messages, model_name: str):
    """Adapt unsupported roles before applying model-specific chat templates."""
    if not is_gemma_model(model_name):
        return messages

    normalized = []
    pending_system = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            pending_system.append(content)
            continue
        if pending_system and role == "user":
            content = "\n\n".join(pending_system + [content])
            pending_system = []
        elif pending_system:
            normalized.append({"role": "user", "content": "\n\n".join(pending_system)})
            pending_system = []
        normalized.append({"role": role, "content": content})

    if pending_system:
        normalized.append({"role": "user", "content": "\n\n".join(pending_system)})
    return normalized


def build_dataset(path, model_name: str):
    raw = load_jsonl(path)
    messages_list = [
        normalize_messages_for_model(
            ex["messages"] + [{"role": "assistant", "content": ex["ideal_response"]}],
            model_name,
        )
        for ex in raw
    ]
    return Dataset.from_dict({"messages": messages_list})


class Gemma3TokenTypeCollator:
    """Add the text-only token_type_ids required by Gemma3 during training."""

    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features):
        batch = self.base_collator(features)
        if "token_type_ids" not in batch and "input_ids" in batch:
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--train-file",  default="data/sft/sft_train_malrule.jsonl")
    parser.add_argument("--val-ratio",   type=float, default=0.1)
    parser.add_argument("--output-dir",  default="checkpoints/sft_llama8b_malrule")
    parser.add_argument("--cache-dir",   default=os.path.expanduser("~/.cache/huggingface"))
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--grad-accum",  type=int,   default=4)
    parser.add_argument("--max-length",  type=int,   default=768)
    parser.add_argument("--full-finetune", action="store_true",
                        help="Full fine-tuning instead of LoRA")
    parser.add_argument("--lora-r",      type=int,   default=16)
    parser.add_argument("--lora-alpha",  type=int,   default=32)
    parser.add_argument("--lora-dropout",type=float, default=0.05)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.cache_dir, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model config (chat format, thinking mode, …) ──────────────────────────
    model_cfg = get_model_cfg(args.model, tokenizer)
    print(f"  response_template : {repr(model_cfg['response_template'])}")
    print(f"  chat_kwargs       : {model_cfg['chat_kwargs']}")
    if model_cfg.get("chat_template"):
        tokenizer.chat_template = model_cfg["chat_template"]

    # Patch apply_chat_template so Qwen3 thinking mode is always disabled
    if model_cfg["chat_kwargs"]:
        _orig_apply = tokenizer.apply_chat_template
        _kwargs = model_cfg["chat_kwargs"]
        def _patched_apply(*args, **kwargs):
            kwargs.update(_kwargs)
            return _orig_apply(*args, **kwargs)
        tokenizer.apply_chat_template = _patched_apply

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("Building datasets...")
    full_dataset = build_dataset(args.train_file, args.model)
    split = full_dataset.train_test_split(test_size=args.val_ratio, seed=42)
    train_dataset = split["train"]
    val_dataset   = split["test"]
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Val:   {len(val_dataset)} examples")

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

    # ── LoRA / Full fine-tuning ───────────────────────────────────────────────
    if args.full_finetune:
        print("Full fine-tuning (no LoRA)")
        for param in model.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}")
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        max_length=args.max_length,
        assistant_only_loss=True,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    if is_gemma_model(args.model):
        trainer.data_collator = Gemma3TokenTypeCollator(trainer.data_collator)

    print("\nStarting training...")
    trainer.train()

    # ── Save final model ──────────────────────────────────────────────────────
    final_path = Path(args.output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nModel saved to {final_path}")


if __name__ == "__main__":
    main()
