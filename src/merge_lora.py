"""
Merge LoRA adapter into base model for vLLM inference.

Usage:
  python merge_lora.py \
    --base-model Qwen/Qwen3-4B \
    --adapter    checkpoints/sft_qwen3_4b_malrule/final \
    --output     checkpoints/sft_qwen3_4b_malrule/merged
"""

import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter",    default="checkpoints/sft_qwen3_4b_malrule/final")
    parser.add_argument("--output",     default="checkpoints/sft_qwen3_4b_malrule/merged")
    parser.add_argument("--cache-dir",  default=os.path.expanduser("~/.cache/huggingface"))
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, cache_dir=args.cache_dir, trust_remote_code=True
    )

    print(f"Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging weights...")
    model = model.merge_and_unload()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to: {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
