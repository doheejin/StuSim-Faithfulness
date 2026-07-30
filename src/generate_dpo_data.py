"""
Build DPO preference pairs for student simulator training.

Chosen  (y+): ideal_response from sft_train_{dataset}.jsonl (judge-filtered)
Rejected (y-): SFT model's own failure cases from the judgment file for the
               target model, where judge_category ∉ C*(feedback_type)

Using the trained SFT model's failures as negatives is strictly better than
baseline model failures: they are hard negatives that reflect the actual
failure modes DPO needs to correct (T2 sycophantic flips, T1 non-flips).

Output: data/sft/dpo_train_{dataset}_{model}.jsonl
  Each line: {
    "problem_id", "feedback_type",
    "prompt": [{"role": ..., "content": ...}],
    "chosen": "<ideal assistant response>",
    "rejected": "<SFT model's failed response>"
  }

Usage:
  # Default input files inferred from --model
  python generate_dpo_data.py --model llama
  python generate_dpo_data.py --model qwen

  # Recommended: all-chosen + auto-balance to T1 count (prevents T2 dominating T1 gradient)
  python generate_dpo_data.py --model llama --all-chosen --balance
  python generate_dpo_data.py --model qwen  --all-chosen --balance

  # Override input files explicitly
  python generate_dpo_data.py --model llama --all-chosen --max-per-type 405 \\
      --sft-data  data/sft/sft_train_malrule.jsonl \\
      --judgments data/results/judgments_sft_llama3_8b_b8g2.csv \\
      --output    data/sft/dpo_train_malrule_llama.jsonl
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent.parent

IDEAL_CATS = {
    "type1_correct": {"correct_flip"},
    "type2_wrong":   {"constructive_pushback", "passive_maintain", "confusion"},
    "type3_generic": {"constructive_pushback", "passive_maintain", "confusion"},
}

# Maps (dataset, model) → judgment CSV path (y- source)
JUDGMENT_FILES = {
    ("malrule", "llama"): "data/results/judgments_sft_llama3_8b_b8g2.csv",
    ("malrule", "qwen"):  "data/results/judgments_sft_qwen3_4b_b8g2.csv",
    ("eedi",    "llama"): "data/results/eedi/judgments_sft_eedi_llama3_8b_b8g2.csv",
    ("eedi",    "qwen"):  "data/results/eedi/judgments_sft_eedi_qwen3_4b_b8g2.csv",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default="malrule", choices=["malrule", "eedi"])
    parser.add_argument("--model",     required=True, choices=["llama", "qwen"],
                        help="Target SFT model — sets default input files and output name suffix")
    parser.add_argument("--sft-data",  default=None,
                        help="y+ source JSONL (default: data/sft/sft_train_{dataset}.jsonl)")
    parser.add_argument("--judgments", default=None,
                        help="y- source CSV  (default: looked up from JUDGMENT_FILES by dataset+model)")
    parser.add_argument("--output",    default=None,
                        help="Output JSONL path (default: data/sft/dpo_train_{dataset}_{model}.jsonl)")
    parser.add_argument("--all-chosen", action="store_true",
                        help="Pair each rejection with ALL available y+ responses (~3× more pairs)")
    parser.add_argument("--balance", action="store_true",
                        help="Auto-cap all feedback types to the T1 pair count (1:1:1 ratio). "
                             "Prevents T2 anti-flip gradient from dominating T1 pro-flip gradient.")
    parser.add_argument("--max-per-type", type=int, default=None,
                        help="Cap pairs per feedback_type to this count (random subsample). "
                             "Overrides --balance if both are given.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for y+ selection and --max-per-type subsampling")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    # ── Resolve paths (explicit args override defaults) ────────────────────────
    sft_path = Path(args.sft_data)  if args.sft_data  else BASE / f"data/sft/sft_train_{args.dataset}.jsonl"
    jpath    = Path(args.judgments) if args.judgments else BASE / JUDGMENT_FILES[(args.dataset, args.model)]
    out_path = Path(args.output)    if args.output    else BASE / f"data/sft/dpo_train_{args.dataset}_{args.model}.jsonl"

    # ── Load train split ───────────────────────────────────────────────────────
    with open(BASE / "data/sft/split.json") as f:
        split = json.load(f)
    train_pids = set(split[args.dataset]["train"])
    print(f"Train problems : {len(train_pids)}")
    print(f"SFT data (y+)  : {sft_path.name}")
    print(f"Judgments (y-) : {jpath.name}")
    print(f"Output         : {out_path}")

    print(f"All-chosen mode : {'yes (~3× pairs)' if args.all_chosen else 'no (1 y+ per rejection)'}")

    # ── Load SFT train data → chosen pool ─────────────────────────────────────
    sft_pool: dict[tuple, list] = defaultdict(list)
    for line in open(sft_path):
        r = json.loads(line)
        sft_pool[(r["problem_id"], r["feedback_type"])].append(r)
    print(f"Unique (pid, ft) keys in sft_train: {len(sft_pool)}")

    # ── Load SFT model judgments → rejected pool ───────────────────────────────
    jdf = pd.read_csv(jpath)
    jdf = jdf[jdf["problem_id"].isin(train_pids)].copy()
    jdf["is_ideal"] = jdf.apply(
        lambda r: r["judge_category"] in IDEAL_CATS.get(r["feedback_type"], set()),
        axis=1,
    )
    bad = jdf[~jdf["is_ideal"]].reset_index(drop=True)
    print(f"SFT model failures (train split): {len(bad)}")

    # ── Build DPO pairs ────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all pairs first (needed for --max-per-type subsampling)
    all_records: dict[str, list] = defaultdict(list)
    n_skipped = 0

    for _, row in bad.iterrows():
        key = (row["problem_id"], row["feedback_type"])
        candidates = sft_pool.get(key)
        if not candidates:
            n_skipped += 1
            continue

        rejected = str(row["student_response"])
        chosen_pool = candidates if args.all_chosen else [rng.choice(candidates)]

        for chosen_ex in chosen_pool:
            chosen = chosen_ex["ideal_response"]
            if chosen == rejected:
                n_skipped += 1
                continue
            all_records[key[1]].append({
                "problem_id":    key[0],
                "feedback_type": key[1],
                "prompt":        chosen_ex["messages"],
                "chosen":        chosen,
                "rejected":      rejected,
            })

    # Resolve cap: --balance auto-sets to T1 count; --max-per-type overrides explicitly
    cap = args.max_per_type
    if cap is None and args.balance:
        cap = len(all_records["type1_correct"])

    if cap is not None:
        for ft in all_records:
            if len(all_records[ft]) > cap:
                all_records[ft] = rng.sample(all_records[ft], cap)
        print(f"max-per-type cap : {cap}")

    stats: dict[str, int] = {}
    n_written = 0
    with open(out_path, "w") as fout:
        for ft in ["type1_correct", "type2_wrong", "type3_generic"]:
            for record in all_records[ft]:
                fout.write(json.dumps(record) + "\n")
                stats[ft] = stats.get(ft, 0) + 1
                n_written += 1

    print(f"\nWritten : {n_written}")
    print(f"Skipped : {n_skipped}  (no y+ match or identical pair)")
    print("Per feedback_type:")
    for ft in ["type1_correct", "type2_wrong", "type3_generic"]:
        print(f"  {ft}: {stats.get(ft, 0)}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
