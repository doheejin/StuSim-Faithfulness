"""
Compute Selective Flip Score (SFS) from a judgments CSV.

SFS is defined as:
    SFS = F_T - (F_M + F_G) / 2

where:
    F_T = flip rate under type1_correct (correct feedback)
    F_M = flip rate under type2_wrong   (misconception-targeted wrong feedback)
    F_G = flip rate under type3_generic (generic wrong feedback)

A flip is any judge label in {correct_flip, sycophantic_flip}.

Usage:
    # Print SFS for one (tag, dataset):
    python src/analysis/sfs.py --dataset malrule --tag qwen4b

    # Print SFS for all seven baseline prompt-only models (Table 3):
    python src/analysis/sfs.py --dataset malrule --table3

    # Compare a trained checkpoint against its prompt-only baseline:
    python src/analysis/sfs.py --dataset malrule \
        --tag sft_qwen3_4b_malrule \
        --baseline-tag simulate_student_authentic_qwen4b
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

FLIP_CATEGORIES = {"correct_flip", "sycophantic_flip"}


def judgments_path(results_dir: Path, dataset: str, tag: str) -> Path:
    folder = results_dir / ("eedi" if dataset == "eedi" else "")
    # Convention: judgments_{tag}.csv. The baseline reflective files on disk
    # are stored as judgments_simulate_student_authentic_{tag}.csv.
    return folder / f"judgments_{tag}.csv"


def read_flip_rates(path: Path, restrict_pairs=None):
    """Return dict {feedback_type: (n, k, rate)}, restricted to matching
    (problem_id, feedback_type) pairs if given."""
    n = defaultdict(int)
    k = defaultdict(int)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ft = row["feedback_type"]
            if restrict_pairs is not None and (row["problem_id"], ft) not in restrict_pairs:
                continue
            n[ft] += 1
            if row["judge_category"] in FLIP_CATEGORIES:
                k[ft] += 1
    out = {}
    for ft in ("type1_correct", "type2_wrong", "type3_generic"):
        rate = k[ft] / n[ft] if n[ft] else float("nan")
        out[ft] = (n[ft], k[ft], rate)
    return out


def sfs(flip_rates) -> float:
    F_T = flip_rates["type1_correct"][2]
    F_M = flip_rates["type2_wrong"][2]
    F_G = flip_rates["type3_generic"][2]
    return F_T - (F_M + F_G) / 2


def print_one(path: Path, label: str, restrict_pairs=None):
    if not path.exists():
        print(f"  [SKIP] not found: {path}")
        return None
    fr = read_flip_rates(path, restrict_pairs=restrict_pairs)
    s = sfs(fr)
    F_T = fr["type1_correct"][2]
    F_M = fr["type2_wrong"][2]
    F_G = fr["type3_generic"][2]
    n_T = fr["type1_correct"][0]
    n_M = fr["type2_wrong"][0]
    n_G = fr["type3_generic"][0]
    print(f"  {label:50s}  F_T={F_T:.3f}  F_M={F_M:.3f}  F_G={F_G:.3f}  "
          f"SFS={s:+.3f}   (n = {n_T}/{n_M}/{n_G})")
    return {"F_T": F_T, "F_M": F_M, "F_G": F_G, "SFS": s, "n": (n_T, n_M, n_G)}


def load_pairs(path: Path):
    if not path.exists():
        return None
    pairs = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs.add((row["problem_id"], row["feedback_type"]))
    return pairs


# Paper's seven prompt-only baseline models (Table 3).
TABLE3_TAGS = [
    ("Qwen3-4B",                    "simulate_student_authentic_qwen4b"),
    ("Llama-3.1-8B-Instruct",       "simulate_student_authentic_llama31_8b"),
    ("gpt-oss-20b",                 "simulate_student_authentic_oss20b"),
    ("Llama-3.3-70B-Instruct",      "simulate_student_authentic_llama33_70b"),
    ("Qwen3-Next-80B-Instruct",     "simulate_student_authentic_qwen80b"),
    ("Qwen3-Next-80B-Thinking",     "simulate_student_authentic_qwen80b_thinking"),
    ("gpt-oss-120b",                "simulate_student_authentic_oss120b"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["malrule", "eedi"])
    ap.add_argument("--results-dir", default="data/results",
                    help="Root of judgments CSVs (default: data/results)")
    ap.add_argument("--tag", default=None,
                    help="Print SFS for judgments_{tag}.csv.")
    ap.add_argument("--baseline-tag", default=None,
                    help="If set with --tag, restrict both to the same "
                         "(problem_id, feedback_type) pairs and print ΔSFS.")
    ap.add_argument("--table3", action="store_true",
                    help="Print SFS across all seven paper baseline models.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)

    if args.table3:
        print(f"\nTable 3 baseline SFS on {args.dataset}:")
        for name, tag in TABLE3_TAGS:
            path = judgments_path(results_dir, args.dataset, tag)
            print_one(path, name)
        return

    if not args.tag:
        ap.error("Provide either --tag or --table3.")

    tag_path = judgments_path(results_dir, args.dataset, args.tag)
    if args.baseline_tag:
        base_path = judgments_path(results_dir, args.dataset, args.baseline_tag)
        if not tag_path.exists():
            sys.exit(f"Not found: {tag_path}")
        if not base_path.exists():
            sys.exit(f"Not found: {base_path}")
        # Matched pairs: intersect judgments' (problem_id, feedback_type).
        adv_pairs = load_pairs(tag_path)
        base_res = print_one(base_path, f"baseline: {args.baseline_tag}",
                             restrict_pairs=adv_pairs)
        adv_res  = print_one(tag_path,  f"trained : {args.tag}",
                             restrict_pairs=adv_pairs)
        if base_res and adv_res:
            print(f"  {'ΔSFS':50s}  {adv_res['SFS'] - base_res['SFS']:+.3f}")
    else:
        print_one(tag_path, args.tag)


if __name__ == "__main__":
    main()
