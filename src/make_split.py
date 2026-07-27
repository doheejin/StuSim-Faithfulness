"""
Create train/test splits for malrule and eedi datasets.

Strategy:
  - Malrule (21 categories, ~50 each):
      Stratified by category, 10 per category → test (210), rest → train (790)
      Used for SFT training AND within-domain evaluation.

  - Eedi (155 categories, highly unbalanced):
      Random 200 → test only (zero-shot generalization, no SFT on eedi)
      Remaining 800 → unused for SFT (cross-domain evaluation only)

SFT pipeline:
  Train : malrule train (790)
  Test 1: malrule test  (210)  — within-domain
  Test 2: eedi test     (200)  — cross-domain generalization

Output: data/sft/split.json
"""

import json
import random
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
OUT  = BASE / "data" / "sft"
OUT.mkdir(parents=True, exist_ok=True)


def stratified_split_by_category(df, test_per_cat=10, seed=42):
    """Malrule: exactly test_per_cat problems per category into test."""
    rng = random.Random(seed)
    train_pids, test_pids = [], []
    for cat, grp in df.groupby("category"):
        pids = grp["problem_id"].tolist()
        rng.shuffle(pids)
        n_test = min(test_per_cat, len(pids))
        test_pids.extend(pids[:n_test])
        train_pids.extend(pids[n_test:])
    return train_pids, test_pids


def random_split(df, n_test=200, seed=42):
    """Eedi: random n_test problems into test."""
    rng = random.Random(seed)
    pids = df["problem_id"].tolist()
    rng.shuffle(pids)
    return pids[n_test:], pids[:n_test]


def main():
    malrule = pd.read_csv(BASE / "data/results/responses_qwen80b_thinking.csv").drop_duplicates("problem_id")
    eedi    = pd.read_csv(BASE / "data/results/eedi/responses_qwen80b_thinking.csv").drop_duplicates("problem_id")

    m_train, m_test = stratified_split_by_category(malrule, test_per_cat=10)
    e_train, e_test = random_split(eedi, n_test=200)

    print(f"Malrule: {len(m_train)} train / {len(m_test)} test")
    print(f"Eedi:    {len(e_test)} test only (zero-shot generalization, no SFT training)")

    m_test_cats = malrule[malrule["problem_id"].isin(m_test)]["category"].nunique()
    print(f"Malrule test: {m_test_cats}/21 categories covered")

    split = {
        "malrule": {"train": m_train, "test": m_test},
        "eedi":    {"train": e_train, "test": e_test},
    }
    out_path = OUT / "split.json"
    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"\nSplit saved to {out_path}")


if __name__ == "__main__":
    main()
