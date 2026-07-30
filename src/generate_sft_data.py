"""
Generate synthetic SFT training data for ideal student simulator behavior.

For each (problem, feedback) triple:
  - T1 (correct feedback): student correctly engages → flips to correct answer
  - T2 (wrong feedback):   student tries feedback but it doesn't match → maintains / pushes back
  - T3 (generic feedback): student is confused, asks for specific hint → maintains

Outputs:
  data/sft/sft_train.jsonl   — training data (malrule, 800 problems)
  data/sft/sft_val.jsonl     — validation data (malrule, 200 problems)

Usage:
  export OPENAI_API_KEY=...
  python generate_sft_data.py [--dataset malrule] [--max-problems 1000] [--dry-run]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

# Add src/ to path so llm_inference can be imported
sys.path.insert(0, str(Path(__file__).parent))

BASE = Path(__file__).parent.parent
OUT  = BASE / "data" / "sft"
OUT.mkdir(parents=True, exist_ok=True)

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a student learning math. You worked on the problem below and \
believe your answer is correct. Your task is to respond to your teacher \
naturally — as a real student would."""

# Generation prompt for T1 (correct feedback targeting the actual misconception)
GEN_T1 = """\
You are helping generate realistic training data for a student simulator.

CONTEXT:
- Problem: {problem_text}
- Student's answer: {student_answer} (this is wrong; correct answer is {correct_answer})
- Student's misconception: {misconception_name} — "{misconception_desc}"
- Teacher's feedback: "{feedback_text}"

This feedback CORRECTLY addresses the student's actual misconception.

TASK: Write a realistic student response (2–5 sentences) where the student:
1. Processes the feedback and recognizes how it relates to their error
2. Corrects their reasoning step by step
3. Arrives at the correct answer: {correct_answer}

Write ONLY the student's response. First-person, natural, conversational. \
Do NOT start with "I" — vary the opening (e.g., "Oh!", "Wait...", "Hmm,", "Ah,").
Do NOT write any explanation or prefix."""

# Generation prompt for T2 (wrong feedback targeting unrelated misconception)
GEN_T2 = """\
You are helping generate realistic training data for a student simulator.

CONTEXT:
- Problem: {problem_text}
- Student's answer: {student_answer} (this is wrong; correct answer is {correct_answer})
- Student's actual misconception: {misconception_name} — "{misconception_desc}"
- Teacher's feedback: "{feedback_text}"

IMPORTANT: This feedback addresses a DIFFERENT misconception, NOT the student's actual error. \
The student genuinely cannot see how this feedback applies to what they did.

TASK: Write a realistic student response (2–5 sentences) where the student:
1. Tries to understand the feedback but it doesn't connect to their thinking
2. Is confused or uncertain — they cannot see how to apply this feedback
3. Maintains their original answer {student_answer} OR expresses they're still not sure

Write ONLY the student's response. First-person, natural, conversational. \
Do NOT start with "I" — vary the opening (e.g., "Hmm,", "Wait,", "I'm not sure...").
Do NOT flip to the correct answer. Do NOT write any explanation or prefix."""

# Generation prompt for T3 (generic feedback: "That's not right. Can you try again?")
GEN_T3 = """\
You are helping generate realistic training data for a student simulator.

CONTEXT:
- Problem: {problem_text}
- Student's answer: {student_answer} (this is wrong; correct answer is {correct_answer})
- Student's actual misconception: {misconception_name} — "{misconception_desc}"
- Teacher's feedback: "That's not the right answer. Can you try again?"

This feedback gives NO information about what went wrong.

TASK: Write a realistic student response (2–4 sentences) where the student:
1. Acknowledges the feedback but doesn't know what specifically to fix
2. Expresses confusion or asks for a hint
3. Keeps their original answer {student_answer} or re-states it with uncertainty

Write ONLY the student's response. First-person, natural, conversational. \
Do NOT start with "I" — vary the opening.
Do NOT flip to the correct answer unless the student explicitly re-checks step by step \
(which is unlikely with no guidance). Do NOT write any explanation or prefix."""


GEN_PROMPTS = {
    "type1_correct": GEN_T1,
    "type2_wrong":   GEN_T2,
    "type3_generic": GEN_T3,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_input_prompt(row):
    """Build the inference-time prompt (same as existing simulate_student.py)."""
    user_content = (
        f"**Problem**: {row['problem_text']}\n"
        f"**Your answer**: {row['malrule_answer']}\n\n"
        f"Your teacher says: \"{row['feedback_text']}\"\n\n"
        "Your teacher then asks: \"So what do you think the answer is?\"\n\n"
        "What would you say to your teacher?"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def generate_ideal_responses(client, row, n_samples=3, dry_run=False):
    """Call GPT-4o-mini to generate n_samples ideal responses via temperature sampling."""
    ft = row["feedback_type"]
    template = GEN_PROMPTS[ft]

    prompt = template.format(
        problem_text      = row["problem_text"],
        student_answer    = row["malrule_answer"],
        correct_answer    = row["correct_answer"],
        misconception_name= row.get("malrule_name", ""),
        misconception_desc= row.get("misconception_desc_used", ""),
        feedback_text     = row["feedback_text"],
    )

    if dry_run:
        return [f"[DRY-RUN] sample {i} for {ft}" for i in range(n_samples)]

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=300,
                n=n_samples,
            )
            return [c.message.content.strip() for c in resp.choices]
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def make_sft_example(row, ideal_response):
    """Return one SFT training example as a dict."""
    return {
        "problem_id":     row["problem_id"],
        "feedback_type":  row["feedback_type"],
        "category":       row.get("category", ""),
        "malrule_name":   row.get("malrule_name", ""),
        "messages":       build_input_prompt(row),
        "ideal_response": ideal_response,
    }


IDEAL_CATEGORIES = {
    "type1_correct": {"correct_flip"},
    "type2_wrong":   {"constructive_pushback", "passive_maintain", "confusion"},
    "type3_generic": {"constructive_pushback", "passive_maintain", "confusion"},
}


def judge_response(client, row, response):
    """Run judge on a generated response. Returns category string."""
    from llm_inference.prompts.studentsim.judge_response import build_conversation
    conv = build_conversation(
        problem_text     = row["problem_text"],
        student_answer   = row["malrule_answer"],
        correct_answer   = row["correct_answer"],
        feedback_text    = row["feedback_text"],
        student_response = response,
    )
    r = client.chat.completions.create(
        model="gpt-4o-mini", messages=conv, temperature=0, max_tokens=200
    )
    try:
        return json.loads(r.choices[0].message.content).get("category", "unknown")
    except Exception:
        return "unknown"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="malrule", choices=["malrule", "eedi"])
    parser.add_argument("--split-file", default="data/sft/split.json")
    parser.add_argument("--n-samples",  type=int, default=3,
                        help="Responses to generate per (problem, feedback_type). "
                             "Each is judged; passing ones → sft_train, failing ones → sft_rejected. "
                             "Paper uses k=3.")
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    # Load pre-defined train split
    with open(BASE / args.split_file) as f:
        split = json.load(f)
    train_pids = set(split[args.dataset]["train"])
    print(f"Train problems: {len(train_pids)}")

    # Load data — train problems only
    if args.dataset == "malrule":
        pattern = str(BASE / "data/results/responses_qwen80b_thinking.csv")
    else:
        pattern = str(BASE / "data/results/eedi/responses_qwen80b_thinking.csv")

    df = pd.read_csv(pattern)
    df = df[df["problem_id"].isin(train_pids)].reset_index(drop=True)
    cost_est = len(df) * args.n_samples * 0.0003 * 2
    print(f"Rows to generate: {len(df)}  n_samples={args.n_samples}  (~${cost_est:.1f} for gen+judge)")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) if not args.dry_run else None

    kept, rejected, errors = [], [], 0

    for i, (_, row) in enumerate(df.iterrows()):
        pid = row["problem_id"]
        ft  = row["feedback_type"]
        try:
            responses = generate_ideal_responses(client, row, n_samples=args.n_samples, dry_run=args.dry_run)

            for response in responses:
                if args.dry_run:
                    kept.append(make_sft_example(row, response))
                    continue

                cat = judge_response(client, row, response)
                if cat in IDEAL_CATEGORIES[ft]:
                    kept.append(make_sft_example(row, response))
                else:
                    # Save rejected examples as DPO negative candidates
                    rejected.append({
                        **make_sft_example(row, response),
                        "judge_category": cat,
                    })

            if (i + 1) % 100 == 0:
                n_filtered = len(rejected)
                print(f"  [{i+1}/{len(df)}] kept={len(kept)} rejected={n_filtered} errors={errors}")
                _save(kept,     OUT / f"sft_train_{args.dataset}.jsonl")
                _save(rejected, OUT / f"sft_rejected_{args.dataset}.jsonl")

        except Exception as e:
            print(f"  ERROR {pid} {ft}: {e}")
            errors += 1

    _save(kept,     OUT / f"sft_train_{args.dataset}.jsonl")
    _save(rejected, OUT / f"sft_rejected_{args.dataset}.jsonl")

    total = len(kept) + len(rejected)
    print(f"\nDone. kept={len(kept)}, rejected={len(rejected)}, errors={errors}")
    if total:
        print(f"Filter rate: {len(rejected)/total*100:.1f}%")
    print(f"Saved to {OUT}/sft_train_{args.dataset}.jsonl")
    print(f"        {OUT}/sft_rejected_{args.dataset}.jsonl")


def _save(examples, path):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


if __name__ == "__main__":
    main()
