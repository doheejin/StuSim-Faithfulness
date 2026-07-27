#!/usr/bin/env python3
"""
Student Simulation Experiment

Tests whether LLMs can serve as authentic student simulators by measuring
misconception faithfulness via the Selective Flip Score (SFS): do simulated
students "flip" (correct their answer) only when teacher feedback addresses
the actual misconception?

Supports two datasets:
  --dataset malrule:  Free-form numeric answers with malrule-based misconceptions
  --dataset eedi:     Multiple-choice (Eedi) with labeled distractor misconceptions

Three-step pipeline + summarize:
  --step feedback:   Generate teacher feedback (Type 1: correct, Type 2: wrong, Type 3: generic)
  --step simulate:   Simulate student responses to each feedback type
  --step judge:      Classify responses into 6-category taxonomy + print analysis
  --step summarize:  Auto-discover all judgment files and produce summary.csv

Usage:
    python run_experiment.py \
        --step feedback --dataset eedi --num-problems 1000 \
        --model openai/gpt-oss-120b --backend vllm --num-gpus 4

    python run_experiment.py \
        --step simulate --dataset eedi --model-tag qwen80b \
        --model Qwen/Qwen3-Next-80B-A3B-Instruct --backend vllm --num-gpus 4

    python run_experiment.py \
        --step judge --dataset eedi --model-tag qwen80b \
        --model openai/gpt-oss-120b --backend vllm --num-gpus 4

    python run_experiment.py --step summarize --dataset eedi
"""

import argparse
import csv
import glob as glob_module
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add this directory to path so llm_inference can be imported
sys.path.insert(0, str(Path(__file__).parent))

from llm_inference.inference_llm import BatchedLLMInference
from llm_inference.prompts import load_prompt
from llm_inference.shared_utils import (
    clean_model_name,
    filter_gpt_oss_responses,
    filter_qwen_responses,
    setup_csv_field_limit,
)
from analysis.sfs import read_flip_rates, sfs

# Base directories (relative to this script)
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EEDI_DIR = BASE_DIR / "eedi-misconception"

# Default data file locations
DEFAULT_RAW_DATA = str(DATA_DIR / "raw_data.csv")
DEFAULT_TAXONOMY = str(DATA_DIR / "oss120b_v3.csv")

# Fixed feedback for Type 3
GENERIC_FEEDBACK = "That's not the right answer. Can you try again?"

# Column definitions
FEEDBACK_COLUMNS = [
    'problem_id', 'category', 'malrule_id', 'malrule_name',
    'problem_text', 'correct_answer', 'malrule_answer',
    'feedback_type', 'feedback_text',
    'wrong_malrule_id', 'misconception_desc_used',
    'raw_response', 'parse_success',
    'model_name', 'timestamp',
]

RESPONSE_COLUMNS = [
    'problem_id', 'category', 'malrule_id', 'malrule_name',
    'problem_text', 'correct_answer', 'malrule_answer',
    'feedback_type', 'feedback_text',
    'wrong_malrule_id', 'misconception_desc_used',
    'student_response', 'raw_response',
    'model_name', 'timestamp',
]

RESPONSE_COLUMNS_MULTITURN = [
    'problem_id', 'category', 'malrule_id', 'malrule_name',
    'problem_text', 'correct_answer', 'malrule_answer',
    'feedback_type', 'feedback_text',
    'wrong_malrule_id', 'misconception_desc_used',
    'initial_student_response', 'initial_raw_response',
    'student_response', 'raw_response',
    'model_name', 'timestamp',
]

JUDGMENT_COLUMNS = [
    'problem_id', 'category', 'malrule_id', 'malrule_name',
    'problem_text', 'correct_answer', 'malrule_answer',
    'feedback_type', 'feedback_text',
    'wrong_malrule_id', 'misconception_desc_used',
    'student_response',
    'judge_category', 'judge_final_answer', 'judge_reasoning',
    'judge_raw_response', 'judge_parse_success',
    'model_name', 'timestamp',
]

VALID_CATEGORIES = {
    'correct_flip', 'sycophantic_flip', 'different_wrong',
    'constructive_pushback', 'passive_maintain', 'confusion',
}


# ─── Path Helpers ────────────────────────────────────────────────────

def get_results_dir(dataset: str, multi_turn: bool = False) -> Path:
    """Return results directory for the given dataset."""
    base_dir = DATA_DIR / ("results_multiturn_holistic" if multi_turn else "results")
    if dataset == "eedi":
        return base_dir / "eedi"
    return base_dir


def get_feedback_file(dataset: str) -> Path:
    return get_results_dir(dataset, multi_turn=False) / "feedback.csv"


def _variant_suffix(prompt_variant: str) -> str:
    """Return file suffix for non-default prompt variants."""
    if prompt_variant == "simulate_student":
        return ""
    # Map display name back to filename (reflective files are stored as authentic on disk)
    file_variant = prompt_variant.replace('_reflective', '_authentic')
    return f"_{file_variant}"


def _tag_suffix(model_tag: str) -> str:
    """Return file suffix for model tag."""
    if not model_tag:
        return ""
    return f"_{model_tag}"


def get_responses_file(dataset: str, prompt_variant: str, model_tag: str = "", multi_turn: bool = False) -> Path:
    suffix = _variant_suffix(prompt_variant) + _tag_suffix(model_tag)
    return get_results_dir(dataset, multi_turn=multi_turn) / f"responses{suffix}.csv"


def get_judgments_file(dataset: str, prompt_variant: str, model_tag: str = "", multi_turn: bool = False) -> Path:
    suffix = _variant_suffix(prompt_variant) + _tag_suffix(model_tag)
    return get_results_dir(dataset, multi_turn=multi_turn) / f"judgments{suffix}.csv"


# ─── Argument Parsing ────────────────────────────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Student Simulation Experiment: Do LLMs understand misconceptions?"
    )
    parser.add_argument("--step", required=True,
                        choices=["feedback", "simulate", "judge", "summarize"],
                        help="Pipeline step to run")
    parser.add_argument("--dataset", type=str, default="malrule",
                        choices=["malrule", "eedi", "all"],
                        help="Dataset to use (default: malrule). Use 'all' with --step summarize to combine both.")
    parser.add_argument("--num-problems", type=int, default=1000,
                        help="Number of problems to sample (feedback step, default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--input-raw", type=str, default=DEFAULT_RAW_DATA,
                        help="Path to raw_data.csv (default: data/raw_data.csv)")
    parser.add_argument("--input-taxonomy", type=str, default=DEFAULT_TAXONOMY,
                        help="Path to oss120b_v3.csv (default: data/oss120b_v3.csv)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output")
    parser.add_argument("--feedback-file", type=str, default=None,
                        help="Override the default feedback CSV path. Used by "
                             "the adversarial-feedback experiment to point at "
                             "paraphrased feedback (e.g. feedback_verbose.csv). "
                             "If unset, uses get_feedback_file(args.dataset).")
    parser.add_argument("--simulate-prompt", type=str, default="simulate_student",
                        help="Prompt variant for simulate step (default: simulate_student)")
    parser.add_argument("--model-tag", type=str, default="",
                        help="Tag appended to output filenames for multi-model runs")
    parser.add_argument("--multi-turn", action="store_true",
                        help="Enable multi-turn mode: student explains chosen answer first, then responds to feedback (default: single-turn)")

    # Model arguments
    parser.add_argument("--model", required=False, default=None, help="Model name")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "gpt", "vllm", "hf"])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--max-tokens', type=int, default=None)
    parser.add_argument('--top-p', type=float, default=None)
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument('--min-p', type=float, default=None)
    parser.add_argument('--enable-thinking', action='store_true')
    parser.add_argument('--gpt-reasoning', type=str, default=None,
                        choices=['low', 'medium', 'high'])
    parser.add_argument('--cache-dir', type=str, default=None)
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--vllm-max-num-seqs', type=int, default=None)
    parser.add_argument('--vllm-max-model-len', type=int, default=None)
    parser.add_argument('--vllm-gpu-memory-utilization', type=float, default=None)

    args = parser.parse_args()

    # Validate: --model required for all steps except summarize
    if args.step != "summarize" and not args.model:
        parser.error("--model is required for feedback/simulate/judge steps")

    return args


# ─── Shared Utilities ─────────────────────────────────────────────────

def determine_backend(model_name: str, backend_arg: str) -> str:
    if backend_arg != "auto":
        return backend_arg
    if model_name.startswith("gpt"):
        return "gpt"
    elif "/" in model_name:
        return "vllm"
    else:
        return "gpt"


def initialize_inference_engine(args, backend: str) -> BatchedLLMInference:
    kwargs = {
        'model_name': args.model, 'backend': backend,
        'batch_size': args.batch_size,
        'num_gpus': args.num_gpus,
    }
    if args.cache_dir is not None: kwargs['cache_dir'] = args.cache_dir
    if args.temperature is not None: kwargs['temperature'] = args.temperature
    if args.max_tokens is not None: kwargs['max_tokens'] = args.max_tokens
    if args.top_p is not None: kwargs['top_p'] = args.top_p
    if args.top_k is not None: kwargs['top_k'] = args.top_k
    if args.min_p is not None: kwargs['min_p'] = args.min_p
    if args.enable_thinking: kwargs['enable_thinking'] = True
    if args.gpt_reasoning is not None: kwargs['gpt_reasoning'] = args.gpt_reasoning
    if args.vllm_max_num_seqs is not None: kwargs['vllm_max_num_seqs'] = args.vllm_max_num_seqs
    if args.vllm_max_model_len is not None: kwargs['vllm_max_model_len'] = args.vllm_max_model_len
    if args.vllm_gpu_memory_utilization is not None:
        kwargs['vllm_gpu_memory_utilization'] = args.vllm_gpu_memory_utilization
    return BatchedLLMInference(**kwargs)


def filter_responses(responses: List[str], model_name: str) -> Tuple[List[str], List[str]]:
    """Apply model-specific response filtering."""
    model_lower = model_name.lower()
    if 'gpt-oss' in model_lower:
        return filter_gpt_oss_responses(responses, verbose=False)
    elif 'qwen' in model_lower and 'thinking' in model_lower:
        return filter_qwen_responses(responses, verbose=False)
    else:
        return responses, responses


def parse_json_response(response: str) -> Optional[Dict]:
    """Parse JSON from LLM response. Returns dict or None."""
    text = response.strip()

    # Try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to find JSON block with balanced braces
    start = text.find('{')
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break

    return None


def find_raw_data(args) -> str:
    """Find raw_data.csv — check explicit arg, then common locations."""
    if args.input_raw and Path(args.input_raw).exists():
        return args.input_raw
    candidates = [
        DATA_DIR / "raw_data.csv",
        Path("raw_data.csv"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    print("Error: Cannot find raw_data.csv. Use --input-raw to specify path.")
    sys.exit(1)


def find_taxonomy(args) -> str:
    """Find oss120b_v3.csv — check explicit arg, then common locations."""
    if args.input_taxonomy and Path(args.input_taxonomy).exists():
        return args.input_taxonomy
    candidates = [
        DATA_DIR / "oss120b_v3.csv",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    print("Error: Cannot find oss120b_v3.csv. Use --input-taxonomy to specify path.")
    sys.exit(1)


def load_completed_ids(csv_path: Path, id_columns: List[str]) -> set:
    """Load completed item IDs from existing CSV for resume support."""
    completed = set()
    if csv_path.exists():
        setup_csv_field_limit()
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                key = tuple(row.get(c, '') for c in id_columns)
                completed.add(key)
    return completed


def print_progress(batch_idx: int, num_batches: int, done: int, total: int,
                   start_time: float, extra: str = ""):
    elapsed = time.time() - start_time
    per_sec = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / per_sec if per_sec > 0 else 0
    eta_str = f"{eta:.0f}s" if eta < 60 else f"{eta/60:.1f}m"
    msg = f"Batch {batch_idx+1}/{num_batches}: {done}/{total} ({100*done/total:.1f}%) | ETA: {eta_str}"
    if extra:
        msg += f" | {extra}"
    print(msg)


# ─── Eedi Dataset Loading ───────────────────────────────────────────

def load_eedi_misconception_mapping() -> Dict[int, str]:
    """Load MisconceptionId -> MisconceptionName from mapping CSV."""
    mapping_path = EEDI_DIR / "misconception_mapping.csv"
    if not mapping_path.exists():
        print(f"Error: Eedi misconception mapping not found: {mapping_path}")
        sys.exit(1)

    mapping = {}
    with open(mapping_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            mid = int(row['MisconceptionId'])
            mapping[mid] = row['MisconceptionName']
    return mapping


def _format_eedi_problem_text(row: Dict) -> str:
    """Format Eedi question with all 4 answer options."""
    text = row['QuestionText']
    text += f"\n\nA. {row['AnswerAText']}"
    text += f"\nB. {row['AnswerBText']}"
    text += f"\nC. {row['AnswerCText']}"
    text += f"\nD. {row['AnswerDText']}"
    return text


def sample_eedi_problems_for_feedback(num_problems: int, seed: int,
                                       misconception_map: Dict[int, str]) -> List[Dict]:
    """
    Load and filter Eedi questions for the feedback step.

    Filters to questions with >=2 different misconceptions among distractors.
    For each usable question, picks one distractor as the "student's answer"
    (Type 1 misconception) and uses another distractor's misconception for Type 2.
    """
    train_path = EEDI_DIR / "train.csv"
    if not train_path.exists():
        print(f"Error: Eedi train data not found: {train_path}")
        sys.exit(1)

    setup_csv_field_limit()

    # Load all questions and filter to usable ones
    usable = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            correct = row['CorrectAnswer']  # A, B, C, D
            options = ['A', 'B', 'C', 'D']
            wrong_options = [o for o in options if o != correct]

            # Get misconception IDs for wrong options
            distractor_misconceptions = {}  # option_letter -> misconception_id
            for opt in wrong_options:
                mid_str = row.get(f'Misconception{opt}Id', '').strip()
                if mid_str and mid_str != '' and mid_str != 'nan':
                    try:
                        mid = int(float(mid_str))
                        if mid in misconception_map:
                            distractor_misconceptions[opt] = mid
                    except (ValueError, TypeError):
                        pass

            # Need at least 2 distractors with DIFFERENT misconceptions
            unique_misconceptions = set(distractor_misconceptions.values())
            if len(unique_misconceptions) < 2:
                continue

            usable.append((row, distractor_misconceptions))

    print(f"  Eedi: {len(usable)} usable questions (≥2 different misconceptions)")

    # Sample
    rng = random.Random(seed)
    if len(usable) > num_problems:
        usable = rng.sample(usable, num_problems)
    rng.shuffle(usable)

    # Convert to pipeline-compatible format
    problems = []
    for i, (row, distractor_misc) in enumerate(usable):
        correct = row['CorrectAnswer']
        correct_text = row[f'Answer{correct}Text']

        # Pick the first distractor with a misconception as the "student's answer"
        # (deterministic given the sorted order)
        sorted_opts = sorted(distractor_misc.keys())
        student_opt = sorted_opts[0]
        student_mid = distractor_misc[student_opt]
        student_text = row[f'Answer{student_opt}Text']

        # Find a different misconception for Type 2
        other_mids = {opt: mid for opt, mid in distractor_misc.items()
                      if mid != student_mid}
        if not other_mids:
            # All distractors have the same misconception — skip
            continue

        # Pick one randomly for Type 2
        wrong_opt = rng.choice(sorted(other_mids.keys()))
        wrong_mid = other_mids[wrong_opt]

        problems.append({
            'problem_id': f"eedi_{row['QuestionId']}",
            'category': row.get('SubjectName', ''),
            'malrule_id': str(student_mid),
            'malrule_name': misconception_map[student_mid],
            'problem_text': _format_eedi_problem_text(row),
            'correct_answer': f"{correct}. {correct_text}",
            'malrule_answer': f"{student_opt}. {student_text}",
            # Store Type 2 info for later use
            '_wrong_malrule_id': str(wrong_mid),
            '_wrong_malrule_name': misconception_map[wrong_mid],
        })

    print(f"  Sampled: {len(problems)} problems across "
          f"{len(set(p['category'] for p in problems))} subjects")

    return problems


# ─── Step 1: Feedback Generation ─────────────────────────────────────

def load_malrule_descriptions(taxonomy_path: str) -> Dict[str, Dict]:
    """Load transformation_description and student_deviation per malrule_id."""
    setup_csv_field_limit()
    descriptions = {}
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            mid = row['malrule_id']
            descriptions[mid] = {
                'transformation_description': row.get('transformation_description', ''),
                'student_deviation': row.get('student_deviation', ''),
                'category': row.get('category', ''),
            }
    return descriptions


def sample_problems_for_feedback(raw_data_path: str, num_problems: int,
                                 seed: int, malrule_descs: Dict) -> List[Dict]:
    """
    Sample problems from raw_data.csv, balanced across categories.
    Skips word_problems (only 1 malrule, can't pair for Type 2).
    Only includes problems whose malrule_id has a description in Phase 1 results.
    """
    setup_csv_field_limit()

    by_category = defaultdict(list)
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row['category']
            mid = row['malrule_id']
            # Skip word_problems and malrules without Phase 1 descriptions
            if cat == 'word_problems':
                continue
            if mid not in malrule_descs:
                continue
            if not row.get('answers_differ', 'True').lower() == 'true':
                continue
            by_category[cat].append(row)

    rng = random.Random(seed)
    per_category = max(1, num_problems // len(by_category))
    sampled = []

    for cat in sorted(by_category.keys()):
        rows = by_category[cat]
        n = min(per_category, len(rows))
        sampled.extend(rng.sample(rows, n))

    # If we need more, sample from remaining
    if len(sampled) < num_problems:
        sampled_set = set(id(r) for r in sampled)
        all_remaining = []
        for cat in sorted(by_category.keys()):
            all_remaining.extend([r for r in by_category[cat] if id(r) not in sampled_set])
        rng.shuffle(all_remaining)
        sampled.extend(all_remaining[:num_problems - len(sampled)])

    sampled = sampled[:num_problems]

    # Assign problem_ids
    for i, row in enumerate(sampled):
        row['problem_id'] = f"p{i:04d}"

    return sampled


def pick_wrong_malrule(problem: Dict, malrule_descs: Dict, rng: random.Random) -> Optional[Tuple[str, str]]:
    """
    Pick a random DIFFERENT malrule from the same category for Type 2 feedback.
    Returns (malrule_id, description) or None if no other malrule available.
    """
    category = problem['category']
    my_id = problem['malrule_id']

    # Find all other malrules in the same category
    candidates = [
        (mid, info)
        for mid, info in malrule_descs.items()
        if info['category'] == category and mid != my_id
        and (info['transformation_description'] or info['student_deviation'])
    ]

    if not candidates:
        return None

    chosen_id, chosen_info = rng.choice(candidates)
    # Prefer student_deviation as the misconception description, fall back to transformation_description
    desc = chosen_info['student_deviation'] or chosen_info['transformation_description']
    return chosen_id, desc


def get_misconception_desc(malrule_id: str, malrule_descs: Dict) -> str:
    """Get the misconception description for a malrule_id."""
    info = malrule_descs.get(malrule_id, {})
    return info.get('student_deviation', '') or info.get('transformation_description', '')


def _build_work_items_malrule(problems: List[Dict], malrule_descs: Dict,
                               seed: int) -> List[Dict]:
    """Build Type 1 + Type 2 work items for malrule dataset."""
    rng = random.Random(seed + 1)
    work_items = []

    for prob in problems:
        my_desc = get_misconception_desc(prob['malrule_id'], malrule_descs)
        if not my_desc:
            continue

        # Type 1: correct misconception
        work_items.append({
            **prob,
            'feedback_type': 'type1_correct',
            'misconception_desc_used': my_desc,
            'wrong_malrule_id': '',
        })

        # Type 2: wrong misconception from same category
        wrong = pick_wrong_malrule(prob, malrule_descs, rng)
        if wrong:
            wrong_id, wrong_desc = wrong
            work_items.append({
                **prob,
                'feedback_type': 'type2_wrong',
                'misconception_desc_used': wrong_desc,
                'wrong_malrule_id': wrong_id,
            })

    return work_items


def _build_work_items_eedi(problems: List[Dict]) -> List[Dict]:
    """Build Type 1 + Type 2 work items for Eedi dataset."""
    work_items = []

    for prob in problems:
        # Type 1: correct misconception (student's actual misconception)
        work_items.append({
            **{k: v for k, v in prob.items() if not k.startswith('_')},
            'feedback_type': 'type1_correct',
            'misconception_desc_used': prob['malrule_name'],
            'wrong_malrule_id': '',
        })

        # Type 2: wrong misconception (different distractor's misconception)
        work_items.append({
            **{k: v for k, v in prob.items() if not k.startswith('_')},
            'feedback_type': 'type2_wrong',
            'misconception_desc_used': prob['_wrong_malrule_name'],
            'wrong_malrule_id': prob['_wrong_malrule_id'],
        })

    return work_items


def run_feedback_step(args):
    """Step 1: Generate teacher feedback for Type 1 and Type 2."""
    print("=" * 70)
    print("STEP 1: FEEDBACK GENERATION")
    print(f"Dataset: {args.dataset}")
    print("Generate targeted teacher feedback (Type 1 + Type 2)")
    print("=" * 70)

    results_dir = get_results_dir(args.dataset)
    output_path = get_feedback_file(args.dataset)

    if output_path.exists() and not args.resume:
        print(f"\nError: Output file exists: {output_path}")
        print("  Use --resume or delete the file first")
        sys.exit(1)

    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data and sample problems based on dataset
    if args.dataset == "eedi":
        print(f"\nLoading Eedi misconception mapping...")
        misconception_map = load_eedi_misconception_mapping()
        print(f"  Loaded {len(misconception_map)} misconception definitions")

        print(f"\nSampling up to {args.num_problems} Eedi problems...")
        problems = sample_eedi_problems_for_feedback(
            args.num_problems, args.seed, misconception_map)

        work_items = _build_work_items_eedi(problems)
    else:
        raw_data_path = find_raw_data(args)
        taxonomy_path = find_taxonomy(args)

        print(f"\nLoading malrule descriptions from {taxonomy_path}...")
        malrule_descs = load_malrule_descriptions(taxonomy_path)
        print(f"  Loaded descriptions for {len(malrule_descs)} malrules")

        print(f"\nSampling {args.num_problems} problems from {raw_data_path}...")
        problems = sample_problems_for_feedback(raw_data_path, args.num_problems,
                                                args.seed, malrule_descs)
        print(f"  Sampled: {len(problems)} problems across "
              f"{len(set(p['category'] for p in problems))} categories")

        work_items = _build_work_items_malrule(problems, malrule_descs, args.seed)

    print(f"  Work items: {len(work_items)} (Type 1 + Type 2)")
    type1_count = sum(1 for w in work_items if w['feedback_type'] == 'type1_correct')
    type2_count = sum(1 for w in work_items if w['feedback_type'] == 'type2_wrong')
    print(f"    Type 1 (correct): {type1_count}")
    print(f"    Type 2 (wrong):   {type2_count}")

    # Resume support
    completed = set()
    if args.resume and output_path.exists():
        completed = load_completed_ids(output_path, ['problem_id', 'feedback_type'])
        print(f"  Resuming: {len(completed)} items already done")

    remaining = [w for w in work_items
                 if (w['problem_id'], w['feedback_type']) not in completed]
    if not remaining:
        print("\nAll items already processed.")
        # Still generate Type 3 rows if needed
        _append_type3_feedback(problems, output_path, args)
        return

    print(f"  Items to process: {len(remaining)}")

    # Initialize
    backend = determine_backend(args.model, args.backend)
    print(f"\nModel: {args.model} | Backend: {backend}")
    inference = initialize_inference_engine(args, backend)
    prompt_builder = load_prompt("studentsim_generate_feedback")

    # Process in batches
    total = len(remaining)
    num_batches = math.ceil(total / args.batch_size)
    write_header = not (args.resume and output_path.exists() and completed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    success_count = 0
    start_time = time.time()

    print(f"\nGenerating feedback for {total} items in {num_batches} batches...")
    print("-" * 70)

    for batch_idx in range(num_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, total)
        batch = remaining[batch_start:batch_end]

        # Build conversations
        conversations = [
            prompt_builder(
                problem_text=w['problem_text'],
                student_answer=w['malrule_answer'],
                misconception_description=w['misconception_desc_used'],
                correct_answer=w['correct_answer'],
            )
            for w in batch
        ]

        # Generate
        batch_predictions = inference.generate(conversations)

        # Filter responses
        batch_predictions, batch_raw = filter_responses(batch_predictions, args.model)

        # Parse and save
        mode = 'w' if write_header and batch_idx == 0 else 'a'
        with open(output_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLUMNS)
            if mode == 'w':
                writer.writeheader()

            for i, item in enumerate(batch):
                raw = batch_raw[i] if i < len(batch_raw) else ""
                filtered = batch_predictions[i] if i < len(batch_predictions) else ""

                parsed = parse_json_response(filtered)
                feedback_text = ""
                parse_success = False

                if parsed and 'feedback' in parsed:
                    feedback_text = parsed['feedback']
                    parse_success = True

                writer.writerow({
                    'problem_id': item['problem_id'],
                    'category': item['category'],
                    'malrule_id': item['malrule_id'],
                    'malrule_name': item['malrule_name'],
                    'problem_text': item['problem_text'],
                    'correct_answer': item['correct_answer'],
                    'malrule_answer': item['malrule_answer'],
                    'feedback_type': item['feedback_type'],
                    'feedback_text': feedback_text,
                    'wrong_malrule_id': item.get('wrong_malrule_id', ''),
                    'misconception_desc_used': item['misconception_desc_used'],
                    'raw_response': raw,
                    'parse_success': str(parse_success),
                    'model_name': args.model,
                    'timestamp': timestamp,
                })
                if parse_success:
                    success_count += 1

        print_progress(batch_idx, num_batches, batch_end, total, start_time,
                       f"Parse OK: {success_count}")

    print("-" * 70)
    print(f"Completed in {time.time()-start_time:.1f}s")
    print(f"Parse success: {success_count}/{total} ({100*success_count/total:.1f}%)")

    # Append Type 3 (generic) rows — no LLM needed
    _append_type3_feedback(problems, output_path, args)

    print(f"Results saved to: {output_path}")


def _append_type3_feedback(problems: List[Dict], output_path: Path, args):
    """Append Type 3 (generic feedback) rows — no LLM call needed."""
    # Check which Type 3 rows already exist
    existing_type3 = set()
    if output_path.exists():
        setup_csv_field_limit()
        with open(output_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('feedback_type') == 'type3_generic':
                    existing_type3.add(row['problem_id'])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_count = 0

    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLUMNS)

        for prob in problems:
            pid = prob['problem_id']
            if pid in existing_type3:
                continue
            writer.writerow({
                'problem_id': pid,
                'category': prob['category'],
                'malrule_id': prob['malrule_id'],
                'malrule_name': prob['malrule_name'],
                'problem_text': prob['problem_text'],
                'correct_answer': prob['correct_answer'],
                'malrule_answer': prob['malrule_answer'],
                'feedback_type': 'type3_generic',
                'feedback_text': GENERIC_FEEDBACK,
                'wrong_malrule_id': '',
                'misconception_desc_used': '',
                'raw_response': '',
                'parse_success': 'True',
                'model_name': '',
                'timestamp': timestamp,
            })
            new_count += 1

    if new_count > 0:
        print(f"  Added {new_count} Type 3 (generic) feedback rows")


# ─── Step 2: Student Simulation ──────────────────────────────────────

def run_simulate_step(args):
    """Step 2: Simulate student responses to all 3 feedback types."""
    print("=" * 70)
    print("STEP 2: STUDENT SIMULATION")
    print(f"Dataset: {args.dataset}")
    print("Simulate student responses to teacher feedback")
    print("=" * 70)

    feedback_file = (Path(args.feedback_file) if args.feedback_file
                     else get_feedback_file(args.dataset))
    if not feedback_file.exists():
        print(f"\nError: Feedback file not found: {feedback_file}")
        print("  Run --step feedback first (or check --feedback-file path)")
        sys.exit(1)
    if args.feedback_file:
        print(f"  Feedback file (override): {feedback_file}")

    prompt_variant = args.simulate_prompt
    model_tag = args.model_tag
    output_path = get_responses_file(args.dataset, prompt_variant, model_tag, args.multi_turn)
    response_columns = RESPONSE_COLUMNS_MULTITURN if args.multi_turn else RESPONSE_COLUMNS
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Prompt variant: {prompt_variant}")
    if model_tag:
        print(f"  Model tag: {model_tag}")
    if args.multi_turn:
        print("  Output directory: results_multiturn_holistic")
    if output_path.exists() and not args.resume:
        print(f"\nError: Output file exists: {output_path}")
        print("  Use --resume or delete the file first")
        sys.exit(1)

    # Load feedback data
    setup_csv_field_limit()
    feedback_rows = []
    with open(feedback_file, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            # Skip rows where feedback generation failed
            if row.get('parse_success', 'False').lower() != 'true':
                continue
            if not row.get('feedback_text', '').strip():
                continue
            feedback_rows.append(row)

    print(f"\nLoaded {len(feedback_rows)} feedback items")

    # Oracle misconception lookup (used only by simulate_student_oracle prompt
    # variant; base/authentic prompts ignore this kwarg). Uses the description
    # from each problem's type1_correct feedback row, which describes the
    # student's TRUE misconception. Constant across the 3 feedback types.
    pid_to_oracle_desc = {
        r['problem_id']: r.get('misconception_desc_used', '') or ''
        for r in feedback_rows
        if r.get('feedback_type') == 'type1_correct'
    }

    by_type = defaultdict(int)
    for r in feedback_rows:
        by_type[r['feedback_type']] += 1
    for ft, count in sorted(by_type.items()):
        print(f"  {ft}: {count}")

    # Resume support
    completed = set()
    if args.resume and output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_header = next(csv.reader(f), [])
        missing_cols = [c for c in response_columns if c not in existing_header]
        if missing_cols:
            print(f"\nError: Existing responses file is missing required columns for this mode: {missing_cols}")
            print("  Run without --resume (new output file) or migrate the existing CSV schema first")
            sys.exit(1)
        completed = load_completed_ids(output_path, ['problem_id', 'feedback_type'])
        print(f"  Resuming: {len(completed)} items already done")

    remaining = [r for r in feedback_rows
                 if (r['problem_id'], r['feedback_type']) not in completed]
    if not remaining:
        print("\nAll items already processed.")
        return

    print(f"  Items to process: {len(remaining)}")

    # Initialize
    backend = determine_backend(args.model, args.backend)
    print(f"\nModel: {args.model} | Backend: {backend}")
    inference = initialize_inference_engine(args, backend)
    
    # Choose prompt builder based on multi-turn mode
    if args.multi_turn:
        print("  Mode: Multi-turn (initial answer + feedback response)")
        initial_prompt_builder = load_prompt("studentsim_simulate_student_initial_answer")
        response_prompt_builder = load_prompt("studentsim_simulate_student_multiturn_response")
    else:
        prompt_builder = load_prompt(f"studentsim_{prompt_variant}")

    # Process in batches
    total = len(remaining)
    num_batches = math.ceil(total / args.batch_size)
    write_header = not (args.resume and output_path.exists() and completed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    start_time = time.time()

    print(f"\nSimulating student responses for {total} items in {num_batches} batches...")
    print("-" * 70)

    for batch_idx in range(num_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, total)
        batch = remaining[batch_start:batch_end]
        initial_predictions = []
        initial_raw = []

        if args.multi_turn:
            # Multi-turn mode: First get initial answer, then response to feedback
            
            # Turn 1: Get initial student answer
            initial_conversations = [
                initial_prompt_builder(
                    problem_text=r['problem_text'],
                    student_answer=r['malrule_answer'],
                )
                for r in batch
            ]
            
            initial_predictions = inference.generate(initial_conversations)
            initial_predictions, initial_raw = filter_responses(initial_predictions, args.model)
            
            # Turn 2: Append teacher feedback to Turn 1 conversation history
            response_conversations = [
                response_prompt_builder(
                    initial_conversation=initial_conversations[i],
                    initial_response=initial_predictions[i] if i < len(initial_predictions) else "",
                    feedback_text=batch[i]['feedback_text'],
                )
                for i in range(len(batch))
            ]
            
            batch_predictions = inference.generate(response_conversations)
            batch_predictions, batch_raw = filter_responses(batch_predictions, args.model)
            
        else:
            # Single-turn mode: original behavior.
            # oracle_misconception kwarg is consumed by the oracle prompt
            # variant and ignored (via **kwargs) by the base/authentic ones.
            conversations = [
                prompt_builder(
                    problem_text=r['problem_text'],
                    student_answer=r['malrule_answer'],
                    feedback_text=r['feedback_text'],
                    oracle_misconception=pid_to_oracle_desc.get(r['problem_id'], ''),
                )
                for r in batch
            ]

            # Generate (free-form text, not JSON)
            batch_predictions = inference.generate(conversations)

            # Filter responses
            batch_predictions, batch_raw = filter_responses(batch_predictions, args.model)

        # Save
        mode = 'w' if write_header and batch_idx == 0 else 'a'
        with open(output_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=response_columns)
            if mode == 'w':
                writer.writeheader()

            for i, item in enumerate(batch):
                raw = batch_raw[i] if i < len(batch_raw) else ""
                filtered = batch_predictions[i] if i < len(batch_predictions) else ""

                row = {
                    'problem_id': item['problem_id'],
                    'category': item['category'],
                    'malrule_id': item['malrule_id'],
                    'malrule_name': item['malrule_name'],
                    'problem_text': item['problem_text'],
                    'correct_answer': item['correct_answer'],
                    'malrule_answer': item['malrule_answer'],
                    'feedback_type': item['feedback_type'],
                    'feedback_text': item['feedback_text'],
                    'wrong_malrule_id': item.get('wrong_malrule_id', ''),
                    'misconception_desc_used': item.get('misconception_desc_used', ''),
                    'student_response': filtered,
                    'raw_response': raw,
                    'model_name': args.model,
                    'timestamp': timestamp,
                }
                if args.multi_turn:
                    row['initial_student_response'] = (
                        initial_predictions[i] if i < len(initial_predictions) else ""
                    )
                    row['initial_raw_response'] = initial_raw[i] if i < len(initial_raw) else ""

                writer.writerow(row)

        print_progress(batch_idx, num_batches, batch_end, total, start_time)

    print("-" * 70)
    print(f"Completed in {time.time()-start_time:.1f}s")
    print(f"Results saved to: {output_path}")


# ─── Step 3: Judge Evaluation ────────────────────────────────────────

def run_judge_step(args):
    """Step 3: Judge student responses and print analysis."""
    print("=" * 70)
    print("STEP 3: JUDGE EVALUATION")
    print(f"Dataset: {args.dataset}")
    print("Classify student responses into 6-category taxonomy")
    print("=" * 70)

    prompt_variant = args.simulate_prompt
    model_tag = args.model_tag
    responses_file = get_responses_file(args.dataset, prompt_variant, model_tag, args.multi_turn)
    print(f"  Prompt variant: {prompt_variant}")
    if model_tag:
        print(f"  Model tag: {model_tag}")
    if args.multi_turn:
        print("  Input/output directory: results_multiturn_holistic")

    if not responses_file.exists():
        print(f"\nError: Responses file not found: {responses_file}")
        print("  Run --step simulate first")
        sys.exit(1)

    output_path = get_judgments_file(args.dataset, prompt_variant, model_tag, args.multi_turn)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.resume:
        print(f"\nError: Output file exists: {output_path}")
        print("  Use --resume or delete the file first")
        sys.exit(1)

    # Load response data
    setup_csv_field_limit()
    response_rows = []
    with open(responses_file, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if not row.get('student_response', '').strip():
                continue
            response_rows.append(row)

    print(f"\nLoaded {len(response_rows)} student responses")
    by_type = defaultdict(int)
    for r in response_rows:
        by_type[r['feedback_type']] += 1
    for ft, count in sorted(by_type.items()):
        print(f"  {ft}: {count}")

    # Resume support
    completed = set()
    if args.resume and output_path.exists():
        completed = load_completed_ids(output_path, ['problem_id', 'feedback_type'])
        print(f"  Resuming: {len(completed)} items already done")

    remaining = [r for r in response_rows
                 if (r['problem_id'], r['feedback_type']) not in completed]
    if not remaining:
        print("\nAll items already processed. Running analysis...")
        _print_analysis(output_path)
        return

    print(f"  Items to process: {len(remaining)}")

    # Initialize
    backend = determine_backend(args.model, args.backend)
    print(f"\nModel: {args.model} | Backend: {backend}")
    inference = initialize_inference_engine(args, backend)
    prompt_builder = load_prompt("studentsim_judge_response")

    # Process in batches
    total = len(remaining)
    num_batches = math.ceil(total / args.batch_size)
    write_header = not (args.resume and output_path.exists() and completed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    success_count = 0
    start_time = time.time()

    print(f"\nJudging {total} responses in {num_batches} batches...")
    print("-" * 70)

    for batch_idx in range(num_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, total)
        batch = remaining[batch_start:batch_end]

        # Build conversations
        conversations = [
            prompt_builder(
                problem_text=r['problem_text'],
                student_answer=r['malrule_answer'],
                correct_answer=r['correct_answer'],
                feedback_text=r['feedback_text'],
                student_response=r['student_response'],
            )
            for r in batch
        ]

        # Generate
        batch_predictions = inference.generate(conversations)

        # Filter responses
        batch_predictions, batch_raw = filter_responses(batch_predictions, args.model)

        # Parse and save
        mode = 'w' if write_header and batch_idx == 0 else 'a'
        with open(output_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=JUDGMENT_COLUMNS)
            if mode == 'w':
                writer.writeheader()

            for i, item in enumerate(batch):
                raw = batch_raw[i] if i < len(batch_raw) else ""
                filtered = batch_predictions[i] if i < len(batch_predictions) else ""

                parsed = parse_json_response(filtered)
                judge_category = ""
                judge_final_answer = ""
                judge_reasoning = ""
                parse_success = False

                if parsed:
                    cat = parsed.get('category', '')
                    if cat in VALID_CATEGORIES:
                        judge_category = cat
                        parse_success = True
                    judge_final_answer = str(parsed.get('final_answer', '') or '')
                    judge_reasoning = parsed.get('reasoning', '')

                writer.writerow({
                    'problem_id': item['problem_id'],
                    'category': item['category'],
                    'malrule_id': item['malrule_id'],
                    'malrule_name': item['malrule_name'],
                    'problem_text': item['problem_text'],
                    'correct_answer': item['correct_answer'],
                    'malrule_answer': item['malrule_answer'],
                    'feedback_type': item['feedback_type'],
                    'feedback_text': item['feedback_text'],
                    'wrong_malrule_id': item.get('wrong_malrule_id', ''),
                    'misconception_desc_used': item.get('misconception_desc_used', ''),
                    'student_response': item['student_response'],
                    'judge_category': judge_category,
                    'judge_final_answer': judge_final_answer,
                    'judge_reasoning': judge_reasoning,
                    'judge_raw_response': raw,
                    'judge_parse_success': str(parse_success),
                    'model_name': args.model,
                    'timestamp': timestamp,
                })
                if parse_success:
                    success_count += 1

        print_progress(batch_idx, num_batches, batch_end, total, start_time,
                       f"Parse OK: {success_count}")

    print("-" * 70)
    print(f"Completed in {time.time()-start_time:.1f}s")
    print(f"Parse success: {success_count}/{total} ({100*success_count/total:.1f}%)")
    print(f"Results saved to: {output_path}")

    # Print analysis
    _print_analysis(output_path)


def _print_analysis(judgments_path: Path):
    """Print the 6x3 distribution table and key metrics."""
    setup_csv_field_limit()

    rows = []
    with open(judgments_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('judge_parse_success', 'False').lower() == 'true':
                rows.append(row)

    if not rows:
        print("\nNo successfully parsed judgments to analyze.")
        return

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Count by (feedback_type, category)
    counts = defaultdict(lambda: defaultdict(int))
    type_totals = defaultdict(int)
    for r in rows:
        ft = r['feedback_type']
        jc = r['judge_category']
        counts[ft][jc] += 1
        type_totals[ft] += 1

    # Print 6x3 table
    feedback_types = ['type1_correct', 'type2_wrong', 'type3_generic']
    categories = ['correct_flip', 'sycophantic_flip', 'different_wrong',
                  'constructive_pushback', 'passive_maintain', 'confusion']

    print("\n--- 6-Category Distribution (counts) ---\n")
    header = f"{'Category':<25}"
    for ft in feedback_types:
        short = ft.replace('type1_', 'T1:').replace('type2_', 'T2:').replace('type3_', 'T3:')
        header += f"  {short:>12}"
    print(header)
    print("-" * (25 + 14 * 3))

    for cat in categories:
        line = f"{cat:<25}"
        for ft in feedback_types:
            c = counts[ft][cat]
            t = type_totals[ft]
            pct = f"{100*c/t:.1f}%" if t > 0 else "N/A"
            line += f"  {c:>5} ({pct:>5})"
        print(line)

    # Totals
    print("-" * (25 + 14 * 3))
    line = f"{'TOTAL':<25}"
    for ft in feedback_types:
        line += f"  {type_totals[ft]:>5}       "
    print(line)

    # Flip rates
    print("\n--- Flip Rates ---\n")
    for ft in feedback_types:
        t = type_totals[ft]
        if t == 0:
            continue
        correct_flips = counts[ft]['correct_flip']
        sycophantic_flips = counts[ft]['sycophantic_flip']
        total_flips = correct_flips + sycophantic_flips
        flip_rate = total_flips / t
        short = ft.replace('type1_', 'T1:').replace('type2_', 'T2:').replace('type3_', 'T3:')
        print(f"  {short:<15} flip_rate = {flip_rate:.3f} "
              f"({total_flips}/{t}, correct={correct_flips}, sycophantic={sycophantic_flips})")

    # SFS (paper metric): F_T - (F_M + F_G) / 2
    fr = read_flip_rates(judgments_path)
    S = sfs(fr)
    print(f"\n--- SFS (Selective Flip Score) = F_T - (F_M + F_G) / 2 = {S:+.3f} ---")


# ─── Step 4: Summarize ───────────────────────────────────────────────

def _parse_judgments_filename(filename: str) -> Tuple[str, str]:
    """
    Parse a judgments filename to extract (prompt_variant, model_tag).
    Examples:
        judgments.csv -> ('simulate_student', '')
        judgments_simulate_student_authentic.csv -> ('simulate_student_reflective', '')
        judgments_qwen80b.csv -> ('simulate_student', 'qwen80b')
        judgments_simulate_student_authentic_qwen80b.csv -> ('simulate_student_reflective', 'qwen80b')
    """
    stem = Path(filename).stem  # e.g. 'judgments_simulate_student_authentic_qwen80b'
    suffix = stem.replace('judgments', '', 1)  # e.g. '_simulate_student_authentic_qwen80b'
    suffix = suffix.lstrip('_')

    if not suffix:
        return 'simulate_student', ''

    # Known prompt variants (check longer first)
    # Maps filename pattern -> output name (authentic renamed to reflective)
    known_variants = [
        ('simulate_student_authentic', 'simulate_student_reflective'),
        ('simulate_student', 'simulate_student'),
    ]
    for file_variant, output_variant in known_variants:
        if suffix == file_variant:
            return output_variant, ''
        if suffix.startswith(file_variant + '_'):
            tag = suffix[len(file_variant) + 1:]
            return output_variant, tag

    # No known variant prefix — must be just a model tag
    return 'simulate_student', suffix


def _model_display_name(judgments_path: Path) -> str:
    """Extract a display name for the student model from a judgments file."""
    setup_csv_field_limit()
    with open(judgments_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
        if row:
            return row.get('model_name', 'unknown')
    return 'unknown'


def _summarize_dataset(dataset: str, multi_turn: bool = False) -> List[dict]:
    """Collect summary rows for a single dataset."""
    results_dir = get_results_dir(dataset, multi_turn=multi_turn)
    pattern = str(results_dir / "judgments*.csv")
    judgment_files = sorted(glob_module.glob(pattern))

    if not judgment_files:
        print(f"  No judgment files found for dataset '{dataset}' in {results_dir}")
        return []

    print(f"\n  Dataset '{dataset}': {len(judgment_files)} judgment files")
    for f in judgment_files:
        print(f"    {Path(f).name}")

    feedback_types = ['type1_correct', 'type2_wrong', 'type3_generic']
    category_names = ['correct_flip', 'sycophantic_flip', 'different_wrong',
                      'constructive_pushback', 'passive_maintain', 'confusion']
    summary_rows = []

    for jf_path in judgment_files:
        jf = Path(jf_path)
        prompt_variant, model_tag = _parse_judgments_filename(jf.name)

        # Get the student model name from the responses file (not the judge model)
        resp_file = get_responses_file(dataset, prompt_variant, model_tag, multi_turn=multi_turn)
        if resp_file.exists():
            student_model = _model_display_name(resp_file)
        else:
            student_model = model_tag or 'unknown'

        # Shorten model name for display
        student_display = clean_model_name(student_model) if '/' in student_model else student_model

        # Load judgments
        setup_csv_field_limit()
        counts = defaultdict(lambda: defaultdict(int))
        type_totals = defaultdict(int)

        with open(jf, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('judge_parse_success', 'False').lower() != 'true':
                    continue
                ft = row['feedback_type']
                jc = row['judge_category']
                counts[ft][jc] += 1
                type_totals[ft] += 1

        for ft in feedback_types:
            t = type_totals.get(ft, 0)
            if t == 0:
                continue
            flips = counts[ft]['correct_flip'] + counts[ft]['sycophantic_flip']
            flip_rate = flips / t if t > 0 else 0

            row_data = {
                'dataset': dataset,
                'student_model': student_display,
                'prompt_variant': prompt_variant,
                'feedback_type': ft,
            }
            for cat in category_names:
                row_data[cat] = counts[ft][cat]
            row_data['total'] = t
            row_data['flip_rate'] = round(flip_rate, 4)
            summary_rows.append(row_data)

    return summary_rows


def run_summarize_step(args):
    """Auto-discover all judgment files and produce summary.csv."""
    print("=" * 70)
    print("STEP: SUMMARIZE")
    print(f"Dataset: {args.dataset}")
    print("Auto-discover judgment files and produce summary.csv")
    print("=" * 70)

    category_names = ['correct_flip', 'sycophantic_flip', 'different_wrong',
                      'constructive_pushback', 'passive_maintain', 'confusion']

    # Collect rows — either one dataset or both
    if args.dataset == 'all':
        datasets = ['malrule', 'eedi']
    else:
        datasets = [args.dataset]

    summary_rows = []
    for ds in datasets:
        summary_rows.extend(_summarize_dataset(ds, multi_turn=args.multi_turn))

    if not summary_rows:
        print("\nNo judgment files found.")
        sys.exit(1)

    # Write summary CSV
    if args.dataset == 'all':
        summary_base = DATA_DIR / ("results_multiturn_holistic" if args.multi_turn else "results")
        summary_path = summary_base / "summary.csv"
    else:
        summary_path = get_results_dir(args.dataset, multi_turn=args.multi_turn) / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_columns = ['dataset', 'student_model', 'prompt_variant', 'feedback_type'] + \
                      category_names + ['total', 'flip_rate']

    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=summary_columns)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"\nSummary written to: {summary_path}")
    print(f"  {len(summary_rows)} rows")

    # Print SFS comparison: SFS = F_T - (F_M + F_G) / 2
    print(f"\n--- SFS Comparison ---\n")
    combos = {}
    for row in summary_rows:
        key = (row['dataset'], row['student_model'], row['prompt_variant'])
        if key not in combos:
            combos[key] = {}
        combos[key][row['feedback_type']] = row['flip_rate']

    print(f"{'Dataset':<10} {'Student Model':<25} {'Prompt':<30} {'F_T':>8} {'F_M':>8} {'F_G':>8} {'SFS':>11}")
    print("-" * 108)
    for (ds, model, variant), rates in sorted(combos.items()):
        F_T = rates.get('type1_correct', 0)
        F_M = rates.get('type2_wrong', 0)
        F_G = rates.get('type3_generic', 0)
        S = F_T - (F_M + F_G) / 2
        print(f"{ds:<10} {model:<25} {variant:<30} {F_T:>8.4f} {F_M:>8.4f} {F_G:>8.4f} {S:>+11.4f}")


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_arguments()

    if args.dataset == 'all' and args.step != 'summarize':
        print("Error: --dataset all is only supported with --step summarize")
        sys.exit(1)

    if args.step == "feedback":
        run_feedback_step(args)
    elif args.step == "simulate":
        run_simulate_step(args)
    elif args.step == "judge":
        run_judge_step(args)
    elif args.step == "summarize":
        run_summarize_step(args)
    else:
        print(f"Unknown step: {args.step}")
        sys.exit(1)


if __name__ == "__main__":
    main()
