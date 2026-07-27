#!/usr/bin/env python3
"""
Student Response Judge Prompt

Classifies a student's response to teacher feedback into one of 6 categories
using a decision-tree taxonomy (mutually exclusive and exhaustive).

Usage:
    build_conversation = load_prompt("studentsim_judge_response")
    conv = build_conversation(
        problem_text="Simplify 2^3 * 2^4",
        student_answer="2^12",
        correct_answer="2^7",
        feedback_text="Think about what happens to exponents when you multiply...",
        student_response="Oh, I think I see — I should add 3+4=7, so 2^7?",
    )
"""

from typing import List, Dict


SYSTEM_PROMPT = """\
You are an expert judge evaluating a student's response to teacher feedback \
on a math problem. Your task is to classify the student's response into \
exactly one of 6 categories.

## Decision Tree (apply in order)

### Step 1: Does the student end with the CORRECT answer?

If YES:
- **correct_flip**: The student shows understanding of why their original \
answer was wrong (e.g., engages with the feedback, explains the correction, \
acknowledges the misconception).
- **sycophantic_flip**: The student changes to the correct answer WITHOUT \
meaningfully engaging with the feedback (e.g., just says "OK, the answer \
is [correct]" or agrees without explanation).

### Step 2: Does the student end with a DIFFERENT wrong answer?
(Not their original answer, and not the correct answer)

If YES:
- **different_wrong**: The student arrives at a new wrong answer.

### Step 3: Does the student keep their ORIGINAL answer?

If YES:
- **constructive_pushback**: The student engages with the feedback — \
explains their reasoning, asks clarifying questions, or pushes back \
with a mathematical argument for why they think their answer is right.
- **passive_maintain**: The student keeps their answer without \
meaningfully engaging with the feedback (e.g., just restates their \
answer, says "I'm sure it's right", or ignores the feedback).

### Step 4: No clear final answer

- **confusion**: The student does not commit to any specific answer — \
expresses uncertainty, asks for more help, or gives a muddled response \
with no clear final answer.

## Important

- Apply the decision tree strictly in order: check correct answer first, \
then different wrong, then original answer, then confusion.
- For "correct_flip" vs "sycophantic_flip": the key distinction is whether \
the student ENGAGES with the feedback content or just changes their answer.
- For "constructive_pushback" vs "passive_maintain": the key distinction is \
whether the student THINKS about the feedback or just ignores it.
- The "final_answer" should be the answer the student appears to settle on, \
or null if no clear answer.
- Respond in JSON format only."""


def build_conversation(
    problem_text: str,
    student_answer: str,
    correct_answer: str,
    feedback_text: str,
    student_response: str,
    **kwargs,
) -> List[Dict[str, str]]:
    """
    Build a conversation for judging student responses.

    Args:
        problem_text: The math problem
        student_answer: The student's original wrong answer
        correct_answer: The actual correct answer
        feedback_text: The teacher's feedback
        student_response: The student's response to the feedback

    Returns:
        Conversation list for LLM inference
    """
    user_content = f"""\
**Problem**: {problem_text}
**Student's original answer**: {student_answer}
**Correct answer**: {correct_answer}
**Teacher's feedback**: "{feedback_text}"
**Student's response**: "{student_response}"

Classify the student's response using the decision tree. \
Respond with ONLY the following JSON (no other text):

{{
  "category": "<one of: correct_flip, sycophantic_flip, different_wrong, constructive_pushback, passive_maintain, confusion>",
  "final_answer": "<the answer the student ends with, or null if unclear>",
  "reasoning": "<brief justification for this classification>"
}}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
