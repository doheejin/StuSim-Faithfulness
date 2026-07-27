#!/usr/bin/env python3
"""
Teacher Feedback Generation Prompt

Generates targeted teacher feedback that addresses a specific misconception,
for use in the student simulation experiment.

Usage:
    build_conversation = load_prompt("studentsim_generate_feedback")
    conv = build_conversation(
        problem_text="Simplify 2^3 * 2^4",
        student_answer="2^12",
        misconception_description="Student multiplies exponents instead of adding them",
        correct_answer="2^7",
    )
"""

from typing import List, Dict


SYSTEM_PROMPT = """\
You are a math teacher. A student made an error on a math problem. \
Generate a short piece of feedback (1-3 sentences) that addresses the \
misconception described below.

## Important rules

- Do NOT reveal the correct answer
- Do NOT solve the problem for the student
- Focus on helping the student understand WHY their approach was wrong
- Reference the specific misconception in your feedback
- Be encouraging but direct
- Respond in JSON format only"""


def build_conversation(
    problem_text: str,
    student_answer: str,
    misconception_description: str,
    correct_answer: str,
    **kwargs,
) -> List[Dict[str, str]]:
    """
    Build a conversation for teacher feedback generation.

    Args:
        problem_text: The math problem
        student_answer: The student's wrong answer
        misconception_description: Description of the misconception to address
        correct_answer: The correct answer (for context, NOT to be revealed)

    Returns:
        Conversation list for LLM inference
    """
    user_content = f"""\
**Problem**: {problem_text}
**Student's answer**: {student_answer}
**Correct answer**: {correct_answer}
**Misconception to address**: {misconception_description}

Generate a short piece of teacher feedback that addresses this misconception \
without revealing the correct answer. \
Respond with ONLY the following JSON (no other text):

{{
  "feedback": "<your feedback to the student>"
}}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
